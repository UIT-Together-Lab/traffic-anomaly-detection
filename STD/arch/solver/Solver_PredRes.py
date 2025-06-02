import os
import gc
import torch.nn as nn
import torch
import math
import numpy as np
from numpy import log10
import tqdm
import torch.optim as optim
import scipy.io as scio
from sklearn.cluster import KMeans

from arch.module.eval_utils import psnr,batch_psnr, l1_metric, l2_metric , min_max_np, calcu_result, reciprocal_metric, log_metric, pairwise_l2_metric,pixel_wise_l2_metric,maxpatch_metric,loss_map, calcu_auc
from arch.module.ResUNet import UResAE
from arch.module.ResNet import ResAE
from arch.module.Basic import BlurFunc
from arch.module.Basic import video_static_motion, video_split_static_and_motion_seq
from arch.model.PredRes_Model import PredRes_AE_Cluster_Model
from arch.module.loss_utils import gradient_loss, gradient_metric
import cv2

class Solver():
    def __init__(self, config, cluster_model=PredRes_AE_Cluster_Model,device_idx=0, model_type='res', checkpoint_path=None):
        self.log_dir = config.log_path
        os.makedirs(self.log_dir, exist_ok=True)
        self.checkpoint_path = checkpoint_path

        self.para_tag = False
        self.device_ids = [0]
        self.eval_device_idx = [0]
        self.device = torch.device("cuda:%d"%device_idx if torch.cuda.is_available() else "cpu")

        self.img_channel = config.img_channel
        self.frames_num = config.clips_length
        self.cluster_num = config.cluster_num
        self.seq_tag = False
        if self.seq_tag:
            motion_channel_out = self.img_channel*( self.frames_num - 1)
        else:
            motion_channel_out=self.img_channel
        
        self.model = cluster_model( 
            static_channel_in=self.img_channel,
            static_channel_out=self.img_channel, 
            static_layer_struct = config.static_layer_struct,
            static_layer_nums= config.static_layer_nums,
            motion_channel_in=self.img_channel*( self.frames_num - 1), 
            motion_channel_out=motion_channel_out, 
            motion_layer_struct= config.motion_layer_struct,
            motion_layer_nums= config.motion_layer_nums,
            img_channel=self.img_channel, 
            frame_nums=self.frames_num, 
            cluster_num=self.cluster_num, 
            blur_ratio=1, 
            seq_tag=self.seq_tag, 
            model_type=model_type ).to(self.device)
        

        self.init_info()

        self.l2_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

        self.optimizer_cluster = optim.Adam( self.model.cluster_par , lr=1e-6)
        self.optimizer = optim.Adam( self.model.ae_par , lr=1e-5)
        self.optimizer_static = optim.Adam( self.model.static_par ,  lr= 5e-5 )
        self.optimizer_motion = optim.Adam( self.model.motion_par ,  lr= 5e-5  )
        
    def train_batch_AE(self, batch_in, alpha=None, loss_appendix=0):
        self.model.train()
        self.model.zero_grad()

        batch_in = batch_in.to(self.device)

        loss_deblur, loss_predict, loss_recon, grad_deblur, grad_predict, grad_recon  = self.model( batch_in,alpha,['F'])

        psnr_predict = 10*log10(1 / loss_predict.mean().item() )
        psnr_deblur =  10*log10(1 / loss_deblur.mean().item() )
        psnr_recon =  10*log10(1 / loss_recon.mean().item() )

        loss = (loss_deblur+loss_predict+0.01*grad_predict)

        loss.mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.info['psnr_predict'].append(psnr_predict)
        self.info['psnr_deblur'].append(psnr_deblur)
        self.info['psnr_recon'].append(psnr_recon)
        self.info['total_loss'].append(loss.mean().item())
        return 

    def train_batch_Static(self, batch_in, alpha=None, loss_appendix=0):
        self.model.train()
        self.model.zero_grad()

        batch_in = batch_in.to(self.device)
        loss_deblur,grad_deblur  = self.model( batch_in,alpha,['S','G'])


        psnr_deblur =  10*log10(1 / loss_deblur.mean().item() )

        loss = loss_deblur 
        loss.mean().backward()
        self.optimizer_static.step()
        self.optimizer_static.zero_grad()

        self.info['total_loss'].append(loss.mean().item())
        self.info['psnr_deblur'].append(psnr_deblur)
        return 
    
    def train_batch_Motion(self, batch_in, alpha=None, loss_appendix=0):
        self.model.train()
        self.model.zero_grad()


        batch_in = batch_in.to(self.device)
        loss_predict,grad_predict  = self.model( batch_in,alpha,['M','G'])

        psnr_predict = 10*log10(1 / loss_predict.mean().item() )
        
        loss = loss_predict + grad_predict
        loss.mean().backward()
        self.optimizer_motion.step()
        self.optimizer_motion.zero_grad()

        del batch_in
        gc.collect()

        self.info['psnr_predict'].append(psnr_predict)
        self.info['total_loss'].append(loss.mean().item())
        return 
 
    def train_batch_Cluster(self, batch_in, alpha=None, loss_appendix=0):
        self.model.train()
        self.model.zero_grad()


        batch_in = batch_in.to(self.device)
        loss_cluster, rep_dist , cat_encoder  = self.model( batch_in,alpha,['S','M','C'])

        loss = 0.1*loss_cluster
        loss.sum().backward()
        self.optimizer_cluster.step()
        self.optimizer_cluster.zero_grad()

        self.info['cluster_loss'].append(loss_cluster.mean().item())
        return 
    
    def init_Cluster(self, training_iter, alpha=None, emmbeding_length=100):
        
        print('start initial cluster centers.......')
        embeddings_bank = []
        for iter_idx in range(emmbeding_length):
            batch_in = next(training_iter)
            self.model.train()
            self.model.zero_grad()
            # if self.para_tag:
            #     static_rep  = nn.parallel.data_parallel(self.model, (batch_in, alpha,['S','M','ini']), device_ids = self.device_ids )
            # else:
            batch_in = batch_in.to(self.device)
            static_rep  = self.model(batch_in, alpha, ['S','M','ini'])
            
            embeddings_bank.append(static_rep.detach().cpu().numpy())   
        
        embeddings_bank = np.concatenate(embeddings_bank, 0)
        kmeans_model = KMeans(n_clusters=self.cluster_num, init="k-means++").fit(embeddings_bank)
        self.model.cluster.cluster_center.data = torch.from_numpy(kmeans_model.cluster_centers_).cuda()
        
        return 
    
    def training_info(self, detail_info):

        for info_keys in self.info.keys():
            if not self.info[info_keys] == []:
                detail_info += ' \t {} : {:.5f} '.format( info_keys, np.stack(self.info[info_keys]).mean() )

        detail_info+='\n'
        

        self.init_info()
        print(detail_info)
        with open( os.path.join( self.log_dir , 'training_log.txt' ) ,'a+') as f:
            f.writelines(detail_info)

        return

    def eval_datasets(self, dataloader, labels_list, epoch=0, list_image_names=None, data_name='data_name', save_name=''):
        self.model.cuda()
        self.model.eval()
        eval_metric_dict = {}
        eval_metric_dict['inv_recon'] = []

        batch_filenames = []

        with torch.no_grad():            
            for batch_idx  in tqdm.tqdm(range(dataloader.fetch_nums)):
                batch_in, list_image_names = dataloader.fetch()
                target_filenames = np.array(list_image_names)[:,-1]

                for filename in target_filenames:
                    batch_filenames.append(filename)
                
                batch_in = batch_in.to(self.device)

                if self.seq_tag:
                    static_in, motion_in, static_target, motion_target = video_split_static_and_motion_seq(batch_in, self.img_channel, self.frames_num )
                else:
                    static_in, motion_in, static_target, motion_target = video_static_motion(batch_in, self.img_channel, self.frames_num )

                self.model.zero_grad()
                if self.para_tag:
                    static_decoder, pred_decoder, loss_cluster_map  = nn.parallel.data_parallel(self.model, (batch_in, None, ['E']), device_ids = self.device_ids )
                else:
                    batch_in = batch_in.to(self.device)
                    static_decoder, pred_decoder, loss_cluster_map  = self.model(batch_in, None, ['E'])

                if self.seq_tag:
                    static_in  =  static_in.repeat(1, self.frames_num-1, 1, 1)
                    static_decoder = static_decoder.repeat(1, self.frames_num-1, 1, 1)
                
                pred_recon = pred_decoder + static_decoder
                pred_target = static_in + motion_target
                loss_pixelwise_cl_re = loss_map( torch.mean((pred_recon-pred_target)**2, [1], keepdim=True) , loss_cluster_map)

                reciprocal_losses = reciprocal_metric( loss_pixelwise_cl_re )
                eval_metric_dict['inv_recon'].append( reciprocal_losses ) 

        auc_list = []
        for eval_keys in eval_metric_dict.keys():            
            eval_metric = np.concatenate(eval_metric_dict[eval_keys])
            #  np.save('results_' + data_name + '.npy', eval_metric, allow_pickle=True)
            # np.save('labels_' + data_name + '.npy', np.array(labels_list), allow_pickle=True)

            # Thêm save_name để tuỳ chỉnh label lưu kết quả
            np.save('results_' + data_name + save_name + '.npy', eval_metric, allow_pickle=True)
            np.save('labels_' + data_name + save_name + '.npy', np.array(labels_list), allow_pickle=True)
            auc = calcu_result(eval_metric, labels_list, converse=False)

        eval_metric_dict['labels'] = labels_list
        detail_info = 'Epoches {} \t  auc {:.5f} \n '.format(epoch, auc)

        print(detail_info)

        print("Visualizing ...")
        norm_result = np.load('./norm_result.npy', allow_pickle=True)
        path_output = './visualized_outputs_' + data_name + '/'
        if not os.path.exists(path_output):
            os.mkdir(path_output)

        for img_path, result, label in tqdm.tqdm(zip(batch_filenames, norm_result, np.concatenate(labels_list))):
            image = cv2.imread(img_path)
            if result > 0.5:
                cv2.putText(image, 'Result: ' + str(round(result, 4)) + ' - Label: ' + str(label), (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(image, 'Result: ' + str(round(result, 4)) + ' - Label: ' + str(label), (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print(f"PATH {os.path.join(path_output,batch_filenames,  img_path.split('/')[-1])}")
            cv2.imwrite(os.path.join(path_output,batch_filenames,  img_path.split('/')[-1]), image)
            
        return
    
    def init_info(self):
        self.info = {}
        self.info['total_loss'] = []
        self.info['cluster_loss'] = []
        self.info['psnr_deblur'] = []
        self.info['psnr_predict'] = []
        self.info['psnr_recon'] = []
        return
    
    def load_model(self,):
        load_path = os.path.join( self.checkpoint_path )
        state = torch.load( load_path )
        self.model.load_state_dict(state['state_dict'])
        print("Checkpoint loaded from {}".format( load_path ) )
        return 
    
    def save_model(self , epoch):
        state = {}
        state['epoches'] = epoch
        state['state_dict'] = self.model.state_dict()
        save_path = os.path.join( self.log_dir, 'E{}_model.pth'.format(epoch) )         
        torch.save( state, save_path )
        print("Checkpoint saved to {}".format( save_path ) )
        return
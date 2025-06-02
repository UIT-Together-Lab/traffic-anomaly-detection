import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
from models.unet import UNet
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate import val

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--show_curve', action='store_true',
                    help='Show and save the psnr curve real-timely, this drops fps.')
parser.add_argument('--show_heatmap', action='store_true',
                    help='Show and save the difference heatmap real-timely, this drops fps.')

args = parser.parse_args()
train_cfg = update_config(args, mode='test')


train_cfg.test_data = '/home/abnormal_detection/VuNgocTu/dataset/avenue/testing/frames/'
train_cfg.print_cfg()
train_cfg.flownet = '2sd'#, 'lite'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = lite_flow.Network()
    flow_net.load_state_dict(torch.load('models/liteFlownet/network-default.pytorch'))

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.




video_folders = os.listdir(train_cfg.test_data)
video_folders.sort()
video_folders = [os.path.join(train_cfg.test_data, aa) for aa in video_folders]

with torch.no_grad():
    for i, folder in enumerate(video_folders):
      name = folder.split('/')[-1]
      fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

      dataset = Dataset.test_dataset(train_cfg, folder)
      motion_writer = cv2.VideoWriter(f'results_motion_testing/{name}_video.avi', fourcc, 30, train_cfg.img_size)
      os.makedirs("./flow_motion_train/"+name,exist_ok=True)
      for j, clips in enumerate(dataset):
          input_frames = clips[0:12, :, :] # (n, 12, 256, 256)
          target_frame = clips[12:15, :, :] # (n, 3, 256, 256)
          input_last = input_frames[9:12, :, :]  # use for flow_loss
          
          input_frames = torch.from_numpy(input_frames).unsqueeze(0).cuda()
          target_frame = torch.from_numpy(target_frame).unsqueeze(0).cuda()
          input_last = torch.from_numpy(input_last).unsqueeze(0).cuda()
          if train_cfg.flownet == 'lite':
              gt_flow_input = torch.cat([input_last, target_frame], 1)
              #pred_flow_input = torch.cat([input_last, G_frame], 1)
              # No need to train flow_net, use .detach() to cut off gradients.
              flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net).detach()
              #flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
          else:
              gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
              #pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)
  
              flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
              #flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()
  
         
          flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
          #print(flow.shape[0])
          for i in range(flow.shape[0]):
              aa = flow_to_color(flow[i], convert_to_bgr=False)
              path = train_cfg.test_data.split('/')[-3] + '_' + str(j)+".jpg"#+flow_strs[i]
              #motion_writer.write(aa)
              cv2.imwrite(f'./flow_motion_test/{name}/{path}.jpg',aa)  # e.g. images/avenue_4_574-575.jpg
              #print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')
                  
          torch.cuda.synchronize()

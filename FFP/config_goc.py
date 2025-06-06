#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from glob import glob
import os

if not os.path.exists('tensorboard_log'):
    os.mkdir('tensorboard_log')
if not os.path.exists('weights_adrone_foggy1'):
    os.mkdir('weights_adrone_foggy1')
if not os.path.exists('results'):
    os.mkdir('results')

share_config = {'mode': 'training',
                'dataset': './UIT_ADrone_Foggy',
                'img_size': (256, 256),
                'data_root': '../'}  # remember the final '/'


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    # assert args.dataset in ('ped2', 'avenue', 'AD_dataset_2022tech', 'AD_dataset_2022'), 'Dataset error.'
    share_config['dataset'] = args.dataset

    if mode == 'train':
        share_config['batch_size'] = args.batch_size
        share_config['train_data'] = share_config['data_root'] + args.dataset + '/train/frames_foggy'
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/test/frames_foggy/'
        share_config['g_lr'] = 0.0002
        share_config['d_lr'] = 0.00002
        share_config['resume'] = glob(f'weights_adrone_foggy1/{args.resume}*')[0] if args.resume else None
        share_config['iters'] = args.iters
        share_config['show_flow'] = args.show_flow
        share_config['save_interval'] = args.save_interval
        share_config['val_interval'] = args.val_interval
        share_config['flownet'] = args.flownet

    elif mode == 'test':
        share_config['test_data'] = share_config['data_root'] + args.dataset + '/test/frames_foggy/'
        share_config['trained_model'] = args.trained_model
        share_config['show_curve'] = args.show_curve
        share_config['show_heatmap'] = args.show_heatmap

    return dict2class(share_config)  # change dict keys to class attributes

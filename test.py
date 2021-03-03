# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import cython

import dataset
import models
from models.omnipose import OmniPose
from models.omnipose import get_Canny_HRNet
from models.frankenstein import get_frankenstein
from models.pose_omni import get_pose_net


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--opts', help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args


def main(args):
    # args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    # model = eval('models.pose_hrnet.get_pose_net')(cfg, is_train=False)
    model = eval('models.pose_omni.get_pose_net')(cfg, is_train=False)
    # model = OmniPose(cfg.MODEL.NUM_JOINTS)
    # model = get_Canny_HRNet(48, cfg.MODEL.NUM_JOINTS, 0.1)
    # model = get_frankenstein(cfg, False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading checkpoint '{}'".format(cfg.TEST.MODEL_FILE))
        checkpoint = torch.load(cfg.TEST.MODEL_FILE)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        # print(checkpoint.keys())

        model_state_dict = model.state_dict()
        new_model_state_dict = {}

        # for k in model_state_dict:
        #     print(k)
        # quit()
        
        for k in checkpoint['state_dict']:
            # print(k)
            if k in model_state_dict and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                new_model_state_dict[k] = checkpoint['state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(checkpoint, strict=False)

        print('begin_epoch', begin_epoch)
        print('best_perf', best_perf)
        print('last_epoch',last_epoch)

        # model_state_dict.update(new_model_state_dict)
        # model.load_state_dict(model_state_dict)

        model.load_state_dict(new_model_state_dict, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        model_state_file = 'models/coco/w48_384×288.pth'
        logger.info('=> loading model from {}'.format(model_state_file))
        # model.load_state_dict(torch.load(model_state_file))

        model_state_dict = torch.load(model_state_file)
        new_model_state_dict = {}
        for k in model_state_dict:
            if k in model_state_dict and model_state_dict[k].size() == model_state_dict[k].size():
                new_model_state_dict[k] = model_state_dict[k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_model_state_dict)

        # model.load_state_dict(checkclpoint['state_dict'])

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion,
             final_output_dir, tb_log_dir)

    print('final_output_dir: ',final_output_dir)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)

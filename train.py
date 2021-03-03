# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

# import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models
from models.omnipose import OmniPose
from models.omnipose import get_Canny_HRNet
from models.frankenstein import get_frankenstein
from models.pose_omni import get_pose_net

import warnings
warnings.filterwarnings("ignore") 


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='experiments/mpii/hrnet/w48_256x256_adam_lr1e-3.yaml', type=str)

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

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    # final_output_dir = 'output/coco/omnipose/OmniPose_HRw48_v3/'
    final_output_dir = 'output/coco/omnipose/OmniPose_HRw48_v3_128_FullySeparable/'

    # logger.info(pprint.pformat(args))
    # logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # model = eval('models.pose_hrnet.get_pose_net')(cfg, is_train=True)
    model = eval('models.pose_omni.get_pose_net')(cfg, is_train=True)
    # model = OmniPose(cfg.MODEL.NUM_JOINTS)
    # model = get_Canny_HRNet(48, cfg.MODEL.NUM_JOINTS, 0.1)
    # model = get_frankenstein(cfg, True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    command_line = 'cp models/' + cfg.MODEL.NAME + '.py ' + final_output_dir
    os.system('mkdir '+ final_output_dir)
    os.system(command_line)
    # shutil.copy2('models/' + cfg.MODEL.NAME + '.py',final_output_dir + cfg.MODEL.NAME + '.py')
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    quit()

    model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_perf_01 = 0.0
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )
    # print(checkpoint_file, os.path.exists(checkpoint_file))
    # quit()
    checkpoint_file = 'output/coco/omnipose/OmniPose_HRw48_v2_128/checkpoint.pth'

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        print(checkpoint['perf'],'epoch',checkpoint['epoch'])
        # begin_epoch = checkpoint['epoch']
        # best_perf = checkpoint['perf']
        # last_epoch = checkpoint['epoch']

        # print(checkpoint.keys())

        model_state_dict = model.state_dict()#checkpoint['state_dict']
        new_model_state_dict = {}
        for k in model_state_dict:
            if k in checkpoint['state_dict'] and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                new_model_state_dict[k] = checkpoint['state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_model_state_dict, strict=False)

        print('begin_epoch', begin_epoch)
        print('best_perf', best_perf)
        print('last_epoch',last_epoch)
        # quit()

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # logger.info("=> loaded checkpoint '{}' (epoch {})".format(
        #     checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=-1
    )

    for i in range(last_epoch):
        lr_scheduler.step()

    # Freeze all layers but WASP module
    # model.requires_grad = False
    # model.waspv2.requires_grad = True
    # model.conv1.requires_grad  = True
    # model.conv2.requires_grad  = True

    # perf_indicator = validate(cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion, \
    #                         final_output_dir, tb_log_dir, writer_dict)

    # quit()

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        print(final_output_dir)

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir)#, writer_dict)


        # evaluate on validation set
        if cfg.DATASET.DATASET == 'mpii':
            perf_indicator, perf_indicator_01 = validate(
                cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion,
                final_output_dir, tb_log_dir, writer_dict)
        elif cfg.DATASET.DATASET == 'coco' or cfg.DATASET.DATASET == 'posetrack':
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion,
                final_output_dir, tb_log_dir, writer_dict)
            perf_indicator_01 = 0

        # quit()

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_perf_01 = perf_indicator_01
            best_model = True

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

        else:
            best_model = False

        print("Best so far: PCKh@0.5 = "+str(best_perf)+", PCKh@0.1 = "+str(best_perf_01))

        # break
        # quit()

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)
    # writer_dict['writer'].close()


if __name__ == '__main__':
    arg = parse_args()
    main(arg)

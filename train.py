# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

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

from config        import cfg
from config        import update_config
from core.loss     import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils   import get_optimizer
from utils.utils   import save_checkpoint
from utils.utils   import create_logger
from utils.utils   import get_model_summary

import dataset
import models

from models.omnipose   import get_omnipose
from models.pose_hrnet import get_pose_net

import warnings
warnings.filterwarnings("ignore") 


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',          help='experiment configure file name',
                        default='experiments/mpii/hrnet/w48_256x256_adam_lr1e-3.yaml', type=str)
    parser.add_argument('--opts',         help="Modify config options using the command-line",
                        default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--modelDir',     help='model directory', type=str, default='')
    parser.add_argument('--logDir',       help='log directory', type=str, default='')
    parser.add_argument('--dataDir',      help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args


def main(args):
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    print('Model will be saved at: ',final_output_dir)
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.MODEL.NAME == 'pose_hrnet':
        model = get_pose_net(cfg, is_train=True)
    elif cfg.MODEL.NAME == 'omnipose':
        model = get_omnipose(cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    command_line = 'cp models/' + cfg.MODEL.NAME + '.py ' + final_output_dir
    os.system('mkdir '+ final_output_dir)
    os.system(command_line)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,}

    dump_input = torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    logger.info(get_model_summary(model, dump_input))

    model = model.cuda()

    # Define loss function and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([transforms.ToTensor(), normalize,]))
    
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([transforms.ToTensor(), normalize, ]) )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY)

    best_perf    = 0.0
    best_perf_01 = 0.0
    best_model   = False
    last_epoch   = -1
    
    optimizer   = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        print('Loading checkpoint with accuracy of 'checkpoint['perf'], 'at epoch ',checkpoint['epoch'])
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        model_state_dict = model.state_dict()
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

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=-1
    )

    for i in range(last_epoch):
        lr_scheduler.step()

    # In case you want to freeze layers prior to WASPv2, uncomment below:
    # model.requires_grad = False
    # model.waspv2.requires_grad = True

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
            
        elif cfg.DATASET.DATASET == 'coco':
            perf_indicator = validate(
                cfg, valid_loader, valid_dataset, cfg.DATASET.DATASET, model, criterion,
                final_output_dir, tb_log_dir, writer_dict)
            perf_indicator_01 = 0
            
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

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.state_dict(), final_model_state_file)

if __name__ == '__main__':
    arg = parse_args()
    main(arg)

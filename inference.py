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

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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
from core.inference import get_final_preds_no_transform


class ColorStyle:
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))

        # Red    = (240,  2,127)
        # Yellow = (255,255,  0)
        # Green  = (169,209,142)
        # Pink   = (252,176,243)
        # Blue   = (0,176,240)
        color_ids = [(0,176,240), (252,176,243), (169,209,142), (255,255,  0), (240,2,127)]

        self.color_ids = []
        for i in range(len(color_ids)):
            self.color_ids.append(tuple(np.array(color_ids[i])/255.))


color = [(252,176,243),(252,176,243),(252,176,243),
            (0,176,240), (0,176,240), (0,176,240),
            (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
            (255,255,0), (255,255,0),(169, 209, 142),
            (169, 209, 142),(169, 209, 142)]

link_pairs = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], \
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], \
    [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]

# Red    = (240,  2,127)
# Yellow = (255,255,  0)
# Green  = (169,209,142)
# Pink   = (252,176,243)
# Blue   = (0,176,240)

point_color = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

artacho_style = ColorStyle(color, link_pairs, point_color)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO predictions')
    parser.add_argument('--dataset', type=str, default='COCO')
    parser.add_argument('--image-path', help='Path of COCO val images',
                        type=str, default='/home/bm3768/Desktop/research/dataset/COCO/val2017/')
    parser.add_argument('--gt-anno', help='Path of COCO val annotation', type=str,
                        default='/home/bm3768/Desktop/research/dataset/COCO/person_keypoints_val2017.json')
    parser.add_argument('--save-path',help="Path to save the visualizations", type=str, default='samples/')
    parser.add_argument('--prediction', help="Prediction file to visualize", type=str, required=True)
    args = parser.parse_args()

    return args


def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

# plot_COCO_image(preds, img_path, save_path, colorstyle.link_pairs, colorstyle.ring_color, colorstyle.color_ids, save=True)
def plot_COCO_image(preds, img_path, save_path, link_pairs, ring_color, color_ids, save=True):
    
    # joints
    # coco = COCO(gt_file)
    # coco_dt = coco.loadRes(data)
    # coco_eval = COCOeval(coco, coco_dt, 'keypoints')
    # coco_eval._prepare()
    # gts_ = coco_eval._gts
    # dts_ = coco_eval._dts
    
    # p = coco_eval.params
    # p.imgIds = list(np.unique(p.imgIds))
    # if p.useCats:
    #     p.catIds = list(np.unique(p.catIds))
    # p.maxDets = sorted(p.maxDets)

    # print(preds.shape)
    # print(preds)
    # print(img_path)
    # quit()

    # loop through images, area range, max detection number
    # catIds = p.catIds if p.useCats else [-1]
    # threshold = 0.3
    # joint_thres = 0.2
    # for catId in catIds:
    #     for imgId in p.imgIds:
    #         # dimention here should be Nxm
    #         gts = gts_[imgId, catId]
    #         dts = dts_[imgId, catId]
    #         inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    #         dts = [dts[i] for i in inds]
    #         if len(dts) > p.maxDets[-1]:
    #             dts = dts[0:p.maxDets[-1]]
    #         if len(gts) == 0 or len(dts) == 0:
    #             continue
            
    #         sum_score = 0
    #         num_box = 0
    #         img_name = str(imgId).zfill(12)
            
    # Read Images
    data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    data_numpy = cv2.resize(data_numpy, (384,288), interpolation = cv2.INTER_AREA)
    # print(data_numpy.shape)
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]
    
    # Plot
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)

    # dt_joints = np.array(dt['keypoints']).reshape(17,-1)
    # print(preds.shape)
    joints_dict = map_joint_dict(preds[0])
    # print(joints_dict)
    # quit()
    
    # stick 
    for k, link_pair in enumerate(link_pairs):
        # if link_pair[0] in joints_dict \
        # and link_pair[1] in joints_dict:
            # if dt_joints[link_pair[0],2] < joint_thres \
            #     or dt_joints[link_pair[1],2] < joint_thres \
            #     or vg[link_pair[0]] == 0 \
            #     or vg[link_pair[1]] == 0:
            #     continue
        # if k in range(6,11):
        #     lw = 1
        # else:
        #     lw = ref / 100.
        lw = 2
        line = mlines.Line2D(
                np.array([joints_dict[link_pair[0]][0],
                          joints_dict[link_pair[1]][0]]),
                np.array([joints_dict[link_pair[0]][1],
                          joints_dict[link_pair[1]][1]]),
                ls='-', lw=lw, alpha=1, color=color_ids[0],)
        line.set_zorder(0)
        ax.add_line(line)
    # black ring
    for k in range(preds.shape[1]):
        # if dt_joints[k,2] < joint_thres \
        #     or vg[link_pair[0]] == 0 \
        #     or vg[link_pair[1]] == 0:
        #     continue
        if preds[0,k,0] > w or preds[0,k,1] > h:
            continue
        radius = 2

        circle = mpatches.Circle(tuple(preds[0,k,:2]), 
                                 radius=radius, 
                                 ec='black', 
                                 fc=ring_color[k], 
                                 alpha=1, 
                                 linewidth=1)
        circle.set_zorder(1)
        ax.add_patch(circle)

    # avg_score = (sum_score / (num_box+np.spacing(1)))*1000

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)
    print(save_path)
    plt.savefig(save_path, format='jpg', bbox_inckes='tight', dpi=100)
    plt.close()


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

    model = eval('models.pose_omni.get_pose_net')(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        logger.info("=> loading checkpoint '{}'".format(cfg.TEST.MODEL_FILE))
        checkpoint = torch.load(cfg.TEST.MODEL_FILE)
        best_perf = checkpoint['perf']

        model_state_dict = model.state_dict()
        new_model_state_dict = {}
        
        for k in checkpoint['state_dict']:
            if k in model_state_dict and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                new_model_state_dict[k] = checkpoint['state_dict'][k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(checkpoint, strict=False)

        print('best_perf', best_perf)

        model.load_state_dict(new_model_state_dict, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        model_state_file = 'models/coco/w48_384×288.pth'
        logger.info('=> loading model from {}'.format(model_state_file))

        model_state_dict = torch.load(model_state_file)
        new_model_state_dict = {}
        for k in model_state_dict:
            if k in model_state_dict and model_state_dict[k].size() == model_state_dict[k].size():
                new_model_state_dict[k] = model_state_dict[k]
            else:
                print('Skipped loading parameter {}'.format(k))

        model.load_state_dict(new_model_state_dict)

    model = model.cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset,
    #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
    #     shuffle=False,
    #     num_workers=cfg.WORKERS,
    #     pin_memory=True
    # )

    transform = transforms.Compose([transforms.ToTensor(),normalize,])

    model.eval()

    files_loc = '/home/bm3768/Desktop/research/dataset/NSL/9303/195KB_1'
    images = os.listdir(files_loc)
    # print(images)
    # quit()

    for idx in range(len(images)):
        print(idx,"/",len(images))
        img_path = os.path.join(files_loc,images[idx])

        data_numpy = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        data_numpy = cv2.resize(data_numpy, (384,288), interpolation = cv2.INTER_AREA)

        # print(data_numpy.shape)

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        # print(data_numpy.shape)

        data_numpy = transform(data_numpy)

        # print(data_numpy.shape)

        input = torch.zeros((1,3,data_numpy.shape[1], data_numpy.shape[2]))
        input[0] = data_numpy

        # print(input.shape)

        input = input.cuda()

        outputs = model(input)

        preds, maxvals = get_final_preds_no_transform(cfg, outputs.detach().cpu().numpy())

        colorstyle = artacho_style

        plot_COCO_image(4*preds, img_path, 'samples/NSL/9303/195KB_1/'+images[idx], colorstyle.link_pairs, colorstyle.ring_color, colorstyle.color_ids, save=True)

        # quit()

        # print(outputs.shape)

        # quit()

        # center   = [184, 184]

        # # img  = np.array(cv2.resize(cv2.imread(img_path),(368,368)), dtype=np.float32)
        # img  = img.transpose(2, 0, 1)
        # img  = torch.from_numpy(img)
        # mean = [128.0, 128.0, 128.0]
        # std  = [256.0, 256.0, 256.0]
        # for t, m, s in zip(img, mean, std):
        #     t.sub_(m).div_(s)

        # img       = torch.unsqueeze(img, 0)

        # self.model.eval()

        # input_var   = img.cuda()

        # heat, limbs = self.model(input_var)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)

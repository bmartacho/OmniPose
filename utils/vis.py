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

import math


from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import torchvision
import json
import cv2

from core.inference import get_max_preds


def save_images(batch_image, batch_joints, batch_joints_vis, output_dir, meta, iteration):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    file_name = meta['image']
    gt_joints = meta['joints']

    gt_file = '/home/bm3768/Desktop/research/dataset/MPII/annot/valid.json'
    with open(gt_file) as f:
        gt = json.load(f)

    color_ids = [(0,176,240), (252,176,243), (169,209,142), (255,255,  0), (240,2,127)]
    point_colors = [(240,2,127),(240,2,127),(240,2,127), 
            (240,2,127), (240,2,127), 
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (252,176,243),(0,176,240),(252,176,243),
            (0,176,240),(252,176,243),(0,176,240),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142),
            (255,255,0),(169, 209, 142)]

    link_pairs = [[5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],]
    line_color = [(252,176,243),(252,176,243),(252,176,243),
            (0,176,240), (0,176,240), (0,176,240),
            (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127), 
            (255,255,0), (255,255,0),(169, 209, 142),
            (169, 209, 142),(169, 209, 142)]

    limb_colors = []
    for i in range(len(color_ids)):
        limb_colors.append(tuple(np.array(color_ids[i])/255.))

    joint_colors = []
    for i in range(len(point_colors)):
        joint_colors.append(tuple(np.array(point_colors[i])/255.))

    images_done = []
    k = 0
    for i in range(batch_joints.shape[0]):
        filename = gt[i]['image']
        i_gt = iteration*batch_joints.shape[0] + i
        if gt[i_gt]['image'] != file_name[i][-13:]:
            print('Different!', gt[i_gt]['image'], file_name[i][-13:])
            quit()
        if Counter(images_done)[file_name[i]] > 0:
            filename = 'output/mpii/omnipose/OmniPose_HRw48_v2/val_images/'+file_name[i][-13:]
            print(filename)
        else:
            filename = file_name[i]
            
        image = cv2.imread(filename, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        h = image.shape[0]
        w = image.shape[1]
        ndarr = image.copy()
        joints = batch_joints[k]
        joints_vis = batch_joints_vis[k]

        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = plt.subplot(1,1,1)
        bk = plt.imshow(image[:,:,::-1])
        bk.set_zorder(-1)

        for j in range(gt_joints.shape[1]):        
            if gt[i_gt]['joints_vis'][j] == 1:
                circle = mpatches.Circle((int(gt[i_gt]['joints'][j][0]),int(gt[i_gt]['joints'][j][1])), 
                                                         radius=int(h/100), 
                                                         ec='black', 
                                                         fc=joint_colors[j], 
                                                         alpha=1, 
                                                         linewidth=1)
                circle.set_zorder(1)
                ax.add_patch(circle)

        for k, link_pair in enumerate(link_pairs):
            if gt[i_gt]['joints_vis'][link_pair[0]] == 1 and gt[i_gt]['joints_vis'][link_pair[1]] == 1:
                line = mlines.Line2D(
                        np.array([gt[i_gt]['joints'][link_pair[0]][0],
                                  gt[i_gt]['joints'][link_pair[1]][0]]),
                        np.array([gt[i_gt]['joints'][link_pair[0]][1],
                                  gt[i_gt]['joints'][link_pair[1]][1]]),
                        ls='-', lw=int(h/100), alpha=1, color=limb_colors[Counter(images_done)[file_name[i]]],)
                line.set_zorder(0)
                ax.add_line(line)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
        plt.margins(0,0)
        plt.savefig(output_dir+'val_images/'+gt[i_gt]['image'][:-3]+'png', 
                       format='png', bbox_inckes='tight', dpi=100)
        plt.close()

        images_done.append(gt[i_gt]['image'])


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

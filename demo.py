# # ------------------------------------------------------------------------------
# # pose.pytorch
# # Copyright (c) 2018-present Microsoft
# # Licensed under The Apache-2.0 License [see LICENSE for details]
# # Written by Bin Xiao (Bin.Xiao@microsoft.com)
# # ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import argparse
# import os
# import pprint

# import torch
# import torch.nn.parallel
# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.transforms as transforms
# from utils.transforms import flip_back

# import _init_paths
# from config import cfg
# from config import update_config
# from core.loss import JointsMSELoss
# from core.function import validate
# from core.inference import gaussian_blur
# from core.inference import taylor
# from utils.utils import create_logger
# from utils.transforms import transform_preds

# import numpy as np

# import time
# import cv2

# import cython

# import dataset
# import models


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train keypoints network')
#     # general
#     parser.add_argument('--cfg', help='experiment configure file name',
#                         default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml', type=str)
#     parser.add_argument('--opts', help="Modify config options using the command-line",
#                         default=None, nargs=argparse.REMAINDER)
#     parser.add_argument('--modelDir', help='model directory', type=str, default='')
#     parser.add_argument('--logDir', help='log directory', type=str, default='')
#     parser.add_argument('--dataDir', help='data directory', type=str, default='')
#     parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

#     args = parser.parse_args()
#     return args


# def main(args):
#     # args = parse_args()
#     update_config(cfg, args)

#     final_output_dir = 'output/mpii/omnipose/baseline'

#     # cudnn related setting
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

#     model = eval('models.pose_hrnet.get_pose_net')(cfg, is_train=False)

#     if cfg.TEST.MODEL_FILE:
#         model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
#     else:
#         model_state_file = os.path.join(
#             final_output_dir, 'final_state.pth'
#         )
#         model_state_file = 'models/coco/w48_384×288.pth'

#         model_state_dict = torch.load(model_state_file)
#         new_model_state_dict = {}
#         for k in model_state_dict:
#             if k in model_state_dict and model_state_dict[k].size() == model_state_dict[k].size():
#                 new_model_state_dict[k] = model_state_dict[k]
#             else:
#                 logging.info('Skipped loading parameter {}'.format(k))

#         # model.load_state_dict(checkclpoint['state_dict'])

#     model = model.cuda()

#     # Data loading code
#     # normalize = transforms.Normalize(
#     #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     # )
#     # valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
#     #     cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
#     #     transforms.Compose([
#     #         transforms.ToTensor(),
#     #         normalize,
#     #     ])
#     # )
#     # valid_loader = torch.utils.data.DataLoader(
#     #     valid_dataset,
#     #     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
#     #     shuffle=False,
#     #     num_workers=cfg.WORKERS,
#     #     pin_memory=True
#     # )

#     # evaluate on validation set
#     # validate(cfg, valid_loader, valid_dataset, model, criterion,
#     #          final_output_dir, tb_log_dir)

#     # switch to evaluate mode
#     model.eval()

#     flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]

#     # num_samples = len(val_dataset)
#     all_preds = np.zeros(
#         (1, cfg.MODEL.NUM_JOINTS, 3),
#         dtype=np.float32
#     )
#     all_boxes = np.zeros((1, 6))
#     image_path = []
#     filenames = []
#     imgnums = []
#     idx = 0
#     with torch.no_grad():
#         end = time.time()

#         img  = np.array(cv2.resize(cv2.imread("sample.jpg"),(256,256)), dtype=np.float32)
#         img  = img.transpose(2, 0, 1)
#         img  = torch.from_numpy(img)
#         mean = [128.0, 128.0, 128.0]
#         std  = [256.0, 256.0, 256.0]
#         for t, m, s in zip(img, mean, std):
#             t.sub_(m).div_(s)

#         img  = torch.unsqueeze(img, 0)

#         input  = img.cuda()

#         # img = cv2.imread("sample.jpg", cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

#         # tbar = tqdm(val_loader)

#         # for i, (input, target, target_weight, meta) in enumerate(tbar):

#         # img = cv2.resize(img,(256,256))

#         # input = torch.from_numpy(img.transpose((2, 0, 1)))

#         # mean = [128.0, 128.0, 128.0]
#         # std  = [256.0, 256.0, 256.0]

#         # for t, m, s in zip(input, mean, std):
#         #     t.sub_(m).div_(s)

#         # print(input.shape)

#         # input  = input.cuda()
#         # target = target.cuda()
#         # target_weight = target_weight.cuda()

#         outputs = model(input)
#         if isinstance(outputs, list):
#             output = outputs[-1]
#         else:
#             output = outputs

#         if cfg.TEST.FLIP_TEST:
#             # this part is ugly, because pytorch has not supported negative index
#             # input_flipped = model(input[:, :, :, ::-1])
#             input_flipped = np.flip(input.cpu().numpy(), 3).copy()
#             input_flipped = torch.from_numpy(input_flipped).cuda()
#             outputs_flipped = model(input_flipped)

#             if isinstance(outputs_flipped, list):
#                 output_flipped = outputs_flipped[-1]
#             else:
#                 output_flipped = outputs_flipped

#             output_flipped = flip_back(output_flipped.cpu().numpy(),flip_pairs)
#             output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

#             output = (output + output_flipped) * 0.5

#         # target = target.cuda(non_blocking=True)
#         # target_weight = target_weight.cuda(non_blocking=True)

#         # loss = criterion(output, target, target_weight)

#         num_images = input.size(0)
#         # measure accuracy and record loss
#         # losses.update(loss.item(), num_images)
#         # _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
#         #                                  target.cpu().numpy())

#         # acc.update(avg_acc, cnt)

#         # measure elapsed time
#         # batch_time.update(time.time() - end)
#         end = time.time()

#         # c = meta['center'].numpy()
#         # s = meta['scale'].numpy()
#         # score = meta['score'].numpy()

#         hm = output.clone().cpu().numpy()

#         batch_heatmaps = output.clone().cpu().numpy()

#         batch_size = batch_heatmaps.shape[0]
#         num_joints = batch_heatmaps.shape[1]
#         width = batch_heatmaps.shape[3]
#         heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
#         idx = np.argmax(heatmaps_reshaped, 2)
#         maxvals = np.amax(heatmaps_reshaped, 2)

#         maxvals = maxvals.reshape((batch_size, num_joints, 1))
#         idx = idx.reshape((batch_size, num_joints, 1))

#         coords = np.tile(idx, (1, 1, 2)).astype(np.float32)

#         coords[:, :, 0] = (coords[:, :, 0]) % width
#         coords[:, :, 1] = np.floor((coords[:, :, 1]) / width)

#         pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
#         pred_mask = pred_mask.astype(np.float32)

#         coords *= pred_mask

#         # coords, maxvals = get_max_preds(hm)
#         heatmap_height = hm.shape[2]
#         heatmap_width = hm.shape[3]

#         # post-processing
#         hm = gaussian_blur(hm, cfg.TEST.BLUR_KERNEL)
#         hm = np.maximum(hm, 1e-10)
#         hm = np.log(hm)
#         for n in range(coords.shape[0]):
#             for p in range(coords.shape[1]):
#                 coords[n,p] = taylor(hm[n][p], coords[n][p])

#         preds = coords.copy()

#         print(preds.shape)

#         # Transform back
#         # for i in range(coords.shape[0]):
#         #     preds[i] = transform_preds(
#         #         coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
#         #     )

#         # return preds, maxvals

#         # preds, maxvals = get_final_preds(
#         #     cfg, output.clone().cpu().numpy(), c, s)

#         all_preds[0:0 + num_images, :, 0:2] = preds[:, :, 0:2]
#         all_preds[0:0 + num_images, :, 2:3] = maxvals
#         # double check this all_boxes parts
#         # all_boxes[0:0 + num_images, 0:2] = c[:, 0:2]
#         # all_boxes[0:0 + num_images, 2:4] = s[:, 0:2]
#         # all_boxes[0:0 + num_images, 4] = np.prod(s*200, 1)
#         # all_boxes[0:0 + num_images, 5] = score
#         image_path.extend(meta['image'])

#         idx += num_images

#         if i % cfg.PRINT_FREQ == 0:

#             prefix = '{}_{}'.format(
#                 os.path.join(output_dir, 'val'), i
#             )
#             save_debug_images(cfg, input, meta, target, pred*4, output,
#                               prefix)

#         # tbar.set_description('Val   Acc: %.6f' % acc.avg)

#         name_values, perf_indicator = val_dataset.evaluate(
#             cfg, all_preds, output_dir, all_boxes, image_path,
#             filenames, imgnums
#         )

#         model_name = cfg.MODEL.NAME
#         if isinstance(name_values, list):
#             for name_value in name_values:
#                 _print_name_value(name_value, model_name)
#         else:
#             _print_name_value(name_values, model_name)



# if __name__ == '__main__':
#     arg = parse_args()
#     main(arg)


import os
import sys
import argparse
import ast
import cv2
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import models
from models.pose_omni import get_pose_net
from misc.HeatmapParser import HeatmapParser

from config import cfg
from config import update_config

from misc.utils import get_multi_scale_size, resize_align_multi_scale, get_multi_stage_outputs
from misc.utils import find_person_id_associations, aggregate_results, get_final_preds, bbox_iou
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation



class omnipose_demo:
    def __init__(self, cfg, args):
        self.c                  = args.channels
        if args.dataset == 'coco':
            self.nof_joints = 17
        elif args.dataset == 'mpii':
            self.nof_joints = 16
        else:
            raise ValueError('Dataset not implemented: ',args.dataset)
        self.checkpoint_path        = args.weights
        self.resolution             = args.image_resolution
        self.interpolation          = cv2.INTER_LINEAR
        self.return_heatmaps        = False
        self.return_bounding_boxes  = True
        self.filter_redundant_poses = True
        self.max_nof_people         = 30
        self.max_batch_size         = 32

        self.model = eval('models.pose_omni.get_pose_net')(cfg, is_train=False)

        if cfg.TEST.MODEL_FILE:
            print("=> loading checkpoint '{}'".format(cfg.TEST.MODEL_FILE))
            checkpoint = torch.load(cfg.TEST.MODEL_FILE)
            begin_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']

            model_state_dict = self.model.state_dict()
            new_model_state_dict = {}
            
            for k in checkpoint['state_dict']:
                if k in model_state_dict and model_state_dict[k].size() == checkpoint['state_dict'][k].size():
                    new_model_state_dict[k] = checkpoint['state_dict'][k]
                else:
                    print('Skipped loading parameter {}'.format(k))
            self.model.load_state_dict(checkpoint, strict=False)

            print('begin_epoch', begin_epoch)
            print('best_perf', best_perf)
            print('last_epoch',last_epoch)

            self.model.load_state_dict(new_model_state_dict, strict=False)
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

            self.model.load_state_dict(new_model_state_dict)

        self.model = self.model.cuda()
        self.model.eval()

        self.output_parser = HeatmapParser(num_joints=self.nof_joints,
                    joint_set=args.dataset, max_num_people=self.max_nof_people,
                    ignore_too_much=True, detection_threshold=0.3)

        self.transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def predict(self, image):
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        ret = self._predict_batch(image[None, ...])
        if len(ret) > 1:  # heatmaps and/or bboxes and joints
            ret = [r[0] for r in ret]
        else:  # joints only
            ret = ret[0]
        return ret

    def _predict_batch(self, image):
        with torch.no_grad():

            heatmaps_list = None
            tags_list = []

            scales = (1,)  # ToDo add support to multiple scales

            scales = sorted(scales, reverse=True)
            base_size, base_center, base_scale = get_multi_scale_size(
                image[0], self.resolution, 1, 1
            )

            for idx, scale in enumerate(scales):
                images = list()
                for img in image:
                    image, size_resized, _, _ = resize_align_multi_scale(
                        img, self.resolution, scale, min(scales), interpolation=self.interpolation
                    )
                    image = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
                    image = image.cuda()
                    images.append(image)
                images = torch.cat(images)

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    self.model, images, with_flip=False, project2image=True, size_projected=size_resized,
                    nof_joints=self.nof_joints, max_batch_size=self.max_batch_size
                )

                heatmaps_list, tags_list = aggregate_results(
                    scale, heatmaps_list, tags_list, heatmaps, tags, with_flip=False, project2image=True
                )

            heatmaps = heatmaps_list.float() / len(scales)
            tags = torch.cat(tags_list, dim=4)

            grouped, scores = self.output_parser.parse(
                heatmaps, tags, adjust=True, refine=True  # ToDo parametrize these two parameters
            )

            # get final predictions
            final_results = get_final_preds(
                grouped, base_center, base_scale, [heatmaps.shape[3], heatmaps.shape[2]]
            )

            if self.filter_redundant_poses:
                final_pts = []
                # for each image
                for i in range(len(final_results)):
                    final_pts.insert(i, list())
                    # for each person
                    for pts in final_results[i]:
                        if len(final_pts[i]) > 0:
                            diff = np.mean(np.abs(np.array(final_pts[i])[..., :2] - pts[..., :2]), axis=(1, 2))
                            if np.any(diff < 3):  # average diff between this pose and another one is less than 3 pixels
                                continue
                        final_pts[i].append(pts)
                final_results = final_pts

            pts = []
            boxes = []
            for i in range(len(final_results)):
                pts.insert(i, np.asarray(final_results[i]))
                if len(pts[i]) > 0:
                    pts[i][..., [0, 1]] = pts[i][..., [1, 0]]  # restoring (y, x) order as in SimpleHRNet
                    pts[i] = pts[i][..., :3]

                    if self.return_bounding_boxes:
                        left_top = np.min(pts[i][..., 0:2], axis=1)
                        right_bottom = np.max(pts[i][..., 0:2], axis=1)
                        # [x1, y1, x2, y2]
                        boxes.insert(i, np.stack(
                            [left_top[:, 1], left_top[:, 0], right_bottom[:, 1], right_bottom[:, 0]], axis=-1
                        ))
                else:
                    boxes.insert(i, [])

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]


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

    parser.add_argument("--filename", "-f",   type=str, default=None)
    parser.add_argument("--channels",         type=int, default=48)
    # parser.add_argument("--hrnet_j", "-j",    help="number of joints", type=int, default=17)
    parser.add_argument("--weights",          type=str, default="./weights/output/coco/omnipose/OmniPose_HRw48/model_best.pth")
    parser.add_argument("--dataset",          type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution (`512` or `640`)", type=int, default=384)
    parser.add_argument("--disable_tracking", action="store_true")
    parser.add_argument("--max_nof_people",   type=int, default=30)
    parser.add_argument("--max_batch_size",   type=int, default=16)
    parser.add_argument("--save_video",       action="store_true")
    parser.add_argument("--video_format",     help=" `MJPG`, `XVID`, `X264`.", type=str, default='MJPG')
    parser.add_argument("--video_framerate",  type=float, default=30)

    args = parser.parse_args()
    return args

def main(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = omnipose_demo(cfg, args)

    frame = cv2.imread('samples/sample.jpg')

    pts = model.predict(frame)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
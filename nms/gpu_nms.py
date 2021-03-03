# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# cimport numpy as np

# assert sizeof(int) == sizeof(np.int32_t)

# from "gpu_nms.hpp":
#     void _nms(np.int32_t*, int*, np.float32_t*, int, int, float, int)

def gpu_nms(dets, thresh, device_id=0):
    boxes_num = dets.shape[0]
    boxes_dim = dets.shape[1]
    keep = np.zeros(boxes_num, dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    sorted_dets = dets[order, :]
    # _nms(&keep[0], &num_out, &sorted_dets[0, 0], boxes_num, boxes_dim, thresh, device_id)
    keep = keep[:num_out]
    return list(order[keep])

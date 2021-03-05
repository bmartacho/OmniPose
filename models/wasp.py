# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #
#                                    OmniPose                                    #
#      Rochester Institute of Technology - Vision and Image Processing Lab       #
#                      Bruno Artacho (bmartacho@mail.rit.edu)                    #
# ------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------ #

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 


class SepConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True, padding_mode='zeros', depth_multiplier=1):
        super(SepConv2d, self).__init__()

        intermediate_channels = in_channels * depth_multiplier

        self.spatialConv = nn.Conv2d(in_channels, intermediate_channels,kernel_size, stride,
             padding, dilation, groups=in_channels, bias=bias, padding_mode=padding_mode)

        self.pointConv = nn.Conv2d(intermediate_channels, out_channels,
             kernel_size=1, stride=1, padding=0, dilation=1, bias=bias, padding_mode=padding_mode)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.spatialConv(x)
        x = self.relu(x)
        x = self.pointConv(x)

        return x

conv_dict = {
    'CONV2D': nn.Conv2d,
    'SEPARABLE': SepConv2d
}

class _AtrousModule(nn.Module):
    def __init__(self, conv_type, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_AtrousModule, self).__init__()
        self.conv = conv_dict[conv_type]
        self.atrous_conv = self.conv(inplanes, planes, kernel_size=kernel_size,
                            stride=1, padding=padding, dilation=dilation, bias=False, padding_mode='zeros')

        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class wasp(nn.Module):
    def __init__(self, inplanes, planes, upDilations=[4,8]):
        super(wasp, self).__init__()
        dilations = [6, 12, 18, 24]
        BatchNorm = nn.BatchNorm2d

        self.aspp1 = _AtrousModule('CONV2D', inplanes, planes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule('CONV2D', planes, planes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule('CONV2D', planes, planes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule('CONV2D', planes, planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(5*planes, planes, 1, bias=False)
        self.conv2 = nn.Conv2d(planes,planes,1,bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        x3 = self.conv2(x3)
        x4 = self.conv2(x4)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_wasp(inplanes, planes, upDilations):
    return wasp(inplanes, planes, upDilations)


class WASPv2(nn.Module):
    def __init__(self, conv_type, inplanes, planes, n_classes=17):
        super(WASPv2, self).__init__()

        # WASP
        dilations = [1, 6, 12, 18]
        # dilations = [1, 12, 24, 36]
        
        # convs = conv_dict[conv_type]

        reduction = planes // 8

        BatchNorm = nn.BatchNorm2d

        self.aspp1 = _AtrousModule(conv_type, inplanes, planes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _AtrousModule(conv_type, planes, planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.relu = nn.ReLU()

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(planes, planes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(planes),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(5*planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, reduction, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(reduction)

        self.last_conv = nn.Sequential(nn.Conv2d(planes+reduction, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(),
                                       nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(planes),
                                       nn.ReLU(),
                                       nn.Conv2d(planes, n_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_features):
        input = x
        x1 = self.aspp1(x)
        x2 = self.aspp2(x1)
        x3 = self.aspp3(x2)
        x4 = self.aspp4(x3)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        # x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

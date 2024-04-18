# -*- coding: utf-8 -*-
# @Time    : 2021/7/8 8:59 上午
# @File    : UCTransNet.py
# @Software: PyCharm
import torch.nn as nn
import torch
import torch.nn.functional as F
from .MFCNet_CTrans import ChannelTransformer


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):  # convolution->BN->ReLU
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class ConvBatchNorm3(nn.Module):  # convolution->BN->ReLU
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

    class ConvBatchNorm(nn.Module):  # convolution->BN->ReLU
        """(convolution => [BN] => ReLU)"""

        def __init__(self, in_channels, out_channels, activation='ReLU'):
            super(ConvBatchNorm, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=3, padding=1)
            self.norm = nn.BatchNorm2d(out_channels)
            self.activation = get_activation(activation)

        def forward(self, x):
            out = self.conv(x)
            out = self.norm(out)
            return self.activation(out)


class ConvBatchNorm5(nn.Module):  # convolution->BN->ReLU
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm5, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=5, padding=2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


# -------------------------3*3 conv-------------------------- #
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with 1 padding "
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


# -------------------------5*5 conv-------------------------- #
def conv5x5(in_planes, out_planes, stride=1):
    "5x5 convolution with 2 padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=5,
                     stride=stride,
                     padding=2,
                     bias=True)


# ------------------------3x3 BasicBlock----------------------#
class BasicBlock3x3(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ------------------------5x5 BasicBlock----------------------#
class BasicBlock5x5(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# -------------------------3x3 ResNet module----------------------#
class ResNet34_3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_3x3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.BasicBlock3x3_1 = BasicBlock3x3(in_channels, out_channels)
        self.BasicBlock3x3_2 = BasicBlock3x3(in_channels, out_channels)

    def forward(self, x):
        # x = x.float()  # 像素转换浮点数
        x = self.BasicBlock3x3_1(x)
        x = self.BasicBlock3x3_2(x)
        return x


# -------------------------5x5 ResNet module----------------------#
class ResNet34_5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_5x5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.BasicBlock5x5_1 = BasicBlock5x5(in_channels, out_channels)
        self.BasicBlock5x5_2 = BasicBlock5x5(in_channels, out_channels)

    def forward(self, x):
        x = self.BasicBlock5x5_1(x)
        x = self.BasicBlock5x5_2(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# idea patch + channel 注意 + 残差  =  观察实验结果
class UpBlock(nn.Module):  # 定义上采样的通道注意
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)  # 上采样，scale_factor=2表示2*2上采样
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)  # 对X进行上采样
        x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension（对跳跃连接的特征图和上采样的特征图进行拼接）
        return self.nConvs(x)


class CA(nn.Module):
    """
    Channel Attention Block
    """

    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))

        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        scale = torch.sigmoid(channel_att_x + channel_att_g).unsqueeze(2).unsqueeze(3).expand_as(x)

        x_after_channel = x * scale  # 对通道进行抑制和增强
        g_after_channel = g * scale

        return x_after_channel, g_after_channel


class SA(nn.Module):
    """
    Spatial Attention Block
    """

    def __init__(self, config, F_g, F_x, img_size=224):
        super().__init__()
        vis = False
        in_channels = config.base_channel  # inchannenl 64
        self.relu = nn.ReLU(inplace=True)
        self.inconv = nn.Conv2d(512, 1, 3, padding=1)  # 1*1卷积
        self.Activation = nn.Sigmoid()  # 1*1激活函数

    def forward(self, g, x):
        g1_sum_tensor = torch.sum(g, dim=1)  # 像素压缩
        g1_sum_tensor = torch.unsqueeze(g1_sum_tensor, dim=1)  # Reshape

        x1_sum_tensor = torch.sum(x, dim=1)  # 像素压缩
        x1_sum_tensor = torch.unsqueeze(x1_sum_tensor, dim=1)

        single_channel = torch.sigmoid(g1_sum_tensor.expand_as(g) + x1_sum_tensor.expand_as(x))  # 单通道空间特征图 + sigmoid

        output_g1 = g * single_channel  # 对空间特征进行抑制和增强
        output_x1 = x * single_channel

        SAB_attention = (output_g1 + output_x1) / 2.0

        return SAB_attention * g + SAB_attention * x


class DCCT(nn.Module):
    """
    DCCT Block
    """

    def __init__(self, config, img_size=224):
        super().__init__()
        vis = False
        in_channels = config.base_channel  # inchannenl 64
        # CCT Module
        self.CCT_3 = ChannelTransformer(config, vis, img_size,
                                        channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                        patchSize=config.patch_sizes)

        self.CCT_5 = ChannelTransformer(config, vis, img_size,
                                        channel_num=[in_channels, in_channels * 2, in_channels * 4, in_channels * 8],
                                        patchSize=config.patch_sizes)

        self.CA4 = CA(F_g=512, F_x=512)
        self.CA3 = CA(F_g=256, F_x=256)
        self.CA2 = CA(F_g=128, F_x=128)
        self.CA1 = CA(F_g=64, F_x=64)

    def forward(self, x1_3, x2_3, x3_3, x4_3, x1_5, x2_5, x3_5, x4_5):
        x1_3, x2_3, x3_3, x4_3, atten_weight1 = self.CCT_3(x1_3, x2_3, x3_3, x4_3)
        x1_5, x2_5, x3_5, x4_5, atten_weight2 = self.CCT_5(x1_5, x2_5, x3_5, x4_5)
        h1_3, h1_5 = self.CA1(x1_3, x1_5)
        x1_3 = x1_3 + h1_3
        x1_5 = x1_5 + h1_5

        h2_3, h2_5 = self.CA2(x2_3, x2_5)
        x2_3 = x2_3 + h2_3
        x2_5 = x2_5 + h2_5

        h3_3, h3_5 = self.CA3(x3_3, x3_5)
        x3_3 = x3_3 + h3_3
        x3_5 = x3_5 + h3_5

        h4_3, h4_5 = self.CA4(x4_3, x4_5)
        x4_3 = x4_3 + h4_3
        x4_5 = x4_5 + h4_5
        
        return x1_3, x2_3, x3_3, x4_3, x1_5, x2_5, x3_5, x4_5


# -------------------------MFCNet Network---------------------#
class MFCNet(nn.Module):
    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel  # inchannenl 64

        # n_channels:3  in_channels：64，convolution->BN->ReLU
        self.inc = ConvBatchNorm(n_channels, in_channels)  # out channenl 64

        # ---------------------------------------------------------------#
        # -----------------------------encoder---------------------------#
        # ---------------------------------------------------------------#

        # stage1
        self.conv1_3x3 = ResNet34_3x3(64, 64)
        self.conv1_5x5 = ResNet34_5x5(64, 64)

        # stage2

        self.conv2_3x3 = ResNet34_3x3(128, 128)
        self.conv2_5x5 = ResNet34_5x5(128, 128)

        # stage3
        self.conv3_3x3 = ResNet34_3x3(256, 256)
        self.conv3_5x5 = ResNet34_5x5(256, 256)

        # stage4
        self.conv4_3x3 = ResNet34_3x3(512, 512)
        self.conv4_5x5 = ResNet34_5x5(512, 512)

        # ---------------------------------------------------------------#
        # -----------------------transformer of channel------------------#
        # ---------------------------------------------------------------#

        self.nConvs1_3 = _make_nConv(64, 128, 2, 'ReLU')
        self.nConvs1_5 = _make_nConv(64, 128, 2, 'ReLU')

        self.nConvs2_5 = _make_nConv(128, 256, 2, 'ReLU')
        self.nConvs2_3 = _make_nConv(128, 256, 2, 'ReLU')

        self.nConvs3_5 = _make_nConv(256, 512, 2, 'ReLU')
        self.nConvs3_3 = _make_nConv(256, 512, 2, 'ReLU')

        # ---------------------------------------------------------------#
        # -----------------------down Max-pooling------------------------#
        # ---------------------------------------------------------------#

        self.pool3x3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.pool5x5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # ---------------------------------------------------------------#
        # -----------------------bridge----------------------------------#
        # ---------------------------------------------------------------#

        self.convbg_1 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)  # 16
        self.bnbg_1 = nn.BatchNorm2d(512)
        self.relubg_1 = nn.ReLU(inplace=True)

        self.convbg_m = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_m = nn.BatchNorm2d(512)
        self.relubg_m = nn.ReLU(inplace=True)

        self.convbg_2 = nn.Conv2d(512, 512, 3, dilation=2, padding=2)
        self.bnbg_2 = nn.BatchNorm2d(512)
        self.relubg_2 = nn.ReLU(inplace=True)

        # ----------------融合函数-------------- #

        self.conv1_Fusion = nn.Conv2d(64 * 2, 64, 3, padding=1)
        self.bn1_Fusion = nn.BatchNorm2d(64)
        self.relu1_Fusion = nn.ReLU(inplace=True)

        self.conv2_Fusion = nn.Conv2d(128 * 2, 128, 3, padding=1)
        self.bn2_Fusion = nn.BatchNorm2d(128)
        self.relu2_Fusion = nn.ReLU(inplace=True)

        self.conv3_Fusion = nn.Conv2d(256 * 2, 256, 3, padding=1)
        self.bn3_Fusion = nn.BatchNorm2d(256)
        self.relu3_Fusion = nn.ReLU(inplace=True)

        self.conv4_Fusion = nn.Conv2d(512 * 2, 512, 3, padding=1)
        self.bn4_Fusion = nn.BatchNorm2d(512)
        self.relu4_Fusion = nn.ReLU(inplace=True)

        self.conv_Fusion = nn.Conv2d(512 * 2, 512, 3, padding=1)
        self.bn_Fusion = nn.BatchNorm2d(512)
        self.relu_Fusion = nn.ReLU(inplace=True)

        # DCCT Module

        self.DCCT = DCCT(config, img_size=224)

        # --------------Spatial Fusion------------------------- #
        self.SAB4 = SA(config, F_g=512, F_x=512, img_size=224)
        self.SAB3 = SA(config, F_g=256, F_x=256, img_size=224)
        self.SAB2 = SA(config, F_g=128, F_x=128, img_size=224)
        self.SAB1 = SA(config, F_g=64, F_x=64, img_size=224)

        # in_channels * 16:1024  in_channels * 4:256
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)

        # in_channels * 8 :512 in_channels * 2: 128
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)

        # in_channels * 4 :256  in_channels:64
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)

        # in_channels * 2 :128  in_channels:64
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()  # if using BCELoss

        # ---------------------------------------------------------------#
        # -----------------------Side Output-----------------------------#
        # ---------------------------------------------------------------#

        self.outconv5 = nn.Conv2d(512, 1, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, 1, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, 1, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.outconv1 = nn.Conv2d(64, 1, 3, padding=1)

        # ---------------------------------------------------------------#
        # -----------------------Bilinear Upsampling---------------------#
        # ---------------------------------------------------------------#

        self.upscore5 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = x.float()
        x0 = self.inc(x)  # image size (224,224)

        # ---------------stage1 encoder------------#
        # image size (224,224)
        h1_3 = self.conv1_3x3(x0)
        h1_5 = self.conv1_5x5(x0)

        # 残差 Patch 注意
        x1_3 = h1_3  # (224,224,64)
        x1_5 = h1_5

        h1_3 = self.pool3x3(h1_3)
        h1_5 = self.pool5x5(h1_5)

        # channel 64->128 image size (112,112)

        h1_3 = self.nConvs1_3(h1_3)
        h1_5 = self.nConvs1_5(h1_5)

        # -------------- stage2 encoder------------ #
        # image size (112,112)
        h2_3 = self.conv2_3x3(h1_3)
        h2_5 = self.conv2_5x5(h1_5)

        x2_3 = h2_3  # (112,112,128)
        x2_5 = h2_5

        # channel 128
        h2_3 = self.pool3x3(h2_3)
        h2_5 = self.pool5x5(h2_5)

        # channel 128->256
        h2_3 = self.nConvs2_3(h2_3)
        h2_5 = self.nConvs2_5(h2_5)

        # -------------- stage3 encoder------------ #

        h3_3 = self.conv3_3x3(h2_3)
        h3_5 = self.conv3_5x5(h2_5)

        x3_3 = h3_3  # (56,56,256)
        x3_5 = h3_5

        # channel 256
        h3_3 = self.pool3x3(h3_3)
        h3_5 = self.pool5x5(h3_5)

        # channel 256->512
        h3_3 = self.nConvs3_3(h3_3)
        h3_5 = self.nConvs3_5(h3_5)

        # -------------- stage4 encoder------------ #
        h4_3 = self.conv4_3x3(h3_3)
        h4_5 = self.conv4_5x5(h3_5)

        x4_3 = h4_3  # (28,28,512)
        x4_5 = h4_5

        # out channel：512
        feature4 = self.relu4_Fusion(
            self.bn4_Fusion(
                self.conv4_Fusion(
                    torch.cat((h4_3, h4_5), 1)
                )
            )
        )

        x5 = self.pool4(feature4)

        # -------------------bridge---------------- #
        hx = self.relubg_1(self.bnbg_1(self.convbg_1(x5)))
        hx = self.relubg_m(self.bnbg_m(self.convbg_m(hx)))
        hbg = self.relubg_2(self.bnbg_2(self.convbg_2(hx)))
        x5 = hbg  # 此新设置------设置残差

        # ===================DCCT================#

        x1_3, x2_3, x3_3, x4_3, x1_5, x2_5, x3_5, x4_5 = \
            self.DCCT(x1_3, x2_3, x3_3, x4_3, x1_5, x2_5, x3_5, x4_5)

        # ===================Spatial Connection Cross Attention（OK）================#
        enhance_feature_x4 = self.SAB4(x4_3, x4_5)
        enhance_feature_x3 = self.SAB3(x3_3, x3_5)
        enhance_feature_x2 = self.SAB2(x2_3, x2_5)
        enhance_feature_x1 = self.SAB1(x1_3, x1_5)

        o4 = self.up4(x5, enhance_feature_x4)  # x5:512    o4:256
        o3 = self.up3(o4, enhance_feature_x3)  # x:512     o3:128
        o2 = self.up2(o3, enhance_feature_x2)  # x:256     o2:64
        o1 = self.up1(o2, enhance_feature_x1)  # x:128     o1:64

        # -------------Side Output-------------
        d5 = self.outconv5(x5)  # side out
        d5 = self.upscore5(d5)  # 14->224

        d4 = self.outconv4(o4)  # side out
        d4 = self.upscore4(d4)  # 28->224

        d3 = self.outconv3(o3)  # side out
        d3 = self.upscore3(d3)  # 56->224

        d2 = self.outconv2(o2)  # side out
        d2 = self.upscore2(d2)  # 112->224

        d1 = self.outconv1(o1)  # 224

        return torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5)

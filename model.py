# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from final_model.SegFormer.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5

from final_model.PPA_my import my_PPA
from third_passage_blocks.CFC_SFC import SFC
from final_model.Multi_Conv.Multi_diagnal_Conv import four_direction_conv
from final_model.SCSegmamba_test.my_MFS_AWL_CSF import my_MFS_AWL_xiugai


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(ConvBlock, self).__init__()

        # self.layer = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.LeakyReLU(),
        #
        #     nn.Conv2d(out_channels, out_channels,kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.LeakyReLU(),
        # )

        self.layer = my_PPA(in_channels, out_channels)

    def forward(self, x):
        return self.layer(x)


class my_PPA_Encoder(nn.Module):
    def __init__(self):
        super(my_PPA_Encoder, self).__init__()

        self.in_chns = 3
        self.ft_chns = [32, 64, 128, 256]
        self.n_class = 1
        self.dropout = [0, 0, 0, 0, 0]
        assert (len(self.ft_chns) == 4)

        self.DownSample = nn.MaxPool2d(2)

        self.stem = ConvBlock(3, self.ft_chns[0], self.dropout[0])
        self.conv1 = ConvBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.conv2 = ConvBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.conv3 = ConvBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])

    def forward(self, x):
        out = []

        x = self.stem(x)  # 卷积 --> 卷积
        out.append(x)
        x = self.DownSample(x)

        x1 = self.conv1(x)  # 卷积 --> 卷积
        out.append(x1)
        x1 = self.DownSample(x1)

        x2 = self.conv2(x1)  # 8,64,64,64
        out.append(x2)
        x2 = self.DownSample(x2)

        x3 = self.conv3(x2)
        out.append(x3)

        return out


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=1, in_channels=[32, 64, 128, 256], embedding_dim=8, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1, )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

        '''定义注意力模块'''
        # self.att2 = four_direction_conv(64, 32)
        # self.att3 = four_direction_conv(128, 64)
        # self.att4 = four_direction_conv(256, 128)
        #
        # '''这里定义融合模块'''
        # self.fusion1 = SFC(32)
        # self.fusion2 = SFC(64)
        # self.fusion3 = SFC(128)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        # '''下面是添加的attention代码'''
        # c2_ = self.att2(c2)
        # c3_ = self.att3(c3)
        # c4_ = self.att4(c4)
        #
        # '''融合模块'''
        # c1 = self.fusion1(c1, c2_)
        # c2 = self.fusion2(c2, c3_)
        # c3 = self.fusion3(c3, c4_)

        '''下面是原先segformer后续的代码'''
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes=1, phi='b0', pretrained=False):
        super(SegFormer, self).__init__()

        '''定义编码部分'''
        # self.backbone = {'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,}[phi](pretrained)
        self.backbone = my_PPA_Encoder()

        '''定义解码部分'''
        '''选第一个(原始)要解除forward里的上采样'''
        # self.decode_head = SegFormerHead(num_classes, [32, 64, 128, 256], 8)
        self.decode_head = my_MFS_AWL_xiugai(in_channels=[32, 64, 128, 256], embedding_dim=8)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # x = self.backbone.forward(inputs)
        x = self.backbone(inputs)

        x = self.decode_head.forward(x)
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return torch.sigmoid(x)


if __name__ == '__main__':
    model = SegFormer().to("cuda")

    data = torch.randn(2, 3, 256, 256).to("cuda")
    print("输入：", data.size())
    out = model(data)

    # print(len(out))

    print("输出：", out.size())
    print('#parameters:', sum(param.numel() for param in model.parameters()))

    # from torchinfo import summary
    # summary(model,input_size=(4,3,256,256),device='cuda')

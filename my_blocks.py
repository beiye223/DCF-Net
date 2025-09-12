import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.fft
import math
from pytorch_wavelets import DWTForward


class my_FCHiLo1(nn.Module):
    def __init__(self, dim, window_size=2, alpha=0.5):
        super().__init__()

        self.ws = window_size

        self.wt = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')


    def hi_lofi(self, x):
        # B, N, C = x.shape
        # H = W = int(N ** 0.5)
        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        low_feats, yH = self.dwt(x)
        print(low_feats.size())
        # low_feats = self.wt(x)

        # 自己添加：
        B,C,H,W = x.shape


        low_up = F.interpolate(low_feats, size=H, mode='nearest')
        high_feats = x - low_up

        # 以下填写low和high通过注意力的代码
        # low_up = att(low_up)
        # high_feat =


        out = torch.cat([low_up, high_feats], dim=1)
        return out

    def forward(self, x):
        return self.hi_lofi(x)


if __name__ == '__main__':
    block = my_FCHiLo1(32)

    input1 = torch.rand(8, 32, 256, 256)

    output = block(input1)

    print("input1 size:", input1.size())
    print("Output size:", output.size())



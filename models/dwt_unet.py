import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from pytorch_wavelets import DWTForward, DWTInverse
from einops import rearrange

from .layers import *

class HMimoUnet(nn.Module):
    def __init__(self, num_res=8, in_channel=3):
        super().__init__()

        base_channel = 32

        self.dwt = DWTForward(J=3, wave='haar', mode='zero')
        self.iwt = DWTInverse(wave='haar', mode='zero')

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, in_channel, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, in_channel, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, in_channel, kernel_size=3, relu=False, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4, in_channel=in_channel)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2, in_channel=in_channel)

    def forward(self, x):
        # x: n c h w
        N, C, H, W = x.shape
        _, (x_1, x_2, x_3) = self.dwt(x)
        # x_1: n c 3 h/2 w/2
        # x_2: n c 3 h/4 w/4
        # x_3: n c 3 h/8 w/8
        assert (N, C, 3, H//2, W//2) == x_1.shape
        assert (N, C, 3, H//4, W//4) == x_2.shape 
        assert (N, C, 3, H//8, W//8) == x_3.shape

        # x_2 = F.interpolate(x, scale_factor=0.5)
        # x_3 = F.interpolate(x_2, scale_factor=0.5)
        x_1 = rearrange(x_1, 'b c m h w -> b (c m) h w')
        x_2 = rearrange(x_2, 'b c m h w -> b (c m) h w')
        x_3 = rearrange(x_3, 'b c m h w -> b (c m) h w')

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_3)

        outputs = list()

        x_ = self.feat_extract[0](x_1)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        o3 = z_ + x_3
        o3 = rearrange(o3, 'b (c m) h w -> b c m h w', m=3)
        outputs.append(o3)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        o2 = z_ + x_2
        o2 = rearrange(o2, 'b (c m) h w -> b c m h w', m=3)
        outputs.append(o2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        o1 = z + x_1
        o1 = rearrange(o1, 'b (c m) h w -> b c m h w', m=3)
        outputs.append(o1)

        return outputs

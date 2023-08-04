import torch
import torch.nn as nn
from functools import partial

from Models import pvt_v2
from timm.models.vision_transformer import _cfg

import numpy as np

#The Adaptive Feature Aggregation Module（AFA）
class FusionModule(nn.Module):
    def __init__(self, in_channels_F2, in_channels_F3, in_channels_F4, *args, **kwargs):
        super(FusionModule, self).__init__(*args, **kwargs)

        self.conv_reduce_F2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_F2, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(32,32),
            nn.SiLU()
        )

        self.conv_reduce_F3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_F3, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(32,32),
            nn.SiLU()
        )

        self.conv_reduce_F4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_F4, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(32,32),
            nn.SiLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(32,32),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(32,32),
            nn.SiLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(64,64),
            nn.SiLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(64,64),
            nn.SiLU()
        )
        
    def forward(self, F2, F3, F4):
        F2_reduced = self.conv_reduce_F2(F2)
        F3_reduced = self.conv_reduce_F3(F3)
        F4_reduced = self.conv_reduce_F4(F4)

        F4_up2 = nn.functional.interpolate(F4_reduced, scale_factor=2, mode='bilinear', align_corners=False)
        F4_conv1 = self.conv1(F4_up2)
        F4_conv2 = self.conv2(F4_up2)

        F3_mul_F4 = F3_reduced * F4_conv1
        concat_F3_F4 = torch.cat((F3_mul_F4, F4_conv2), dim=1)
        M2 = self.conv3(concat_F3_F4)

        F2_up2 = nn.functional.interpolate(F3_reduced, scale_factor=2, mode='bilinear', align_corners=False)
        F2_conv1 = self.conv1(F2_up2)

        F4_up4 = nn.functional.interpolate(F4_reduced, scale_factor=4, mode='bilinear', align_corners=False)
        F4_conv3 = self.conv1(F4_up4)

        F2_mul_F3_F4 = F2_reduced * F2_conv1 * F4_conv3
        M2_up = nn.functional.interpolate(M2, scale_factor=2, mode='bilinear', align_corners=False)
        
        concat_M1_M2 = torch.cat((F2_mul_F3_F4, M2_up), dim=1)
        M_out = self.conv4(concat_M1_M2)

        return M_out

#The ResidualBlock Module
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 2, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(out_channels // 2, out_channels)
        )

        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels // 2, out_channels)
        )

    def forward(self, x):
        main_out = self.main_conv(x)
        skip_out = self.skip_conv(x)

        if main_out.shape[2:] != skip_out.shape[2:]:
            skip_out = nn.functional.interpolate(skip_out, size=main_out.shape[2:], mode='bilinear', align_corners=False)

        out = main_out + skip_out
        out = nn.SiLU(inplace=True)(out)

        return out

#The Selective Global-to-Local Fusion Head Module（SGLFH）
class ResPVTHead(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(ResPVTHead, self).__init__()

        self.concat_reduce = nn.Sequential(
            nn.Conv2d(in_channels1 + in_channels2, in_channels1, kernel_size=1, padding=0),
            nn.GroupNorm(in_channels2 // 2, in_channels2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels1, in_channels1, kernel_size=1, padding=0),
            nn.GroupNorm(in_channels2 // 2, in_channels2),
            nn.SiLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x2 = nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2), dim=1)
        x = self.concat_reduce(x)
        out = self.upsample(x)
        return out

#The FCN Branch
class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 4, 4],
        n_levels_down=6,
        n_levels_up=6,
        n_RBs=2,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(ResidualBlock(ch, min_channel_mult * min_level_channels))
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2))
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(ResidualBlock(ch, ch), ResidualBlock(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    ResidualBlock(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                    )
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))


    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h

#The Transformer Branch
class TB(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = pvt_v2.PyramidVisionTransformerV2(
            patch_size=4,
            embed_dims=[64, 128, 320, 512], 
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
        )

        checkpoint = torch.load("pvt_v2_b3.pth") 

        backbone.default_cfg = _cfg()
        backbone.load_state_dict(checkpoint)
        self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]

        for i in [1, 4, 7, 10]:
            self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

        self.FM = FusionModule(128, 320, 512)
        self.RB = ResidualBlock(64, 64)
        self.ResPVTHead = ResPVTHead(64, 64)

    def get_pyramid(self, x):
        pyramid = []
        B = x.shape[0]
        for i, module in enumerate(self.backbone):
            if i in [0, 3, 6, 9]:
                x, H, W = module(x)
            elif i in [1, 4, 7, 10]:
                for sub_module in module:
                    x = sub_module(x, H, W)
            else:
                x = module(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                pyramid.append(x)

        return pyramid

    def forward(self, x):
        pyramid = self.get_pyramid(x)

        M1_out = self.FM(pyramid[1], pyramid[2], pyramid[3])

        O2 = self.RB(pyramid[0])

        output = self.ResPVTHead(M1_out, O2)

        return output

#The DuaPSNet
class DuaPSNet(nn.Module):
    def __init__(self, size=352):

        super().__init__()

        self.TB = TB()

        self.FCB = FCB(in_resolution=size)
        self.PH = nn.Sequential(
            ResidualBlock(64 + 32, 64), ResidualBlock(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )
        self.up_tosize = nn.Upsample(size=size)

    def forward(self, x):
        x1 = self.TB(x)
        x2 = self.FCB(x)
        x1 = self.up_tosize(x1)
        x = torch.cat((x1, x2), dim=1)
        out = self.PH(x)

        return out


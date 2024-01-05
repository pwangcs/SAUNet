import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import einops
import scipy.io as scio
from model.saconv import SAConv
from model.utils import LayerNorm2d, SimpleGate, EuclideanProj



######################################### Self-Attention Convolution (SAC) Bloack ###########################

class SACBlock(nn.Module):
    def __init__(self, c, FFN_Expand=2, lightweight=False, drop_out_rate=0.):
        super(SACBlock,self).__init__()

        if not lightweight:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4, bias=True)
            )
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
            )

        # Convolutional Spatial Attention
        self.csa = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            SAConv(in_channels=c, out_channels=c, base_size=8, kernel_size=3, padding=1, stride=1, group=c, bias=True)
        )
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.conv11 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Feed-Forward Network
        ffn_channel = FFN_Expand * c
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            SimpleGate(),
            nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

        # SimpleGate
        # self.sg = SimpleGate()

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        # x = self.dwconv(x)
        # x = self.sg(x)
        # x = self.csa(x)* self.sca(x)
        x = self.dwconv(x) * self.csa(x) * self.sca(x)
        x = self.conv11(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.ffn(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class BasicBlock(nn.Module):
    def __init__(self, color_channels=1, width=64, middle_blk_num=6, enc_blk_nums=[1,2,4], dec_blk_nums=[1,1,1], first_stage=False):
        super(BasicBlock, self).__init__()
        self.first_stage = first_stage
        self.embedding = nn.Conv2d(in_channels=color_channels, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        if not first_stage:
            self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.mapping = nn.Conv2d(in_channels=width, out_channels=color_channels, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            if not first_stage:
                self.convs1.append(
                    nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
                )
            self.encoders.append(
                nn.Sequential(
                    *[SACBlock(chan, FFN_Expand=2, lightweight=False) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(
                *[SACBlock(chan, FFN_Expand=2, lightweight=False) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.convs2.append(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
            )
            self.decoders.append(
                nn.Sequential(
                    *[SACBlock(chan, FFN_Expand=2, lightweight=False) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, last_decs=None):
        B, C, H, W = inp.shape

        inp = self.check_image_size(inp)

        x = self.embedding(inp)

        encs = []
        decs = []

        if not self.first_stage:
            for encoder, down, conv1, last_dec in zip(self.encoders, self.downs, self.convs1, last_decs[::-1]):
                x = conv1(torch.cat([x, last_dec], dim=1))
                x = encoder(x)
                encs.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x)
                encs.append(x)
                x = down(x)

        x = self.middle_blks(x)

        for decoder, up, conv2, enc_skip in zip(self.decoders, self.ups, self.convs2, encs[::-1]):
            x = up(x)
            x = conv2(torch.cat([x, enc_skip], dim=1))
            x = decoder(x)
            decs.append(x)

        x = self.mapping(x)
        x = x + inp

        return x[:, :, :H, :W], decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class HyPaNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=7, c=64, lightweight=False):
        super(HyPaNet, self).__init__()
        self.out_nc = out_nc
        self.feature1 = nn.Conv2d(in_channels=in_nc, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.feature2 = nn.Conv2d(in_channels=c, out_channels=out_nc, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.norm = LayerNorm2d(c)

        if not lightweight:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4, bias=True)
            )
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c, bias=True),
            )
        # Convolutional Spatial Attention
        self.csa = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            SAConv(in_channels=c, out_channels=c, base_size=8, kernel_size=3, padding=1, stride=1, group=c, bias=True)
        )        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        self.conv11 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.norm(self.feature1(x))
        # x = self.dwconv(x)
        # x = self.sg(x)
        # x = self.csa(x)* self.sca(x)
        x = self.dwconv(x) * self.csa(x) * self.sca(x)
        x = self.feature2(self.conv11(x))
        y = self.softplus(x)

        return y



class SAUNet(nn.Module):
    def __init__(self, imag_size, meas_size, img_channels=1, channels=64, mid_blocks=6, enc_blocks=[1,2,4], dec_blocks=[1,1,1], stages=7, matrix_train=True, only_test=False):
        super(SAUNet, self).__init__()
        self.stages = stages
        self.imag_size = imag_size
        self.only_test = only_test

        self.H = nn.Parameter(torch.ones(meas_size[0], imag_size[0]), requires_grad=matrix_train)
        self.W = nn.Parameter(torch.ones(meas_size[1], imag_size[1]), requires_grad=matrix_train)
        torch.nn.init.normal_(self.H, mean=0, std=0.1)
        torch.nn.init.normal_(self.W, mean=0, std=0.1)

        self.global_estimator = HyPaNet(in_nc=img_channels, out_nc=stages, c=64)
        self.stage_estimator = HyPaNet(in_nc=img_channels, out_nc=img_channels, c=64)
        self.cons = nn.Parameter(torch.Tensor([0.5]).repeat(stages))

        self.denoisers = nn.ModuleList([])
        self.denoisers.append(BasicBlock(color_channels=img_channels,width=channels, middle_blk_num=mid_blocks,
                                    enc_blk_nums=enc_blocks,dec_blk_nums=dec_blocks,first_stage=True))
        for i in range(stages-1):
            self.denoisers.append(BasicBlock(color_channels=img_channels,width=channels, middle_blk_num=mid_blocks,
                                    enc_blk_nums=enc_blocks,dec_blk_nums=dec_blocks))

    def forward(self, X):
        """
        :input X: [b,h,w]
        """
        X = X.unsqueeze(1)
        b, c, h, w = X.shape

        H = self.H
        W = self.W

        HT = torch.transpose(H, 0, 1).contiguous()
        WT = torch.transpose(W, 0, 1).contiguous()

        Y = torch.matmul(torch.matmul(H.repeat((b,c,1,1)),X),WT.repeat((b,c,1,1)))
        X = torch.matmul(torch.matmul(HT.repeat((b,c,1,1)),Y),W.repeat((b,c,1,1)))

        mu_all = self.global_estimator(X)   # [b, stages, h, w]

        for i in range(self.stages):

            delta_mu = self.stage_estimator(X)
            mu = mu_all[:,i:i+1,:,:] + self.cons[i]*delta_mu

            Z = EuclideanProj(X,Y,H,W,HT,WT,mu)

            if self.only_test and Z.shape[0]>1:
                Z = einops.rearrange(Z,'(a b) 1 h w-> 1 1 (a h) (b w)',a=2,b=2)
            if i==0:
               X, features = self.denoisers[i](Z)
            else:     
                X, features = self.denoisers[i](Z,features)
            if self.only_test and X.shape[2]>h:
                X = einops.rearrange(X,'1 1 (a h) (b w)-> (a b) 1 h w',a=2,b=2)

        X = X.squeeze(1)

        return X, H, W, HT, WT

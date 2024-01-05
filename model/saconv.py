import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from model.utils import SimpleGate


class SAConv(nn.Module):
    def __init__(self, in_channels, out_channels, base_size=8, kernel_size=3, padding=1, stride=1, group=1, bias=True):
        super(SAConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_size = base_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.group = group
        self.bias = bias
        
        self.fuse = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0, stride=1,groups=1,bias=True)

        interc = min(self.in_channels, 32)
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.base_size, 1, padding=0, stride=1))
        self.attention2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.base_size, kernel_size=1)

        self.map_bata = nn.Parameter(torch.zeros((1, self.base_size, 1, 1)) + 1e-2, requires_grad=True)

        self.convlist = nn.ModuleList()
        for i in range(self.base_size):
            self.convlist.append(nn.Conv2d(in_channels=self.in_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            padding=self.padding,
                                            stride=self.stride,
                                            groups=self.group,
                                            bias=self.bias))

    def forward(self,x):
        b,c,h,w = x.shape
        map = self.attention1(x)*self.map_bata + self.attention2(x)
        x = self.fuse(x)
        y = map[:,0:1,:,:]*self.convlist[0](x) \
            + map[:,1:2,:,:]*self.convlist[1](x) \
            + map[:,2:3,:,:]*self.convlist[2](x) \
            + map[:,3:4,:,:]*self.convlist[3](x) \
            + map[:,4:5,:,:]*self.convlist[4](x) \
            + map[:,5:6,:,:]*self.convlist[5](x) \
            + map[:,6:7,:,:]*self.convlist[6](x) \
            + map[:,7:8,:,:]*self.convlist[7](x) \
            # + map[:,8:9,:,:]*self.convlist[8](x) \
            # + map[:,9:10,:,:]*self.convlist[9](x) \
            # + map[:,10:11,:,:]*self.convlist[10](x) \
            # + map[:,11:12,:,:]*self.convlist[11](x) \
            # + map[:,12:13,:,:]*self.convlist[12](x) \
            # + map[:,13:14,:,:]*self.convlist[13](x) \
            # + map[:,14:15,:,:]*self.convlist[14](x) \
            # + map[:,15:16,:,:]*self.convlist[15](x) 
        # y = torch.zeros((b,self.out_channels,h,w)).to(x.device)
        # for j in range(self.base_size):
        #     y += map[:,j:j+1,:,:]*self.convlist[j](x)
        return y





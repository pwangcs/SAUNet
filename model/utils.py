import torch
import torch.nn as nn

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # print('mu, var', mu.mean(), var.mean())
        # d.append([mu.mean(), var.mean()])
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimpleCA(nn.Module):
    def __init__(self, channel):
        super(SimpleCA, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

    def forward(self, x):
        y = self.sca(x)
        return x * y


def EuclideanProj(X,Y,H,W,HT,WT,mu):
    b, c, _, _ = X.shape
    Delta_Y = Y - torch.matmul(torch.matmul(H.repeat((b,c,1,1)),X),WT.repeat((b,c,1,1)))
    Delta_X = torch.matmul(torch.matmul(HT.repeat((b,c,1,1)),Delta_Y),W.repeat((b,c,1,1)))
    Z = X + torch.div(Delta_X, mu+1)
    return Z

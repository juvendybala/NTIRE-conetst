import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from utils import set_manual_seed

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        # print(weight.shape)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    


class NAFBlock(nn.Module):
    def __init__(self, c=32, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            # nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
            #           groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = nn.GELU()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # self.norm1 = nn.BatchNorm2d(num_features=c)
        # self.norm2 = nn.BatchNorm2d(num_features=c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
            x = inp
            # print(x.shape)
            x = self.norm1(x)

            x = self.conv1(x)
            x = self.conv2(x)
            x = self.sg(x)
            x = x * self.sca(x)
            x = self.conv3(x)

            x = self.dropout1(x)

            y = inp + x * self.beta

            x = self.conv4(self.norm2(y))
            x = self.sg(x)
            x = self.conv5(x)

            x = self.dropout2(x)

            return y + x * self.gamma

class DropPath(nn.Module):
    def __init__(self, drop_rate, module, seed=42):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module
        self.rng = np.random.RandomState(seed)

    def forward(self, x):
        if self.training and self.rng.rand() < self.drop_rate:
            return x

        new_x = self.module(x)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        if self.training and factor != 1.:
            new_x = x + factor * (new_x - x)
        return new_x


class NAFSR(nn.Module):
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_out_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.body = nn.Sequential(*[DropPath(drop_rate=drop_path_rate, module=NAFBlock(c=width,drop_out_rate=drop_out_rate)) for _ in range(num_blks)])
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')
        feat = self.intro(inp)
        feat = self.body(feat)
        out = self.up(feat)
        out = out + inp_hr
        return out
    


if __name__ == "__main__":
    set_manual_seed(0)
    if(torch.cuda.is_available):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = NAFSR(up_scale=2, num_blks=6, img_channel=4).to(device=device)
    model.eval()
    with torch.no_grad():
        result = []
        for i in range(10):
            input = torch.rand(1, 4, 128, 128).to(device=device)
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            print(f"the {i}th iteration time is: {inference_time:.2f}ms")
            result.append(inference_time)
        print(f"the average inference time is {sum(result) / len(result)} ms")
    # output = model(input)
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"可训练参数数量: {total_params}")
    # print(output.shape)
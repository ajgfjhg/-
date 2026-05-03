'''#以下为使用预训练权重的 ConvNeXt V2 模型定义
import timm
import torch.nn as nn

def convnext_v2_tiny(num_class):
    # 使用 timm 加载带权重的 ConvNeXt V2
    return timm.create_model('convnextv2_tiny', pretrained=False, num_classes=num_class)
'''
#以下为 ConvNeXt V2 不使用预训练权重的模型定义
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):

    #专门为卷积层设计的 LayerNorm。
    #支持两种数据格式: 
    #'channels_last' (N, H, W, C) 或 'channels_first' (N, C, H, W)

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 针对 [N, C, H, W] 维度的归一化逻辑
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    # ConvNeXt V2 核心：Global Response Normalization
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.LayerNorm(dim, eps=1e-6) # 这里后面会通过 permute 处理
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # [N, H, W, C] -> [N, C, H, W]
        return input + x

class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=29, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.downsample_layers = nn.ModuleList() 
        # 第一层 Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first") # 使用自定义 LayerNorm
        )
        self.downsample_layers.append(stem)
        
        # 之后的下采样层
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = self.norm(x.mean([-2, -1])) # 全局池化
        return self.head(x)

def convnext_v2_tiny(num_class):
    return ConvNeXtV2(num_classes=num_class)

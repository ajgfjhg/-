"""#以下为使用预训练权重的 ViT 模型定义
import timm
import torch.nn as nn

def vit_tiny(num_class):
    # ViT 手写结构很难加载权重，通过 timm 获得 ImageNet 预训练能力
    return timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=num_class)
"""
#以下为不使用预训练权重的 ViT 模型定义
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    #将图片切成小块 (Patches) 并进行线性投影 
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=29, 
                 embed_dim=384, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 类别占位符 (CLS Token) 和 位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # 拼接 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 加上位置信息
        x = self.pos_drop(x + self.pos_embed)

        # 进入 Transformer 模块
        x = self.blocks(x)
        
        x = self.norm(x)
        # 只取 CLS Token 的输出进行分类
        return self.head(x[:, 0])

def vit_tiny(num_class):
    # 针对 1.5w 张图的数据集，适当调高了 dropout 比例
    return VisionTransformer(
        img_size=224, 
        patch_size=16, 
        embed_dim=384, 
        depth=6, 
        num_heads=8, 
        num_classes=num_class,
        dropout=0.2,       # 提高随机失活率
        mlp_ratio=4.0      # 保持 MLP 膨胀率
    )

"""#以下为使用预训练权重的swin模型定义
import timm
import torch.nn as nn

def swin_tiny(num_class):
    # 保留你的接口逻辑，内部调用 timm 获取预训练权重
    return timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_class)
    """
#以下为不使用预训练权重的swin模型定义
# Swin.py
import torch
import torch.nn as nn
import timm

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=29, pretrained=False):
        #基于 Swin Transformer Tiny 的分类模型
        #num_classes: 分类数量（你的项目是 29）
        #pretrained: 是否使用 ImageNet 预训练权重
        super(SwinTransformerModel, self).__init__()
        
        # 使用 timm 创建 swin_tiny 模型
        # swin_tiny_patch4_window7_224 是最常用的版本，输入大小为 224x224
        print(f"正在加载 Swin Transformer 预训练模型 (pretrained={pretrained})...")
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x):
        # x 的输入维度预期为 [Batch_Size, 3, 224, 224]
        return self.model(x)

def swin_tiny(num_class):
    #供 train_model.py 调用的接口函数
    return SwinTransformerModel(num_classes=num_class, pretrained=False)

if __name__ == '__main__':
    # 简单的模型测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化模型
    model = swin_tiny(num_class=29).to(device)
    # 模拟一张 224x224 的图片输入
    test_input = torch.randn(1, 3, 224, 224).to(device)
    # 前向传播
    output = model(test_input)
    print("\n--- 模型测试结果 ---")
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape} (预期为 [1, 29])")

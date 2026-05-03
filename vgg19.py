# vgg19.py
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, input_channel=3, num_class=29):
        super(VGG, self).__init__()

        # 定义卷积层部分 (VGG19 特点是 16 个卷积层)
        # 即使 img_size 改变，卷积层结构也是固定的
        self.conv1 = self._make_conv_block(input_channel, 64)
        self.conv2 = self._make_conv_block(64, 64, pool=True)
        
        self.conv3 = self._make_conv_block(64, 128)
        self.conv4 = self._make_conv_block(128, 128, pool=True)
        
        self.conv5 = self._make_conv_block(128, 256)
        self.conv6 = self._make_conv_block(256, 256)
        self.conv7 = self._make_conv_block(256, 256)
        self.conv8 = self._make_conv_block(256, 256, pool=True)
        
        self.conv9 = self._make_conv_block(256, 512)
        self.conv10 = self._make_conv_block(512, 512)
        self.conv11 = self._make_conv_block(512, 512)
        self.conv12 = self._make_conv_block(512, 512, pool=True)
        
        self.conv13 = self._make_conv_block(512, 512)
        self.conv14 = self._make_conv_block(512, 512)
        self.conv15 = self._make_conv_block(512, 512)
        self.conv16 = self._make_conv_block(512, 512, pool=True)

        # 【关键点 1】：自适应池化层
        # 无论输入的图片多大，经过上面的卷积池化后，输出会被强制调整为 7x7 尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 【关键点 2】：分类层（全连接层）
        # 512（通道数）* 7 * 7（指定的宽高）= 25088
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_class),
        )

        # 方便 forward 中循环调用
        self.conv_list = nn.ModuleList([
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, 
            self.conv6, self.conv7, self.conv8, self.conv9, self.conv10,
            self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, self.conv16
        ])

    def _make_conv_block(self, in_c, out_c, pool=False):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 经过 16 个卷积块
        for conv in self.conv_list:
            x = conv(x)
        
        # 经过自适应池化
        x = self.avgpool(x)
        
        # 展平数据：从 [batch, 512, 7, 7] 变成 [batch, 25088]
        x = torch.flatten(x, 1)
        
        # 经过全连接层
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # 测试模型是否能跑通
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG(num_class=29).to(device)
    # 模拟一个 64x64 的输入
    test_input = torch.randn(1, 3, 64, 64).to(device)
    output = model(test_input)
    print("模型输出形状:", output.shape) # 应该是 [1, 29]

def vgg19(num_class):
    return VGG(num_class=num_class)
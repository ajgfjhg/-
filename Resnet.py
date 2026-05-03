import torch
from torch import nn
import torch.nn.functional as F
'''
两层结构，18层和34层网络使用
'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                               kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(Bottleneck,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                               kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels=out_channels,
                out_channels=out_channels*self.expansion,
                               kernel_size=1,stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample = downsample

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,block_nums,num_class=10):
        super(ResNet,self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=3,
            out_channels=self.in_channel,kernel_size=7,
                               stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self.ResLayer(block, 64, block_nums[0])
        self.layer2 = self.ResLayer(block, 128, block_nums[1])
        self.layer3 = self.ResLayer(block, 256, block_nums[2])
        self.layer4 = self.ResLayer(block, 512, block_nums[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion,num_class)



    def ResLayer(self,block,channel,block_num,stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel* block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel,
                          out_channels=channel*block.expansion,
                          kernel_size=1,stride=stride),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers = []
        block_temp = block(self.in_channel,channel,
                           downsample=downsample,stride=stride)
        layers.append(block_temp)

        self.in_channel = channel*block.expansion

        for _ in range(1,block_num):
            block_temp = block(self.in_channel,channel)
            layers.append(block_temp)

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

def resnet18(num_class):
    net18 = ResNet(BasicBlock,[2, 2, 2, 2],num_class=num_class)
    return net18
def resnet34(num_class):
    resnet34 = ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class)
    return resnet34
def resnet50(num_class):
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], num_class=num_class)
    return resnet50
def resnet101(num_class):
    resnet101 = ResNet(Bottleneck, [3, 4, 23, 3], num_class=num_class)
    return resnet101
# net101 = ResNet(Bottleneck, [3, 4, 23, 3], num_class=10)
def resnet152(num_class=59):
     resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], num_class=num_class)
     return resnet152
if __name__ == '__main__':
    # a = torch.rand(6,64,56,56)
    # downsample = nn.Sequential(
    #     nn.Conv2d(in_channels=64,out_channels=128,stride=2,
    #               kernel_size=3,padding=1),
    #     nn.BatchNorm2d(128)
    # )#当stride != 1 or inchannel != out_channel 通道数要发生变化或特征图大小发生变化
    # net = BasicBlock(64,128,2,downsample)
    # print(net)
    # print(net(a).shape)

    # a = torch.rand(6, 256, 56, 56)
    # downsample = nn.Sequential(
    #     nn.Conv2d(in_channels=64, out_channels=256, stride=1,
    #               kernel_size=3, padding=1),
    #     nn.BatchNorm2d(256)
    # )  # 当stride != 1 or inchannel != out_channel*4 通道数要发生变化或特征图大小发生变化
    # net = Bottleneck(256, 64, 1, None)
    # print(net)
    # print(net(a).shape)

    resnet50 = resnet50(33)
    print(resnet50)
    pass




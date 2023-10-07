import torch.nn as nn
import torch

#18层和34层的残差结构
class BasicBlock(nn.Module):
    #残差结构当中主分支若采用的卷积核的个数    决定了输出的维度 
    expansion = 1   
    #downsample=None  下采样函数 对应虚线的残差结构 1x1的卷积核
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)#kernel_size卷积核大小。bias=False不使用偏执参数
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()       #定义了激活函数
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample    #定义下采样方法   默认等于none
    #正向传播的过程
    def forward(self, x):
        identity = x    #identity 捷径分支上的输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x) #卷积层1    
        out = self.bn1(out) #BN层1
        out = self.relu(out)

        out = self.conv2(out)#卷积层2
        out = self.bn2(out)#BN层2

        out += identity #输出+ 上捷径上的输出之后 通过relu激活函数
        out = self.relu(out)

        return out

#50层101层的残差结构
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    #正向传播过程
    def forward(self, x):  #x 输入的特征矩阵
        identity = x    #identity捷径分支
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

        out += identity
        out = self.relu(out)

        return out

#定义ResNet的整个网络的框架部分
class ResNet(nn.Module):
        #blocks_num, 对应的列表参数
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=8,#训练集的类别个数
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 #定义特征的深度

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, #kernel_size=7 7x7的卷积核
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#为了使输出特征矩阵的高和宽缩减到原来的一半，这里的padding为1
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)#全连接层

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1): #第一层卷积核的个数channel 。block_num=3多少个残差结构
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion: #对于18，34层来说 block.expansion等于1.对于50，101层来说 block.expansion等于4
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,#输入特征矩阵的深度
                                channel,#残差结构主分之第一层卷积核个数
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)#非关键字参数的方式 写入到nn.Sequential
#正向传播过程
    def forward(self, x):
        x = self.conv1(x)#卷积层1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)#最大池化下采样3x3的

        x = self.layer1(x)#对应的conv2
        x = self.layer2(x)#对应的conv3
        x = self.layer3(x)#对应的conv4
        x = self.layer4(x)#对应的conv5

        if self.include_top:
            x = self.avgpool(x)#平均池化下采样
            x = torch.flatten(x, 1)#进行展平处理
            x = self.fc(x)#进行全连接

        return x
#定义34层的网络
def resnet34(num_classes=8, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=8, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=8, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=8, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

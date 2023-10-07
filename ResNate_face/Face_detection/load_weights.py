import os
import torch
import torch.nn as nn
from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    net = resnet34()
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 8)

    # option2
    # net = resnet34(num_classes=8)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    #
    # for key in del_key:
    #     del pre_weights[key]
    #
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
# 首先，它导入了需要的模块和类，包括os、torch和torch.nn，以及resnet34模型。
# 接下来，在main函数中，首先确定所使用的设备（CPU或GPU）。
# 然后，代码加载了预训练的权重文件resnet34-pre.pth，并使用assert语句确保该文件存在。如果文件不存在，将抛出异常并终止程序执行。
# 接下来，有两种选项用于修改模型的全连接层结构。
# 选项1：
# 代码创建一个resnet34模型的实例net。
# 然后，通过load_state_dict方法加载预训练的权重文件到模型中，同时指定map_location参数将模型加载到所选择的设备上。
# 最后，修改模型的全连接层结构，将输入通道数替换为8。
# 选项2：
# 代码通过创建一个resnet34模型的实例net，并设置num_classes参数为8来加以修改。
# 然后，使用torch.load方法加载权重文件到变量pre_weights中，并使用map_location参数将权重加载到所选择的设备上。
# 然后，代码通过遍历权重字典中的键，删除所有与全连接层相关的键。
# 最后，使用load_state_dict方法加载修改后的权重字典到模型中，并通过strict=False来避免对加载过程中的严格匹配。
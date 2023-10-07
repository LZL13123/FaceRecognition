#torch模块是PyTorch深度学习框架的核心模块，提供了张量操作、模型构建、模型训练和模型预测等功能。
# 在这段代码中，torch模块用于构建和加载ResNet-34模型，进行模型的预测操作
#matplotlib.pyplot模块提供了绘图和可视化相关的功能，例如显示图像、绘制曲线等。
# 在这段代码中，matplotlib.pyplot模块用于显示原始图像，并展示识别结果
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from model import resnet34
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 定义类别和输出中文
class_dict = {
    "0": "此人像有胡子",
    "1": "性别:女",
    "2": "是长头发",
    "3": "性别:男",
    "4": "没有胡子",
    "5": "是短头发",
    "6": "配戴眼镜",
    "7": "不配戴眼镜",
}

# 定义图像转换操作：使用transforms.Compose()函数将多个数据转换操作组合起来
# 形成一个转换操作的序列。通过调整图像大小、中心裁剪、转换为张量，以及归一化等操作将原始图像转换为模型所需的输入形式。
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#使用GPu或者使用cpu

    model = resnet34(num_classes=8).to(device)
    weights_path = "resNet34.pth"  # 模型权重文件路径
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()

    # 选择图片文件夹路径
    img_folder_path = "image"  # 图片文件夹路径
    assert os.path.exists(img_folder_path), "folder: '{}' does not exist.".format(img_folder_path)

    # 遍历文件夹中的每个文件，通过os.listdir()函数遍历指定路径下的所有文件
    for filename in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, filename) #使用os.path.join()函数将图像路径和文件名拼接起来，得到完整的图像文件路径

        if os.path.isfile(img_path):
            # 读取图像并进行预处理
            img = Image.open(img_path)
            img_original = img.copy()  # 创建图像的副本以显示原始图像
    #进行结果解析和显示
    # 对预测输出的tensor进行解析和处理。
    # 通过torch.argmax()函数找到概率最高的类别索引，并通过class_dict 字典获取对应的人脸特征描述。
    # 通过打印输出显示识别结果
            img = data_transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)

            with torch.no_grad():
                output = torch.squeeze(model(img)).cpu()
                predict = torch.softmax(output, dim=0)
                predict = predict.numpy()

                # 显示预测结果
                print("模型识别"+"图片{}人脸特征如下:".format(filename))
                for i in range(len(predict)):
                    if predict[i] > 0.1:
                        print("       "+class_dict[str(i)])

            # 显示原始图片：使用matplotlib.pyplot库的plt.imshow()方法显示原始图像。
            # 通过plt.axis('off')方法去除坐标轴的显示。
            plt.imshow(img_original)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()






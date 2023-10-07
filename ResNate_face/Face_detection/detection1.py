#此代码是将模型识别输出结果到features.txt中
import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from model import resnet34
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet34(num_classes=8).to(device)
    weights_path = r"D:\Software\ResNet\image processing\resNet34.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    
    feature_file = open("features.txt", "w", encoding='utf-8')

    img_folder_path = r"C:\Users\LZL\Desktop\imageprocessing\ResNate_face\Face_detection\image"
    assert os.path.exists(img_folder_path), "folder: '{}' does not exist.".format(img_folder_path)

    for filename in os.listdir(img_folder_path):
        img_path = os.path.join(img_folder_path, filename)

        if os.path.isfile(img_path):
            img = Image.open(img_path)
            img_original = img.copy()

            img = data_transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)

            with torch.no_grad():
                output = torch.squeeze(model(img)).cpu()
                predict = torch.softmax(output, dim=0)
                predict = predict.numpy()

                feature_file.write("图片{}人脸特征如下:\n".format(filename))
                for i in range(len(predict)):
                    if predict[i] > 0.1:
                        feature_file.write("       "+class_dict[str(i)]+"\n")

            plt.imshow(img_original)
            plt.axis('off')
            plt.show()
            
    feature_file.close()

if __name__ == '__main__':
    main()

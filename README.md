# FaceRecognition
项目描述：通过实习学习PyTorch框架和ResNet模型，我自己做了一个简单的人脸识别项目。这个项目涉及数据收集、预处理、模型训练和识别预测。我使用PyTorch构建了ResNet模型，用于提取和识别人脸特征。
在数据处理方面，我对图像进行了缩放、裁剪、增强和归一化等操作，以确保数据的准确性。项目中的识别脚本具有GUI界面，可以可视化识别图片，还能批量识别图片并将结果保存到txt文件中。最后，我还实现了实时人脸特征检测，可以调用摄像头进行处理，并通过GUI界面进行实时可视化。
face_data文件夹是存放数据集的文件夹,其中split.py脚本是为了对训练集和测试集进行分类。
ResNate_face文件夹是存放代码数据文件夹
class_indices.json--类别信息
detection_GUI.py--图形化界面检测人脸特征
detection.py--控制台输出检测人脸信息
detection1.py--将模型识别输出结果到features.txt中
Detector.py--调用摄像头进行3s的一次人脸特征检测
model.py--有ResNet18-101模型代码的脚本
resNet34.pth--训练好的模型
train.py--训练脚本(没有使用迁移学习的方法训练)

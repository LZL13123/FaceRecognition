# import os
# import torch
# import tkinter as tk
# from tkinter import filedialog
# from PIL import ImageTk, Image
# from torchvision import transforms
# from model import resnet34
#
# class_dict = {
#     "0": "此人像有胡子",
#     "1": "性别:女",
#     "2": "是长头发",
#     "3": "性别:男",
#     "4": "没有胡子",
#     "5": "是短头发",
#     "6": "配戴眼镜",
#     "7": "不配戴眼镜",
# }
#
# data_transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = resnet34(num_classes=8).to(device)
#
# weights_path = "resNet34.pth"
# assert os.path.exists(weights_path), "文件 '{}' 不存在.".format(weights_path)
# model.load_state_dict(torch.load(weights_path, map_location=device))
#
# model.eval()
# img_path = None
#
#
# def choose_image():
#     global img_path
#     img_path = filedialog.askopenfilename(initialdir="image", title="选择图片文件",
#                                           filetypes=(("jpg files", "*.png"), ("all files", "*.*")))
#     if img_path:
#         img = Image.open(img_path)
#         img_original = img.copy()
#         img_original = ImageTk.PhotoImage(img_original)
#         image_label.configure(image=img_original)
#         image_label.image = img_original
#
#
# def analyse_image():
#     if img_path:
#         img = Image.open(img_path)
#         img = data_transform(img)
#         img = torch.unsqueeze(img, 0)
#         img = img.to(device)
#
#         with torch.no_grad():
#             output = torch.squeeze(model(img)).cpu()
#             predict = torch.softmax(output, dim=0)
#             predict = predict.numpy()
#
#             result_text.delete("1.0", tk.END)
#             for i in range(len(predict)):
#                 if predict[i] > 0.1:
#                     result_text.insert(tk.END, class_dict[str(i)] + "\n")
#
#
# window = tk.Tk()
# window.title("人像识别")
# window.geometry("800x400")
#
# # 创造左侧的面板并添加内容
# left_panel = tk.Frame(window)
# choose_button = tk.Button(left_panel, text="选择图片", command=choose_image)
# choose_button.pack(pady=10)
# analyse_button = tk.Button(left_panel, text="点击识别", command=analyse_image)
# analyse_button.pack(pady=10)
# image_label = tk.Label(left_panel)
# image_label.pack()
# left_panel.pack(side=tk.LEFT)
#
# # 创建右侧的面板并添加内容
# right_panel = tk.Frame(window)
# result_label = tk.Label(right_panel, text="识别结果：")
# result_label.pack()
# result_text = tk.Text(right_panel, height=22, width=30)
# result_text.pack()
# right_panel.pack(side=tk.RIGHT)
#
# window.mainloop()

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
from model import resnet34
from torchvision.transforms import transforms
import torch
import torch.nn.functional as F


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.classes_to_results = {
            "Class_1": "此人像有胡子",
            "Class_2": "性别：女",#
            "Class_3": "是长头发",
            "Class_4": "性别：男",#
            "Class_5": "没有胡子",
            "Class_6": "是短头发",
            "Class_7": "佩戴眼镜",
            "Class_8": "不配戴眼镜",
        }

        self.model = self.load_model()
        self.last_pred_time = 0

        self.vcap = cv2.VideoCapture(0)
        self.width = self.vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)

        self.label = tk.Label(window, width=30,justify='left',anchor='w',font=14,fg='red')
        self.label.grid(row=0, column=1)

        self.update_frame()
        self.window.mainloop()

    def load_model(self):
        model = resnet34(num_classes=len(self.classes_to_results)).to(self.device)
        weights_path = r"resNet34.pth"
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def update_frame(self):
        ret, frame = self.vcap.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            if time.time() - self.last_pred_time >= 3:  # If it's been 3 seconds since the last prediction
                preds = self.predict(frame)
                if preds:
                    pred_text = '人脸特征模型识别结果：\n'
                    for i, pred in enumerate(preds, 1):
                        pred_text += f'{i}. {pred}\n'
                        print(f'{i}. {pred}')
                    self.label.config(text=pred_text)
                else:
                    self.label.config(text='No result with probability > 0.01')
                self.last_pred_time = time.time()

        self.window.after(1, self.update_frame)

    def predict(self, frame):
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = data_transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        with torch.no_grad():
            predictions = F.softmax(self.model(img).cpu(), dim=1)
        results = []
        for i, pred in enumerate(predictions[0]):
            if pred.item() > 0.2:
                class_name = "Class_" + str(i + 1)
                result = self.classes_to_results.get(class_name)
                results.append(result)
        return results


if __name__ == '__main__':
    App(tk.Tk(), "Face Real time video")

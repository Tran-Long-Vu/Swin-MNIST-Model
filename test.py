# check gpu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn

import torch
cuda = torch.cuda.is_available()
print("GPU:", cuda)

#init cnn Model()
import torch.nn as nn
import torch.nn.functional as F

# Formula to calculate shape as we go through layer by layer = [(X - F + 2P)/S] + 1
# Here,
# X = Width / Height
# F = Kernel size
# P = Padding
# S = Strides (default = 1)

#Our input to the first layer is going to be [batchsize,1,28,28]
#substitute, =[(28 - 5 + 2(0))/1] + 1
#             =[(23)/1] + 1
#             =23 + 1
#             =24


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1,16,5) #(channels,output,kernel_size)   [Batch_size,1,28,28]  --> [Batch_size,16,24,24]
        #print(self.conv1)
        self.mxp1 = nn.MaxPool2d(2)   #                                 [Batch_size,16,24,24] --> [Batch_size,16,24/2,24/2] --> [Batch_size,16,12,12]
        #print(self.mxp1)
        self.conv2 = nn.Conv2d(16,24,5) #                               [Batch_size,16,12,12] --> [Batch_size,24,8,8]
        #print(self.conv2)
        self.mxp2 = nn.MaxPool2d(2)   #                                 [Batch_size,24,8,8] ---> [Batch_size,32,8/2,8/2] ---> [Batch_size,24,4,4]
        #print(self.mxp2)
        self.linear1 = nn.Linear(24 * 4 * 4, 100)                       #input shape --> 100 outputs
        #print(self.linear1)
        self.linear2 = nn.Linear(100,10)                                #100 inputs --> 10 outputs
        #print(self.linear2)

    def forward(self,x):
        X = self.mxp1(F.relu(self.conv1(x)))
        print(X)
        X = self.mxp2(F.relu(self.conv2(X)))
        print(X)
        X = X.view(-1, 24 * 4 * 4)  #reshaping to input shape
        print(X)
        X = F.relu(self.linear1(X))
        print(X)
        X = self.linear2(X)
        print(X)
        print(F.log_softmax(X, dim=1))
        return F.log_softmax(X, dim=1)

cnn = Model()

if cuda:
    cnn.cuda()


print(cnn)

# Load the saved model state dictionary into:
# A: GPU cnn.load_state_dict(torch.load("1st-model.pth"))
# B: GPU

cnn.load_state_dict(torch.load("1st-model.pth", map_location=torch.device('cpu')))
cnn.to(torch.device('cpu'))

cnn.eval()




# recording a video
import torchvision.transforms as transforms # using torchvision
from PIL import Image # using PIL
import cv2
from matplotlib import pyplot as plt


# format the roi frame to image.
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


cap = cv2.VideoCapture(0) # device zero.

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        roi = frame[300:400, 400:500]
        cv2.rectangle(frame, (400, 300), (500, 400), (0, 255, 0), 2)

        # Preprocess the ROI image
        roi_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_image = transform(roi_image)
        roi_image = roi_image.unsqueeze(0)

        # Perform image prediction
        with torch.no_grad():
            outputs = cnn(roi_image)  # Assuming 'cnn' is your CNN model
            _, predicted_class = torch.max(outputs.data, 1)
            cv2.putText(
                frame,
                "Predicted Number: " + str(predicted_class.item()),
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            cv2.imshow('Frame', frame)
            cv2.imshow('ROI', roi)
            # print("Predicted Number:", predicted_class.item())
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



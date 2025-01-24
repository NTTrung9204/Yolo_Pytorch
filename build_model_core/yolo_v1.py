# import torch
# import torch.nn as nn

# class YOLOV1Model(nn.Module):
#     def __init__(self, cell_size: int, num_classes: int, num_bbox: int) -> None:
#         super(YOLOV1Model, self).__init__()

#         self.S = cell_size
#         self.C = num_classes
#         self.B = num_bbox

#     def forward(self, inputs: torch.Tensor):
#         x = nn.Conv2d(3, 64, 7, 2)(inputs)
#         x = nn.MaxPool2d(64, 2)(x)

#         x = nn.Conv2d(64, 192, 3)(x)
#         x = nn.MaxPool2d(64, 2)(x)

#         x = nn.Conv2d(192, 128, 1)(x)
#         x = nn.Conv2d(128, 256, 3)(x)
#         x = nn.Conv2d(256, 256, 1)(x)
#         x = nn.Conv2d(256, 512, 3)(x)
#         x = nn.MaxPool2d(64, 2)(x)
        
#         x = nn.Conv2d(512, 256, 1)(x)
#         x = nn.Conv2d(256, 512, 3)(x)
#         x = nn.Conv2d(512, 256, 1)(x)
#         x = nn.Conv2d(256, 512, 3)(x)
#         x = nn.Conv2d(512, 256, 1)(x)
#         x = nn.Conv2d(256, 512, 3)(x)
#         x = nn.Conv2d(512, 256, 1)(x)
#         x = nn.Conv2d(256, 512, 3)(x)
#         x = nn.Conv2d(512, 512, 1)(x)
#         x = nn.Conv2d(512, 1024, 3)(x)
#         x = nn.MaxPool2d(64, 2)(x)

#         x = nn.Conv2d(1024, 512, 1)(x)
#         x = nn.Conv2d(512, 1024, 3)(x)
#         x = nn.Conv2d(1024, 512, 1)(x)
#         x = nn.Conv2d(512, 1024, 3)(x)
#         x = nn.Conv2d(1024, 1024, 3, 2)(x)

#         x = nn.Conv2d(1024, 1024, 3)(x)
#         x = nn.Conv2d(1024, 1024, 3)(x)

#         x = nn.Flatten()(x)
#         x = nn.Linear(self.S * self.S * 1024, 4096)

#         x = nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))

#         x = torch.reshape(x, (-1, self.S, self.S, self.C + self.B * 5))

import torch
import torch.nn as nn

class YOLOV1Model(nn.Module):
    def __init__(self, cell_size: int, num_classes: int, num_bbox: int) -> None:
        super(YOLOV1Model, self).__init__()

        self.S = cell_size
        self.C = num_classes
        self.B = num_bbox

        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 192, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(192, 128, 1)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 512, 3)
        self.conv9 = nn.Conv2d(512, 256, 1)
        self.conv10 = nn.Conv2d(256, 512, 3)
        self.conv11 = nn.Conv2d(512, 256, 1)
        self.conv12 = nn.Conv2d(256, 512, 3)
        self.conv13 = nn.Conv2d(512, 256, 1)
        self.conv14 = nn.Conv2d(256, 512, 3)
        self.conv15 = nn.Conv2d(512, 512, 1)
        self.conv16 = nn.Conv2d(512, 1024, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv17 = nn.Conv2d(1024, 512, 1)
        self.conv18 = nn.Conv2d(512, 1024, 3)
        self.conv19 = nn.Conv2d(1024, 512, 1)
        self.conv20 = nn.Conv2d(512, 1024, 3)
        self.conv21 = nn.Conv2d(1024, 1024, 3, 2)

        self.conv22 = nn.Conv2d(1024, 1024, 3)
        self.conv23 = nn.Conv2d(1024, 1024, 3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.S * self.S * 1024, 4096)
        self.fc2 = nn.Linear(4096, self.S * self.S * (self.C + self.B * 5))

    def forward(self, inputs: torch.Tensor):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.pool4(x)

        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)

        x = self.conv22(x)
        x = self.conv23(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        x = torch.reshape(x, (-1, self.S, self.S, self.C + self.B * 5))
        return x

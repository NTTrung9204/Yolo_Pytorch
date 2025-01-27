from build_model_core.yolo_v1 import YOLOV1Model
from torchsummary import summary
import torch
from DatasetBuilder import DatasetBuilder
from torch.utils.data import DataLoader
import os
from utils import plot_image_with_boxes, test_model
from yolo_v1_loss import YOLOLoss
from torchvision import transforms

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLOV1Model(7, 5, 2).to(device)

# summary(model, input_size=(3, 448, 448), batch_size=128)

# Đường dẫn dữ liệu test
dataset_images_path = "dataset/train/images/"
dataset_labels_path = "dataset/train/labels/"

annotations_list = []
images_name_list = []

for images in os.listdir(dataset_images_path):
    images_name_list.append(dataset_images_path + images)

for annotation in os.listdir(dataset_labels_path):
    annotations_list.append(dataset_labels_path + annotation)

transform = transforms.Compose(
    [
        transforms.Resize(448),
        transforms.ToTensor(),
    ]
)

# Tạo dataset và dataloader
test_data = DatasetBuilder(annotations_list=annotations_list, images_list=images_name_list, transform=transform)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

# Khởi tạo mô hình và load trọng số đã huấn luyện
loss = YOLOLoss(7, 1, 2)
model = YOLOV1Model(7, 1, 2)
model.load_state_dict(torch.load("yolo_v1_model.pth"))

# # Test mô hình
test_model(model, test_loader, class_names=["face"])

# for image, target in test_loader:
#     pred = model(image)

#     print(pred.shape)

#     print(loss(pred, target))

#     break
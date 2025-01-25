import torch
import torch.nn as nn
import torch.optim as optim
from build_model_core.yolo_v1 import YOLOV1Model
from DatasetBuilder import DatasetBuilder
from torch.utils.data import DataLoader
import os
from yolo_v1_loss import YOLOLoss

def train_model(model, dataloader, loss_fn, optimizer, num_epochs=10, save_path="yolo_v1_model.pth"):
    """
    Hàm huấn luyện mô hình YOLOv1.

    Args:
        model: Mô hình YOLOv1.
        dataloader: DataLoader chứa dữ liệu huấn luyện.
        loss_fn: Hàm loss YOLOLoss.
        optimizer: Optimizer để cập nhật tham số.
        num_epochs: Số epoch huấn luyện.
        save_path: Đường dẫn lưu mô hình.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (images, grids) in enumerate(dataloader):
            # Đưa dữ liệu lên GPU nếu có
            images = images.to(device)
            grids = grids.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(images)

            # Tính loss
            loss = loss_fn(predictions, grids)

            # Backward pass và cập nhật tham số
            loss.backward()
            optimizer.step()

            # Tích lũy loss
            epoch_loss += loss.item()

            # # Hiển thị thông tin
            # if batch_idx % 10 == 0:
            #     print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Hiển thị loss mỗi epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Average Loss: {epoch_loss / len(dataloader):.4f}")

    # Lưu mô hình sau khi huấn luyện
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Đường dẫn dữ liệu
    dataset_images_path = "dataset/train/images/"
    dataset_labels_path = "dataset/train/labels/"

    annotations_list = []
    images_name_list = []

    for images in os.listdir(dataset_images_path):
        images_name_list.append(dataset_images_path + images)

    for annotation in os.listdir(dataset_labels_path):
        annotations_list.append(dataset_labels_path + annotation)

    # Tạo dataset và dataloader
    my_data = DatasetBuilder(annotations_list=annotations_list, images_list=images_name_list)
    my_loader = DataLoader(my_data, batch_size=4, shuffle=True)

    # Khởi tạo mô hình, loss, và optimizer
    model = YOLOV1Model(7, 1, 2)
    loss_fn = YOLOLoss(7, 1, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Huấn luyện mô hình
    train_model(model, my_loader, loss_fn, optimizer, num_epochs=30, save_path="yolo_v1_model.pth")

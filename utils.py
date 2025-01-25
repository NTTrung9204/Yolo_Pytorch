import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_image_with_boxes(images, predictions, class_names=None):
    batch_size = images.size(0)
    images = (images * 255).byte()  # Convert image back to [0, 255]
    predictions = predictions.detach()

    for b in range(batch_size):
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(images[b].permute(1, 2, 0).cpu().numpy().astype(np.uint8))

        S = predictions.size(1)  # Grid size
        for i in range(S):
            for j in range(S):
                # Confidence threshold
                print(predictions[b, i, j, 4])
                if predictions[b, i, j, 4] > 0.01:  # Only consider confident predictions
                    x_center = predictions[b, i, j, 0] * 448
                    y_center = predictions[b, i, j, 1] * 448
                    width = predictions[b, i, j, 2] * 448
                    height = predictions[b, i, j, 3] * 448
                    class_id = 0  # Get class ID

                    x = x_center - width / 2
                    y = y_center - height / 2

                    # Draw bounding box
                    rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    # Add label
                    if class_names:
                        ax.text(x, y, f'{class_names[class_id]}', color='red', fontsize=12)

        ax.axis('off')
        plt.show()


def test_model(model, dataloader, class_names):
    """
    Hàm test mô hình YOLOv1.

    Args:
        model: Mô hình YOLOv1.
        dataloader: DataLoader chứa dữ liệu test.
        class_names: Danh sách tên các lớp.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for images, grids in dataloader:
            images = images.to(device)
            predictions = model(images)  # Model output

            # Visualize predictions
            plot_image_with_boxes(images.cpu(), predictions.cpu(), class_names=class_names)
            break  # Test 1 batch only

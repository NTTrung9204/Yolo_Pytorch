import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, cell_size: int, num_classes: int, num_bbox: int, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super(YOLOLoss, self).__init__()

        self.S = cell_size  # Grid size (SxS)
        self.C = num_classes  # Number of classes
        self.B = num_bbox  # Number of bounding boxes per grid cell
        self.lambda_coord = lambda_coord  # Weight for Localization Loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object Confidence Loss

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute YOLO loss.

        :param predictions: Tensor of predictions, shape: [batch_size, S, S, B * 5 + C]
        :param targets: Tensor of ground truth, shape: [batch_size, S, S, 5]
        :return: Total loss
        """
        batch_size = predictions.size(0)

        # Reshape predictions and targets
        pred_boxes = predictions[..., :self.B * 5].reshape(batch_size, self.S, self.S, self.B, 5)  # [x, y, w, h, confidence]
        pred_classes = predictions[..., self.B * 5:]  # Class probabilities

        target_boxes = targets[..., :5].unsqueeze(3)  # [x, y, w, h, class_id]
        target_classes = targets[..., 4].long()  # Class IDs

        # Adjust target_classes to be 0-based only for non-zero class IDs
        target_classes_adjusted = torch.where(target_classes > 0, target_classes - 1, target_classes)

        # Create one-hot encoding for adjusted target classes
        target_classes_one_hot = F.one_hot(target_classes_adjusted, num_classes=self.C).float()

        # Calculate IoU for each bbox
        ious = torch.stack(
            [self.calculate_iou(pred_boxes[..., b, :4], target_boxes[..., 0, :4]) for b in range(self.B)],
            dim=3
        )  # Shape: [batch_size, S, S, B]

        # Select the bbox with the highest IoU
        best_iou, best_bbox = torch.max(ious, dim=3, keepdim=True)  # Shape: [batch_size, S, S, 1]
        best_iou = best_iou.squeeze(-1)

        # Create masks for the best bbox
        obj_mask = (targets[..., 4] > 0).unsqueeze(-1)  # Object presence mask
        noobj_mask = ~obj_mask  # No-object mask

        best_pred_boxes = pred_boxes.gather(3, best_bbox.unsqueeze(-1).expand(-1, -1, -1, -1, 5)).squeeze(3)

        obj_mask = obj_mask.squeeze(-1)

        epsilon = 1e-6  # Small value to prevent invalid sqrt
        clamped_best_pred_boxes = torch.clamp(best_pred_boxes[..., 2:4], min=epsilon)  # Ensure w, h >= epsilon
        clamped_target_boxes = torch.clamp(target_boxes[..., 0, 2:4], min=epsilon)    # Ensure w, h >= epsilon

        # Calculate (w, h) loss with clamped values
        coord_loss = torch.sum(
            obj_mask * (
                torch.sum((best_pred_boxes[..., 0:2] - target_boxes[..., 0, 0:2]) ** 2, -1) +  # (x, y)
                torch.sum((torch.sqrt(clamped_best_pred_boxes) - torch.sqrt(clamped_target_boxes)) ** 2, -1)  # (w, h)
            )
        )

        # Confidence Loss
        obj_confidence_loss = torch.sum(
            obj_mask * (best_pred_boxes[..., 4] - 1) ** 2
        )
        # noobj_confidence_loss = torch.sum(
        #     noobj_mask * (best_pred_boxes[..., 4] - 0) ** 2
        # )
        # confidence_loss = obj_confidence_loss + self.lambda_noobj * noobj_confidence_loss
        confidence_loss = obj_confidence_loss

        # Classification Loss
        class_loss = torch.sum(
            obj_mask * torch.sum((pred_classes - target_classes_one_hot) ** 2, dim=-1)
        )

        # Total Loss
        total_loss = (coord_loss + confidence_loss + class_loss) / batch_size

        return total_loss

    def calculate_iou(self, pred_boxes, target_boxes):
        """
        Tính toán Intersection over Union (IoU) giữa các box dự đoán và box thực tế.

        :param pred_boxes: Tensor chứa các bounding box dự đoán, shape: (B, 4)
        :param target_boxes: Tensor chứa các bounding box thực tế, shape: (1, 4)
        :return: Tensor chứa giá trị IoU cho mỗi box, shape: (B,)
        """
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

        # Tính diện tích của các box
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

        # Tính diện tích của phần giao nhau
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0)) * torch.max(inter_y2 - inter_y1, torch.tensor(0.0))

        # Tính IoU
        union_area = pred_area + target_area - inter_area
        iou = inter_area / union_area

        return iou
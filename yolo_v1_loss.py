import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(
        self,
        cell_size: int,
        num_classes: int,
        num_bbox: int,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ) -> None:
        super(YOLOLoss, self).__init__()

        self.S = cell_size  # Grid size (SxS)
        self.C = num_classes  # Number of classes
        self.B = num_bbox  # Number of bounding boxes per grid cell
        self.lambda_coord = lambda_coord  # Weight for Localization Loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object Confidence Loss
        self.mse = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute YOLO loss.

        :param predictions: Tensor of predictions, shape: [batch_size, S, S, B * 5 + C]
        :param targets: Tensor of ground truth, shape: [batch_size, S, S, B * 5 + C]
        :return: Total loss
        """
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou1 = self.calculate_iou(
            predictions[..., self.C + 1 : self.C + 5],
            targets[..., self.C + 1 : self.C + 5],
        )
        iou2 = self.calculate_iou(
            predictions[..., self.C + 6 : self.C + 10],
            targets[..., self.C + 1 : self.C + 5],
        )

        ious = torch.cat(
            [iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0
        )  # [[iou1],[iou2]]

        # bbox
        iou_max_val, best_bbox = torch.max(ious, dim=0)

        # print(targets[..., self.C].shape)

        # I_obj_ij
        actual_box = targets[..., self.C].unsqueeze(
            3
        )  # (-1,S,S,1)

        # print("actual_box", actual_box[0])

        # box coords
        box_pred = actual_box * (
            (
                best_bbox * predictions[..., self.C + 6 : self.C + 10]
                + (1 - best_bbox) * predictions[..., self.C + 1 : self.C + 5]
            )
        )

        box_pred[..., 2:4] = torch.sign(box_pred[..., 2:4]) * (
            torch.sqrt(torch.abs(box_pred[..., 2:4] + 1e-6))
        )

        box_target = actual_box * targets[..., self.C + 1 : self.C + 5]

        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # print(torch.flatten(box_pred, end_dim=-2))
        # print(torch.flatten(box_target, end_dim=-2))

        box_coord_loss = self.mse(
            torch.flatten(box_pred, end_dim=-2), torch.flatten(box_target, end_dim=-2)
        )

        # print("box_coord_loss", box_coord_loss)

        # object loss
        pred_box = (
            best_bbox * predictions[..., self.C + 5 : self.C + 6]
            + (1 - best_bbox) * predictions[..., self.C : self.C + 1]
        )

        obj_loss = self.mse(
            torch.flatten(actual_box * pred_box),
            torch.flatten(actual_box * targets[..., self.C : self.C + 1]),
        )

        # no object loss
        no_obj_loss = self.mse(
            torch.flatten(
                (1 - actual_box) * predictions[..., self.C : self.C + 1], start_dim=1
            ),
            torch.flatten(
                (1 - actual_box) * targets[..., self.C : self.C + 1], start_dim=1
            ),
        )

        no_obj_loss += self.mse(
            torch.flatten(
                (1 - actual_box) * predictions[..., self.C + 5 : self.C + 6],
                start_dim=1,
            ),
            torch.flatten(
                (1 - actual_box) * targets[..., self.C : self.C + 1], start_dim=1
            ),
        )

        # class loss
        class_loss = self.mse(
            torch.flatten((actual_box) * predictions[..., : self.C], end_dim=-2),
            torch.flatten((actual_box) * targets[..., : self.C], end_dim=-2),
        )

        loss = (
            self.lambda_coord * box_coord_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )

        return loss

    def calculate_iou(self, pred_boxes, target_boxes, format = "midpoint"):
        """
        Tính toán Intersection over Union (IoU) giữa các box dự đoán và box thực tế.

        :param pred_boxes: Tensor chứa các bounding box dự đoán, shape: (B, 4)
        :param target_boxes: Tensor chứa các bounding box thực tế, shape: (1, 4)
        :return: Tensor chứa giá trị IoU cho mỗi box, shape: (B,)
        """
        if format == "midpoint":
        
            box1_x1 = pred_boxes[...,0:1] - pred_boxes[...,2:3]/2
            box1_x2 = pred_boxes[...,0:1] + pred_boxes[...,2:3]/2
            box1_y1 = pred_boxes[...,1:2] - pred_boxes[...,3:4]/2
            box1_y2 = pred_boxes[...,1:2] + pred_boxes[...,3:4]/2
            
            box2_x1 = target_boxes[...,0:1] - target_boxes[...,2:3]/2
            box2_x2 = target_boxes[...,0:1] + target_boxes[...,2:3]/2
            box2_y1 = target_boxes[...,1:2] - target_boxes[...,3:4]/2
            box2_y2 = target_boxes[...,1:2] + target_boxes[...,3:4]/2
            
            
        x1 = torch.max(box1_x1, box2_x1)[0]
        y1 = torch.max(box1_y1, box2_y1)[0]
        x2 = torch.min(box1_x2, box2_x2)[0]
        y2 = torch.min(box1_y2, box2_y2)[0]
        
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1 = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2 = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
        
        union = box1 + box2 - inter + 1e-6
        
        iou = inter/union
        
        return iou

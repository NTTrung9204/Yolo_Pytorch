import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class DatasetBuilder(nn.Module):
    def __init__(
        self, annotations_list: list[str], images_list: list[str], transform=None
    ) -> None:
        super(DatasetBuilder, self).__init__()

        self.annotations_list: list[str] = annotations_list
        self.images_list: list[str] = images_list
        self.img_size = 448
        self.transform = transform
        self.B = 2
        self.C = 1
        self.S = 7

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, index):
        image_path = self.images_list[index]
        annotation_path = self.annotations_list[index]

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        with open(annotation_path, "r") as file:
            annotations = file.readlines()

        grid = torch.zeros((self.S, self.S, self.C + self.B * 5))

        for annotation in annotations:
            parts = annotation.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            j, i = int(x_center * self.S), int(y_center * self.S)

            grid_x = self.S * x_center - j
            grid_y = self.S * y_center - i

            width_grid, height_grid = self.S * width, self.S * height

            if grid[i, j, self.C] == 0:

                grid[i, j, self.C] = 1

                box_coord = torch.tensor([grid_x, grid_y, width_grid, height_grid])

                grid[i, j, self.C + 1: self.C + 5] = box_coord

                grid[i, j, int(class_id)] = 1

        return image, grid

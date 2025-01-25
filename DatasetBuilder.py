import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class DatasetBuilder(nn.Module):
    def __init__(self, annotations_list: list[str], images_list: list[str]) -> None:
        super(DatasetBuilder, self).__init__()

        self.annotations_list: list[str] = annotations_list
        self.images_list: list[str] = images_list
        self.img_size = 448

    def __len__(self) -> int:
        return len(self.images_list)
    
    def __getitem__(self, index):
        image_path = self.images_list[index]
        annotation_path = self.annotations_list[index]
        
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        
        image = np.array(image) / 255.0
        image = torch.tensor(image).float()
        # image = image.permute(2, 0, 1)
        image = image.permute(2, 1, 0)
        
        with open(annotation_path, 'r') as file:
            annotations = file.readlines()

        grid_size = 7
        grid = torch.zeros((grid_size, grid_size, 5))
        
        for annotation in annotations:
            parts = annotation.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            
            grid_x = int(x_center * grid_size)
            grid_y = int(y_center * grid_size)
            
            grid[grid_x, grid_y, 0] = x_center
            grid[grid_x, grid_y, 1] = y_center
            grid[grid_x, grid_y, 2] = width
            grid[grid_x, grid_y, 3] = height
            grid[grid_x, grid_y, 4] = class_id + 1
        
        return image, grid

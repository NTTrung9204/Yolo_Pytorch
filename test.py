from build_model_core.yolo_v1 import YOLOV1Model
from torchsummary import summary
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLOV1Model(7, 5, 2).to(device)

summary(model, input_size=(3, 448, 488), batch_size=128)
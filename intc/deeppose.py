"""
DeepPose model for COCO keypoint estimation.
Input:  (B, 3, 256, 256) cropped person image
Output: (B, 17*2) normalized keypoint coordinates in [-1, 1]
"""
import torch
import torch.nn as nn
import torchvision.models as models


class DeepPose(nn.Module):
    def __init__(self, n_keypoints=17, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # strip fc
        self.fc = nn.Linear(2048, n_keypoints * 2)
        self.n_keypoints = n_keypoints

    def forward(self, x):
        f = self.features(x).flatten(1)   # (B, 2048)
        return self.fc(f)                  # (B, 34)  in [-1,1] range after tanh at loss

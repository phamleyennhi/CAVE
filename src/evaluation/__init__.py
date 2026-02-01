import os
import torch
import numpy as np
from torchvision import transforms

ATTRIBUTION_PATH = "PATH/TO/ATTRIBUTIONS"
IMG_ATTRIBUTION_PATH = "PATH/TO/IMG_ATTRIBUTIONS"


def to_torch_tensor(matrix):
    if isinstance(matrix, np.ndarray):
        return torch.from_numpy(matrix)
    return matrix 

def to_numpy_image(img):
    img = np.array(img)
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img -= img.min();img /= img.max()
    return img


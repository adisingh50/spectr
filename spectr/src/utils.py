"""File containing generenic utility functions for the repository"""

import torch
import torch.nn as nn


def get_num_parameters(model: nn.Module) -> int:
    """Returns number of individual trainable parameters

    Args:
        model: A Pytorch model object

    Returns:
        total_params: Total number of individual trainable parameters in the model
    """
    total_params = 0
    for param in list(model.parameters()):
        individual_params = 1
        for shape_dim in list(param.size()):
            individual_params *= shape_dim
        total_params += individual_params

    return total_params


def get_classId_from_rgb(rgb: torch.Tensor) -> int:
    """Obtains a unique classId from a given rgb pixel value. Primarily will be applied on a semantically segmented
    label image.

    Args:
        rgb (torch.Tensor): rgb tensor, length 3.

    Raises:
        ValueError: if rgb input is not of length 3.

    Returns:
        int: the unique classId corresponding to the rgb pixel value.
    """
    if (rgb.shape[0] != 3 or len(rgb.shape) > 1):
        raise ValueError("rbg tensor must be of size 3")

    rgb_to_classId_map = {
        (0, 0, 0): 0,
        (111, 74, 0): 1,
        (81, 0, 81): 2,
        (128, 64, 128): 3,
        (244, 35, 232): 4,
        (250, 170, 160): 5,
        (230, 150, 140): 6,
        (70, 70, 70): 7,
        (102, 102, 156): 8,
        (190, 153, 153): 9,
        (180, 165, 180): 10,
        (150, 100, 100): 11,
        (150, 120, 90): 12,
        (153, 153, 153): 13,
        (250, 170, 30): 14,
        (220, 220, 0): 15,
        (107, 142, 35): 16,
        (152, 251, 152): 17,
        (70, 130, 180): 18,
        (220, 20, 60): 19,
        (255, 0, 0): 20,
        (0, 0, 142): 21,
        (0, 0, 70): 22,
        (0, 60, 100): 23,
        (0, 0, 90): 24,
        (0, 0, 110): 25,
        (0, 80, 100): 26,
        (0, 0, 230): 27,
        (119, 11, 32): 28,
        (0, 0, 142): 29
    }

    rgb_tuple = tuple(rgb.tolist())
    classId = rgb_to_classId_map[rgb_tuple]
    return classId

if __name__ == "__main__":
    rgb = torch.Tensor([0, 0, 0])
    classId = get_classId_from_rgb(rgb)
    print(classId)
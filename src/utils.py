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

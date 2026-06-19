import torch
from typing import Union

def accuracy_score(y_true: Union[torch.Tensor, list, "np.ndarray"],
                   y_pred: Union[torch.Tensor, list, "np.ndarray"]) -> float:
    """
    Compute the accuracy: fraction of matching elements in y_true and y_pred.
    Both inputs may be torch.Tensor, list, or numpy.ndarray.
    """
    y_true_t = torch.as_tensor(y_true)
    y_pred_t = torch.as_tensor(y_pred)
    return (y_true_t == y_pred_t).float().mean().item()


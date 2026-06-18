import torch
from typing import Optional

def to_categorical(x: torch.Tensor, n_col: Optional[int] = None) -> torch.Tensor:
    """
    Perform one-hot encoding on a 1D integer tensor `x`. If `n_col` is not provided, infer it from the max value in `x`.
    """
    # Easy way out: torch.nn.functional.one_hot
    # (had to do .to(torch.float32) to meet the test case)
    """
    if n_col is not None:
        return torch.nn.functional.one_hot(x, n_col).to(torch.float32)
    else:
        return torch.nn.functional.one_hot(x).to(torch.float32)
    """

    # A bit harder way: manually!

    # Get the number of classes (aka n_col)
    num_classes = n_col if n_col is not None else (torch.max(x).item() + 1)

    # Implement one-hot encoding

    # Debug this
    return_list = []
    # baseline is 0 — hard-coded below
    for val in x.tolist():
        # val IS the distance from zero encoding
        val_list = []
        
        if val != 0:
            val_list += ([0] * val)
        
        val_list += [1] # add the 1

        if len(val_list) != num_classes:
            val_list += ([0] * (abs(len(val_list) - num_classes)))
        
        return_list.append(val_list)
    
    return torch.tensor(return_list, dtype=torch.float32)

    """
    First attempt: passes test one, fails test two
    Diagnosis: Baseline should be zero

    baseline = torch.min(x).item() # this is what [1, 0, ..., 0] is
    for val in x.tolist():
        if val == baseline:
            return_val_list = [1]
            return_val_list += ([0] * (num_classes - 1))
        else:
            diff = val - baseline # gives how far away from baseline val is
            # useful for then encoding that distance in the one-hot encoding tensor that gets returned
            return_val_list = [0] * (diff)
            return_val_list += [1]
            return_val_list += ([0] * (num_classes - len(return_val_list)))
        
        return_list.append(return_val_list)
    
    return torch.tensor(return_list, dtype=torch.float32)
    """



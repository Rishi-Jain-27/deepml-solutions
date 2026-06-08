import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    
    # Check if T_t is invertible
    if (T_t.ndim != 2 or T_t.shape[0] != T_t.shape[1]) and (T_t.shape[0] != torch.linalg.matrix_rank(T_t).item()):
        return torch.tensor(-1)
    
    # Check if S_t is invertible
    if (S_t.ndim != 2 or S_t.shape[0] != S_t.shape[1]) and (S_t.shape[0] != torch.linalg.matrix_rank(S_t).item()):
        return torch.tensor(-1)
    
    return ((torch.linalg.inv(T_t)) @ A_t @ S_t)

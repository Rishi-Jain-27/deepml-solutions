import torch

def ridge_loss(X: torch.Tensor, w: torch.Tensor, y_true: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Implements the Ridge Regression Loss Function using PyTorch.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        w: Weight vector of shape (n_features,)
        y_true: True target values of shape (n_samples,)
        alpha: Regularization parameter (lambda)
    
    Returns:
        The Ridge loss value as a scalar tensor
    """
    n, f = X.shape

    # 1. Calculate MSE
    pred = X @ w.unsqueeze(-1) # shape = (n_samples, 1)
    mse = (1/n) * torch.sum((pred - y_true.unsqueeze(-1))**2) # shape () bc torch.sum with no dim collapses everything to a single value

    # 2. Calculate regularization term
    reg_term = alpha * torch.sum(w**2, dim=-1).squeeze(-1) # squeeze to bring it to scalar as well

    # 3. Add & return
    return (mse + reg_term)

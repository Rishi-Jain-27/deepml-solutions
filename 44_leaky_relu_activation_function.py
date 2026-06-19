import torch

def leaky_relu(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Implements the Leaky ReLU activation function using PyTorch.
    
    Args:
        z: Input tensor (scalar or any shape)
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        Output tensor after applying Leaky ReLU
    """
    if z.item() <= 0:
        return torch.tensor(z.item() * alpha, dtype=torch.float32)
    return z

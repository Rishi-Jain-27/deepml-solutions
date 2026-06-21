import torch

def kernel_function(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Computes the linear kernel between two input vectors.
    The linear kernel is defined as the dot product (inner product) of two vectors.
    
    Args:
        x1: First input tensor (1D vector)
        x2: Second input tensor (1D vector)
    
    Returns:
        Scalar tensor representing the linear kernel (dot product)
    """
    return torch.dot(x1, x2)

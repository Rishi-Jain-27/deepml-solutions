import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    x = torch.linalg.eigvals(matrix)
    sorted_values, indices = torch.sort(torch.real(x), descending=True)
    return sorted_values

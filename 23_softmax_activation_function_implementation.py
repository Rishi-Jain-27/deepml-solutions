import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    scores_t = torch.tensor(scores, dtype=torch.float32)
    ez = torch.exp(scores_t - torch.max(scores_t))
    return (ez/torch.sum(ez, axis=-1, keepdim=True)).tolist()
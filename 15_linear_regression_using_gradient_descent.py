import torch

def linear_regression_gradient_descent(X, y, alpha, iterations) -> torch.Tensor:
    """
    Perform linear regression using gradient descent with PyTorch autograd.

    Args:
        X: Feature matrix (m, n) - can be tensor or array-like
        y: Target vector (m,) - can be tensor or array-like  
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D tensor of shape (n,)
    """
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True) # weights & biases
    
    for epoch in range(iterations):
        # 1. Zero grad
        if theta.grad is not None:
            theta.grad.zero_()
        # 2. Forward pass
        y_pred = X_t @ theta
        # 3. Calc loss
        loss = (1/(2*m)) * torch.sum((y_pred - y_t)**2)
        # 4. Backprop
        loss.backward()
        # 5. Gradient descent
        with torch.no_grad():
            if theta is not None:
                theta -= alpha * theta.grad
        
    
    return theta.detach()


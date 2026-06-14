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
    # Convert X and y to tensors
    X_t = torch.as_tensor(X, dtype=torch.float32) # (m, n)
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1) # (m, 1)

    # Create theta
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)
    # theta = (n, 1)
    # X_t @ theta = shape of (m, 1)
    # That can be used in a loss calculation with y_t which has
    # shape of (m, 1)

    # Training loop
    # Loop iterations times
    for epoch in range(iterations):
        # 1. Zero grad
        # optimizer.zero_grad()
        if theta.grad is not None:
            theta.grad.zero_()

        # 2. Forward pass
        prediction = X_t @ theta

        # 3. Calc loss
        loss = (1/(2 * m)) * torch.sum((prediction - y_t)**2)

        # 4. Backpropagation
        loss.backward()

        # 5. Gradient descent
        # optimizer.step()
        with torch.no_grad():
            if theta.grad is not None:
                theta -= (alpha * theta.grad)

    # Return theta (weights and biases)
    theta = theta.squeeze(1)
    return theta.detach()


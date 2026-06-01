import torch

def pca(data, k) -> torch.Tensor:
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    data_t = torch.tensor(data).T

    # 1. Standardize the data
    standardized_data = (data_t - torch.mean(data_t, dim=1).unsqueeze(dim=1))/(torch.std(data_t, dim=1).unsqueeze(dim=1))

    # 2. Get covariance matrix
    cov_matrix = torch.cov(standardized_data)

    # 3. Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

    # 4. Fix eigenvector direction
    for eigvect in eigenvectors.T: # .T because each column is an eigvect
        mask = torch.abs(eigvect) > 1e-10 # boolean mask of items in the vector that match our condition

        # Create a list of the indexes of the true value
        true_val_idxs = torch.nonzero(mask)

        # Check if there's at least one match
        if true_val_idxs.numel() > 0:
            # If negative, make the eigvect negative
            # true_val_idxs[0][0] == idx of the true val
            if eigvect[true_val_idxs[0][0]] < 0: eigvect *= -1
    
    # 5. Select Principal Components
    # Sort the eigenvalues, greatest to least, in a list.
    # Get the indices
    indices = torch.argsort(eigenvalues, descending=True)

    # Get only the top k indices
    indices = indices[:k]

    # Assemble results from eigenvectors
    results = eigenvectors[:, indices]

    return results

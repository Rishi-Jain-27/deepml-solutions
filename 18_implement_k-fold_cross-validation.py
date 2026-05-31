import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Return train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds
        shuffle: Whether to shuffle indices before splitting
    
    Returns:
        List of (train_indices, test_indices) tuples, where each is a list of ints
    """
    n_samples_per_fold = n_samples // k
    fold_sizes = [n_samples_per_fold] * k

    # Spread the remainder across the first folds
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    
    # Generate a list of indices representing the dataset
    indices = [i for i in range(n_samples)]

    # Implement shuffle=True requirement
    if shuffle: np.random.shuffle(indices)
    
    # Set up the dual lists approach
    result = []
    test_list = indices
    train_list = indices

    # Iterate through len(fold_sizes) number of times
    for i in range(len(fold_sizes)):
        # the test fold is the first fold_sizes[i] indices
        fold_length = fold_sizes[i]
        test_fold = test_list[:fold_length]

        # then get rid of those indices from the train_list
        test_list = test_list[fold_length:]

        # Now train_fold is every indice that isn't in test_fold
        train_fold = [idx for idx in train_list if idx not in test_fold]

        result.append((train_fold, test_fold))
     
    return result

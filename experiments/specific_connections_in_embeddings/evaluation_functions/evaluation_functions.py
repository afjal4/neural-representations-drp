import numpy as np
from scipy.linalg import block_diag

def generate_kernel_matrix(embeddings, kernel_function):
    num_embeddings = embeddings.shape[0]
    kernel_matrix = np.zeros((num_embeddings, num_embeddings))

    for i in range(num_embeddings):
        for j in range(num_embeddings):
            kernel_matrix[i, j] = kernel_function(embeddings[i], embeddings[j])

    return kernel_matrix

def kernel_matrix_error(kernel_matrix, target_matrix = None, norm='fro'):
    if target_matrix is None:
        target_matrix = block_diag(*[np.ones((4, 4)) for _ in range(4)])

    return np.linalg.norm(kernel_matrix - target_matrix, ord=norm)
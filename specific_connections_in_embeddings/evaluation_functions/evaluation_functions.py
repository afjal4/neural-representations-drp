import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

def get_answer(words, labels):
    """
    words: list of words (16,)

    labels: list of 0, 1, 2, 3, category of words

    returns:
        4 lists of words in each category
    """
    words = np.array(words)
    labels = np.array(labels)
    for i in range(4):
        print(words[labels == i])

def create_kernel_mat(words, fn, model, **kwargs):
    """
    words: list of words

    fn: kernel function

    model: dict mapping words to vectors

    returns:
        kernel matrix calculated using kernel function
    """
    mat = []
    for i in words:
        row = []
        for j in words:
            row.append(fn(model[i], model[j], **kwargs))
        mat.append(row)
    return mat

def plot_kernel_mat(mat, labels, cmap='magma'):
    plt.imshow(mat, cmap=cmap)
    plt.xticks(ticks=np.arange(16), labels=labels, rotation=90)
    plt.yticks(ticks=np.arange(16), labels=labels)
    plt.colorbar()
    plt.show()

def normalized_dot_fn(x, y, eps=1e-8):
    """
    Cosine-like similarity: normalized dot product.
    Returns scalar.
    """
    nx = np.linalg.norm(x) + eps
    ny = np.linalg.norm(y) + eps
    return float((x @ y) / (nx * ny))

def rbf_kernel_fn(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

def compute_gamma(words, model):
    """
    Compute gamma = 1 / median(||x_i - x_j||^2)

    words: list of words

    model: dict mapping word -> vector
    """
    vecs = np.stack([model[w] for w in words], axis=0)  # (N, d)
    N = vecs.shape[0]

    dists2 = []
    for i in range(N):
        for j in range(i + 1, N):
            d2 = np.sum((vecs[i] - vecs[j]) ** 2)
            dists2.append(d2)

    median_dist2 = np.median(dists2)
    gamma = 1.0 / (median_dist2 + 1e-12)

    return gamma

def kmeans_order_from_K(K, n_clusters=4, random_state=0):
    """
    Cluster data using kmeans.

    K: (N, N) similarity / kernel matrix

    returns:
        order: list of reordered indices
    """

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(K)

    centers = km.cluster_centers_

    dist = np.linalg.norm(K - centers[labels], axis=1)
    order = np.argsort(labels + 1e-6 * dist)

    return order.tolist()

def apply_order(x, order):
    """
    x: (N, N) matrix OR (N,) vector/list

    order: list of indices

    returns:
        reordered x
    """
    order = np.asarray(order)
    x = np.asarray(x)

    if x.ndim == 2:
        return x[np.ix_(order, order)]
    elif x.ndim == 1:
        return x[order]
    else:
        raise ValueError("x must be 1D or 2D")
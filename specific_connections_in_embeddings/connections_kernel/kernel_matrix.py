from scipy.cluster import SpectralClustering
from scipy.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cdist


def kernel_func(embeddings, ker_type):
    # this method will do the kernel function
    


    X = embeddings  # shape (16, d)
    if ker_type == "cosine":
        similarity = cosine_similarity(X)
    elif ker_type == "RBF":
        sq_dists = cdist(X, X, 'sqeuclidean')
        dists = pdist(X, 'euclidean')
        sigma = np.median(dists)
        gamma = 1 / (2 * sigma**2)
        similarity = np.exp(-gamma * sq_dists)
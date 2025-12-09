from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
def kernel_func(embeddings):
    #this method will define the kernel function

    X = embeddings  # shape (16, d)
    similarity = cosine_similarity(X)

    clustering = SpectralClustering(
        n_clusters=4,
        affinity='precomputed',
        random_state=0
    ).fit(similarity)

    groups = clustering.labels_
    return groups
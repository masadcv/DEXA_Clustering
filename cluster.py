from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def get_kmeans_clusters(num_clusters, embeddings):
    """
    Get kmeans clusters for the embeddings

    Args:
        num_clusters: int, number of clusters
        embeddings: np.array, embeddings
    
    Returns:
        kmeans: KMeans, kmeans model
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    return kmeans


def get_tsne_embeddings(embeddings, n_components=2):
    """
    Get tsne embeddings for the embeddings
    
    Args:
        embeddings: np.array, embeddings
        n_components: int, number of components
        
    Returns:
        embeddings_nd: np.array, tsne embeddings
    """
    tsne = TSNE(n_components=n_components, random_state=0)
    embeddings_nd = tsne.fit_transform(embeddings)
    return embeddings_nd

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def get_kmeans_clusters(num_clusters, embeddings):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    return kmeans


def get_tsne_embeddings(embeddings, n_components=2):
    tsne = TSNE(n_components=n_components, random_state=0)
    embeddings_nd = tsne.fit_transform(embeddings)
    return embeddings_nd

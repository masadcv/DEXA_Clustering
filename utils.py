import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {file_path}")


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def plot_clusters(output, embeddings_2d, cluster_labels):
    plt.figure(figsize=(10, 10))
    num_clusters = len(np.unique(cluster_labels))
    for i in range(num_clusters):
        plt.scatter(
            embeddings_2d[cluster_labels == i, 0],
            embeddings_2d[cluster_labels == i, 1],
            label=f"Cluster {i}",
        )
    plt.legend()
    plt.title("Clusters of embeddings")
    plt.savefig(os.path.join(output, "embeddings_clusters_assignment.png"))


def save_n_images_per_cluster(
    output, image_paths, cluster_labels, images_per_cluster=10
):
    clusters_image_folder = os.path.join(output, "clusters_images")
    os.makedirs(clusters_image_folder, exist_ok=True)

    num_clusters = len(np.unique(cluster_labels))
    for i in range(num_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_image_paths = [image_paths[idx] for idx in cluster_indices]

        # output folder
        output_folder = os.path.join(clusters_image_folder, f"cluster_{i}")
        os.makedirs(output_folder, exist_ok=True)

        for src_path in cluster_image_paths[:images_per_cluster]:
            trg_path = os.path.join(output_folder, os.path.basename(src_path))
            shutil.copy(src_path, trg_path)
        print(f"Cluster {i} images are saved to {output_folder}")


# def process_clusters(output, num_clusters, embeddings, image_paths, kmeans):
#     # project the embeddings to 2D space
#     tsne = TSNE(n_components=2, random_state=0)
#     embeddings_2d = tsne.fit_transform(embeddings)
#     # embeddings_2d = tsne(embeddings[:, :10], dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, seed_positions=np.array([]))

#     # plot the clusters with embeddings and cluster ids
#     plt.figure(figsize=(10, 10))
#     for i in range(num_clusters):
#         plt.scatter(
#             embeddings_2d[kmeans.labels_ == i, 0],
#             embeddings_2d[kmeans.labels_ == i, 1],
#             label=f"Cluster {i}",
#         )
#     plt.legend()
#     plt.title("Clusters of embeddings")
#     plt.savefig(os.path.join(output, "embeddings_clusters_assignment.png"))

#     images_per_cluster = 10
#     clusters_image_folder = os.path.join(output, "clusters_images")
#     os.makedirs(clusters_image_folder, exist_ok=True)
#     for i in range(num_clusters):
#         cluster_indices = np.where(kmeans.labels_ == i)[0]
#         cluster_image_paths = [image_paths[idx] for idx in cluster_indices]

#         # output folder
#         output_folder = os.path.join(clusters_image_folder, f"cluster_{i}")
#         os.makedirs(output_folder, exist_ok=True)

#         for src_path in cluster_image_paths[:images_per_cluster]:
#             trg_path = os.path.join(output_folder, os.path.basename(src_path))
#             shutil.copy(src_path, trg_path)
#         print(f"Cluster {i} images are saved to {output_folder}")

import argparse
import torchxrayvision as xrv
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import shutil

from utils import load_json, save_json
from datasets import NIH_Dataset


def get_vae_embeddings(args):
    # embeddings fetching loop
    if (
        args.use_cache
        and os.path.exists(os.path.join(args.output, "nih_embeddings.npy"))
        and os.path.exists(os.path.join(args.output, "nih_labels.npy"))
    ):
        embeddings = np.load(os.path.join(args.output, "nih_embeddings.npy"))
        labels = np.load(os.path.join(args.output, "nih_labels.npy"))
        image_paths = load_json(os.path.join(args.output, "nih_image_paths.json"))[
            "image_paths"
        ]
    else:
        if args.use_cache:
            print("Cache not found, recomputing embeddings..")

        # National Institutes of Health ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
        d_nih = NIH_Dataset(imgpath=args.dataset_path)  # , transform=transform)
        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            d_nih, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = xrv.autoencoders.ResNetAE(weights="101-elastic").to(
            device
        )  # NIH chest X-ray8
        model = model.eval()

        # get size for model output
        input_image = torch.randn(1, 1, 224, 224).to(device)
        output = model.encode(input_image)

        # prepare output containers for embeddings and labels
        len_save = args.num_images
        embeddings = (
            np.zeros((len(d_nih), output.flatten().shape[0]))
            if len_save is None
            else np.zeros((len_save, output.flatten().shape[0]))
        )
        labels = (
            np.zeros((len(d_nih), d_nih[0]["lab"].shape[0]), dtype=int)
            if len_save is None
            else np.zeros((len_save, d_nih[0]["lab"].shape[0]), dtype=int)
        )
        image_paths = []

        # run inference loop through the dataset
        for idx, frame_data in enumerate(tqdm(dataloader)):
            if len_save is not None and idx > len_save - 1:
                break

            input_image = frame_data["img"].to(device)

            # take images and pass through VAE encoder to get embeddings_            input_image = frame_data["img"].to(device)
            z = model.encode(input_image)

            # save embeddings and labels
            embeddings[idx, :] = z.flatten().cpu().detach().numpy()
            labels[idx, :] = frame_data["lab"].flatten().cpu().detach().numpy()
            image_paths += frame_data["img_path"]

        # save embeddings and labels
        np.save(os.path.join(args.output, "nih_embeddings.npy"), embeddings)
        np.save(os.path.join(args.output, "nih_labels.npy"), labels)
        save_json(
            os.path.join(args.output, "nih_image_paths.json"),
            {"image_paths": image_paths},
        )

    return embeddings, labels, image_paths


def get_kmeans_clusters(args, embeddings):
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(embeddings)
    return kmeans


def process_clusters(args, embeddings, labels, image_paths, kmeans):

    # project the embeddings to 2D space
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    # embeddings_2d = tsne(embeddings[:, :10], dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, seed_positions=np.array([]))

    # plot the clusters with embeddings and cluster ids
    plt.figure(figsize=(10, 10))
    for i in range(args.num_clusters):
        plt.scatter(
            embeddings_2d[kmeans.labels_ == i, 0],
            embeddings_2d[kmeans.labels_ == i, 1],
            label=f"Cluster {i}",
        )
    plt.legend()
    plt.title("Clusters of embeddings")
    plt.savefig(os.path.join(args.output, "embeddings_clusters_assignment.png"))

    # plot the clusters with embeddings and labels
    plt.figure(figsize=(10, 10))
    for i in range(np.max(labels) + 1):
        plt.scatter(
            embeddings_2d[labels == i, 0],
            embeddings_2d[labels == i, 1],
            label=f"Label {i}",
        )
    plt.legend()
    plt.title("Clusters of embeddings with disease label")
    plt.savefig(os.path.join(args.output, "embeddings_labels_assignment.png"))

    images_per_cluster = 10
    clusters_image_folder = os.path.join(args.output, "clusters_images")
    os.makedirs(clusters_image_folder, exist_ok=True)
    for i in range(args.num_clusters):
        cluster_indices = np.where(kmeans.labels_ == i)[0]
        cluster_image_paths = [image_paths[idx] for idx in cluster_indices]

        # output folder
        output_folder = os.path.join(clusters_image_folder, f"cluster_{i}")
        os.makedirs(output_folder, exist_ok=True)

        for src_path in cluster_image_paths[:images_per_cluster]:
            trg_path = os.path.join(output_folder, os.path.basename(src_path))
            shutil.copy(src_path, trg_path)
        print(f"Cluster {i} images are saved to {output_folder}")


def main(args):
    embeddings, labels, image_paths = get_vae_embeddings(args)
    labels = np.argmax(labels, axis=1)
    kmeans = get_kmeans_clusters(args, embeddings)
    np.save(os.path.join(args.output, "kmeans_labels.npy"), kmeans.labels_)

    process_clusters(args, embeddings, labels, image_paths, kmeans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", default=None, type=int)
    parser.add_argument("--num_clusters", default=14, type=int)
    parser.add_argument(
        "--dataset_path",
        default="/data/home/xaw004/Dataset/NIH_CXR/images-224",
        type=str,
    )
    parser.add_argument("--output", default="./output", type=str)
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main(args)

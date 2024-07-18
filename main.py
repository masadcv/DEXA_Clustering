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

transform = torchvision.transforms.Compose(
    [
        # xrv.datasets.XRayCenterCrop(),
        # xrv.datasets.XRayResizer(224),
        torchvision.transforms.ToTensor(),
    ]
)


def unnormalize(img, maxval):
    """Scales images to be roughly [-1024 1024]."""

    img = (img / 1024 + 1) / 2 * maxval

    return img


def get_vae_embeddins(args):
    # embeddings fetching loop
    if args.use_cache:
        if os.path.exists(
            os.path.join(args.output, "nih_embeddings.npy")
        ) and os.path.exists(os.path.join(args.output, "nih_labels.npy")):
            embeddings = np.load(os.path.join(args.output, "nih_embeddings.npy"))
            labels = np.load(os.path.join(args.output, "nih_labels.npy"))
    else:
        args.dataset_path = "/data/KCLData/Datasets/NIH_CXR/images-224/"
        # National Institutes of Health ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
        d_nih = xrv.datasets.NIH_Dataset(
            imgpath=args.dataset_path
        )  # , transform=transform)
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
        input_image = torch.randn(1, 1, 224, 224)
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

        # run inference loop through the dataset
        for idx, frame_data in tqdm(enumerate(dataloader)):
            if len_save is not None and idx > len_save - 1:
                break

            # take images and pass through VAE encoder to get embeddings
            input_image = frame_data["img"].to(device)
            z = model.encode(input_image)

            # save embeddings and labels
            embeddings[idx, :] = z.flatten().cpu().detach().numpy()
            labels[idx, :] = frame_data["lab"].flatten().cpu().detach().numpy()

        # save embeddings and labels
        np.save(os.path.join(args.output, "nih_embeddings.npy"), embeddings)
        np.save(os.path.join(args.output, "nih_labels.npy"), labels)
    
    return embeddings, labels

def get_kmeans_clusters(args, embeddings):
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(embeddings)
    return kmeans


def main(args):
    embeddings, labels = get_vae_embeddins(args)
    kmeans = get_kmeans_clusters(args, embeddings)
    np.save(os.path.join(args.output, "kmeans_labels.npy"), kmeans.labels_)

    # project the embeddings to 2D space
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    # plot the clusters with embeddings
    plt.figure(figsize=(10, 10))
    for i in range(args.num_clusters):
        plt.scatter(
            embeddings_2d[kmeans.labels_ == i, 0],
            embeddings_2d[kmeans.labels_ == i, 1],
            label=f"Cluster {i}",
        )
    plt.legend()
    plt.title("Clusters of embeddings")
    plt.savefig(os.path.join(args.output, "embeddings_clusters.png"))



    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", default=None, type=int)
    parser.add_argument("--num_clusters", default=10, type=int)
    parser.add_argument(
        "--dataset_path",
        default="/data/home/xaw004/Dataset/NIH_CXR/images-224",
        type=str,
    )
    parser.add_argument("--output", default="./output", type=str)
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()
    main(args)

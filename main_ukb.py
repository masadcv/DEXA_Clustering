import argparse
import os

import numpy as np
import torch

from cluster import get_kmeans_clusters, get_tsne_embeddings
from datasets import DexaDatasetUKB, NIH_Dataset, transform_dexaukb_xrv
from embeddings import get_embeddings
from model_zoo import model_name_to_func
from utils import plot_clusters, save_n_images_per_cluster


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model_name_to_func[args.model]().to(device)
    model.eval()

    if args.dataset == "nih":
        dataset = NIH_Dataset(imgpath=args.dataset_path)
    elif args.dataset == "ukb":
        dataset = DexaDatasetUKB(
            root=args.dataset_path,
            transform=transform_dexaukb_xrv,
            num_images=args.num_images,
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )

    embeddings, image_paths = get_embeddings(
        device=device,
        model_func=model.encode if "vae" in args.model else model,
        dataloader=dataloader,
        output=args.output,
        use_cache=args.use_cache,
        num_images=args.num_images,
    )
    kmeans = get_kmeans_clusters(args.num_clusters, embeddings)
    np.save(os.path.join(args.output, "kmeans_labels.npy"), kmeans.labels_)

    embeddings_2d = get_tsne_embeddings(embeddings, n_components=2)
    np.save(os.path.join(args.output, "tsne_embeddings_2d.npy"), embeddings_2d)

    plot_clusters(args.output, embeddings_2d, kmeans.labels_)
    save_n_images_per_cluster(
        args.output, image_paths, kmeans.labels_, images_per_cluster=100
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xrv_vae", type=str)
    parser.add_argument("--dataset", default="ukb", type=str)
    parser.add_argument("--num_images", default=None, type=int)
    parser.add_argument("--num_clusters", default=4, type=int)
    parser.add_argument(
        "--dataset_path",
        # default="/data/home/xaw004/Dataset/NIH_CXR/images-224",
        # default="/data/Coding/ImageClusteringDEXA/DummyDataset",
        default="/data/Coding/ImageClusteringDEXA/white_background_data/",
        type=str,
    )
    parser.add_argument("--output", default="./output", type=str)
    parser.add_argument("--use_cache", action="store_true")
    args = parser.parse_args()
    if args.num_images is not None:
        args.output = f"{args.output}_{args.num_images}images"
    args.output = f"{args.output}_{args.num_clusters}clusters"
    os.makedirs(args.output, exist_ok=True)
    main(args)

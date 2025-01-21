import argparse
import os

import numpy as np
import torch
from PIL import Image

from datasets import DexaDatasetUKB
from utils import check_and_remove_white_background, is_background_white


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dataset = DexaDatasetUKB(root=args.dataset_path, transform=None)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )

    total = 0
    white_background = 0

    for data in dataloader:
        total += 1
        image_path = data["img_path"][0]
        image = np.squeeze(data["img"].detach().numpy()).astype(np.uint8)

        if is_background_white(image):
            white_background += 1
            output_path = os.path.join(
                args.output, os.path.basename(image_path).replace(".dcm", "_or.png")
            )
            Image.fromarray(image).save(output_path)
            print(f"Removing background from {image_path}")
            image = check_and_remove_white_background(image)

            # write path
            output_path = os.path.join(
                args.output, os.path.basename(image_path).replace(".dcm", "_rm.png")
            )
            Image.fromarray(image).save(output_path)

    print(f"Total images: {total}")
    print(f"White background images: {white_background}")
    print(f"Non-white background images: {total - white_background}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/data/Coding/ImageClusteringDEXA/DummyDataset",
        type=str,
    )
    parser.add_argument("--output", default="./output_removebk", type=str)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    main(args)

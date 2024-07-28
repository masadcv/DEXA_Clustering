import os

import numpy as np
import torch
from tqdm import tqdm

from utils import load_json, save_json


def get_embeddings(device, model_func, dataloader, output, use_cache, num_images=None):
    # embeddings fetching loop
    if use_cache and os.path.exists(os.path.join(output, "ukb_dexa_embeddings.npy")):
        embeddings = np.load(os.path.join(output, "ukb_dexa_embeddings.npy"))
        image_paths = load_json(os.path.join(output, "ukb_dexa_image_paths.json"))[
            "image_paths"
        ]
    else:
        if use_cache:
            print("Cache not found, recomputing embeddings..")

        # get size for model output
        input_image = torch.randn(1, 1, 224, 224).to(device)
        model_output = model_func(input_image)

        # prepare output containers for embeddings and labels
        len_save = num_images
        embeddings = (
            np.zeros((len(dataloader.dataset), model_output.flatten().shape[0]))
            if len_save is None
            else np.zeros((len_save, model_output.flatten().shape[0]))
        )
        image_paths = []

        # run inference loop through the dataset
        for idx, frame_data in enumerate(tqdm(dataloader)):
            if len_save is not None and idx > len_save - 1:
                break

            input_image = frame_data["img"].to(device)

            # take imgs and pass through VAE encoder to get embeddings_            input_image = frame_data["img"].to(device)
            z = model_func(input_image)

            # save embeddings and labels
            embeddings[idx, :] = z.flatten().cpu().detach().numpy()
            image_paths += frame_data["img_path"]

        # save embeddings and labels
        np.save(os.path.join(output, "ukb_dexa_embeddings.npy"), embeddings)
        save_json(
            os.path.join(output, "ukb_dexa_image_paths.json"),
            {"image_paths": image_paths},
        )

    return embeddings, image_paths

import json
import os
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage


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
            if src_path.endswith(".dcm"):
                # convert dcm to png for viewing
                img, _ = LoadImage()(src_path)
                img = img.squeeze().numpy()
                trg_path = os.path.join(
                    output_folder, os.path.basename(src_path).replace(".dcm", ".png")
                )
                plt.imsave(trg_path, img, cmap="gray")
            else:
                trg_path = os.path.join(output_folder, os.path.basename(src_path))
                shutil.copy(src_path, trg_path)
        print(f"Cluster {i} images are saved to {output_folder}")


def check_and_remove_white_background(image_orig):
    image = image_orig.copy()

    bg_val = np.bincount(image.flatten()).argmax()
    is_background_white = bg_val > 255 / 2

    if is_background_white:
        # pad 1 pixel on each side with value bg_val
        image = cv2.copyMakeBorder(
            image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(bg_val)
        )

        # gray to color
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Create a mask for GrabCut algorithm
        mask = np.zeros(image.shape[:2], np.uint8)

        # Define the rectangle for initial mask
        rect = (1, 1, image.shape[1] - 1, image.shape[0] - 1)

        # Define the background and foreground models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Apply GrabCut algorithm
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        # Modify the mask to isolate the foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

        mask2 = mask2[1:-1, 1:-1]

        # Apply the mask to the image
        foreground = image_orig * mask2
    else:
        foreground = image_orig

    return foreground


def is_background_white(image):
    bg_val = np.bincount(image.flatten()).argmax()
    return bg_val > 255 / 2


def lin_stretch_img(img, low_prc, high_prc, do_ignore_minmax=True):
    """
    Apply linear "stretch" - low_prc percentile goes to 0,
    and high_prc percentile goes to 255.
    The result is clipped to [0, 255] and converted to np.uint8

    Additional feature:
    When computing high and low percentiles, ignore the minimum and maximum intensities (assumed to be outliers).
    """
    # For ignoring the outliers, replace them with the median value
    if do_ignore_minmax:
        tmp_img = img.copy()
        med = np.median(img)  # Compute median
        tmp_img[img == img.min()] = med
        tmp_img[img == img.max()] = med
    else:
        tmp_img = img

    lo, hi = np.percentile(
        tmp_img, (low_prc, high_prc)
    )  # Example: 1% - Low percentile, 99% - High percentile

    if lo == hi:
        return np.full(
            img.shape, 128, np.uint8
        )  # Protection: return gray image if lo = hi.

    stretch_img = (img.astype(float) - lo) * (
        255 / (hi - lo)
    )  # Linear stretch: lo goes to 0, hi to 255.
    stretch_img = stretch_img.clip(0, 255).astype(
        np.uint8
    )  # Clip range to [0, 255] and convert to uint8
    return stretch_img

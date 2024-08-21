import json
import os
import shutil
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.transforms import LoadImage
from PIL import Image


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {file_path}")


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def make_json_safe(data):
    """
    Recursively converts a Python dictionary into a JSON-safe dictionary.
    Handles torch.Tensor and numpy.ndarray by converting them to lists or scalars.
    """
    if isinstance(data, dict):
        return {str(k): make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(i) for i in data]
    elif isinstance(data, set):
        return [make_json_safe(i) for i in data]  # Convert sets to lists
    elif isinstance(data, tuple):
        return [make_json_safe(i) for i in data]  # Convert tuples to lists
    elif isinstance(data, torch.Tensor):
        return (
            data.tolist() if data.ndim > 0 else data.item()
        )  # Convert tensors to lists or scalars
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    else:
        return str(data)  # Convert non-serializable objects to strings


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


def extract_zip_files(root, working_dir, remove_zip, ext):
    os.makedirs(working_dir, exist_ok=True)
    # recursively parse the root directory and sub directories and check for zip files
    # if zip files found, unzip them in the same directory
    count_images_extracted = 0
    for root_local, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(".zip"):
                unzip_path = os.path.join(
                    working_dir, os.path.basename(root_local), file.split(".zip")[0]
                )
                print("Found zip file: ", file)
                # check if any file with the extension is already unzipped
                # if yes, skip unzipping
                if os.path.exists(unzip_path) and any(
                    [f.endswith(ext) for f in os.listdir(unzip_path)]
                ):
                    print("Files already unzipped")
                    continue
                print("Unzipping the file")
                count_images_extracted += 1
                zip_path = os.path.join(root_local, file)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(unzip_path)
                if remove_zip:
                    os.remove(zip_path)


def load_dataset_paths(root, remove_unused_dcm, ext):
    # recursively parse the root directory and sub directories for files with the extension
    dataset = []
    for root_local, _, files in os.walk(root):
        for file in files:
            if file.endswith(ext):
                dataset.append(os.path.join(root_local, file))
            else:
                if remove_unused_dcm and file.endswith(".dcm"):
                    os.remove(os.path.join(root_local, file))
    return dataset


def convert_dcm_to_images(input_folder, remove_dcm=True):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".dcm"):
                img, metadata = LoadImage()(os.path.join(root, file))
                img = img.squeeze().numpy().astype(np.uint8)
                img = check_and_remove_white_background(img)
                img = lin_stretch_img(img, 0.1, 99.9)
                # orient the image such that largest dimension is the height
                # in DEXA full body scans largest dimension indicates the height of the patient
                if img.shape[0] < img.shape[1]:
                    img = img.T
                metadata = make_json_safe(metadata)
                metadata["file_name"] = file.replace(".dcm", ".png")
                save_json(os.path.join(root, file.replace(".dcm", ".json")), metadata)
                Image.fromarray(img).save(
                    os.path.join(root, file.replace(".dcm", ".png"))
                )
                if remove_dcm:
                    os.remove(os.path.join(root, file))


def process_dexa_data_ukb(
    input_folder,
    output_folder,
    dicom_str="11.12.1.dcm",
    remove_zip=False,
    remove_unused_dcm=True,
):
    extract_zip_files(input_folder, output_folder, remove_zip=remove_zip, ext=dicom_str)
    load_dataset_paths(
        output_folder, remove_unused_dcm=remove_unused_dcm, ext=dicom_str
    )
    convert_dcm_to_images(output_folder, remove_dcm=remove_unused_dcm)


def make_tarball(input_folder, output_folder, tarball_name):
    shutil.make_archive(
        os.path.join(output_folder, tarball_name), "gztar", input_folder
    )

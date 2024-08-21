import os
import zipfile

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from monai.transforms import EnsureChannelFirst, LoadImage
from PIL import Image

import utils
from transforms import ResizeWithPadding
from utils import check_and_remove_white_background, lin_stretch_img

transform_nih = torchvision.transforms.Compose(
    [
        # xrv.datasets.XRayCenterCrop(),
        # xrv.datasets.XRayResizer(224),
        torchvision.transforms.ToTensor(),
    ]
)

transform_dexaukb_xrv = torchvision.transforms.Compose(
    [
        # EnsureChannelFirst(),
        ResizeWithPadding(224, 224),
        # normalize range from [0, 255] to [-1024, 1024]
        torchvision.transforms.Lambda(lambda x: unnormalize(x, 1024)),
    ]
)


def unnormalize(img, maxval):
    """Scales images to be roughly [-1024 1024]."""

    img = (img / 1024 + 1) / 2 * maxval

    return img


class NIH_Dataset(xrv.datasets.NIH_Dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        imgid = self.csv["Image Index"].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        sample["img_path"] = img_path
        return sample


class DexaDatasetUKB:
    def __init__(
        self,
        root,
        working_dir="./data",
        dicom_str="11.12.1.dcm",
        remove_zip=False,
        remove_unused_dcm=True,
        transform=None,
        num_images=None,
    ):
        self.root = root
        self.transform = transform
        self.dicom_str = dicom_str
        self.extract_zip_files(
            root, working_dir, remove_zip, ext=dicom_str, num_images=num_images
        )
        self.dataset = self.load_dataset_paths(
            root=working_dir, remove_unused_dcm=remove_unused_dcm, ext=dicom_str
        )

    def extract_zip_files(self, root, working_dir, remove_zip, ext, num_images=None):
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
                    if num_images is not None and count_images_extracted >= num_images:
                        print("Number of images extracted reached the limit")
                        return

    def load_dataset_paths(self, root, remove_unused_dcm, ext):
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

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        img, metadata = LoadImage()(img_path)
        if img.ndim > 2:
            img = img.squeeze()
        # orient the image such that largest dimension is the height
        # in DEXA full body scans largest dimension indicates the height of the patient
        if img.shape[0] < img.shape[1]:
            img = img.T

        img_np = img.numpy().astype(np.uint8)
        img_np = check_and_remove_white_background(img_np)

        # linear stretch the image
        img_np = lin_stretch_img(img_np, 0.1, 99.9)

        img = torch.from_numpy(img_np).float().unsqueeze(0)

        if self.transform:
            img = self.transform(img)
        return dict(img=img, metadata=metadata, img_path=img_path)

    def __len__(self):
        return len(self.dataset)


class DexaDatasetImagesUKB:
    def __init__(
        self,
        root,
        ext="png",
        transform=None,
        num_images=None,
    ):
        self.root = root
        self.transform = transform
        self.ext = ext
        self.dataset = self.load_dataset_paths(root=root, ext=ext)

        if num_images is not None:
            print(f"Limiting dataset to {num_images} images")
            self.dataset = self.dataset[:num_images]

    def load_dataset_paths(self, root, ext):
        # recursively parse the root directory and sub directories for files with the extension
        dataset = []
        for root_local, _, files in os.walk(root):
            for file in files:
                if file.endswith(ext):
                    dataset.append(os.path.join(root_local, file))
        return dataset

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        # img, metadata = LoadImage()(img_path)
        img = np.asarray(Image.open(img_path))
        metadata = utils.load_json(img_path.replace(".png", ".json"))

        if img.ndim > 2:
            img = img.squeeze()

        img = torch.from_numpy(img).float().unsqueeze(0)

        if self.transform:
            img = self.transform(img)
        return dict(img=img, metadata=metadata, img_path=img_path)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    # dataset = DexaDatasetUKB(
    #     root="/data/Coding/ImageClusteringDEXA/DummyDataset",
    #     dicom_str="11.12.1.dcm",
    #     transform=transform_dexaukb_xrv,
    # )

    # data = dataset[0]

    # from matplotlib import pyplot as plt

    # plt.imshow(data["image"][0], cmap="gray")
    # plt.savefig("test.png")
    # print(data["image"].shape)
    # print(data["image"].min(), data["image"].max())
    # print()

    dataset = DexaDatasetImagesUKB(
        root="/data/Coding/ImageClusteringDEXA/DEXA_Clustering/output_data",
        ext="png",
        transform=None,
    )

    data = dataset[0]
    
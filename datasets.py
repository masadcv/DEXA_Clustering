import os
import zipfile

import torchvision
import torchxrayvision as xrv
from monai.transforms import EnsureChannelFirst, LoadImage

from transforms import ResizeWithPadding

transform_nih = torchvision.transforms.Compose(
    [
        # xrv.datasets.XRayCenterCrop(),
        # xrv.datasets.XRayResizer(224),
        torchvision.transforms.ToTensor(),
    ]
)

transform_dexaukb_xrv = torchvision.transforms.Compose(
    [
        EnsureChannelFirst(),
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
        dicom_str="11.12.1.dcm",
        remove_zip=False,
        remove_unused_dcm=True,
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.dicom_str = dicom_str
        self.dataset = self.load_dataset_paths(
            root, dicom_str, remove_zip=remove_zip, remove_unused_dcm=remove_unused_dcm
        )

    def load_dataset_paths(self, root, ext, remove_zip, remove_unused_dcm):
        # recursively parse the root directory and sub directories and check for zip files
        # if zip files found, unzip them in the same directory
        # then parse the directory for files with the extension
        for root_local, _, files in os.walk(root):
            for file in files:
                if file.endswith(".zip"):
                    unzip_path = os.path.join(root_local, file.split(".zip")[0])
                    print("Found zip file: ", file)
                    # check if any file with the extension is already unzipped
                    # if yes, skip unzipping
                    if any([f.endswith(ext) for f in os.listdir(unzip_path)]):
                        print("Files already unzipped")
                        continue
                    print("Unzipping the file")
                    zip_path = os.path.join(root_local, file)
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(unzip_path)
                    if remove_zip:
                        os.remove(zip_path)

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
        if self.transform:
            img = self.transform(img)
        return dict(img=img, metadata=metadata, img_path=img_path)

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = DexaDatasetUKB(
        root="/data/Coding/ImageClusteringDEXA/DummyDataset",
        dicom_str="11.12.1.dcm",
        transform=transform_dexaukb_xrv,
    )

    data = dataset[0]

    from matplotlib import pyplot as plt

    plt.imshow(data["image"][0], cmap="gray")
    plt.savefig("test.png")
    print(data["image"].shape)
    print(data["image"].min(), data["image"].max())
    print()

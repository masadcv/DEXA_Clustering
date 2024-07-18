import torchxrayvision as xrv
import os
import torchvision

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

class NIH_Dataset(xrv.datasets.NIH_Dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        imgid = self.csv["Image Index"].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        sample["img_path"] = img_path
        return sample
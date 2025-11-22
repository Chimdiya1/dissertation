import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch
import random

class PrePostDataset(Dataset):
    def __init__(self, pre_dir, post_dir, mask_dir, augment=True):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.mask_dir = mask_dir

        self.files = [f for f in os.listdir(pre_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.augment = augment

        # Base transform to tensor only
        self.to_tensor = T.ToTensor()
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        pre_path = os.path.join(self.pre_dir, filename)
        post_filename = filename.replace("pre_disaster", "post_disaster").replace("pre_", "post_")
        post_path = os.path.join(self.post_dir, post_filename)

        mask_filename = post_filename.replace("post_disaster", "post_disaster_mask")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")  # grayscale mask

        if self.augment:
            # synchronized augmentations
            if random.random() > 0.5:
                pre_img = T.functional.hflip(pre_img)
                post_img = T.functional.hflip(post_img)
                mask_img = T.functional.hflip(mask_img)

            if random.random() > 0.5:
                pre_img = T.functional.vflip(pre_img)
                post_img = T.functional.vflip(post_img)
                mask_img = T.functional.vflip(mask_img)

            angle = random.uniform(-20, 20)
            pre_img = T.functional.rotate(pre_img, angle)
            post_img = T.functional.rotate(post_img, angle)
            mask_img = T.functional.rotate(mask_img, angle)

            # Color jitter for images only
            pre_img = self.color_jitter(pre_img)
            post_img = self.color_jitter(post_img)

        # convert to tensor
        pre_img = self.to_tensor(pre_img)
        post_img = self.to_tensor(post_img)
        mask_img = self.to_tensor(mask_img)

        #  FIX: Binarize mask
        mask_img = (mask_img > 0.001).float()

        return {
            "pre": pre_img,
            "post": post_img,
            "mask": mask_img,
            "filename": filename
        }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    pre_path = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/images/pre_disaster_patch"
    post_path = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/images/post_disaster_patch"
    mask_path = "/Users/chimdiaanyiam/Desktop/school/dissertation/train/masks_512"

    dataset = PrePostDataset(pre_path, post_path, mask_path, augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    batch = next(iter(loader))
    print(batch["pre"].shape, batch["post"].shape, batch["mask"].shape)

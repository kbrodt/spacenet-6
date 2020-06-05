import albumentations as A
import torch
import numpy as np

from utils import read_img_ski


p = 0.5
albu_train = A.Compose([
    A.CropNonEmptyMaskIfExists(512, 512),

    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
    A.RandomRotate90(p=p),

    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
    ], p=p),
    
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.IAAAdditiveGaussianNoise(p=1),

        A.Blur(p=1),
    ], p=p),
])


albu_dev = A.Compose([
    A.PadIfNeeded(1024, 1024, border_mode=0),
])


def train_transform(img, mask):
    data = albu_train(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    
    img = torch.from_numpy(img).permute(2, 0, 1)
    mask = torch.from_numpy(mask).permute(2, 0, 1)

    return img, mask


def dev_transform(img, mask):
    data = albu_dev(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    
    img = torch.from_numpy(img).permute(2, 0, 1)
    mask = torch.from_numpy(mask).permute(2, 0, 1)
    
    return img, mask


def to_cat(mask):
    mask[..., 0] -= mask[..., 1]
    mask = np.concatenate([np.zeros_like(mask[..., [0]]), mask], -1).argmax(-1)[..., None]
    
    return mask

    
class CloudsDS(torch.utils.data.Dataset):
    def __init__(self, items, root, transform, w3m=False, use_softmax=False):
        self.items = items
        self.root = root
        self.transform = transform
        self.w3m = w3m
        self.use_softmax = use_softmax

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items.iloc[index]

        path_to_img = item.image
        path_to_mask = item.label
        img = read_img_ski(path_to_img).astype('float32')/255
        mask = read_img_ski(path_to_mask).astype('float32')/255
        if self.use_softmax:  # 0-mask, 1-border, 2-contact
            mask = to_cat(mask)
        elif not self.w3m:
            mask = mask[..., None]
        
        return self.transform(img, mask)


def collate_fn(x):
    x, y = list(zip(*x))

    return torch.stack(x), torch.stack(y)

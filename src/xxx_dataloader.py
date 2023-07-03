'''a template for dataloader
you need to rewrite your transform and dataloader function to replace the xxx with the name of dataset ,and replace you own dataset
'''

import torchvision.transforms as tt
import torch

from lib.img_dataset import ImgBinaryDataset
from lib.processing_utils import get_mean_std

xxx_transforms_train = tt.Compose(
    [
        tt.RandomRotation(30),
        tt.Resize((144, 144)),
        tt.RandomHorizontalFlip(),
        tt.ColorJitter(brightness=0.3),
        tt.RandomCrop((112, 112)),
        tt.ToTensor(),
        tt.RandomErasing(p=0.5, scale=(0.05, 0.33)),
        tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)

xxx_transforms_test = tt.Compose(
    [
        tt.Resize((112, 112)),
        # tt.RandomCrop(112, 112),
        tt.ToTensor(),
        tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)


def xxx_dataloader(train, args):
    # dataset and data loader
    if train:
        surf_dataset = ImgBinaryDataset(data_transform=xxx_transforms_train)


    else:

        surf_dataset = surf_dataset = ImgBinaryDataset(data_transform=xxx_transforms_test)

    surf_data_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=args.batch_size,
        shuffle=True)

    return surf_data_loader

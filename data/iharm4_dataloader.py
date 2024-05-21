import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

from .transforms import (
    TripleCompose, TripleRandomCrop,
    TripleNormalize, TripleToTensor)
from .iharmony4_h5_dataset import IH5Dataset


def train_dataloader(
        subset,
        archive='datasets/ihm4/IHD_train_256.h5',
        use_subarch=True,
        batch_size=64,
        num_workers=0
    ):
    transform = TripleCompose([
        TripleToTensor(),
        TripleNormalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    ])
    dataloader = DataLoader(
        IH5Dataset(
            archive=archive,
            transform=transform,
            mode='train',
            subset=subset,
            use_subarch=use_subarch
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def valid_dataloader(
        subset,
        archive='datasets/ihm4/IHD_test_256.h5',
        use_subarch=True,
        batch_size=64,
        num_workers=0
    ):
    transform = TripleCompose([
        TripleToTensor(),
        TripleNormalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
    ])
    dataloader = DataLoader(
        IH5Dataset(
            archive=archive,
            transform=transform,
            mode='test',
            subset=subset,
            use_subarch=use_subarch
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# def test_dataloader(path, batch_size=1, num_workers=0):
#     image_dir = os.path.join(path, 'test')
#     dataloader = DataLoader(
#         DeblurDataset(image_dir, is_test=True),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True
#     )
#     return dataloader

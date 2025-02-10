# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
# Copyright (c) 2025 TMQ Authors
# SPDX-License-Identifier: MPL-2.0
#
# Adapted from Torchvision CalTech dataset loader

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
import os
import os.path
import shutil
import math
import random
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets

import sys


class SubSet101(VisionDataset):
    def __init__(self, root, index=None, labels=None, categories=None):
        super().__init__(root, transform=None, target_transform=None)
        self.index = index
        self.y = labels
        self.categories = categories

    def __getitem__(self, index:int) -> Tuple[Any,Any]:
        img_path = os.path.join(self.root, "101_ObjectCategories", self.categories[self.y[index]], f"image_{self.index[index]:04d}.jpg")

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index)



class SubSet256(VisionDataset):
    def __init__(self, root, index=None, labels=None, categories=None):
        super().__init__(root, transform=None, target_transform=None)
        self.index = index
        self.y = labels
        self.categories = categories
        self.transform = None
        self.target_transform = None

    def __getitem__(self, index:int) -> Tuple[Any,Any]:
        img_path = os.path.join(self.root, "256_ObjectCategories", self.categories[self.y[index]], f"{self.y[index] + 1:03d}_{self.index[index]:04d}.jpg")

        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        target = self.y[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.index)



class CaltechBase:

    def num_classes(self) -> int:
        return len(self.categories)

    def train_test_split(self, train_weight, deterministic=True) -> Tuple[Any, Any]:
        assert train_weight > 0 and train_weight < 1, "Illegal training/test ratio"
        test_weight = 1 - train_weight
        test_index = []
        train_index = []
        test_labels = []
        train_labels = []
        for i, c in enumerate(self.categories):
            if self.histogram[i] > 0:
                num_test_items = int(math.ceil(self.histogram[i] * test_weight))
                all_items = self.index[self.offsets[i]:self.offsets[i]+self.histogram[i]]
                num_train_items = len(all_items) - num_test_items
                if not deterministic:
                    random.shuffle(all_items)
                test_index.extend(all_items[0:num_test_items])
                train_index.extend(all_items[num_test_items:])
                test_labels.extend(num_test_items * [i])
                train_labels.extend(num_train_items * [i])
        train_set = self._make_subset(self.root, train_index, train_labels, self.categories)
        test_set = self._make_subset(self.root, test_index, test_labels, self.categories)
        return train_set, test_set

    def _scan(self, data_dir):
        self.categories = sorted(os.listdir(data_dir))
        self.offsets = [0]
        self.histogram = torch.zeros(len(self.categories), dtype=torch.int64)
        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len([item for item in os.listdir(os.path.join(data_dir, c)) if item.endswith(".jpg")])
            self.index.extend(range(1, n + 1))
            self.offsets.append(len(self.index))
            self.y.extend(n * [i])
            self.histogram[i] = n
        max_item = torch.max(self.histogram).item()
        self.category_weights = max_item / self.histogram.to(torch.float32)


    def _make_subset(self, root, indices, labels, categories):
        raise Exception("Implement in derived classes")


class Caltech101(CaltechBase, SubSet101):
    """
    This is a "cleaned up" version of the CalTech-101 dataset.

    As we are not primarily interested in compatibility to the CalTech-101 labels but only on
    training/test performance of quantized models, we remove a few "dirty classes" here.
    """
    def __init__(self, root, download:bool=False):
        super().__init__(os.path.join(root, "caltech101"))
        os.makedirs(self.root, exist_ok=True)
        if download:
            self.download()
        if not self._check_integrity():
            raise Exception("Error downloading dataset")
        self._cleanup_and_scan()

    def filename(self, index:int) -> str:
        return os.path.join(self.root, "101_ObjectCategories", self.categories[self.y[index]], f"image_{self.index[index]:04d}.jpg")

    def label(self, index:int) -> int:
        return self.y[index]

    def category(self, index:int) -> str:
        return self.categories[self.y[index]]

    def download(self):
        # Check if we downloaded already
        if self._check_integrity():
            print("Downloaded already")
            return

        download_and_extract_archive(
            "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp",
            self.root,
            filename="101_ObjectCategories.tar.gz",
            md5="b224c7392d521a49829488ab0f1120d9",
        )

    def _check_integrity(self):
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def _cleanup_and_scan(self):
        data_dir = os.path.join(self.root, "101_ObjectCategories")
        if not os.path.exists(data_dir):
            raise Exception("Cannot find training data")

        # Remove things that do not really add value for our purposes
        if os.path.exists(os.path.join(data_dir, "BACKGROUND_Google")):
            shutil.rmtree(os.path.join(data_dir, "BACKGROUND_Google"))

        self._scan(data_dir)

    def _make_subset(self, root, indices, labels, categories):
        return SubSet101(root, indices, labels, categories)


class Caltech256(CaltechBase, SubSet256):
    """
    This is a "cleaned up" version of the CalTech-256 dataset.

    As we are not primarily interested in compatibility to the CalTech-256 labels but only on
    training/test performance of quantized models, we remove a few "dirty classes" here.
    """
    def __init__(self, root, download:bool=False):
        super().__init__(os.path.join(root, "caltech256"))
        os.makedirs(self.root, exist_ok=True)
        if download:
            self.download()
        if not self._check_integrity():
            raise Exception("Error downloading dataset")
        self._cleanup_and_scan()


    def download(self) -> None:
        # Check if we downloaded already
        if self._check_integrity():
            print("Downloaded already")
            return

        # Download if not the case
        download_and_extract_archive(
            "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK",
            self.root,
            filename="256_ObjectCategories.tar",
            md5="67b4f42ca05d46448c6bb8ecd2220f6d",
        )

    def _check_integrity(self):
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))

    def _cleanup_and_scan(self):
        data_dir = os.path.join(self.root, "256_ObjectCategories")
        if not os.path.exists(data_dir):
            raise Exception("Cannot find training data")

        # Remove things that do not really add value for our purposes
        if os.path.exists(os.path.join(data_dir, "257.clutter")):
            shutil.rmtree(os.path.join(data_dir, "257.clutter"))
        if os.path.exists(os.path.join(data_dir, "056.dog/greg")):
            shutil.rmtree(os.path.join(data_dir, "056.dog/greg"))

        self._scan(data_dir)

    def _make_subset(self, root, indices, labels, categories):
        return SubSet256(root, indices, labels, categories)


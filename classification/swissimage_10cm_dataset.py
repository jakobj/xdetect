import glob
import numpy as np
import os
import torch
import re
from skimage import io


class SWISSIMAGE10cmDataset(torch.utils.data.Dataset):

    def __init__(self, asset_dir, examples_dir, transform=None):

        positive_examples = self.get_examples_as_array(asset_dir, os.path.join(examples_dir, 'positive'))
        negative_examples = self.get_examples_as_array(asset_dir, os.path.join(examples_dir, 'negative'))

        # if len(positive_examples) != len(negative_examples):
        #     raise RuntimeError("imbalanced dataset - make sure the number of positive and negative examples are identical")

        self.data = np.vstack([positive_examples, negative_examples])
        self.labels = np.hstack([np.ones(len(positive_examples), dtype=np.long), np.zeros(len(negative_examples), dtype=np.long)])

        self.transform = transform

    @staticmethod
    def get_examples_as_array(asset_dir, examples_dir):
        annotated_assets = SWISSIMAGE10cmDataset.determine_annotated_assets(examples_dir)

        data = []
        for asset in annotated_assets:
            rgx = re.compile(os.path.join(examples_dir, f"{asset}-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).png"))
            img = io.imread(os.path.join(asset_dir, f"{asset}.tif"))
            for fn in sorted(glob.glob(os.path.join(examples_dir, f"{asset}*.png"))):
                match = rgx.search(fn)
                y, x, yDy, xDx = int(match[1]), int(match[2]), int(match[3]), int(match[4])
                data.append(img[y:yDy, x:xDx])

        return np.array(data)

    @staticmethod
    def determine_annotated_assets(examples_dir):
        annotated_assets = set()
        rgx = re.compile(os.path.join(examples_dir, f"(swissimage.*)-[0-9]+_[0-9]+_[0-9]+_[0-9]+.png"))
        for fn in glob.glob(os.path.join(examples_dir, f"*.png")):
            match = rgx.search(fn)
            annotated_assets.add(match[1])
        return annotated_assets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.data[idx].copy()

        if self.transform is not None:
            return self.transform(img), self.labels[idx]

        return img, self.labels[idx]

import numpy as np
import torch


class SWISSIMAGE10cmDataset(torch.utils.data.Dataset):

    def __init__(self, fn_positive, fn_negative, transform=None):

        positive_examples = np.load(fn_positive)
        negative_examples = np.load(fn_negative)
        if len(positive_examples) != len(negative_examples):
            raise RuntimeError("imbalanced dataset - make sure the number of positive and negative examples are identical")

        self.data = np.vstack([positive_examples, negative_examples])
        self.labels = np.hstack([np.ones(len(positive_examples), dtype=np.long), np.zeros(len(negative_examples), dtype=np.long)])

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        img = self.data[idx].copy()

        if self.transform is not None:
            return self.transform(img), self.labels[idx]

        return img, self.labels[idx]

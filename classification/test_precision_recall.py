import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os
from skimage import io
import torch
from torch.utils.data import DataLoader
import torchvision
import re
import sys

import lib_classification

sys.path.insert(0, "../annotation/")
from annotate import identifier_from_asset


THRESHOLD = 0.9999


def compute_precision_and_recall(*, n_rows, n_cols, bboxes_ground_truth, bboxes):
    a = np.zeros((n_rows, n_cols), dtype=bool)
    for bbox in bboxes_ground_truth:
        a[bbox[0] : bbox[2], bbox[1] : bbox[3]] = True
    b = np.zeros((n_rows, n_cols), dtype=bool)
    for bbox in bboxes:
        b[bbox[0] : bbox[2], bbox[1] : bbox[3]] = True
    true_positives = np.sum(np.logical_and(a, b))
    precision = true_positives / np.sum(b)
    recall = true_positives / np.sum(a)
    return precision, recall


if __name__ == "__main__":
    asset_dir = "../data/"
    examples_dir = f"../data_annotated_50px_test/"

    asset = "swissimage-dop10_2018_2600-1200_0.1_2056.tif"

    img = io.imread(os.path.join(asset_dir, asset))

    target_bboxes_ground_truth = lib_classification.determine_target_bboxes_ground_truth(
        asset_dir=asset_dir,
        examples_dir=examples_dir,
        identifier=identifier_from_asset(asset),
    )
    target_bboxes = lib_classification.determine_target_bboxes(img=img, threshold=THRESHOLD)

    precision, recall = compute_precision_and_recall(
        n_rows=len(img),
        n_cols=len(img[0]),
        bboxes_ground_truth=target_bboxes_ground_truth,
        bboxes=target_bboxes,
    )
    print(f"precision: {precision:.04f}, recall: {recall:.04f}")

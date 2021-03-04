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

from test_overview import determine_target_bboxes, draw_bboxes

sys.path.insert(0, "../annotation/")
from annotate import identifier_from_asset


def determine_target_bboxes_ground_truth(*, asset_dir, examples_dir, identifier):
    target_bboxes = []
    rgx = re.compile(
        os.path.join(
            examples_dir,
            "positive",
            f"{identifier}_0.1_2056-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).png",
        )
    )
    for fn in sorted(
        glob.glob(os.path.join(examples_dir, "positive", f"{identifier}*.png"))
    ):
        match = rgx.search(fn)
        y, x, yDy, xDx = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        target_bboxes.append((y, x, yDy, xDx))
    return target_bboxes


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

    fn_test_image = "../data/swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    img = io.imread(fn_test_image)

    target_bboxes_ground_truth = determine_target_bboxes_ground_truth(
        asset_dir=asset_dir,
        examples_dir=examples_dir,
        identifier=identifier_from_asset(os.path.basename(fn_test_image)),
    )
    target_bboxes = determine_target_bboxes(img=img, threshold=0.9994)

    precision, recall = compute_precision_and_recall(
        n_rows=len(img),
        n_cols=len(img[0]),
        bboxes_ground_truth=target_bboxes_ground_truth,
        bboxes=target_bboxes,
    )
    print(f"precision: {precision:.04f}, recall: {recall:.04f}")

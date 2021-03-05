import numpy as np
import os
from skimage import io
import sys

from config import DETECTION_THRESHOLD
import lib_classification

sys.path.insert(0, "../annotation/")
from annotate import identifier_from_asset


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
    asset_dir = "../data/assets/"
    examples_dir = f"../data/crosswalks/"
    model_file = "./crosswalks.torch"
    asset = "swissimage-dop10_2018_2600-1200_0.1_2056.tif"

    img = io.imread(os.path.join(asset_dir, asset))

    target_bboxes_ground_truth = lib_classification.determine_target_bboxes_ground_truth(
        asset_dir=asset_dir,
        examples_dir=os.path.join(examples_dir, "test"),
        identifier=identifier_from_asset(asset),
    )
    target_bboxes = lib_classification.determine_target_bboxes(img=img, model_file=model_file, threshold=DETECTION_THRESHOLD)

    precision, recall = compute_precision_and_recall(
        n_rows=len(img),
        n_cols=len(img[0]),
        bboxes_ground_truth=target_bboxes_ground_truth,
        bboxes=target_bboxes,
    )
    print(f"precision: {precision:.04f}, recall: {recall:.04f}")

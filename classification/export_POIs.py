import csv
import glob
import json
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


def load_metadata(*, asset_dir, identifier):
    with open(os.path.join(asset_dir, f"{identifier}.json"), 'r') as f:
        return json.load(f)


def compute_midpoint(bbox):
    return bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2


def compute_coords_from_pixel_location(*, point, n_img_rows, n_img_cols, coords_bbox):
    n_img_rows = 10_000
    n_img_cols = 10_000
    slope_rows = (coords_bbox[1] - coords_bbox[3]) / n_img_rows
    slope_cols = (coords_bbox[2] - coords_bbox[0]) / n_img_cols
    return np.round(coords_bbox[3] + slope_rows * point[0], 7), np.round(coords_bbox[0] + slope_cols * point[1], 7)


if __name__ == '__main__':

    asset_dir = "../data/"
    assets = ["swissimage-dop10_2018_2600-1200_0.1_2056.tif", "swissimage-dop10_2018_2598-1200_0.1_2056.tif"]

    coordinates = []
    for asset in assets:
        img = io.imread(os.path.join(asset_dir, asset))
        # img = img[:2000, :2000]

        identifier = identifier_from_asset(asset)
        metadata = load_metadata(asset_dir=asset_dir, identifier=identifier)
        coords_bbox_asset = metadata['bbox']

        target_bboxes = determine_target_bboxes(img=img, threshold=0.999)
        for bbox in target_bboxes:
            midpoint = compute_midpoint(bbox)
            coordinates.append(compute_coords_from_pixel_location(point=midpoint, n_img_rows=len(img), n_img_cols=len(img[0]), coords_bbox=coords_bbox_asset))

    with open('exported_POIs.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for c in coordinates:
            writer.writerow(c)

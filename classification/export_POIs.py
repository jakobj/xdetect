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

from test import determine_target_bboxes, draw_bboxes

sys.path.insert(0, "../annotation/")
from annotate import identifier_from_asset


def load_metadata(*, asset_dir, identifier):
    with open(os.path.join(asset_dir, f"{identifier}.json"), 'r') as f:
        return json.load(f)


def compute_midpoint(bbox):
    return bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2


def compute_coords_from_pixel_location(*, point, n_img_rows, n_img_cols, coords_bbox):
    slope_rows = (coords_bbox[1] - coords_bbox[3]) / n_img_rows
    slope_cols = (coords_bbox[2] - coords_bbox[0]) / n_img_cols
    return np.round(coords_bbox[3] + slope_rows * point[0], 7), np.round(coords_bbox[0] + slope_cols * point[1], 7)


FILE_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Classified POIs - {class_label}</name>
  {placemarks}
</Document>
</kml>
"""

PLACEMARK_TEMPLATE = """\
  <Placemark id="{name}">
    <name>{name}</name>
    <Point>
      <coordinates>{coords_E},{coords_N},0.00</coordinates>
    </Point>
  </Placemark>"""

if __name__ == '__main__':

    class_label = 'Crosswalk'

    fn_asset = "../data/swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    img = io.imread(fn_asset)

    identifier = identifier_from_asset(os.path.basename(fn_asset))
    metadata = load_metadata(asset_dir=os.path.dirname(fn_asset), identifier=identifier)
    coords_bbox_asset = metadata['bbox']

    target_bboxes = determine_target_bboxes(img, threshold=0.9999)
    coordinates = []
    for bbox in target_bboxes:
        midpoint = compute_midpoint(bbox)
        coords = compute_coords_from_pixel_location(point=midpoint, n_img_rows=len(img), n_img_cols=len(img[0]), coords_bbox=coords_bbox_asset)
        coordinates.append(coords)

    placemarks = []
    for i, coords in enumerate(coordinates):
        placemark = PLACEMARK_TEMPLATE.format(name=f"{class_label}-{i}", coords_N=coords[0], coords_E=coords[1])
        placemarks.append(placemark)

    kml_data = FILE_TEMPLATE.format(class_label=class_label, placemarks='\n'.join(placemarks))

    with open('exported_POIs.kml', 'w') as f:
        f.write(kml_data)

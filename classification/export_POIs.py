import csv
import json
import numpy as np
import os
# import pandas as pd
import geopandas as gpd
import shapely.geometry
from skimage import io
import sys

from test_precision_recall import THRESHOLD
import lib_classification

sys.path.insert(0, "../annotation/")
from annotate import identifier_from_asset, asset_prefix_from_asset


gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'


def load_metadata(*, asset_dir, identifier):
    with open(os.path.join(asset_dir, f"{identifier}.json"), 'r') as f:
        return json.load(f)


def compute_midpoint(bbox):
    return bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2


def compute_coordinates_from_bbox(*, bbox, n_img_rows, n_img_cols, coordinates_bbox):
    def convert_y(y):
        return np.round(coordinates_bbox[3] + slope_rows * y, 7)
    def convert_x(x):
        return np.round(coordinates_bbox[0] + slope_cols * x, 7)

    n_img_rows = 10_000
    n_img_cols = 10_000
    slope_rows = (coordinates_bbox[1] - coordinates_bbox[3]) / n_img_rows
    slope_cols = (coordinates_bbox[2] - coordinates_bbox[0]) / n_img_cols
    return (convert_y(bbox[0]), convert_x(bbox[1]), convert_y(bbox[2]), convert_x(bbox[3]))


def create_polygon_from_coordinates(coordinates):
    # return (coordinates[0], coordinates[1]), (coordinates[0], coordinates[3]), (coordinates[1], coordinates[3]), (coordinates[1], coordinates[1]), (coordinates[0], coordinates[1])
    return (coordinates[1], coordinates[0]), (coordinates[3], coordinates[0]), (coordinates[3], coordinates[2]), (coordinates[1], coordinates[2]), (coordinates[1], coordinates[0])


if __name__ == '__main__':

    asset_dir = "../data/"
    assets = ["swissimage-dop10_2018_2598-1198_0.1_2056.tif",
              "swissimage-dop10_2018_2598-1199_0.1_2056.tif",
              "swissimage-dop10_2018_2598-1200_0.1_2056.tif",
              "swissimage-dop10_2018_2599-1198_0.1_2056.tif",
              "swissimage-dop10_2018_2599-1199_0.1_2056.tif",
              "swissimage-dop10_2018_2599-1200_0.1_2056.tif",
              "swissimage-dop10_2018_2600-1198_0.1_2056.tif",
              "swissimage-dop10_2018_2600-1199_0.1_2056.tif",
              "swissimage-dop10_2018_2600-1200_0.1_2056.tif"]
    export_dir = "../data_exported/"

    for asset in assets:
        print(f"  processing asset '{asset}'")
        img = io.imread(os.path.join(asset_dir, asset))
        img = img[:3000, :3000]

        identifier = identifier_from_asset(asset)
        metadata = load_metadata(asset_dir=asset_dir, identifier=identifier)
        coordinates_bbox_asset = metadata['bbox']

        target_bboxes = lib_classification.determine_target_bboxes(img=img, threshold=THRESHOLD)
        coordinates = []
        for bbox in target_bboxes:
            coordinates.append(compute_coordinates_from_bbox(bbox=bbox, n_img_rows=len(img), n_img_cols=len(img[0]), coordinates_bbox=coordinates_bbox_asset))

        polygons = []
        for coords in coordinates:
            polygons.append(shapely.geometry.Polygon(create_polygon_from_coordinates(coords)))

        asset_prefix = asset_prefix_from_asset(asset)
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))
        gdf = gdf.set_crs("EPSG:4326")
        fn = os.path.join(export_dir, f'ROIs-{asset_prefix}.kml')
        gdf.to_file(fn, driver='KML')
        print(f"    -> exported to {fn}")
        exit()

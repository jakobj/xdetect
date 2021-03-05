import json
import geopandas as gpd
import glob
import numpy as np
import os
import pathlib
import re
import shapely.geometry
from skimage import io
import sys

from config import DETECTION_THRESHOLD
import lib_classification

sys.path.insert(0, "../annotation/")
from annotate import identifier_from_asset, asset_prefix_from_asset, asset_from_filename


def load_metadata(*, asset_dir, identifier):
    with open(os.path.join(asset_dir, f"{identifier}.json"), 'r') as f:
        return json.load(f)


def compute_coordinates_from_bbox(*, bbox, n_img_rows, n_img_cols, coordinates_bbox):
    def convert_y(y):
        return np.round(coordinates_bbox[3] + slope_rows * y, 7)

    def convert_x(x):
        return np.round(coordinates_bbox[0] + slope_cols * x, 7)

    slope_rows = (coordinates_bbox[1] - coordinates_bbox[3]) / n_img_rows
    slope_cols = (coordinates_bbox[2] - coordinates_bbox[0]) / n_img_cols
    return (convert_y(bbox[0]), convert_x(bbox[1]), convert_y(bbox[2]), convert_x(bbox[3]))


def create_polygon_from_coordinates(coordinates):
    # return (coordinates[0], coordinates[1]), (coordinates[0], coordinates[3]), (coordinates[2], coordinates[3]), (coordinates[2], coordinates[1]), (coordinates[0], coordinates[1])
    return (coordinates[1], coordinates[0]), (coordinates[3], coordinates[0]), (coordinates[3], coordinates[2]), (coordinates[1], coordinates[2]), (coordinates[1], coordinates[2])


def determine_exported_assets(export_dir):
    exported_assets = set()
    rgx = re.compile(".*/ROIs-(swissimage.*).geojson")
    for fn in glob.glob(os.path.join(export_dir, "ROIs*.geojson")):
        match = rgx.search(fn)
        exported_assets.add(f"{match[1]}.tif")
    return exported_assets


if __name__ == '__main__':

    asset_dir = "../data/assets/"
    export_dir = "../data/export/"
    model_file = "./crosswalks.torch"

    exported_assets = determine_exported_assets(export_dir)

    for fn in glob.glob(os.path.join(asset_dir, "swissimage*.tif")):
        asset = asset_from_filename(fn)

        if asset in sorted(exported_assets):
            inp = ''
            while inp not in ('y', 'n'):
                inp = input(f"  asset '{asset}' already exported - reexport? (y/n) ")
            if inp == 'n':
                print(f"  skipping '{asset}'")
                continue

        print(f"  processing asset '{asset}'")
        img = io.imread(os.path.join(asset_dir, asset))

        identifier = identifier_from_asset(asset)
        metadata = load_metadata(asset_dir=asset_dir, identifier=identifier)
        coordinates_bbox_asset = metadata['bbox']

        target_bboxes = lib_classification.determine_target_bboxes(img=img, model_file=model_file, threshold=DETECTION_THRESHOLD)
        coordinates = []
        for bbox in target_bboxes:
            coordinates.append(compute_coordinates_from_bbox(bbox=bbox, n_img_rows=len(img), n_img_cols=len(img[0]), coordinates_bbox=coordinates_bbox_asset))

        polygons = []
        for coords in coordinates:
            polygons.append(shapely.geometry.Polygon(create_polygon_from_coordinates(coords)))

        asset_prefix = asset_prefix_from_asset(asset)
        fn = os.path.join(export_dir, f'ROIs-{asset_prefix}.geojson')
        if len(polygons) > 0:
            gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polygons))
            gdf = gdf.set_crs("epsg:4326")
            gdf.to_file(fn, driver='GeoJSON')
            print(f"    -> exported to {fn}")
        else:
            # if nothing was detected, we write an empty file to avoid
            # reanalyzing the corresponding asset
            pathlib.Path(fn).touch()
            print(f"    -> written empty file {fn}")

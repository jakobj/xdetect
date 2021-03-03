import os
import requests
import re

URL = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissimage-dop10/items?bbox={bbox}"


def get_features_from_bbox(bbox):
    r = requests.get(URL.format(bbox=",".join(str(f) for f in bbox)))
    if r.status_code != 200:
        raise RuntimeError(r.text)

    features = r.json()["features"]
    print(f"  found {len(features)} features for bounding box {bbox}")
    return features


def get_assets_from_features(features, *, output_dir):
    for feature in features:
        identifier = feature["id"]
        rgx = re.compile(f"{identifier}_0\.1_.*\.tif")

        for a in feature["assets"]:
            if rgx.search(a) is not None:
                fn = os.path.join(output_dir, a)
                if not os.path.isfile(fn):
                    print(f"  downloading asset '{a}' -> {fn}")
                    file_response = requests.get(feature["assets"][a]["href"])
                    with open(fn, "wb") as f:
                        f.write(file_response.content)
                else:
                    print(f"  skipping asset '{a}' - file exists")


def get_assets_from_bbox(bbox, output_dir):
    return get_assets_from_features(get_features_from_bbox(bbox), output_dir=output_dir)

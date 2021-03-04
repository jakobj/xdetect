import os
import json
import requests
import re

URL = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissimage-dop10/items?bbox={bbox}"


def get_json(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()


def get_features_from_bbox(bbox):
    """Return all features for bounding box"""

    url = URL.format(bbox=",".join(str(f) for f in bbox))
    features = []
    while True:
        r_json = get_json(url)
        features += r_json["features"]
        more_features = False
        for link in r_json['links']:
            if link['rel'] == 'next':
                url = link['href']
                more_features = True

        if not more_features:
            break

    print(f"  found {len(features)} features for bounding box {bbox}")
    return features


def get_assets_from_features(*, features, output_dir):
    """Download 0.1cm assets (images) for each feature and store locally."""

    for feature in features:
        identifier = feature["id"]

        with open(os.path.join(output_dir, f"{identifier}.json"), "w") as f:
            json.dump(feature, f)

        rgx = re.compile(f"{identifier}_0\.1_.*\.tif")
        for asset in feature["assets"]:
            if rgx.search(asset) is not None:
                fn = os.path.join(output_dir, asset)
                if not os.path.isfile(fn):
                    print(f"  downloading asset '{asset}' -> {fn}")
                    file_response = requests.get(feature["assets"][asset]["href"])
                    with open(fn, "wb") as f:
                        f.write(file_response.content)
                else:
                    print(f"  skipping asset '{asset}' - file exists")


def get_assets_from_bbox(*, bbox, output_dir):
    """Download all 0.1cm assets (images) for bounding box and store locally."""

    return get_assets_from_features(
        features=get_features_from_bbox(bbox), output_dir=output_dir
    )

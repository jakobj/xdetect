import glob
import numpy as np
import os
from skimage import io

from annotate import (
    asset_from_filename,
    asset_prefix_from_asset,
    save_patch,
    mkdirp,
)
import config


def count_examples(*, examples_dir, asset_prefix):
    return len(glob.glob(os.path.join(examples_dir, f"{asset_prefix}*.png")))


def generate_negative_examples(*, asset_dir, examples_dir):

    mkdirp(os.path.join(examples_dir, "negative"))

    for fn in glob.glob(os.path.join(asset_dir, "*_0.1_*.tif")):

        asset = asset_from_filename(fn)
        asset_prefix = asset_prefix_from_asset(asset)
        n_positive_examples = count_examples(
            examples_dir=os.path.join(examples_dir, "positive"),
            asset_prefix=asset_prefix,
        )
        if n_positive_examples == 0:
            continue

        n_negative_examples = count_examples(
            examples_dir=os.path.join(examples_dir, "negative"),
            asset_prefix=asset_prefix,
        )
        if n_negative_examples > config.REL_NUMBER_NEGATIVE_EXAMPLES * n_positive_examples:
            raise RuntimeError(
                f"too many negative examples for asset '{asset}' - remove a few"
            )

        img = io.imread(fn)

        n_rows = img.shape[0] // config.MINIMAL_EDGE_LENGTH
        n_cols = img.shape[1] // config.MINIMAL_EDGE_LENGTH

        assert n_rows * config.MINIMAL_EDGE_LENGTH == img.shape[0]
        assert n_cols * config.MINIMAL_EDGE_LENGTH == img.shape[1]

        print(f"  generating negative examples for asset '{asset}'")
        while n_negative_examples < config.REL_NUMBER_NEGATIVE_EXAMPLES * n_positive_examples:

            row = np.random.randint(n_rows)
            col = np.random.randint(n_cols)

            bbox = (
                row * config.MINIMAL_EDGE_LENGTH,
                col * config.MINIMAL_EDGE_LENGTH,
                (row + 1) * config.MINIMAL_EDGE_LENGTH,
                (col + 1) * config.MINIMAL_EDGE_LENGTH,
            )
            patch = img[bbox[0] : bbox[2], bbox[1] : bbox[3]]

            save_patch(
                patch=patch,
                output_dir=os.path.join(examples_dir, "negative"),
                asset_prefix=asset_prefix,
                bbox=bbox,
            )

            n_negative_examples += 1


if __name__ == "__main__":
    asset_dir = "../data/assets/"
    examples_dir = f"../data/crosswalks/"

    generate_negative_examples(asset_dir=asset_dir, examples_dir=examples_dir)

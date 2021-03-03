import glob
import numpy as np
import os
from skimage import io

from annotate import MINIMAL_EDGE_LENGTH, asset_from_file_name, save_patch, mkdirp


REL_NUMBER_NEGATIVE_EXAMPLES = 2


def count_examples(examples_dir, *, asset):
    return len(glob.glob(os.path.join(examples_dir, f"{asset}*.png")))


def generate_negative_examples(asset_dir, examples_dir):

    mkdirp(os.path.join(examples_dir, 'negative'))

    for fn in glob.glob(os.path.join(asset_dir, "*_0.1_*.tif")):

        asset = asset_from_file_name(fn)
        n_positive_examples = count_examples(os.path.join(examples_dir, 'positive'), asset=asset)
        if n_positive_examples == 0:
            continue

        n_negative_examples = count_examples(os.path.join(examples_dir, 'negative'), asset=asset)
        if n_negative_examples > REL_NUMBER_NEGATIVE_EXAMPLES * n_positive_examples:
            raise RuntimeError(
                f"too many negative examples for asset '{asset}' - remove a few"
            )

        img = io.imread(fn)

        n_rows = img.shape[0] // MINIMAL_EDGE_LENGTH
        n_cols = img.shape[1] // MINIMAL_EDGE_LENGTH

        assert n_rows * MINIMAL_EDGE_LENGTH == img.shape[0]
        assert n_cols * MINIMAL_EDGE_LENGTH == img.shape[1]

        while n_negative_examples < REL_NUMBER_NEGATIVE_EXAMPLES * n_positive_examples:

            row = np.random.randint(n_rows)
            col = np.random.randint(n_cols)

            bbox = (
                row * MINIMAL_EDGE_LENGTH,
                col * MINIMAL_EDGE_LENGTH,
                (row + 1) * MINIMAL_EDGE_LENGTH,
                (col + 1) * MINIMAL_EDGE_LENGTH,
            )
            patch = img[bbox[0] : bbox[2], bbox[1] : bbox[3]]

            save_patch(
                patch,
                output_dir=os.path.join(examples_dir, "negative"),
                asset=asset,
                bbox=bbox,
            )

            n_negative_examples += 1


if __name__ == "__main__":
    asset_dir = "../data/"
    examples_dir = f"../data_annotated_{MINIMAL_EDGE_LENGTH}px/"

    generate_negative_examples(asset_dir, examples_dir)

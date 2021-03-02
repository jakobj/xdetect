import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from skimage import io


from annotate import determine_annotated_assets


def get_examples_as_array(original_dir, input_dir):
    annotated_assets = determine_annotated_assets(input_dir)

    data = []
    for asset in annotated_assets:
        rgx = re.compile(os.path.join(input_dir, f"{asset}-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).png"))
        img = io.imread(os.path.join(original_dir, f"{asset}.tif"))
        for fn in sorted(glob.glob(os.path.join(input_dir, f"{asset}*.png"))):
            match = rgx.search(fn)
            y, x, yDy, xDx = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            data.append(img[y:yDy, x:xDx])

    return np.array(data)


if __name__ == '__main__':

    original_dir = "./data/"
    input_dir = "./data_annotated/"
    output_dir = "./datasets/"

    positive_examples = get_examples_as_array(original_dir, os.path.join(input_dir, 'positive'))
    negative_examples = get_examples_as_array(original_dir, os.path.join(input_dir, 'negative'))

    np.save(os.path.join(output_dir, "first_dataset_positive.npy"), positive_examples)
    np.save(os.path.join(output_dir, "first_dataset_negative.npy"), negative_examples)

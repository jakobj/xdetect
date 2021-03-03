import glob
import matplotlib.pyplot as plt
import os
import re
from skimage import io

import api_wrapper


MINIMAL_EDGE_LENGTH = 100


def devide_into_patches(patch, *, bbox):

    n_rows = bbox[2] - bbox[0]
    n_cols = bbox[3] - bbox[1]

    Dy = n_rows // 2
    Dx = n_cols // 2

    if Dy % MINIMAL_EDGE_LENGTH == 0:
        Dy_lower = Dy
        Dy_upper = Dy
    else:
        Dy_lower = Dy - (Dy % MINIMAL_EDGE_LENGTH)
        Dy_upper = Dy_lower + MINIMAL_EDGE_LENGTH

    if Dx % MINIMAL_EDGE_LENGTH == 0:
        Dx_lower = Dx
        Dx_upper = Dx
    else:
        Dx_lower = Dx - (Dx % MINIMAL_EDGE_LENGTH)
        Dx_upper = Dx_lower + MINIMAL_EDGE_LENGTH

    patches = [
        patch[:Dy_upper, :Dx_upper],
        patch[:Dy_upper, Dx_lower:],
        patch[Dy_lower:, :Dx_upper],
        patch[Dy_lower:, Dx_lower:],
    ]
    bboxes = [
        (bbox[0], bbox[1], bbox[0] + Dy_upper, bbox[1] + Dx_upper),
        (bbox[0], bbox[1] + Dx_lower, bbox[0] + Dy_upper, bbox[3]),
        (bbox[0] + Dy_lower, bbox[1], bbox[2], bbox[1] + Dx_upper),
        (bbox[0] + Dy_lower, bbox[1] + Dx_lower, bbox[2], bbox[3]),
    ]

    for patch_i, bbox_i in zip(patches, bboxes):
        assert len(patch_i) == bbox_i[2] - bbox_i[0]
        assert len(patch_i[0]) == bbox_i[3] - bbox_i[1]

    return patches, bboxes


def save_patch(patch, *, output_dir, asset, bbox):
    assert len(patch) == MINIMAL_EDGE_LENGTH
    plt.clf()
    plt.imshow(patch)
    fn_out = os.path.join(
        output_dir, asset + f"-{'_'.join(str(i) for i in bbox)}.png",
    )
    print(f"  saving example -> {fn_out}")
    plt.savefig(fn_out)
    plt.close("all")


def determine_target_locations(patch, *, bbox):
    target_locations = []

    keys = 'uijk'
    def onpress(event):
        if event.key == keys[0]:
            target_locations.append((bbox[0], bbox[1]))
        elif event.key == keys[1]:
            target_locations.append((bbox[0], bbox[3] - 1))
        elif event.key == keys[2]:
            target_locations.append((bbox[2] - 1, bbox[1]))
        elif event.key == keys[3]:
            target_locations.append((bbox[2] - 1, bbox[3] - 1))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.suptitle(
        "please click all tiles with the target category", color="r", fontsize=20
    )
    ax.imshow(patch, rasterized=True)
    ax.axhline(len(patch) // 2, lw=3, color="r")
    ax.axvline(len(patch[0]) // 2, lw=3, color="r")
    fig.canvas.mpl_connect("key_press_event", onpress)
    plt.show()
    plt.close("all")

    return target_locations


def is_loc_in_bbox(loc, bbox):
    return (loc[0] >= bbox[0] and loc[0] < bbox[2]) and (
        loc[1] >= bbox[1] and loc[1] < bbox[3]
    )


def process_patch(patch, *, output_dir, asset, bbox):

    assert bbox[0] % MINIMAL_EDGE_LENGTH == 0
    assert bbox[1] % MINIMAL_EDGE_LENGTH == 0

    if len(patch) <= MINIMAL_EDGE_LENGTH:
        save_patch(patch, output_dir=os.path.join(output_dir, 'positive'), asset=asset, bbox=bbox)
        return

    target_locations = determine_target_locations(patch, bbox=bbox)

    patches, bboxes = devide_into_patches(patch, bbox=bbox)
    for patch_i, bbox_i in zip(patches, bboxes):
        for target_loc_i in target_locations:
            if is_loc_in_bbox(target_loc_i, bbox_i):
                process_patch(
                    patch_i,
                    output_dir=output_dir,
                    asset=asset,
                    bbox=bbox_i,
                )
                break  # since this patch has been processed we can
                # move on to next patch even if multiple target
                # locations were indicated for this patch


def determine_annotated_assets(output_dir):
    annotated_assets = set()
    rgx = re.compile("(swissimage-dop10_[0-9]{4}_[0-9]{4}-[0-9]{4}_0.1_[0-9]{4}).*.png")
    for fn in glob.glob(os.path.join(output_dir, "*.png")):
        annotated_assets.add(rgx.search(fn)[1])
    return annotated_assets


def asset_from_file_name(fn):
    return re.search("\/(swissimage.*)\.tif", fn)[1]


def generate_positive_examples_from_assets(asset_dir, examples_dir):

    annotated_assets = determine_annotated_assets(os.path.join(examples_dir, "positive"))

    for fn in glob.glob(os.path.join(asset_dir, "*_0.1_*.tif")):
        asset = asset_from_file_name(fn)
        if asset not in annotated_assets:
            print(f"  annotating {fn}")
            img = io.imread(fn)
            process_patch(
                img,
                output_dir=examples_dir,
                asset=asset,
                bbox=(0, 0, 10_000, 10_000),
            )
        else:
            print(f"  skipping {fn} - already annotated")


if __name__ == "__main__":

    asset_dir = "../data/"
    examples_dir = "../data_annotated/"

    # bbox notation (E, N, E + DE, N + DN)
    bbox = (7.42390, 46.93353, 7.45145, 46.95342)
    api_wrapper.get_assets_from_bbox(bbox, output_dir=asset_dir)
    generate_positive_examples_from_assets(asset_dir, examples_dir)

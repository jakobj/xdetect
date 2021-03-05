import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import re
from skimage import io

import api_wrapper
import config


def devide_into_patches(*, patch, bbox):

    n_rows = bbox[2] - bbox[0]
    n_cols = bbox[3] - bbox[1]

    Dy = n_rows // 2
    Dx = n_cols // 2

    if Dy % config.MINIMAL_EDGE_LENGTH == 0:
        Dy_lower = Dy
        Dy_upper = Dy
    else:
        Dy_lower = Dy - (Dy % config.MINIMAL_EDGE_LENGTH)
        Dy_upper = Dy_lower + config.MINIMAL_EDGE_LENGTH

    if Dx % config.MINIMAL_EDGE_LENGTH == 0:
        Dx_lower = Dx
        Dx_upper = Dx
    else:
        Dx_lower = Dx - (Dx % config.MINIMAL_EDGE_LENGTH)
        Dx_upper = Dx_lower + config.MINIMAL_EDGE_LENGTH

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


def save_patch(*, patch, output_dir, asset_prefix, bbox):
    assert len(patch) == config.MINIMAL_EDGE_LENGTH
    plt.clf()
    plt.imshow(patch)
    fn_out = os.path.join(
        output_dir, f"{asset_prefix}-{'_'.join(str(i) for i in bbox)}.png",
    )
    print(f"  saving example -> {fn_out}")
    plt.savefig(fn_out)
    plt.close("all")


def determine_target_locations(*, patch, bbox, annotated_patches):
    target_locations = []

    keys = "uijk"

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
        f"please select all tiles with the target category ({'/'.join(keys)})",
        color="r",
        fontsize=20,
    )
    ax.imshow(patch, rasterized=True)
    ax.axhline(len(patch) // 2, lw=3, color="r")
    ax.axvline(len(patch[0]) // 2, lw=3, color="r")
    draw_annotated_patches(ax=ax, bbox=bbox, annotated_patches=annotated_patches)
    fig.canvas.mpl_connect("key_press_event", onpress)
    plt.show()
    plt.close("all")

    return target_locations


def draw_annotated_patches(*, ax, bbox, annotated_patches):
    for annotated_patch in annotated_patches:
        if is_loc_in_bbox(loc=annotated_patch[:2], bbox=bbox) and is_loc_in_bbox(
            loc=annotated_patch[2:], bbox=bbox
        ):
            y = annotated_patch[0] - bbox[0]
            Dy = annotated_patch[2] - bbox[0] - y
            x = annotated_patch[1] - bbox[1]
            Dx = annotated_patch[3] - bbox[1] - x
            ax.add_patch(Rectangle((x, y), Dx, Dy, color="b", alpha=0.4))


def is_loc_in_bbox(*, loc, bbox):
    return (loc[0] >= bbox[0] and loc[0] < bbox[2]) and (
        loc[1] >= bbox[1] and loc[1] < bbox[3]
    )


def process_patch(*, patch, output_dir, asset_prefix, bbox, annotated_patches):

    assert bbox[0] % config.MINIMAL_EDGE_LENGTH == 0
    assert bbox[1] % config.MINIMAL_EDGE_LENGTH == 0

    if len(patch) <= config.MINIMAL_EDGE_LENGTH:
        save_patch(
            patch=patch,
            output_dir=os.path.join(output_dir, "positive"),
            asset_prefix=asset_prefix,
            bbox=bbox,
        )
        return

    target_locations = determine_target_locations(
        patch=patch, bbox=bbox, annotated_patches=annotated_patches
    )

    patches, bboxes = devide_into_patches(patch=patch, bbox=bbox)
    for patch_i, bbox_i in zip(patches, bboxes):
        for target_loc_i in target_locations:
            if is_loc_in_bbox(loc=target_loc_i, bbox=bbox_i):
                process_patch(
                    patch=patch_i,
                    output_dir=output_dir,
                    asset_prefix=asset_prefix,
                    bbox=bbox_i,
                    annotated_patches=annotated_patches,
                )
                break  # since this patch has been processed we can
                # move on to next patch even if multiple target
                # locations were indicated for this patch


def identifier_from_asset(asset):
    return re.search("(swissimage.*)_0.1_.*\.tif", asset)[1]


def identifier_from_filename(fn):
    return identifier_from_asset(os.path.basename(fn))


def asset_prefix_from_asset(asset):
    return os.path.splitext(asset)[0]


def asset_prefix_from_filename(fn):
    return asset_prefix_from_asset(os.path.basename(fn))


def asset_from_filename(fn):
    return os.path.basename(fn)


def generate_positive_examples_from_assets(*, asset_dir, examples_dir, assets):

    mkdirp(examples_dir)
    mkdirp(os.path.join(examples_dir, "positive"))

    for asset in assets:
        asset_prefix = asset_prefix_from_asset(asset)
        annotated_patches = determine_annotated_patches(
            examples_dir=examples_dir, asset_prefix=asset_prefix
        )
        if len(annotated_patches) > 0:
            print(f"  asset '{asset}' has already been annotated")
            inp = ""
            while inp not in ("y", "n"):
                inp = input("  reannotate asset? (y/n) ")
            if inp == "y":
                pass
            elif inp == "n":
                continue

        print(f"  annotating asset '{asset}'")
        img = io.imread(os.path.join(asset_dir, asset))
        process_patch(
            patch=img,
            output_dir=examples_dir,
            asset_prefix=asset_prefix,
            bbox=(0, 0, 10_000, 10_000),
            annotated_patches=annotated_patches,
        )


def mkdirp(directory):
    os.makedirs(directory, exist_ok=True)


def determine_annotated_patches(*, examples_dir, asset_prefix):
    annotated_patches = []
    rgx = re.compile(
        os.path.join(
            examples_dir,
            "positive",
            f"{asset_prefix}-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).png",
        )
    )
    for fn in sorted(
        glob.glob(os.path.join(examples_dir, "positive", f"{asset_prefix}*.png"))
    ):
        match = rgx.search(fn)
        y, x, yDy, xDx = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        annotated_patches.append([y, x, yDy, xDx])
    return annotated_patches


if __name__ == "__main__":

    asset_dir = "../data/"
    examples_dir = f"../data_annotated_{MINIMAL_EDGE_LENGTH}px/"

    # bbox notation (E, N, E + DE, N + DN)
    bbox = (7.42390, 46.93353, 7.45145, 46.95342)
    api_wrapper.get_assets_from_bbox(bbox=bbox, output_dir=asset_dir)
    assets = api_wrapper.get_assets_from_bbox(bbox=bbox, output_dir=asset_dir)
    generate_positive_examples_from_assets(
        asset_dir=asset_dir, examples_dir=examples_dir, assets=assets,
    )

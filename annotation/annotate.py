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


def compute_midpoint(bbox):
    return bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2


def determine_target_locations(*, patch, bbox, sub_bboxes, annotated_bboxes):
    target_locations = set()
    rectangles = {}

    keys = "uijk"

    def onpress(event):
        if event.key in keys:
            idx = keys.index(event.key)
            midpoint = compute_midpoint(sub_bboxes[idx])
            if midpoint not in target_locations:
                target_locations.add(midpoint)
                r = event.canvas.figure.axes[0].add_patch(
                    Rectangle(
                        (sub_bboxes[idx][1] - bbox[1], sub_bboxes[idx][0] - bbox[0]),
                        sub_bboxes[idx][3] - sub_bboxes[idx][1],
                        sub_bboxes[idx][2] - sub_bboxes[idx][0],
                        color='g',
                        alpha=0.4,
                        zorder=1
                    )
                )
                rectangles[midpoint] = r
                event.canvas.draw_idle()
            else:
                target_locations.remove(midpoint)
                rectangles[midpoint].remove()
                event.canvas.draw_idle()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.suptitle(
        f"please select all tiles with the target category ({'/'.join(keys)})",
        color="r",
        fontsize=20,
    )
    ax.imshow(patch, rasterized=True)
    ax.axhline(len(patch) // 2, lw=3, color="0.8", zorder=2)
    ax.axvline(len(patch[0]) // 2, lw=3, color="0.8", zorder=2)
    draw_annotated_bboxes(ax=ax, bbox=bbox, annotated_bboxes=annotated_bboxes)
    fig.canvas.mpl_connect("key_press_event", onpress)
    plt.show()
    plt.close("all")

    return target_locations


def draw_annotated_bboxes(*, ax, bbox, annotated_bboxes):
    for annotated_bbox in annotated_bboxes:
        midpoint = compute_midpoint(annotated_bbox)
        if is_loc_in_bbox(loc=midpoint, bbox=bbox) and is_loc_in_bbox(
            loc=midpoint, bbox=bbox
        ):
            y = annotated_bbox[0] - bbox[0]
            Dy = annotated_bbox[2] - bbox[0] - y
            x = annotated_bbox[1] - bbox[1]
            Dx = annotated_bbox[3] - bbox[1] - x
            ax.add_patch(Rectangle((x, y), Dx, Dy, color="b", alpha=0.4))


def is_loc_in_bbox(*, loc, bbox):
    return (loc[0] >= bbox[0] and loc[0] < bbox[2]) and (
        loc[1] >= bbox[1] and loc[1] < bbox[3]
    )


def process_patch(*, patch, output_dir, asset_prefix, bbox, annotated_bboxes):

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

    sub_patches, sub_bboxes = devide_into_patches(patch=patch, bbox=bbox)
    target_locations = determine_target_locations(
        patch=patch, bbox=bbox, sub_bboxes=sub_bboxes, annotated_bboxes=annotated_bboxes
    )

    for patch_i, bbox_i in zip(sub_patches, sub_bboxes):
        for target_loc_i in target_locations:
            if is_loc_in_bbox(loc=target_loc_i, bbox=bbox_i):
                process_patch(
                    patch=patch_i,
                    output_dir=output_dir,
                    asset_prefix=asset_prefix,
                    bbox=bbox_i,
                    annotated_bboxes=annotated_bboxes,
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
        annotated_bboxes = determine_annotated_bboxes(
            examples_dir=examples_dir, asset_prefix=asset_prefix
        )
        if len(annotated_bboxes) > 0:
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
            annotated_bboxes=annotated_bboxes,
        )


def mkdirp(directory):
    os.makedirs(directory, exist_ok=True)


def determine_annotated_bboxes(*, examples_dir, asset_prefix):
    annotated_bboxes = []
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
        annotated_bboxes.append([y, x, yDy, xDx])
    return annotated_bboxes


if __name__ == "__main__":

    asset_dir = "../data/assets/"
    examples_dir = f"../data/crosswalks/"

    # bbox notation (E, N, E + DE, N + DN)
    bbox = (7.42390, 46.93353, 7.45145, 46.95342)
    api_wrapper.get_assets_from_bbox(bbox=bbox, output_dir=asset_dir)
    assets = api_wrapper.get_assets_from_bbox(bbox=bbox, output_dir=asset_dir)
    generate_positive_examples_from_assets(
        asset_dir=asset_dir, examples_dir=examples_dir, assets=assets,
    )

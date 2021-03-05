import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from skimage import io
import sys

from config import DETECTION_THRESHOLD, MINIMAL_EDGE_LENGTH
import lib_classification

sys.path.insert(0, "../annotation/")
from annotate import asset_prefix_from_asset, identifier_from_asset, mkdirp, save_patch


def store_locations(
    *, examples_dir, img, asset_prefix, locations
):
    mkdirp(examples_dir)
    for loc in locations:
        save_patch(
            patch=img[
                loc[0] : loc[2],
                loc[1] : loc[3],
            ],
            output_dir=examples_dir,
            asset_prefix=asset_prefix,
            bbox=loc,
        )


def event_is_close(*, event, location):
    return (location[0] - 5 <= event.ydata and event.ydata < location[0] + 5) and (
        location[1] - 5 <= event.xdata and event.xdata < location[1] + 5
    )


def add_grid(*, ax, n_rows, n_cols):
    for i in range(n_rows):
        ax.axhline(i * MINIMAL_EDGE_LENGTH, color="0.8", lw=0.5, zorder=-1, alpha=0.5)
    for j in range(n_cols):
        ax.axvline(j * MINIMAL_EDGE_LENGTH, color="0.8", lw=0.5, zorder=-1, alpha=0.5)


def draw_bboxes(*, ax, bboxes, edgecolor):
    for bbox in bboxes:
        ax.add_patch(
            Rectangle(
                (bbox[1], bbox[0]),
                MINIMAL_EDGE_LENGTH,
                MINIMAL_EDGE_LENGTH,
                edgecolor=edgecolor,
                fill=False,
            )
        )


if __name__ == "__main__":

    asset_dir = "../data/assets/"
    examples_dir = "../data/crosswalks/"
    model_file = "./crosswalks.torch"
    asset = "swissimage-dop10_2018_2598-1198_0.1_2056.tif"
    # asset = "swissimage-dop10_2018_2598-1199_0.1_2056.tif"
    # asset = "swissimage-dop10_2018_2598-1200_0.1_2056.tif"
    # asset = "swissimage-dop10_2018_2599-1198_0.1_2056.tif"
    # asset = "swissimage-dop10_2018_2599-1199_0.1_2056.tif"
    # asset = "swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    # asset = "swissimage-dop10_2018_2598-1201_0.1_2056.tif"

    img = io.imread(os.path.join(asset_dir, asset))
    # img = img[:1000, :1000]

    target_bboxes_ground_truth = lib_classification.determine_target_bboxes_ground_truth(
        asset_dir=asset_dir,
        examples_dir=os.path.join(examples_dir, "positive"),
        identifier=identifier_from_asset(asset),
    )
    target_bboxes = lib_classification.determine_target_bboxes(img=img, model_file=model_file, threshold=DETECTION_THRESHOLD)

    ax_ref = [None]
    potential_location = [None, None]
    missed_locations = set()
    missclassified_locations = set()
    patches = {}

    def onclick(event):
        potential_location[0] = event.ydata
        potential_location[1] = event.xdata

    def onrelease(event):
        if event_is_close(event=event, location=potential_location):
            y = int(event.ydata // MINIMAL_EDGE_LENGTH) * MINIMAL_EDGE_LENGTH
            x = int(event.xdata // MINIMAL_EDGE_LENGTH) * MINIMAL_EDGE_LENGTH
            bbox = (y, x, y + MINIMAL_EDGE_LENGTH, x + MINIMAL_EDGE_LENGTH)
            if (bbox not in missclassified_locations) and (bbox not in missed_locations):
                if event.button == 1:
                    missed_locations.add(bbox)
                    color = 'b'
                elif event.button == 3:
                    missclassified_locations.add(bbox)
                    color = 'r'

                p = ax_ref[0].add_patch(
                    Rectangle(
                        (x, y),
                        MINIMAL_EDGE_LENGTH,
                        MINIMAL_EDGE_LENGTH,
                        color=color,
                        alpha=0.4,
                    )
                )
                patches[bbox] = p
                event.canvas.draw_idle()
            else:
                if event.button == 1:
                    missed_locations.remove(bbox)
                elif event.button == 3:
                    missclassified_locations.remove(bbox)
                patches[bbox].remove()
                event.canvas.draw_idle()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_ref[0] = ax
    ax.imshow(img, zorder=-2)
    draw_bboxes(ax=ax, bboxes=target_bboxes_ground_truth, edgecolor="b")
    draw_bboxes(ax=ax, bboxes=target_bboxes, edgecolor="r")
    add_grid(
        ax=ax,
        n_rows=len(img) // MINIMAL_EDGE_LENGTH,
        n_cols=len(img[0]) // MINIMAL_EDGE_LENGTH,
    )
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("button_release_event", onrelease)
    plt.show()

    store_locations(
        examples_dir=os.path.join(examples_dir, 'positive'),
        img=img,
        asset_prefix=asset_prefix_from_asset(asset),
        locations=missed_locations,
    )
    store_locations(
        examples_dir=os.path.join(examples_dir, 'missclassified'),
        img=img,
        asset_prefix=asset_prefix_from_asset(asset),
        locations=missclassified_locations,
    )

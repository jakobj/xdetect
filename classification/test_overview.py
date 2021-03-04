import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader
import torchvision
import sys

from classifier import ConvNet
from train import normalize


from train import MINIMAL_EDGE_LENGTH

sys.path.insert(0, "../annotation/")
from annotate import asset_prefix_from_filename, mkdirp, save_patch


class SegmentedImage(torch.utils.data.Dataset):
    def __init__(self, img, transform=None):

        self.patches = self.segment_image(img)
        self.transform = transform

    def segment_image(self, img):

        assert img.shape[0] % MINIMAL_EDGE_LENGTH == 0
        assert img.shape[1] % MINIMAL_EDGE_LENGTH == 0
        assert img.shape[0] == img.shape[1]

        n_rows = int(img.shape[0] // self.effective_edge_width)
        n_cols = n_rows

        patches = np.empty(
            (n_rows, n_cols, MINIMAL_EDGE_LENGTH, MINIMAL_EDGE_LENGTH, 3),
            dtype=np.float32,
        )
        for i in range(n_rows):
            for j in range(n_cols):
                patches[i, j] = (
                    img[self.get_rows(i, len(img)), self.get_cols(j, len(img[i]))]
                    / 255.0
                )
        return patches

    def get_rows(self, i, n_img_rows):
        rows = slice(
            i * self.effective_edge_width,
            i * self.effective_edge_width + MINIMAL_EDGE_LENGTH,
        )
        if rows.stop > n_img_rows:
            rows = slice(n_img_rows - MINIMAL_EDGE_LENGTH, n_img_rows)
        return rows

    def get_cols(self, j, n_img_cols):
        cols = slice(
            j * self.effective_edge_width,
            j * self.effective_edge_width + MINIMAL_EDGE_LENGTH,
        )
        if cols.stop > n_img_cols:
            cols = slice(n_img_cols - MINIMAL_EDGE_LENGTH, n_img_cols)
        return cols

    @property
    def effective_edge_width(self):
        stride_ratio = 0.66  # see http://hw.oeaw.ac.at/0xc1aa500e%200x00373589.pdf
        return int(stride_ratio * MINIMAL_EDGE_LENGTH)

    @property
    def n_rows(self):
        return len(self.patches)

    @property
    def n_cols(self):
        return len(self.patches[0])

    def __len__(self):
        return self.n_rows * self.n_cols

    def __getitem__(self, idx):

        row = idx // self.n_rows
        col = idx % self.n_rows

        img = self.patches[row, col].copy()

        if self.transform is not None:
            return self.transform(img)

        return img


def determine_target_bboxes(*, img, threshold=0.5):

    dataset = SegmentedImage(
        img,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(normalize),
            ]
        ),
    )
    data_loader = DataLoader(dataset, batch_size=dataset.n_cols, shuffle=False)

    model = ConvNet()
    model.load_state_dict(torch.load("./first_model.torch"))

    target_bboxes = []
    with torch.no_grad():
        for i, x in enumerate(data_loader):

            outputs = model(x.detach())
            probs = torch.nn.functional.softmax(outputs, dim=1)

            rows = dataset.get_rows(i, len(img))
            for j in range(dataset.n_cols):
                if probs[j, 1] > threshold:
                    cols = dataset.get_cols(j, len(img[0]))
                    target_bboxes.append((rows.start, cols.start, rows.stop, cols.stop))

    return target_bboxes


def store_missclassified_locations(
    *, missclassifications_dir, img, asset_prefix, missclassified_locations
):
    mkdirp(missclassifications_dir)
    for missclassified_loc_i in missclassified_locations:
        save_patch(
            patch=img[
                missclassified_loc_i[0] : missclassified_loc_i[2],
                missclassified_loc_i[1] : missclassified_loc_i[3],
            ],
            output_dir=missclassifications_dir,
            asset_prefix=asset_prefix,
            bbox=missclassified_loc_i,
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

    missclassifications_dir = "../data_annotated_50px/missclassified/"
    # fn_test_image = "../data/swissimage-dop10_2018_2599-1198_0.1_2056.tif"
    fn_test_image = "../data/swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    img = io.imread(fn_test_image)

    target_bboxes = determine_target_bboxes(img=img, threshold=0.9999)

    ax_ref = [None]
    potential_location = [None, None]
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
            if bbox not in missclassified_locations:
                missclassified_locations.add(bbox)
                p = ax_ref[0].add_patch(
                    Rectangle(
                        (x, y),
                        MINIMAL_EDGE_LENGTH,
                        MINIMAL_EDGE_LENGTH,
                        color="b",
                        alpha=0.4,
                    )
                )
                patches[bbox] = p
                event.canvas.draw_idle()
            else:
                missclassified_locations.remove(bbox)
                patches[bbox].remove()
                event.canvas.draw_idle()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_ref[0] = ax
    ax.imshow(img, zorder=-2)
    draw_bboxes(ax=ax, bboxes=target_bboxes, edgecolor="b")
    add_grid(
        ax=ax,
        n_rows=len(img) // MINIMAL_EDGE_LENGTH,
        n_cols=len(img[0]) // MINIMAL_EDGE_LENGTH,
    )
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("button_release_event", onrelease)
    plt.show()

    store_missclassified_locations(
        missclassifications_dir=missclassifications_dir,
        img=img,
        asset_prefix=asset_prefix_from_filename(fn_test_image),
        missclassified_locations=missclassified_locations,
    )

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from skimage import io
import skimage.transform
import torch
from torch.utils.data import DataLoader
import torchvision
import sys

from classifier import ConvNet
from train import normalize, validation
from swissimage_10cm_dataset import SWISSIMAGE10cmDataset


from train import MINIMAL_EDGE_LENGTH

sys.path.insert(0, '../annotation/')
from annotate import asset_from_file_name, mkdirp, save_patch


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

        patches = np.empty((n_rows, n_cols, MINIMAL_EDGE_LENGTH, MINIMAL_EDGE_LENGTH, 3), dtype=np.float32)
        for i in range(n_rows):
            rows = slice(i * self.effective_edge_width, i * self.effective_edge_width + MINIMAL_EDGE_LENGTH)
            if rows.stop > len(img):
                rows = slice(len(img) - MINIMAL_EDGE_LENGTH, len(img))
            for j in range(n_cols):
                cols = slice(j * self.effective_edge_width, j * self.effective_edge_width + MINIMAL_EDGE_LENGTH)
                if cols.stop > len(img[i]):
                    cols = slice(len(img[i]) - MINIMAL_EDGE_LENGTH, len(img[i]))
                patches[i, j] = img[rows, cols] / 255.

        return patches

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


def create_heatmap(img, *, threshold=0.5):

    dataset = SegmentedImage(img, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(normalize),
    ]))
    data_loader = DataLoader(dataset, batch_size=dataset.n_cols, shuffle=False)

    model = ConvNet()
    model.load_state_dict(torch.load("./first_model.torch"))

    heatmap = np.zeros((img.shape[0], img.shape[1]))
    with torch.no_grad():
        for i, x in enumerate(data_loader):

            outputs = model(x.detach())
            probs = torch.nn.functional.softmax(outputs, dim=1)

            for j in range(dataset.n_cols):
                heatmap[i * dataset.effective_edge_width:i * dataset.effective_edge_width + MINIMAL_EDGE_LENGTH,
                        j * dataset.effective_edge_width:j * dataset.effective_edge_width + MINIMAL_EDGE_LENGTH] = (probs[j, 1] > threshold)

    return heatmap


def store_missclassified_locations(missclassifications_dir, *, img, asset, missclassified_locations):
    mkdirp(missclassifications_dir)
    for missclassified_loc_i in missclassified_locations:
        save_patch(img[missclassified_loc_i[0]:missclassified_loc_i[2], missclassified_loc_i[1]:missclassified_loc_i[3]], output_dir=missclassifications_dir, asset=asset, bbox=missclassified_loc_i)


if __name__ == '__main__':

    missclassifications_dir = "../data_annotated_50px/missclassified/"
    fn_test_image = "../data/swissimage-dop10_2018_2599-1198_0.1_2056.tif"
    # fn_test_image = "../data/swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    img = io.imread(fn_test_image)
    # img = img[:1000, :1000]

    heatmap = create_heatmap(img, threshold=0.99)
    assert img.shape[:2] == heatmap.shape

    ax_ref = [None]
    missclassified_locations = []
    def onclick(event):
        y = int(event.ydata // MINIMAL_EDGE_LENGTH) * MINIMAL_EDGE_LENGTH
        x = int(event.xdata // MINIMAL_EDGE_LENGTH) * MINIMAL_EDGE_LENGTH
        missclassified_locations.append((y, x, y + MINIMAL_EDGE_LENGTH, x + MINIMAL_EDGE_LENGTH))
        ax_ref[0].add_patch(Rectangle((x, y), MINIMAL_EDGE_LENGTH, MINIMAL_EDGE_LENGTH, color='b', alpha=0.4))
        event.canvas.draw_idle()

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0., 0., 1., 1.])
    ax_ref[0] = ax
    ax.imshow(img)
    ax.imshow(heatmap, alpha=0.4, cmap='Reds', vmin=0, vmax=1)
    # plt.savefig('test.png', dpi=1200)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    store_missclassified_locations(missclassifications_dir, img=img, asset=asset_from_file_name(fn_test_image), missclassified_locations=missclassified_locations)

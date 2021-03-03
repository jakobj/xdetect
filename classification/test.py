import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import skimage.transform
import torch
from torch.utils.data import DataLoader
import torchvision


from classifier import ConvNet
from train import normalize, validation
from swissimage_10cm_dataset import SWISSIMAGE10cmDataset


MINIMAL_EDGE_LENGTH = 50


class SegmentedImage(torch.utils.data.Dataset):

    def __init__(self, img, transform=None):

        assert img.shape[0] % MINIMAL_EDGE_LENGTH == 0
        assert img.shape[1] % MINIMAL_EDGE_LENGTH == 0

        self.n_rows = len(img) // MINIMAL_EDGE_LENGTH
        self.n_cols = len(img) // MINIMAL_EDGE_LENGTH
        self.patches = self.segment_image(img)

        self.transform = transform

    def segment_image(self, img):
        patches = np.empty((self.n_rows, self.n_cols, MINIMAL_EDGE_LENGTH, MINIMAL_EDGE_LENGTH, 3), dtype=np.float32)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                patches[i, j] = img[i * MINIMAL_EDGE_LENGTH:(i + 1) * MINIMAL_EDGE_LENGTH,
                                    j * MINIMAL_EDGE_LENGTH:(j + 1) * MINIMAL_EDGE_LENGTH]

        return patches

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

    heatmap = np.zeros(img.shape[:2])
    with torch.no_grad():
        for i, x in enumerate(data_loader):

            outputs = model(x.detach())
            probs = torch.nn.functional.softmax(outputs, dim=1)

            for j in range(dataset.n_cols):
                heatmap[i * MINIMAL_EDGE_LENGTH:(i + 1) * MINIMAL_EDGE_LENGTH,
                        j * MINIMAL_EDGE_LENGTH:(j + 1) * MINIMAL_EDGE_LENGTH] = (probs[j, 1] > threshold)

    return heatmap


if __name__ == '__main__':

    # fn_test_image = "../data/swissimage-dop10_2018_2599-1198_0.1_2056.tif"
    fn_test_image = "../data/swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    img = io.imread(fn_test_image)

    heatmap = create_heatmap(img, threshold=0.99)
    assert img.shape[:2] == heatmap.shape

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_axes([0., 0., 1., 1.])
    ax.imshow(img)
    ax.imshow(heatmap, alpha=0.4, cmap='Reds', vmin=0, vmax=1)
    # plt.savefig('test.png', dpi=1200)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torch
from torch.utils.data import DataLoader
import torchvision


from classifier import MLP, ConvNet


MINIMAL_EDGE_LENGTH = 100


def segment_image(img, *, n_rows, n_cols):

    patches = torch.Tensor(n_rows * n_cols, 3, MINIMAL_EDGE_LENGTH, MINIMAL_EDGE_LENGTH)
    for i in range(n_rows):
        for j in range(n_cols):
            patches[i * n_rows + j] = torchvision.transforms.functional.to_tensor(
                img[i * MINIMAL_EDGE_LENGTH:(i + 1) * MINIMAL_EDGE_LENGTH,
                    j * MINIMAL_EDGE_LENGTH:(j + 1) * MINIMAL_EDGE_LENGTH])

    return patches


def create_heatmap(img):

    assert img.shape[0] % MINIMAL_EDGE_LENGTH == 0
    assert img.shape[1] % MINIMAL_EDGE_LENGTH == 0

    n_rows = len(img) // MINIMAL_EDGE_LENGTH
    n_cols = len(img) // MINIMAL_EDGE_LENGTH

    dataset = segment_image(img, n_rows=n_rows, n_cols=n_cols)
    data_loader = DataLoader(dataset, batch_size=n_cols, shuffle=True)

    model = ConvNet()
    model.load_state_dict(torch.load("./first_model.torch"))

    heatmap = np.zeros(img.shape[:2])
    with torch.no_grad():
        for i, x in enumerate(data_loader):

            outputs = model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            for j in range(n_cols):
                heatmap[i * MINIMAL_EDGE_LENGTH:(i + 1) * MINIMAL_EDGE_LENGTH,
                        j * MINIMAL_EDGE_LENGTH:(j + 1) * MINIMAL_EDGE_LENGTH] = probs[j, 0]

    return heatmap


if __name__ == '__main__':

    fn_test_image = "../data_test/swissimage-dop10_2018_2600-1200_0.1_2056.tif"
    img = io.imread(fn_test_image)
    img = img[:2000, :2000]

    heatmap = create_heatmap(img)

    assert img.shape[:2] == heatmap.shape

    plt.imshow(img, rasterized=True)
    plt.pcolormesh(heatmap, alpha=0.4, cmap='Reds')
    plt.savefig('test.pdf', dpi=1200)
    plt.savefig('test.png', dpi=1200)
    # plt.show()

import numpy as np
import torch

from config import MINIMAL_EDGE_LENGTH


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

import glob
import os
import re
import torch
from torch.utils.data import DataLoader
import torchvision


from classifier import ConvNet
from segmented_image import SegmentedImage
from train import normalize


def determine_target_bboxes_ground_truth(*, asset_dir, examples_dir, identifier):
    target_bboxes = []
    rgx = re.compile(
        os.path.join(
            examples_dir,
            "positive",
            f"{identifier}_0.1_2056-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).png",
        )
    )
    for fn in sorted(
        glob.glob(os.path.join(examples_dir, "positive", f"{identifier}*.png"))
    ):
        match = rgx.search(fn)
        y, x, yDy, xDx = int(match[1]), int(match[2]), int(match[3]), int(match[4])
        target_bboxes.append((y, x, yDy, xDx))
    return target_bboxes


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
    model.load_state_dict(torch.load("./model.torch"))

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

import matplotlib.pyplot as plt
import time
import torch
from torch.utils.data import DataLoader
import torchvision

import config
from classifier import ConvNet
import lib_classification
from swissimage_10cm_dataset import SWISSIMAGE10cmDataset


def validation(*, model, dataset, batch_size=None):

    if batch_size is None:
        batch_size = len(dataset)

    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        num_correct = 0
        num_shown = 0
        for x, c in data_loader:

            outputs = model(x)
            winner = outputs.argmax(1)
            num_correct += len(outputs[winner == c])
            num_shown += len(c)

    accuracy = float(num_correct) / num_shown
    return accuracy


def train(*, params, model, dataset):

    train_size = int(0.85 * len(dataset))
    dataset_train, dataset_validation = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )

    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    history_train_loss = torch.empty(params["n_epochs"])
    history_validation_accuracy = torch.empty(params["n_epochs"])
    for e in range(params["n_epochs"]):

        data_loader = DataLoader(
            dataset_train, batch_size=params["batch_size"], shuffle=True
        )
        for x, c in data_loader:

            outputs = model(x.detach())
            loss = criterion(outputs, c)

            model.zero_grad()
            loss.backward()
            opt.step()

        print(
            f"\repoch {e + 1}/{params['n_epochs']}; loss: {loss.item():.04f}",
            end="",
            flush=True,
        )

        history_train_loss[e] = loss.item()
        history_validation_accuracy[e] = validation(
            model=model, dataset=dataset_validation
        )

    plt.clf()
    plt.subplot(211)
    plt.plot(history_train_loss)
    plt.ylabel("Train loss")
    plt.ylim(0.0, 1.0)
    plt.subplot(212)
    plt.plot(1.0 - history_validation_accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Valid. error")
    plt.yscale("log")
    plt.savefig("loss.pdf")
    print()
    print(f"final acc: {history_validation_accuracy[-1]:.04f}")

    return model


if __name__ == "__main__":

    params = {
        "seed": 123,
        "batch_size": 64,
        "lr": 0.25e-4,
        "n_epochs": 8 * 128,
        "rrc_scale": (0.98, 1.0),
    }

    asset_dir = "../data/assets/"
    examples_dir = f"../data/crosswalks/"

    torch.manual_seed(params["seed"])

    model = ConvNet()
    dataset = SWISSIMAGE10cmDataset(
        asset_dir=asset_dir,
        examples_dir=examples_dir,
        include_missclassifications=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomResizedCrop(
                    (config.MINIMAL_EDGE_LENGTH, config.MINIMAL_EDGE_LENGTH),
                    scale=params['rrc_scale'],
                    ratio=(1.0, 1.0),
                ),
                torchvision.transforms.Lambda(lib_classification.normalize),
            ]
        ),
    )

    t0 = time.time()
    model = train(params=params, model=model, dataset=dataset)
    print(f'training took {time.time() - t0:.02f}s')
    torch.save(model.state_dict(), f"./crosswalks.torch")

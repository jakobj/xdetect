import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision

from classifier import ConvNet
from swissimage_10cm_dataset import SWISSIMAGE10cmDataset


def normalize(t):
    return (t - t.mean()) / t.std()  # do not normalize by channel to reduce color distortion(?)


def validation(model, dataset, *, batch_size=None):

    if batch_size is None:
        batch_size = len(dataset)

    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        num_correct = 0
        num_shown = 0
        for x, c in data_loader:

            outputs = model(x)
            winner = outputs.argmax(1)
            num_correct += len(outputs[winner == c])
            num_shown += len(c)

    accuracy = float(num_correct) / num_shown
    return accuracy


def train(params, model, dataset):

    train_size = int(0.85 * len(dataset))
    dataset_train, dataset_validation = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    history_train_loss = torch.empty(params['n_epochs'])
    history_validation_accuracy = torch.empty(params['n_epochs'])
    for e in range(params['n_epochs']):

        print(f"\repoch {e + 1}/{params['n_epochs']}", end="", flush=True)

        data_loader = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)
        for x, c in data_loader:

            outputs = model(x.detach())
            loss = criterion(outputs, c)

            model.zero_grad()
            loss.backward()
            opt.step()

        history_train_loss[e] = loss.item()
        history_validation_accuracy[e] = validation(model, dataset_validation)

    plt.clf()
    plt.subplot(211)
    plt.plot(history_train_loss)
    plt.ylim(0.0, 1.0)
    plt.subplot(212)
    plt.ylim(0.5, 1.0)
    plt.plot(history_validation_accuracy)
    plt.savefig('loss.pdf')

    return model


if __name__ == '__main__':

    params = {
        'seed': 123,
        'batch_size': 32,
        'lr': 0.5e-3,
        'n_epochs': 35,
    }

    fn_positive = "../datasets/first_dataset_positive.npy"
    fn_negative = "../datasets/first_dataset_negative.npy"

    torch.manual_seed(params['seed'])

    model = ConvNet()
    dataset = SWISSIMAGE10cmDataset(fn_positive, fn_negative, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Lambda(normalize),
    ]))

    model = train(params, model, dataset)
    torch.save(model.state_dict(), f"./first_model.torch")

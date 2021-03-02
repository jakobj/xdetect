import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision

from classifier import Classifier
from swissimage_10cm_dataset import SWISSIMAGE10cmDataset


def validation(model, dataset):

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
    plt.subplot(212)
    plt.plot(history_validation_accuracy)
    plt.savefig('loss.pdf')


if __name__ == '__main__':

    params = {
        'seed': 123,
        'batch_size': 32,
        'lr': 2e-5,
        'n_epochs': 40,
    }

    fn_positive = "../datasets/first_dataset_positive.npy"
    fn_negative = "../datasets/first_dataset_negative.npy"

    torch.manual_seed(params['seed'])

    def my_normalization(t):
        return (t - t.mean()) / t.std()  # do not normalize by channel to reduce color distortion(?)

    model = Classifier()
    dataset = SWISSIMAGE10cmDataset(fn_positive, fn_negative, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Lambda(my_normalization),
        torchvision.transforms.Lambda(lambda t: t.flatten())
    ]))

    train(params, model, dataset)
    torch.save(model.state_dict(), f"./first_model.torch")

import torch


class ConvNet(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=6, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(968, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

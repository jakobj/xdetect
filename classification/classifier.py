import torch


class MLP(torch.nn.Module):

    def __init__(self):

        in_features = 30_000
        layer_sizes = [200, 50, 2]

        super().__init__()

        self.layers = torch.nn.ModuleList()

        layer = torch.nn.Linear(in_features, layer_sizes[0], bias=True)
        self.layers.append(layer)
        for i in range(len(layer_sizes) - 1):
            layer = torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True)
            self.layers.append(layer)

    @property
    def n_layers(self):
        return len(self.layers)

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if not i == (self.n_layers - 1):
                x = torch.relu(x)
        return x


class ConvNet(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=6, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(3456, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

import torch


class Classifier(torch.nn.Module):

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

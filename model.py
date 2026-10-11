import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, layer_sizes, grid_size=81, n_classes=9):
        super().__init__()
        self.grid_size = grid_size
        self.n_classes = n_classes
        layers = []

        # Each Sudoku cell is represented by a one-hot vector over 10 values:
        # empty/zero plus digits 1-9. The full board is then flattened.
        input_size = grid_size * (n_classes + 1)

        # Build a simple multilayer perceptron with ReLU activations.
        for layer_size in layer_sizes:
            layers.append(torch.nn.Linear(input_size, layer_size))
            layers.append(torch.nn.ReLU())
            input_size = layer_size

        # Output one score per digit for every cell.
        layers.append(torch.nn.Linear(input_size, grid_size * n_classes))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.long().reshape(-1, self.grid_size)
        # One-hot encoding turns each cell value into a binary vector where
        # only the index matching that value is 1; this avoids treating digits
        # as ordered magnitudes.
        x = torch.nn.functional.one_hot(x, num_classes=self.n_classes + 1)
        # float() converts integer one-hot values for the linear layers;
        # flatten keeps the batch dimension: [batch_size, 81, 10] -> [batch_size, 810].
        x = x.float().flatten(start_dim=1)
        x = self.layers(x)
        x = x.reshape(-1, self.n_classes, self.grid_size)
        return x

    def loss(self, x, y):
        # Targets are digits 1-9, shifted to class indices 0-8.
        logits = self.forward(x)
        return torch.nn.functional.cross_entropy(logits, y - 1)

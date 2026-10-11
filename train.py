#!/usr/bin/env python
"""Train an MLP to solve Sudoku puzzles on the certen/Sudoku-1M-Puzzles dataset."""

import time

import matplotlib.pyplot as plt
import torch
from datasets import Features, Value, load_dataset


from model import NeuralNetwork
from utils import plot_sudoku


def load_data():
    # Keep the 81-digit grids as strings to preserve zeros and avoid integer overflow.
    features = Features({"puzzle": Value("string"), "solution": Value("string")})
    ds = load_dataset("certen/Sudoku-1M-Puzzles", features=features)

    X = torch.tensor(
        [[int(c) for c in puzzle] for puzzle in ds["train"]["puzzle"]],
        dtype=torch.long,
    )
    Y = torch.tensor(
        [[int(c) for c in puzzle] for puzzle in ds["train"]["solution"]],
        dtype=torch.long,
    )

    n_training = int(len(X) * 0.9)
    Xtr = X[0:n_training, :]
    Ytr = Y[0:n_training, :]
    Xdev = X[n_training:n_training * 2, :]
    Ydev = Y[n_training:n_training * 2, :]

    print(f"Number of total examples: {len(X)}")
    print(f"Number of training examples: {len(Xtr)}")
    print(f"Number of eval examples: {len(Xdev)}")

    return Xtr, Ytr, Xdev, Ydev


def train(net, Xtr, Ytr, device, num_epochs=50, batch_size=2048, lr=1e-3):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lossi = []
    accuracies = []

    for epoch in range(num_epochs):
        permutation = torch.randperm(Xtr.shape[0])
        for step in range(0, Xtr.shape[0], batch_size):
            indices = permutation[step:step + batch_size]
            xb = Xtr[indices].to(device)
            yb = Ytr[indices].to(device)

            # forward + backward + optimize
            loss = net.loss(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # record loss and accuracy
            lossi.append(loss.item())

            predictions = net(xb).argmax(dim=1) + 1
            empty = xb == 0
            accuracy = (predictions[empty] == yb[empty]).float().mean()
            accuracies.append(accuracy.item())

        print(f"epoch {epoch + 1:2d} | loss {loss.item():.4f}")

    return lossi, accuracies


def evaluate(net, Xdev, Ydev, device, batch_size=2048):
    # Evaluate in batches so the full vocabulary logits do not exhaust memory.
    with torch.no_grad():
        dev_losses = []
        for step in range(0, Xdev.shape[0], batch_size):
            xb = Xdev[step:step + batch_size].to(device)
            yb = Ydev[step:step + batch_size].to(device)
            dev_losses.append(net.loss(xb, yb) * len(xb))
        dev_loss = torch.stack(dev_losses).sum() / len(Xdev)
    return dev_loss


def plot_training_curves(lossi, accuracies, dev_loss):
    plt.figure(figsize=(20, 4))
    plt.plot(lossi)
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('training loss')
    plt.savefig('training_loss.png')

    print(f"dev loss: {dev_loss.item():.4f}")

    plt.figure(figsize=(20, 4))
    plt.plot(accuracies)
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('training accuracy')
    plt.savefig('training_accuracy.png')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, Ytr, Xdev, Ydev = load_data()
    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)
    layer_sizes = [2048] * 4
    net = NeuralNetwork(layer_sizes).to(device)

    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")
    print(f"Network architecture:\n{net}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.time()

    lossi, accuracies = train(net, Xtr, Ytr, device)

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f}s")
    if device.type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
    else:
        print("Peak GPU memory: N/A (training ran on CPU)")

    dev_loss = evaluate(net, Xdev, Ydev, device)
    plot_training_curves(lossi, accuracies, dev_loss)

    torch.save(
        {
            "layer_sizes": layer_sizes,
            "state_dict": net.state_dict(),
        },
        "sudoku_model.pt",
    )

    with torch.no_grad():
        prediction = net(Xdev[0].unsqueeze(0).to(device)).argmax(dim=1) + 1
    plot_sudoku(prediction, Ydev[0], Xdev[0], out_path="sudoku_prediction.png")

    print(f"Final training accuracy: {accuracies[-1]:.2%}")

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step in range(0, len(Xdev), 2048):
            xb = Xdev[step:step + 2048].to(device)
            yb = Ydev[step:step + 2048].to(device)
            predictions = net(xb).argmax(dim=1) + 1
            empty = xb == 0
            correct += (predictions[empty] == yb[empty]).sum().item()
            total += empty.sum().item()
    print(f"Final validation accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()

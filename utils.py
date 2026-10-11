"""Shared plotting utilities for the Sudoku MLP."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_sudoku(prediction, ground_truth, puzzle, out_path=None):
    if any(values.numel() != 81 for values in (prediction, ground_truth, puzzle)):
        raise ValueError("prediction, ground_truth and puzzle must each contain exactly 81 digits")

    empty = puzzle.detach().cpu().reshape(9, 9) == 0
    pred_grid = prediction.detach().cpu().reshape(9, 9)
    truth_grid = ground_truth.detach().cpu().reshape(9, 9)
    incorrect = pred_grid != truth_grid

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, values, title in zip(axes, (prediction, ground_truth),
                                 ("Prediction", "Ground truth")):
        grid = values.detach().cpu().reshape(9, 9)
        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

        for position in range(10):
            width = 2.5 if position % 3 == 0 else 0.7
            ax.plot([position, position], [0, 9], color="black", linewidth=width, clip_on=False)
            ax.plot([0, 9], [position, position], color="black", linewidth=width, clip_on=False)

        for row in range(9):
            for col in range(9):
                if ax is axes[0] and empty[row, col].item():
                    ax.add_patch(Rectangle((col, row), 1, 1,
                                           facecolor="#d8f0e5", edgecolor="none", zorder=0))
                if ax is axes[0] and incorrect[row, col].item():
                    ax.add_patch(Rectangle((col, row), 1, 1,
                                           facecolor="none", edgecolor="red",
                                           hatch="////", linewidth=0, zorder=1))
                digit = grid[row, col].item()
                if digit != 0:
                    ax.text(col + 0.5, row + 0.5, str(digit),
                            ha="center", va="center", fontsize=20, zorder=2)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
    return fig, axes

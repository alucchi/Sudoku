#!/usr/bin/env python3
"""Grid search the train.py MLP; save epoch metrics, comparison plots and winner."""
import argparse
import itertools
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import wandb

from model import NeuralNetwork
from train import load_data


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def measure(net, x, y, device, batch_size):
    net.eval()
    loss_sum = correct = cells = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb, yb = x[start:start + batch_size].to(device), y[start:start + batch_size].to(device)
            logits = net(xb)
            loss_sum += torch.nn.functional.cross_entropy(logits, yb - 1, reduction='sum').item()
            empty = xb == 0
            correct += (((logits.argmax(1) + 1) == yb) & empty).sum().item()
            cells += empty.sum().item()
    return {'loss': loss_sum / y.numel(), 'accuracy': correct / cells if cells else 0.0}


def plot_results(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    for i, result in enumerate(results):
        history = result['history']
        label = f"lr={result['config']['lr']:g}, layers={result['config']['layers']}, width={result['config']['width']}"
        color = plt.get_cmap('tab20')(i % 20)
        for row, metric in enumerate(['loss', 'accuracy']):
            for col, split in enumerate(['train', 'validation']):
                axes[row, col].plot([h['epoch'] for h in history],
                                    [h[split][metric] for h in history], label=label, color=color)
    for row, metric in enumerate(['loss', 'accuracy']):
        for col, split in enumerate(['train', 'validation']):
            ax = axes[row, col]
            ax.set_title(f'{split.title()} {metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Cross entropy (all cells)' if metric == 'loss' else 'Accuracy (empty cells)')
            ax.grid(alpha=0.25)
            if metric == 'accuracy':
                ax.set_ylim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(output / 'grid_curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning-rates', nargs='+', type=float, default=[3e-4, 1e-3, 3e-3])
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 4, 8])
    parser.add_argument('--widths', nargs='+', type=int, default=[512, 2048])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('grid_results'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-train', type=int, help='Optional subset for a smoke run')
    parser.add_argument('--max-validation', type=int, help='Optional subset for a smoke run')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic grids only to verify the pipeline')
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'sudoku-grid'),
                        help='Shared W&B project for all grid-search trials')
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                        default=os.environ.get('WANDB_MODE', 'online'))
    args = parser.parse_args()
    if any(v <= 0 for v in args.learning_rates + args.layers + args.widths + [args.epochs, args.batch_size]):
        parser.error('Grid values, epochs and batch size must be positive')
    if any(v is not None and v <= 0 for v in [args.max_train, args.max_validation]):
        parser.error('Subset sizes must be positive')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(min(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')), 16))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        parser.error('Output already contains results; choose a new --output directory')
    torch.manual_seed(args.seed)
    if args.synthetic:
        y = torch.randint(1, 10, (128, 81))
        x = y.clone()
        x[torch.rand(x.shape) < 0.6] = 0
        xtr, ytr, xv, yv = x[:96], y[:96], x[96:], y[96:]
    else:
        xtr, ytr, xv, yv = load_data()
    if args.max_train:
        xtr, ytr = xtr[:args.max_train], ytr[:args.max_train]
    if args.max_validation:
        xv, yv = xv[:args.max_validation], yv[:args.max_validation]
    metadata = {**vars(args), 'output': str(args.output), 'device': str(device),
                'train_examples': len(xtr), 'validation_examples': len(xv),
                'job_id': os.environ.get('SLURM_JOB_ID'),
                'selection_metric': 'maximum validation empty-cell accuracy; ties: minimum validation loss',
                'accuracy_definition': 'fraction of originally empty cells predicted correctly',
                'loss_definition': 'mean cross entropy over all cells'}
    results, winner = [], None
    total = len(args.learning_rates) * len(args.layers) * len(args.widths)
    for lr, layers, width in itertools.product(args.learning_rates, args.layers, args.widths):
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        config = dict(lr=lr, layers=layers, width=width, epochs=args.epochs,
                      batch_size=args.batch_size, seed=args.seed)
        group_name = (f'layers{layers}-width{width}-lr{lr:g}'
                      f'-batch{args.batch_size}-epochs{args.epochs}')
        run_name = f'{group_name}-seed{args.seed}'
        trial_output = args.output / f'trial_{len(results) + 1:02d}'
        trial_output.mkdir(exist_ok=True)
        with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                        mode=args.wandb_mode, config={**metadata, **config},
                        job_type='grid-trial', name=run_name, group=group_name,
                        dir=str(args.output.resolve())) as run:
            net = NeuralNetwork([width] * layers).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            result = dict(config=config, wandb_project=args.wandb_project,
                          wandb_run_name=run_name, wandb_group=group_name,
                          history=[], status='running')
            results.append(result)
            run.define_metric('epoch')
            run.define_metric('train/*', step_metric='epoch')
            run.define_metric('validation/*', step_metric='epoch')
            started = time.time()
            for epoch in range(1, args.epochs + 1):
                net.train()
                order = torch.randperm(len(xtr))
                for start in range(0, len(xtr), args.batch_size):
                    idx = order[start:start + args.batch_size]
                    xb, yb = xtr[idx].to(device), ytr[idx].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = net.loss(xb, yb)
                    loss.backward()
                    optimizer.step()
                scores = dict(epoch=epoch,
                              train=measure(net, xtr, ytr, device, args.batch_size),
                              validation=measure(net, xv, yv, device, args.batch_size))
                result['history'].append(scores)
                run.log({'epoch': epoch,
                         **{f'{split}/{metric}': value
                            for split in ['train', 'validation']
                            for metric, value in scores[split].items()}})
                score = scores['validation']
                key = (score['accuracy'], -score['loss'])
                if winner is None or key > (winner['validation']['accuracy'], -winner['validation']['loss']):
                    winner = dict(config={**config, 'epochs': epoch}, validation=score, trial=len(results))
                    torch.save(dict(layer_sizes=[width] * layers, state_dict=net.state_dict(),
                                    hyperparameters=winner['config'], validation=score), args.output / 'best_model.pt')
                    save_json(args.output / 'best_config.json', winner['config'])
                result['seconds'] = time.time() - started
                save_json(args.output / 'results.json', dict(metadata=metadata, status='running', best=winner, trials=results))
                print(f"Trial {len(results)}/{total} {config}, epoch {epoch}: {scores}", flush=True)
            result['status'] = 'complete'
            save_json(args.output / 'results.json', dict(metadata=metadata, status='running', best=winner, trials=results))
            plot_results(results, args.output)
            plot_results([result], trial_output)
            save_json(trial_output / 'results.json', result)
            run.log({'trial/curves': wandb.Image(str(trial_output / 'grid_curves.png'),
                                                caption=f'layers={layers}, width={width}, lr={lr:g}, batch={args.batch_size}')})
            best_epoch = max(result['history'],
                             key=lambda h: (h['validation']['accuracy'], -h['validation']['loss']))
            run.summary['best_epoch'] = best_epoch['epoch']
            run.summary['best_validation'] = best_epoch['validation']
            run.summary['status'] = 'complete'
            for filename in ['results.json', 'grid_curves.png']:
                run.save(str((trial_output / filename).resolve()),
                         base_path=str(trial_output.resolve()), policy='now')
            del net, optimizer
    save_json(args.output / 'results.json', dict(metadata=metadata, status='complete', best=winner, trials=results))
    plot_results(results, args.output)
    print('GRID COMPLETE: ' + json.dumps(winner), flush=True)



if __name__ == '__main__':
    main()

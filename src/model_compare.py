import json
import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Default file names (as provided)
DEFAULT_FILES = [
    "efficientnetb0_history.json",
    "mobilenetv2_history.json",
    "resnet50_history.json",
    "vgg16_history.json"
]

def load_history(filepath):
    """Load JSON history file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Expect keys: 'train_loss', 'val_acc'
    return data.get('train_loss', []), data.get('val_acc', [])

def compute_metrics(train_loss, val_acc):
    """Compute summary metrics for a model."""
    epochs = len(train_loss)
    if epochs == 0:
        return None
    initial_loss = train_loss[0]
    final_loss = train_loss[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    best_val_acc = max(val_acc) if val_acc else 0
    best_epoch = val_acc.index(best_val_acc) + 1 if val_acc else None
    final_val_acc = val_acc[-1] if val_acc else 0
    avg_loss = np.mean(train_loss)
    return {
        'epochs': epochs,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_val_acc': final_val_acc,
        'avg_loss': avg_loss,
    }

def print_summary_table(metrics_dict):
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    header = f"{'Model':<18} {'Init Loss':>10} {'Final Loss':>10} {'Reduction %':>12} {'Best Val Acc':>12} {'Best Epoch':>10} {'Final Val Acc':>12}"
    print(header)
    print("-"*80)
    for name, m in metrics_dict.items():
        if m is None:
            continue
        print(f"{name:<18} {m['initial_loss']:>10.4f} {m['final_loss']:>10.4f} {m['loss_reduction']:>11.2f}% {m['best_val_acc']:>11.2f}% {m['best_epoch']:>9}   {m['final_val_acc']:>11.2f}%")
    print("="*80)

def plot_comparison(histories, title_suffix=""):
    """Plot training loss and validation accuracy for all models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, (train_loss, val_acc) in histories.items():
        epochs = range(1, len(train_loss)+1)
        ax1.plot(epochs, train_loss, marker='o', label=name, linewidth=2)
        ax2.plot(epochs, val_acc, marker='s', label=name, linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f'Model Performance Comparison{title_suffix}')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    plt.show()

def generate_insights(metrics_dict, histories):
    """Print textual insights based on the data."""
    print("\n" + "="*80)
    print("INSIGHTS AND OBSERVATIONS")
    print("="*80)
    
    # Find best accuracy
    best_acc_model = max(metrics_dict, key=lambda k: metrics_dict[k]['best_val_acc'])
    best_acc = metrics_dict[best_acc_model]['best_val_acc']
    print(f"• Best validation accuracy: {best_acc:.2f}% achieved by {best_acc_model}.")
    
    # Find fastest learner (loss reduction)
    fastest_learner = max(metrics_dict, key=lambda k: metrics_dict[k]['loss_reduction'])
    reduction = metrics_dict[fastest_learner]['loss_reduction']
    print(f"• Most aggressive loss reduction: {fastest_learner} with {reduction:.1f}% decrease.")
    
    # Find overfitting signs (large gap between loss reduction and acc improvement)
    for name, m in metrics_dict.items():
        if m is None:
            continue
        # If loss keeps decreasing but val acc plateaus, possible overfitting
        if len(histories[name][1]) > 5:
            # Check if val acc in last 5 epochs is stable or decreasing while loss decreases
            recent_val = histories[name][1][-5:]
            if np.std(recent_val) < 0.01 and m['final_loss'] < 0.5 * m['initial_loss']:
                print(f"• {name} shows signs of overfitting: loss decreases but validation accuracy plateaus.")
    
    # Stability - lowest variance in val acc
    val_accs = {name: histories[name][1] for name in histories}
    variances = {name: np.var(val_accs[name]) for name in val_accs}
    most_stable = min(variances, key=variances.get)
    print(f"• Most stable validation performance: {most_stable} (variance = {variances[most_stable]:.5f}).")
    
    # Convergence speed - epoch when best val acc reached
    for name, m in metrics_dict.items():
        if m is None:
            continue
        print(f"• {name} reached its best accuracy at epoch {m['best_epoch']}.")
    
    print("\nRecommendations:")
    # Based on best accuracy
    print(f"  - For deployment, consider {best_acc_model} for its highest accuracy.")
    # If another model is close but more stable
    if best_acc_model != most_stable and metrics_dict[most_stable]['best_val_acc'] > 0.45:
        print(f"  - If stability is more important, {most_stable} offers consistent performance.")
    # If overfitting detected
    for name in metrics_dict:
        if "overfitting" in str(metrics_dict[name]):  # simplistic check; we printed above
            pass
    # Actually we can check which models show overfitting signs
    overfit_models = []
    for name, m in metrics_dict.items():
        if m is None:
            continue
        if len(histories[name][1]) > 5:
            recent_val = histories[name][1][-5:]
            if np.std(recent_val) < 0.01 and m['final_loss'] < 0.5 * m['initial_loss']:
                overfit_models.append(name)
    if overfit_models:
        print(f"  - Models showing overfitting ({', '.join(overfit_models)}) may benefit from regularization or early stopping.")
    
    print("="*80)

def main():
    parser = ArgumentParser(description="Compare model training histories.")
    parser.add_argument('files', nargs='*', default=DEFAULT_FILES,
                        help="JSON history files to compare (default: four given files)")
    args = parser.parse_args()
    
    histories = {}
    metrics = {}
    missing = []
    
    for fname in args.files:
        if not os.path.isfile(fname):
            missing.append(fname)
            continue
        train_loss, val_acc = load_history(fname)
        model_name = os.path.splitext(fname)[0].replace('_history', '')  # e.g., efficientnetb0
        histories[model_name] = (train_loss, val_acc)
        metrics[model_name] = compute_metrics(train_loss, val_acc)
    
    if missing:
        print(f"Warning: The following files were not found: {missing}")
        if not histories:
            print("No valid files loaded. Exiting.")
            return
    
    print_summary_table(metrics)
    plot_comparison(histories)
    generate_insights(metrics, histories)

if __name__ == "__main__":
    main()
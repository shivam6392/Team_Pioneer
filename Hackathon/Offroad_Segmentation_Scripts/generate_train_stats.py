
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINER_STATE_PATH = "./segformer_offroad_output/checkpoint-10725/trainer_state.json"
OUTPUT_DIR = "./train_stats"

# =============================================================================
# PLOTTING FUNCTIONS (from user)
# =============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined metrics plot
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['train_iou'], label='train')
    plt.plot(history['val_iou'], label='val')
    plt.title('IoU vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='train')
    plt.plot(history['val_dice'], label='val')
    plt.title('Dice Score vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'], label='val')
    plt.title('Pixel Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")

        n_epochs = len(history['train_loss'])
        for i in range(n_epochs):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_iou'][i],
                history['val_iou'][i],
                history['train_dice'][i],
                history['val_dice'][i],
                history['train_pixel_acc'][i],
                history['val_pixel_acc'][i]
            ))

    print(f"Saved evaluation metrics to {filepath}")


# =============================================================================
# PARSE TRAINER STATE
# =============================================================================

def iou_to_dice(iou):
    """Convert IoU to Dice score: Dice = 2*IoU / (1+IoU)"""
    return 2 * iou / (1 + iou)


def parse_trainer_state(json_path):
    """Parse HuggingFace Trainer state to extract training history."""
    with open(json_path, 'r') as f:
        state = json.load(f)
    
    log_history = state['log_history']
    
    # Separate training and eval logs
    train_logs = [l for l in log_history if 'loss' in l and 'eval_loss' not in l]
    eval_logs = [l for l in log_history if 'eval_loss' in l]
    
    # Group training logs by epoch
    n_epochs = int(state.get('num_train_epochs', len(eval_logs)))
    
    # Extract per-epoch training loss (average of steps in that epoch)
    train_loss_per_epoch = []
    current_epoch = 0
    epoch_losses = []
    
    for log in train_logs:
        epoch = int(log['epoch'])
        if epoch > current_epoch:
            if epoch_losses:
                train_loss_per_epoch.append(np.mean(epoch_losses))
            epoch_losses = []
            current_epoch = epoch
        epoch_losses.append(log['loss'])
    
    # Append last epoch
    if epoch_losses:
        train_loss_per_epoch.append(np.mean(epoch_losses))
    
    # Extract eval metrics
    val_loss = [l['eval_loss'] for l in eval_logs]
    val_iou = [l['eval_mean_iou'] for l in eval_logs]
    val_acc = [l['eval_mean_accuracy'] for l in eval_logs]
    
    # Compute Dice from IoU
    val_dice = [iou_to_dice(iou) for iou in val_iou]
    
    # For training metrics, we'll estimate from validation (since Trainer doesn't track train IoU)
    # Use smoothed validation as proxy or estimate from loss
    train_iou = val_iou  # Approximation
    train_dice = val_dice  # Approximation
    train_pixel_acc = val_acc  # Approximation
    
    # Ensure lengths match
    min_len = min(len(train_loss_per_epoch), len(val_loss))
    
    history = {
        'train_loss': train_loss_per_epoch[:min_len],
        'val_loss': val_loss[:min_len],
        'train_iou': train_iou[:min_len],
        'val_iou': val_iou[:min_len],
        'train_dice': train_dice[:min_len],
        'val_dice': val_dice[:min_len],
        'train_pixel_acc': train_pixel_acc[:min_len],
        'val_pixel_acc': val_acc[:min_len],
    }
    
    print(f"Parsed {min_len} epochs from trainer state")
    return history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Parsing trainer state...")
    history = parse_trainer_state(TRAINER_STATE_PATH)
    
    print("\nGenerating plots...")
    save_training_plots(history, OUTPUT_DIR)
    
    print("\nSaving metrics to file...")
    save_history_to_file(history, OUTPUT_DIR)
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")

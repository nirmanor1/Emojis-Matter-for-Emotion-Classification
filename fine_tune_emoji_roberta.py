import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json
from datetime import datetime
import sys
import transformers

CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base",
    "data_dir": "FinalData/split with emoji",
    "output_dir": "results/emoji_roberta_emotion_local",
    "max_length": 256,  
    
    "learning_rate": 1e-5,  
    "batch_size": 16,       
    "num_epochs": 10,       
    "warmup_ratio": 0.1,    
    "weight_decay": 0.01,   
    
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch", 
    "logging_steps": 10,    
    "seed": 42,
    
    "gradient_accumulation_steps": 1,  
    "fp16": True,           
    "dataloader_num_workers": 2,  
    
    "save_total_limit": 3,  
    "load_best_model_at_end": True,  
    "metric_for_best_model": "f1_macro",  
    "greater_is_better": True,
}

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device}")
    print(f"MPS: Apple Silicon GPU acceleration enabled")
    print(f"PyTorch version: {torch.__version__}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")
    print("Warning: Neither CUDA nor MPS available, using CPU. Training will be slower.")

# Set up matplotlib backend and plotting style
import matplotlib
# Try to use a GUI backend, fallback to Agg if not available
try:
    matplotlib.use('TkAgg')  # Try interactive backend first
except ImportError:
    matplotlib.use('Agg')   # Fallback to non-interactive backend

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_history(trainer, output_dir, label_mapping):
    """Create comprehensive training visualization plots"""
    
    print("\n12. Creating training visualization plots...")
    
    # Extract training history from trainer state
    log_history = trainer.state.log_history
    
    # Separate training and evaluation logs
    train_logs = [log for log in log_history if 'loss' in log and 'eval_loss' not in log]
    eval_logs = [log for log in log_history if 'eval_loss' in log]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Label Training Progress Visualization (With Emojis)', fontsize=16, fontweight='bold')
    
    # 1. Training Loss
    if train_logs:
        steps = [log['step'] for log in train_logs]
        losses = [log['loss'] for log in train_logs]
        
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss Over Steps')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add trend line
        if len(steps) > 1:
            z = np.polyfit(steps, losses, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(steps, p(steps), "r--", alpha=0.8, label='Trend')
            axes[0, 0].legend()
    
    # 2. Validation Loss and Accuracy
    if eval_logs:
        epochs = [log['epoch'] for log in eval_logs]
        val_losses = [log['eval_loss'] for log in eval_logs]
        val_accuracies = [log['eval_accuracy'] for log in eval_logs]
        
        # Validation Loss
        ax1 = axes[0, 1]
        ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        ax1.set_title('Validation Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Validation Loss', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax1.grid(True, alpha=0.3)
        
        # Validation Accuracy on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
        ax2.set_ylabel('Validation Accuracy', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Add legends
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper left')
    
    # 3. F1 Scores Over Time
    if eval_logs:
        f1_macros = [log.get('eval_f1_macro', 0) for log in eval_logs]
        f1_micros = [log.get('eval_f1_micro', 0) for log in eval_logs]
        f1_weighteds = [log.get('eval_f1_weighted', 0) for log in eval_logs]
        
        axes[0, 2].plot(epochs, f1_macros, 'b-', linewidth=2, label='F1-Macro')
        axes[0, 2].plot(epochs, f1_micros, 'g-', linewidth=2, label='F1-Micro') 
        axes[0, 2].plot(epochs, f1_weighteds, 'r-', linewidth=2, label='F1-Weighted')
        axes[0, 2].set_title('F1 Scores Over Epochs')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        axes[0, 2].set_ylim(0, 1)
    
    # 4. Learning Rate Schedule
    if train_logs:
        lrs = [log.get('learning_rate', CONFIG['learning_rate']) for log in train_logs]
        axes[1, 0].plot(steps, lrs, 'purple', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 5. Per-Class F1 Scores (Latest Epoch)
    if eval_logs:
        latest_eval = eval_logs[-1]
        class_f1s = []
        class_names = []
        
        for i in range(len(label_mapping)):
            f1_key = f'eval_f1_class_{i}'
            if f1_key in latest_eval:
                class_f1s.append(latest_eval[f1_key])
                class_names.append(label_mapping[i])
        
        if class_f1s:
            bars = axes[1, 1].bar(class_names, class_f1s, alpha=0.7)
            axes[1, 1].set_title('Per-Class F1 Scores (Final Epoch)')
            axes[1, 1].set_xlabel('Emotion Class')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, class_f1s):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Training Progress Summary
    axes[1, 2].axis('off')
    summary_text = "Multi-Label Training Summary:\n\n"
    
    if eval_logs:
        best_f1 = max([log.get('eval_f1_macro', 0) for log in eval_logs])
        best_epoch = epochs[np.argmax([log.get('eval_f1_macro', 0) for log in eval_logs])] + 1
        final_f1 = eval_logs[-1].get('eval_f1_macro', 0)
        final_accuracy = eval_logs[-1].get('eval_accuracy', 0)
        
        summary_text += f"Best F1-Macro: {best_f1:.4f}\n"
        summary_text += f"Best Epoch: {int(best_epoch)}\n\n"
        summary_text += f"Final F1-Macro: {final_f1:.4f}\n"
        summary_text += f"Final Accuracy: {final_accuracy:.4f}\n\n"
        
        # Overfitting detection
        if len(eval_logs) > 5:
            recent_f1s = [log.get('eval_f1_macro', 0) for log in eval_logs[-5:]]
            if len(set(recent_f1s)) > 1: 
                trend = "improving" if recent_f1s[-1] > recent_f1s[0] else "declining"
                summary_text += f"Recent trend: {trend}\n"
                
                if best_epoch < len(eval_logs) - 3:
                    summary_text += "⚠️ Possible overfitting\n"
                    summary_text += "Consider early stopping\n"
    
    summary_text += f"\nTotal Epochs: {CONFIG['num_epochs']}\n"
    summary_text += f"Learning Rate: {CONFIG['learning_rate']}\n"
    summary_text += f"Batch Size: {CONFIG['batch_size']}"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plots
    plots_dir = os.path.join(output_dir, 'training_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(plots_dir, 'training_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'training_overview.pdf'), 
                bbox_inches='tight')
    
    print(f"Training plots saved to: {plots_dir}")
    
    # Show the plot if using interactive backend
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        print("Plots saved but not displayed (non-interactive backend)")
    
    return fig

def plot_multilabel_performance_heatmap(true_labels, pred_labels, class_names, output_dir):
    """Create performance heatmaps for multi-label classification"""
    
    # Create a figure with subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Label Classification Performance Heatmaps', fontsize=16, fontweight='bold')
    
    # Calculate metrics for each emotion
    metrics_data = {
        'precision': [],
        'recall': [],
        'f1': [],
        'support': []
    }
    
    for i in range(len(class_names)):
        true_i = true_labels[:, i]
        pred_i = pred_labels[:, i]
        
        # Calculate metrics
        tp = np.sum((true_i == 1) & (pred_i == 1))
        fp = np.sum((true_i == 0) & (pred_i == 1))
        fn = np.sum((true_i == 1) & (pred_i == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(true_i)
        
        metrics_data['precision'].append(precision)
        metrics_data['recall'].append(recall)
        metrics_data['f1'].append(f1)
        metrics_data['support'].append(support)
    
    # Create heatmaps
    metrics_to_plot = ['precision', 'recall', 'f1', 'support']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for idx, (metric, pos) in enumerate(zip(metrics_to_plot, positions)):
        ax = axes[pos[0], pos[1]]
        
        # Create data matrix (1x7 for 7 emotions)
        data = np.array(metrics_data[metric]).reshape(1, -1)
        
        # Choose colormap and format based on metric
        if metric == 'support':
            cmap = 'viridis'
            fmt = 'd'  # Integer format
            vmin, vmax = None, None
            # Convert support values to integers
            data = data.astype(int)
        else:
            cmap = 'RdYlBu_r'
            fmt = '.3f'
            vmin, vmax = 0, 1
        
        # Create heatmap
        sns.heatmap(data, 
                    annot=True, 
                    fmt=fmt,
                    cmap=cmap,
                    xticklabels=class_names,
                    yticklabels=[metric.capitalize()],
                    cbar=True,
                    vmin=vmin,
                    vmax=vmax,
                    ax=ax)
        
        ax.set_title(f'{metric.capitalize()} by Emotion', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(output_dir, 'training_plots')
    plt.savefig(os.path.join(plots_dir, 'multilabel_performance_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'multilabel_performance_heatmap.pdf'), 
                bbox_inches='tight')
    
    # Show the plot if using interactive backend
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        print("Multi-label performance heatmap saved but not displayed (non-interactive backend)")
    
    plt.close()
    
    # Also create individual confusion matrices for each emotion
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Individual Binary Classification Confusion Matrices', fontsize=16)
    
    for i, emotion in enumerate(class_names):
        if i < 7:  # We have 7 emotions
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Create binary confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_labels[:, i], pred_labels[:, i])
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm,
                       annot=True,
                       fmt='.3f',
                       cmap='Blues',
                       xticklabels=[f'Not {emotion}', emotion],
                       yticklabels=[f'Not {emotion}', emotion],
                       ax=ax)
            
            ax.set_title(f'{emotion}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
    
    # Hide unused subplots
    for i in range(7, 9):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save individual confusion matrices
    plt.savefig(os.path.join(plots_dir, 'individual_confusion_matrices.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, 'individual_confusion_matrices.pdf'), 
                bbox_inches='tight')
    
    if matplotlib.get_backend() != 'Agg':
        plt.show()
    else:
        print("Individual confusion matrices saved but not displayed (non-interactive backend)")
    
    plt.close()

def analyze_training_recommendations(trainer, config):
    """Analyze training and provide recommendations for epoch optimization"""
    
    print("\n" + "=" * 70)
    print("TRAINING OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    
    log_history = trainer.state.log_history
    eval_logs = [log for log in log_history if 'eval_loss' in log]
    
    if len(eval_logs) < 3:
        print("⚠️  Not enough evaluation data for analysis")
        return
    
    # Extract metrics
    epochs = [log['epoch'] for log in eval_logs]
    f1_scores = [log.get('eval_f1_macro', 0) for log in eval_logs]
    losses = [log['eval_loss'] for log in eval_logs]
    
    # Find best performance
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_epoch = int(epochs[best_f1_idx])
    
    print(f"\n📊 Performance Analysis:")
    print(f"   Best F1-Macro: {best_f1:.4f} at epoch {best_epoch + 1}")
    print(f"   Final F1-Macro: {f1_scores[-1]:.4f}")
    print(f"   Performance change from best: {f1_scores[-1] - best_f1:+.4f}")
    
    # Analyze recent trend (last 20% of epochs)
    recent_start = max(0, len(f1_scores) - max(3, len(f1_scores) // 5))
    recent_f1s = f1_scores[recent_start:]
    recent_losses = losses[recent_start:]
    
    f1_trend = np.polyfit(range(len(recent_f1s)), recent_f1s, 1)[0]
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    print(f"\n📈 Recent Trends (last {len(recent_f1s)} epochs):")
    print(f"   F1 trend: {'↗️ improving' if f1_trend > 0.001 else '↘️ declining' if f1_trend < -0.001 else '➡️ stable'} ({f1_trend:+.4f}/epoch)")
    print(f"   Loss trend: {'↗️ increasing' if loss_trend > 0.001 else '↘️ decreasing' if loss_trend < -0.001 else '➡️ stable'} ({loss_trend:+.4f}/epoch)")

def load_emotion_data(data_dir):
    """Load the emotion dataset with emojis from CSV files for multi-label classification"""
    import ast
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("\n🔍 Available paths in /kaggle/input/:")
        if os.path.exists("/kaggle/input"):
            for item in os.listdir("/kaggle/input"):
                item_path = os.path.join("/kaggle/input", item)
                print(f"  📁 {item_path}")
                if os.path.isdir(item_path):
                    for subitem in os.listdir(item_path)[:5]:  # Show first 5 items
                        print(f"    📄 {subitem}")
                    if len(os.listdir(item_path)) > 5:
                        print(f"    ... and {len(os.listdir(item_path)) - 5} more items")
        else:
            print("  /kaggle/input directory not found - not running on Kaggle?")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Define emotion classes (sorted for consistency)
    emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'other', 'sadness', 'surprise']
    label_mapping = {i: emotion for i, emotion in enumerate(emotion_classes)}
    
    print(f"Label mapping: {label_mapping}")
    print(f"🏷️  MULTI-LABEL MODE: Using all emotions, not just the first one")
    
    # Load datasets
    datasets = {}
    
    # Mapping from split names in CSV to expected names
    split_mapping = {'train': 'train', 'validation': 'val', 'test': 'test'}
    
    for csv_split, split_name in split_mapping.items():
        csv_path = os.path.join(data_dir, f"{csv_split}.csv")
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        texts = df['text'].tolist()
        tweet_ids = df['id'].tolist()
        emojis = df['emojis'].tolist()
        
        # Process emotion labels - CREATE MULTI-LABEL BINARY VECTORS
        labels = []
        single_label_count = 0
        multi_label_count = 0
        
        for emotions_str in df['aggregated_emotions']:
            emotions_list = ast.literal_eval(emotions_str)
            
            # Create binary vector for all emotions
            label_vector = [0.0] * len(emotion_classes)
            for emotion in emotions_list:
                if emotion in emotion_classes:
                    emotion_idx = emotion_classes.index(emotion)
                    label_vector[emotion_idx] = 1.0
            
            labels.append(label_vector)
            
            # Track statistics 
            if len(emotions_list) == 1:
                single_label_count += 1
            else:
                multi_label_count += 1
            
            # Validate emotion count 
            if len(emotions_list) > 4:
                print(f"Warning: Sample has {len(emotions_list)} emotions (more than expected 1-4): {emotions_list}")
        
        datasets[split_name] = {
            'text': texts,
            'labels': labels,
            'tweet_id': tweet_ids,
            'emojis': emojis
        }
        
        print(f"{split_name.capitalize()} set: {len(texts)} samples")
        print(f"  Single-emotion samples: {single_label_count} ({single_label_count/len(texts)*100:.1f}%)")
        print(f"  Multi-emotion samples: {multi_label_count} ({multi_label_count/len(texts)*100:.1f}%)")
        
        # Show label distribution (count of each emotion)
        emotion_counts = [0] * len(emotion_classes)
        for label_vector in labels:
            for i, is_present in enumerate(label_vector):
                if is_present == 1.0:
                    emotion_counts[i] += 1
        
        print(f"  Emotion frequencies:")
        for i, emotion in enumerate(emotion_classes):
            count = emotion_counts[i]
            percentage = (count / len(texts)) * 100
            print(f"    {emotion}: {count} ({percentage:.1f}%)")
    
    return datasets, label_mapping

def create_dataset_dict(datasets):
    """Convert to HuggingFace DatasetDict"""
    dataset_dict = {}
    
    for split, data in datasets.items():
        dataset_dict[split] = Dataset.from_dict(data)
    
    return DatasetDict(dataset_dict)

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the texts"""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding=False,
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute comprehensive metrics for multi-label evaluation"""
    from sklearn.metrics import multilabel_confusion_matrix, hamming_loss, jaccard_score
    
    predictions, labels = eval_pred
    
    # Convert predictions to binary (apply sigmoid threshold at 0.5)
    predictions_sigmoid = torch.sigmoid(torch.from_numpy(predictions)).numpy()
    predictions_binary = (predictions_sigmoid > 0.5).astype(float)
    
    # Ensure each sample has at least one predicted emotion - use "other" as fallback
    other_emotion_idx = 4  # "other" is at index 4 in emotion_classes
    for i in range(len(predictions_binary)):
        if np.sum(predictions_binary[i]) == 0:  # No emotions predicted
            # Set the emotion with highest probability or "other" if all are very low
            if np.max(predictions_sigmoid[i]) > 0.1:  # If there's a reasonable confidence
                best_emotion_idx = np.argmax(predictions_sigmoid[i])
                predictions_binary[i][best_emotion_idx] = 1.0
            else:  # All confidences very low, default to "other"
                predictions_binary[i][other_emotion_idx] = 1.0
    
    labels = np.array(labels)
    
    # Multi-label accuracy metrics
    # Exact match ratio 
    exact_match_ratio = np.mean(np.all(predictions_binary == labels, axis=1))
    
    # Hamming loss
    hamming_loss_score = hamming_loss(labels, predictions_binary)
    
    # Jaccard score
    jaccard_scores = []
    for i in range(len(labels)):
        y_true_i = set(np.where(labels[i] == 1)[0])
        y_pred_i = set(np.where(predictions_binary[i] == 1)[0])
        
        if len(y_true_i) == 0 and len(y_pred_i) == 0:
            jaccard_scores.append(1.0)  # Both empty sets
        elif len(y_true_i.union(y_pred_i)) == 0:
            jaccard_scores.append(0.0)  # No union
        else:
            intersection = len(y_true_i.intersection(y_pred_i))
            union = len(y_true_i.union(y_pred_i))
            jaccard_scores.append(intersection / union)
    
    jaccard_score_mean = np.mean(jaccard_scores)
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for i in range(labels.shape[1]):  # For each emotion class
        class_labels = labels[:, i]
        class_preds = predictions_binary[:, i]
        
        if np.sum(class_labels) > 0:  # Only if class is present in dataset
            precision, recall, f1, support = precision_recall_fscore_support(
                class_labels, class_preds, average='binary', zero_division=0
            )
            per_class_metrics[f'precision_class_{i}'] = float(precision)
            per_class_metrics[f'recall_class_{i}'] = float(recall)
            per_class_metrics[f'f1_class_{i}'] = float(f1)
            per_class_metrics[f'support_class_{i}'] = int(np.sum(class_labels))
        else:
            per_class_metrics[f'precision_class_{i}'] = 0.0
            per_class_metrics[f'recall_class_{i}'] = 0.0
            per_class_metrics[f'f1_class_{i}'] = 0.0
            per_class_metrics[f'support_class_{i}'] = 0
    
    # Macro averages (average across all classes)
    macro_precision = np.mean([per_class_metrics[f'precision_class_{i}'] for i in range(labels.shape[1])])
    macro_recall = np.mean([per_class_metrics[f'recall_class_{i}'] for i in range(labels.shape[1])])
    macro_f1 = np.mean([per_class_metrics[f'f1_class_{i}'] for i in range(labels.shape[1])])
    
    # Micro averages (calculate globally)
    micro_precision = precision_recall_fscore_support(
        labels.ravel(), predictions_binary.ravel(), average='binary', zero_division=0
    )[0]
    micro_recall = precision_recall_fscore_support(
        labels.ravel(), predictions_binary.ravel(), average='binary', zero_division=0
    )[1]
    micro_f1 = precision_recall_fscore_support(
        labels.ravel(), predictions_binary.ravel(), average='binary', zero_division=0
    )[2]
    
    # Sample-wise F1 (average F1 score per sample)
    sample_f1_scores = []
    for i in range(len(labels)):
        if np.sum(labels[i]) > 0:  
            sample_f1 = f1_score(labels[i], predictions_binary[i], average='binary', zero_division=0)
            sample_f1_scores.append(sample_f1)
    
    sample_f1_mean = np.mean(sample_f1_scores) if sample_f1_scores else 0.0
    
    return {
        'exact_match_ratio': exact_match_ratio,
        'hamming_loss': hamming_loss_score,
        'accuracy': 1 - hamming_loss_score,  
        'jaccard_score': jaccard_score_mean, 
        'f1_macro': macro_f1,
        'f1_micro': micro_f1,
        'f1_sample_wise': sample_f1_mean,
        'precision_macro': macro_precision,
        'recall_macro': macro_recall,
        'precision_micro': micro_precision,
        'recall_micro': micro_recall,
        **per_class_metrics
    }



def save_multilabel_evaluation_results_to_file(val_results, test_results, label_mapping, 
                                              val_true_labels, test_true_labels, 
                                              val_pred_labels, test_pred_labels, 
                                              class_names, output_dir, config, 
                                              start_time, end_time, trainer):
    """Save comprehensive multi-label evaluation results to a text file"""
    
    print("\nSaving comprehensive multi-label evaluation results to file...")
    
    # Create results text
    results_text = []
    
    # Header
    results_text.append("=" * 70)
    results_text.append("COMPREHENSIVE MULTI-LABEL EVALUATION RESULTS")
    results_text.append("=" * 70)
    results_text.append("")
    
    # Training info
    results_text.append(f"Training completed in: {end_time - start_time}")
    results_text.append(f"Model saved to: {output_dir}")
    results_text.append(f"Dataset: {config['data_dir']}")
    results_text.append(f"Model: {config['model_name']} (Multi-label classification)")
    results_text.append(f"Number of epochs: {config['num_epochs']}")
    results_text.append(f"Learning rate: {config['learning_rate']}")
    results_text.append(f"Batch size: {config['batch_size']}")
    results_text.append(f"Loss function: Binary Cross-Entropy with Sigmoid")
    results_text.append("")
    
    # Overall multi-label metrics
    results_text.append("-" * 50)
    results_text.append("OVERALL MULTI-LABEL METRICS")
    results_text.append("-" * 50)
    results_text.append(f"{'Metric':<20} {'Validation':<12} {'Test':<12}")
    results_text.append("-" * 50)
    results_text.append(f"{'Exact Match Ratio':<20} {val_results['eval_exact_match_ratio']:<12.4f} {test_results['eval_exact_match_ratio']:<12.4f}")
    results_text.append(f"{'Hamming Loss':<20} {val_results['eval_hamming_loss']:<12.4f} {test_results['eval_hamming_loss']:<12.4f}")
    results_text.append(f"{'Accuracy (1-Ham.)':<20} {val_results['eval_accuracy']:<12.4f} {test_results['eval_accuracy']:<12.4f}")
    results_text.append(f"{'Jaccard Score':<20} {val_results['eval_jaccard_score']:<12.4f} {test_results['eval_jaccard_score']:<12.4f}")
    results_text.append(f"{'F1-Macro':<20} {val_results['eval_f1_macro']:<12.4f} {test_results['eval_f1_macro']:<12.4f}")
    results_text.append(f"{'F1-Micro':<20} {val_results['eval_f1_micro']:<12.4f} {test_results['eval_f1_micro']:<12.4f}")
    results_text.append(f"{'F1-Sample-wise':<20} {val_results['eval_f1_sample_wise']:<12.4f} {test_results['eval_f1_sample_wise']:<12.4f}")
    results_text.append(f"{'Precision-Macro':<20} {val_results['eval_precision_macro']:<12.4f} {test_results['eval_precision_macro']:<12.4f}")
    results_text.append(f"{'Recall-Macro':<20} {val_results['eval_recall_macro']:<12.4f} {test_results['eval_recall_macro']:<12.4f}")
    results_text.append("")
    
    # Per-emotion metrics
    results_text.append("-" * 70)
    results_text.append("PER-EMOTION METRICS (TEST SET)")
    results_text.append("-" * 70)
    results_text.append(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    results_text.append("-" * 70)
    
    for i, label in label_mapping.items():
        precision = test_results.get(f'eval_precision_class_{i}', 0)
        recall = test_results.get(f'eval_recall_class_{i}', 0)
        f1 = test_results.get(f'eval_f1_class_{i}', 0)
        support = test_results.get(f'eval_support_class_{i}', 0)
        results_text.append(f"{label:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}")
    
    results_text.append("")
    
    # Validation set per-emotion metrics
    results_text.append("-" * 70)
    results_text.append("PER-EMOTION METRICS (VALIDATION SET)")
    results_text.append("-" * 70)
    results_text.append(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    results_text.append("-" * 70)
    
    for i, label in label_mapping.items():
        precision = val_results.get(f'eval_precision_class_{i}', 0)
        recall = val_results.get(f'eval_recall_class_{i}', 0)
        f1 = val_results.get(f'eval_f1_class_{i}', 0)
        support = val_results.get(f'eval_support_class_{i}', 0)
        results_text.append(f"{label:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}")
    
    results_text.append("")
    
    # Multi-label statistics
    results_text.append("-" * 50)
    results_text.append("MULTI-LABEL STATISTICS (TEST SET)")
    results_text.append("-" * 50)
    
    # Calculate exact matches and partial matches
    exact_matches = np.sum(np.all(test_pred_labels == test_true_labels, axis=1))
    total_samples = len(test_true_labels)
    
    # Calculate label-wise statistics
    n_labels_true = np.sum(test_true_labels, axis=1)
    n_labels_pred = np.sum(test_pred_labels, axis=1)
    n_labels_correct = np.sum(test_pred_labels * test_true_labels, axis=1)
    
    results_text.append(f"Total samples: {total_samples}")
    results_text.append(f"Exact matches: {exact_matches} ({exact_matches/total_samples*100:.1f}%)")
    results_text.append(f"")
    results_text.append(f"Label count statistics:")
    results_text.append(f"  Avg labels per sample (true): {np.mean(n_labels_true):.2f}")
    results_text.append(f"  Avg labels per sample (pred): {np.mean(n_labels_pred):.2f}")
    results_text.append(f"  Avg correct labels per sample: {np.mean(n_labels_correct):.2f}")
    results_text.append("")
    
    # Individual binary classification reports
    results_text.append("-" * 50)
    results_text.append("BINARY CLASSIFICATION REPORTS BY EMOTION (TEST SET)")
    results_text.append("-" * 50)
    
    from sklearn.metrics import classification_report
    
    for i, emotion in enumerate(class_names):
        if np.sum(test_true_labels[:, i]) > 0: 
            results_text.append(f"\n{emotion.upper()} (Binary Classification):")
            results_text.append("-" * 30)
            report = classification_report(
                test_true_labels[:, i], 
                test_pred_labels[:, i],
                target_names=[f'Not {emotion}', emotion],
                digits=4,
                zero_division=0
            )
            results_text.extend(report.split('\n'))
    
    results_text.append("")
    
    # Model configuration details
    results_text.append("-" * 50)
    results_text.append("MODEL CONFIGURATION")
    results_text.append("-" * 50)
    for key, value in config.items():
        results_text.append(f"{key}: {value}")
    results_text.append("")
    
    # Label mapping
    results_text.append("-" * 50)
    results_text.append("LABEL MAPPING")
    results_text.append("-" * 50)
    for idx, label in label_mapping.items():
        results_text.append(f"{idx}: {label}")
    results_text.append("")
    
    # Multi-label dataset statistics
    results_text.append("-" * 50)
    results_text.append("MULTI-LABEL DATASET STATISTICS")
    results_text.append("-" * 50)
    results_text.append(f"Validation samples: {len(val_true_labels)}")
    results_text.append(f"Test samples: {len(test_true_labels)}")
    results_text.append("")
    
    # Emotion frequency in test set
    results_text.append("Emotion frequencies in test set:")
    for i, emotion in enumerate(class_names):
        count = np.sum(test_true_labels[:, i])
        percentage = (count / len(test_true_labels)) * 100
        results_text.append(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Multi-label distribution
    results_text.append("")
    results_text.append("Multi-label distribution in test set:")
    label_counts = np.sum(test_true_labels, axis=1)
    for n_labels in range(1, 5): 
        count = np.sum(label_counts == n_labels)
        if count > 0:
            percentage = (count / len(test_true_labels)) * 100
            results_text.append(f"  {n_labels} label(s): {count} samples ({percentage:.1f}%)")
    
    # Also show if there are any samples with more than 4 labels
    max_labels = int(np.max(label_counts))
    if max_labels > 4:
        for n_labels in range(5, max_labels + 1):
            count = np.sum(label_counts == n_labels)
            if count > 0:
                percentage = (count / len(test_true_labels)) * 100
                results_text.append(f"  {n_labels} label(s): {count} samples ({percentage:.1f}%)")
    
    results_text.append("")
    results_text.append("=" * 70)
    results_text.append(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append("=" * 70)
    
    # Save to file
    results_file_path = os.path.join(output_dir, 'multilabel_evaluation_results.txt')
    with open(results_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_text))
    
    print(f"Multi-label evaluation results saved to: {results_file_path}")
    
    # Also save as JSON for programmatic access
    json_results = {
        'training_info': {
            'duration': str(end_time - start_time),
            'model_path': output_dir,
            'dataset': config['data_dir'],
            'model_name': config['model_name'],
            'config': config,
            'multi_label': True,
            'loss_function': 'Binary Cross-Entropy with Sigmoid'
        },
        'validation_metrics': val_results,
        'test_metrics': test_results,
        'label_mapping': label_mapping,
        'class_names': class_names,
        'multilabel_statistics': {
            'total_samples': len(test_true_labels),
            'exact_matches': int(np.sum(np.all(test_pred_labels == test_true_labels, axis=1))),
            'avg_labels_per_sample_true': float(np.mean(np.sum(test_true_labels, axis=1))),
            'avg_labels_per_sample_pred': float(np.mean(np.sum(test_pred_labels, axis=1))),
            'emotion_frequencies': {
                emotion: int(np.sum(test_true_labels[:, i]))
                for i, emotion in enumerate(class_names)
            }
        },
        'dataset_stats': {
            'validation_samples': len(val_true_labels),
            'test_samples': len(test_true_labels)
        },
        'generation_timestamp': datetime.now().isoformat()
    }
    
    json_file_path = os.path.join(output_dir, 'multilabel_evaluation_results.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"Multi-label evaluation results (JSON) saved to: {json_file_path}")
    
    return results_file_path, json_file_path

def save_test_results_to_csv(test_dataset, test_pred_labels, test_true_labels, class_names, output_dir):
    """Save test results to CSV with tweet ID, predicted labels, true labels, and emojis"""
    
    print("\nSaving test results to CSV...")
    
    # Get tweet IDs and emojis from the test dataset
    tweet_ids = test_dataset['tweet_id']
    emojis = test_dataset['emojis']
    
    def labels_to_emotions(label_vector, class_names):
        """Convert binary label vector to list of emotion names"""
        emotions = []
        for i, is_present in enumerate(label_vector):
            if is_present == 1.0:
                emotions.append(class_names[i])
        
        # Should never be empty due to prediction post-processing, but safety check
        if not emotions:
            print(f"WARNING: Empty emotion prediction found! This should not happen. Label vector: {label_vector}")
            emotions = ['other']
        
        return emotions
    
    # Prepare CSV data
    csv_data = []
    for i, tweet_id in enumerate(tweet_ids):
        # Get predicted and true emotion lists
        pred_emotions = labels_to_emotions(test_pred_labels[i], class_names)
        true_emotions = labels_to_emotions(test_true_labels[i], class_names)
        
        csv_data.append({
            'tweet_id': tweet_id,
            'emojis': emojis[i],
            'predicted_emotions': str(pred_emotions),  # Convert to string for CSV
            'true_emotions': str(true_emotions),
            # Add individual binary predictions for each emotion
            **{f'pred_{emotion}': int(test_pred_labels[i][j]) for j, emotion in enumerate(class_names)},
            **{f'true_{emotion}': int(test_true_labels[i][j]) for j, emotion in enumerate(class_names)}
        })
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(csv_data)
    
    # Reorder columns for better readability
    column_order = ['tweet_id', 'emojis', 'predicted_emotions', 'true_emotions']
    # Add individual emotion columns (predicted first, then true)
    for emotion in class_names:
        column_order.extend([f'pred_{emotion}', f'true_{emotion}'])
    
    results_df = results_df[column_order]
    
    # Save to CSV
    csv_file_path = os.path.join(output_dir, 'test_results_detailed.csv')
    results_df.to_csv(csv_file_path, index=False, encoding='utf-8')
    
    print(f"Test results CSV saved to: {csv_file_path}")
    print(f"CSV contains {len(results_df)} samples with columns:")
    print(f"  - tweet_id: Original tweet identifier")
    print(f"  - emojis: Emoji content from the tweet")
    print(f"  - predicted_emotions: List of predicted emotions")
    print(f"  - true_emotions: List of true emotions")
    print(f"  - Individual binary columns for each emotion (pred_/true_)")
    
    return csv_file_path

def detect_environment_and_adjust_paths(base_config):
    """Detect if running on Kaggle or locally and adjust paths accordingly"""
    config = base_config.copy()
    # for running in kaggle online server
    if os.path.exists("/kaggle"):
        print("🔍 Detected Kaggle environment")
        if not os.path.exists(config['data_dir']) and os.path.exists("/kaggle/input"):
            print("  → Adjusting data directory for Kaggle structure")
            kaggle_inputs = os.listdir("/kaggle/input")
            if kaggle_inputs:
                config['data_dir'] = f"/kaggle/input/{kaggle_inputs[0]}"
        if not config['output_dir'].startswith("/kaggle"):
            config['output_dir'] = "/kaggle/working"
    else:
        print("🏠 Detected local environment")
        # Ensure local paths are relative to current working directory
        if not os.path.isabs(config['data_dir']):
            config['data_dir'] = os.path.join(os.getcwd(), config['data_dir'])
        if not os.path.isabs(config['output_dir']):
            config['output_dir'] = os.path.join(os.getcwd(), config['output_dir'])
    
    # Verify data directory exists
    if not os.path.exists(config['data_dir']):
        print(f"❌ Data directory not found: {config['data_dir']}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"📁 Available directories:")
        try:
            for item in os.listdir(os.getcwd()):
                if os.path.isdir(item):
                    print(f"  📁 {item}")
        except:
            pass
        raise FileNotFoundError(f"Data directory not found: {config['data_dir']}")
    
    print(f"✅ Data directory confirmed: {config['data_dir']}")
    print(f"📁 Output directory: {config['output_dir']}")
    
    return config

def get_device_optimized_config(base_config, device):
    """Optimize training configuration based on the available device"""
    config = base_config.copy()
    
    if device.type == "mps":
        print("🍎 Applying MPS optimizations for Apple Silicon...")
        config["fp16"] = False  
        config["dataloader_num_workers"] = 0 
        config["gradient_accumulation_steps"] = max(1, config.get("gradient_accumulation_steps", 1))
        print("   - Disabled FP16 (not supported on MPS)")
        print("   - Set dataloader workers to 0 for stability")
        
    elif device.type == "cuda":
        print("🚀 Applying CUDA optimizations...")
        config["fp16"] = base_config.get("fp16", True)  
        config["dataloader_num_workers"] = base_config.get("dataloader_num_workers", 2)
        print(f"   - FP16 enabled: {config['fp16']}")
        print(f"   - Dataloader workers: {config['dataloader_num_workers']}")
        
    else:  # CPU
        # CPU optimizations
        print("💻 Applying CPU optimizations...")
        config["fp16"] = False  
        config["dataloader_num_workers"] = min(4, base_config.get("dataloader_num_workers", 2))
        config["batch_size"] = min(config["batch_size"], 8) 
        print("   - Disabled FP16 (not supported on CPU)")
        print(f"   - Reduced batch size to {config['batch_size']} for CPU")
        print(f"   - Dataloader workers: {config['dataloader_num_workers']}")
    
    return config

def main():
    # Set random seed for reproducibility
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(CONFIG['seed'])
    
    print("=" * 60)
    print("Fine-tuning RoBERTa on Emotion Classification (Final Dataset with Emojis)")
    print("Optimized for Multi-Platform Support (CPU/CUDA/MPS)")
    print("=" * 60)
    
    # Detect environment and adjust paths
    path_adjusted_config = detect_environment_and_adjust_paths(CONFIG)
    
    # Get device-optimized configuration
    optimized_config = get_device_optimized_config(path_adjusted_config, device)
    
    # Print system information
    print(f"\n🖥️  System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Transformers version: {transformers.__version__}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   Working directory: {os.getcwd()}")
    
    # Load data
    print("\n1. Loading data...")
    datasets, label_mapping = load_emotion_data(optimized_config['data_dir'])
    dataset_dict = create_dataset_dict(datasets)
    
    # Load tokenizer and model
    print("\n2. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(optimized_config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        optimized_config['model_name'],
        num_labels=len(label_mapping),
        id2label=label_mapping,
        label2id={v: k for k, v in label_mapping.items()},
        problem_type="multi_label_classification",  # Enable multi-label classification
        use_safetensors=True  # Force use of safetensors to avoid torch.load security issue
    )
    
    # Move model to device
    model.to(device)
    
    print(f"Model loaded: {optimized_config['model_name']}")
    print(f"Number of labels: {len(label_mapping)}")
    print(f"🏷️  Multi-label classification mode enabled")
    print(f"🔥  Using Binary Cross-Entropy loss with sigmoid activation")
    
    # Tokenize datasets
    print("\n3. Tokenizing data...")
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_function(examples, tokenizer, optimized_config['max_length']),
        batched=True,
        remove_columns=['text']  # Only remove text column, keep labels, tweet_id, and emojis
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    print("\n4. Setting up training...")
    
    # Create output directory
    os.makedirs(optimized_config['output_dir'], exist_ok=True)
    
    print("\n🔧 Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=optimized_config['output_dir'],
        num_train_epochs=optimized_config['num_epochs'],
        per_device_train_batch_size=optimized_config['batch_size'],
        per_device_eval_batch_size=optimized_config['batch_size'],
        learning_rate=optimized_config['learning_rate'],
        # Device-optimized settings
        logging_steps=optimized_config.get('logging_steps', 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=optimized_config['seed'],
        report_to=[],  # Explicitly empty list
        # Device-specific optimizations
        dataloader_pin_memory=False,
        dataloader_num_workers=optimized_config['dataloader_num_workers'],
        fp16=optimized_config['fp16'],
        remove_unused_columns=True,
        push_to_hub=False,
        # Model saving settings
        load_best_model_at_end=optimized_config.get('load_best_model_at_end', False),
        save_total_limit=optimized_config.get('save_total_limit', 1),
        gradient_accumulation_steps=optimized_config.get('gradient_accumulation_steps', 1),
    )
    print("✅ Training arguments created!")
    
    # Create trainer
    print("\n🔧 Creating trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['val'],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        print("✅ Trainer created successfully!")
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        raise
    
    # Training
    print("\n5. Starting training...")
    print(f"Training samples: {len(tokenized_datasets['train'])}")
    print(f"Validation samples: {len(tokenized_datasets['val'])}")
    print(f"Test samples: {len(tokenized_datasets['test'])}")
    print(f"\n🚀 Training configuration:")
    print(f"   Device: {device}")
    print(f"   Batch size: {optimized_config['batch_size']}")
    print(f"   Epochs: {optimized_config['num_epochs']}")
    print(f"   Learning rate: {optimized_config['learning_rate']}")
    print(f"   FP16: {optimized_config['fp16']}")
    print(f"   Workers: {optimized_config['dataloader_num_workers']}")
    
    start_time = datetime.now()
    
    # Handle 0 epochs case (baseline evaluation)
    if optimized_config['num_epochs'] == 0:
        print(f"\n🔍 BASELINE EVALUATION MODE (0 epochs)")
        print("="*50)
        print("Skipping training - evaluating untrained model for baseline comparison")
        print("="*50)
        train_result = None
        end_time = datetime.now()
        print(f"\n✅ Baseline model ready for evaluation!")
    else:
        print(f"\n⏰ Starting training now...")
        print("\n" + "="*50)
        print("TRAINING STARTING - YOU SHOULD SEE PROGRESS BELOW")
        print("="*50)
        
        try:
            print("\n🚀 Calling trainer.train()...")
            train_result = trainer.train()
            print(f"\n✅ Training completed successfully!")
        except KeyboardInterrupt:
            print(f"\n⏹️ Training interrupted by user")
            raise
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            print(f"💡 Troubleshooting:")
            print(f"   1. Try batch_size=4 or batch_size=2")
            print(f"   2. Reduce num_epochs to 3")
            print(f"   3. Check GPU memory with !nvidia-smi")
            import traceback
            traceback.print_exc()
            raise
        end_time = datetime.now()
    
    print(f"\nCompleted in: {end_time - start_time}")
    
    # Save model
    if optimized_config['num_epochs'] == 0:
        print("\n📋 Saving baseline model for comparison...")
    else:
        print("\n💾 Saving trained model...")
    
    trainer.save_model()
    tokenizer.save_pretrained(optimized_config['output_dir'])
    
    # Save additional metadata
    metadata = {
        'label_mapping': label_mapping,
        'config': optimized_config,
        'original_config': CONFIG,
        'device_used': str(device),
        'model_name': optimized_config['model_name'],
        'num_labels': len(label_mapping),
        'max_length': optimized_config['max_length'],
        'emotion_classes': ['anger', 'disgust', 'fear', 'joy', 'other', 'sadness', 'surprise'],
        'created_timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'data_dir': optimized_config['data_dir'],
            'description': 'RoBERTa model fine-tuned on multi-label emotion classification with emojis'
        },
        'multi_label': True,
        'platform_optimizations': {
            'fp16_enabled': optimized_config['fp16'],
            'dataloader_workers': optimized_config['dataloader_num_workers'],
            'device_type': device.type
        }
    }
    
    # Save metadata as JSON
    metadata_path = os.path.join(optimized_config['output_dir'], 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Model metadata saved to: {metadata_path}")
    
    # Evaluation on validation set
    print("\n10. Evaluating on validation set...")
    val_results = trainer.evaluate(eval_dataset=tokenized_datasets['val'])
    
    # Evaluation on test set
    print("\n11. Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets['test'])
    
    # Get detailed predictions for multi-label evaluation
    val_predictions = trainer.predict(tokenized_datasets['val'])
    test_predictions = trainer.predict(tokenized_datasets['test'])
    
    # Convert predictions to binary using sigmoid threshold (0.5)
    val_pred_sigmoid = torch.sigmoid(torch.from_numpy(val_predictions.predictions)).numpy()
    test_pred_sigmoid = torch.sigmoid(torch.from_numpy(test_predictions.predictions)).numpy()
    
    val_pred_labels = (val_pred_sigmoid > 0.5).astype(float)
    test_pred_labels = (test_pred_sigmoid > 0.5).astype(float)
    
    # Ensure each sample has at least one predicted emotion - use "other" as fallback
    other_emotion_idx = 4  # "other" is at index 4 in emotion_classes
    
    # Fix validation predictions
    empty_val_count = 0
    for i in range(len(val_pred_labels)):
        if np.sum(val_pred_labels[i]) == 0:  # No emotions predicted
            empty_val_count += 1
            if np.max(val_pred_sigmoid[i]) > 0.1:  # If there's reasonable confidence
                best_emotion_idx = np.argmax(val_pred_sigmoid[i])
                val_pred_labels[i][best_emotion_idx] = 1.0
            else:  # All confidences very low, default to "other"
                val_pred_labels[i][other_emotion_idx] = 1.0
    
    # Fix test predictions
    empty_test_count = 0
    for i in range(len(test_pred_labels)):
        if np.sum(test_pred_labels[i]) == 0:  # No emotions predicted
            empty_test_count += 1
            if np.max(test_pred_sigmoid[i]) > 0.1:  # If there's reasonable confidence
                best_emotion_idx = np.argmax(test_pred_sigmoid[i])
                test_pred_labels[i][best_emotion_idx] = 1.0
            else:  # All confidences very low, default to "other"
                test_pred_labels[i][other_emotion_idx] = 1.0
    
    if empty_val_count > 0:
        print(f"📊 Fixed {empty_val_count} empty validation predictions (set to highest confidence or 'other')")
    if empty_test_count > 0:
        print(f"📊 Fixed {empty_test_count} empty test predictions (set to highest confidence or 'other')")
    
    val_true_labels = np.array(val_predictions.label_ids)
    test_true_labels = np.array(test_predictions.label_ids)
    
    # Print comprehensive results
    print("\n" + "=" * 70)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nTraining completed in: {end_time - start_time}")
    print(f"Model saved to: {optimized_config['output_dir']}")
    
    # Overall multi-label metrics
    print("\n" + "-" * 50)
    print("OVERALL MULTI-LABEL METRICS")
    print("-" * 50)
    print(f"{'Metric':<20} {'Validation':<12} {'Test':<12}")
    print("-" * 50)
    print(f"{'Exact Match Ratio':<20} {val_results['eval_exact_match_ratio']:<12.4f} {test_results['eval_exact_match_ratio']:<12.4f}")
    print(f"{'Hamming Loss':<20} {val_results['eval_hamming_loss']:<12.4f} {test_results['eval_hamming_loss']:<12.4f}")
    print(f"{'Accuracy (1-Ham.)':<20} {val_results['eval_accuracy']:<12.4f} {test_results['eval_accuracy']:<12.4f}")
    print(f"{'Jaccard Score':<20} {val_results['eval_jaccard_score']:<12.4f} {test_results['eval_jaccard_score']:<12.4f}")
    print(f"{'F1-Macro':<20} {val_results['eval_f1_macro']:<12.4f} {test_results['eval_f1_macro']:<12.4f}")
    print(f"{'F1-Micro':<20} {val_results['eval_f1_micro']:<12.4f} {test_results['eval_f1_micro']:<12.4f}")
    print(f"{'F1-Sample-wise':<20} {val_results['eval_f1_sample_wise']:<12.4f} {test_results['eval_f1_sample_wise']:<12.4f}")
    print(f"{'Precision-Macro':<20} {val_results['eval_precision_macro']:<12.4f} {test_results['eval_precision_macro']:<12.4f}")
    print(f"{'Recall-Macro':<20} {val_results['eval_recall_macro']:<12.4f} {test_results['eval_recall_macro']:<12.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 70)
    print("PER-CLASS METRICS (TEST SET)")
    print("-" * 70)
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
    print("-" * 70)
    
    for i, label in label_mapping.items():
        precision = test_results.get(f'eval_precision_class_{i}', 0)
        recall = test_results.get(f'eval_recall_class_{i}', 0)
        f1 = test_results.get(f'eval_f1_class_{i}', 0)
        support = test_results.get(f'eval_support_class_{i}', 0)
        print(f"{label:<12} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<12}")
    
    # Multi-label statistics
    print("\n" + "-" * 50)
    print("MULTI-LABEL STATISTICS (TEST SET)")
    print("-" * 50)
    
    # Calculate exact matches and partial matches
    exact_matches = np.sum(np.all(test_pred_labels == test_true_labels, axis=1))
    total_samples = len(test_true_labels)
    
    # Calculate label-wise statistics
    n_labels_true = np.sum(test_true_labels, axis=1)
    n_labels_pred = np.sum(test_pred_labels, axis=1)
    n_labels_correct = np.sum(test_pred_labels * test_true_labels, axis=1)
    
    print(f"Total samples: {total_samples}")
    print(f"Exact matches: {exact_matches} ({exact_matches/total_samples*100:.1f}%)")
    print(f"")
    print(f"Label count statistics:")
    print(f"  Avg labels per sample (true): {np.mean(n_labels_true):.2f}")
    print(f"  Avg labels per sample (pred): {np.mean(n_labels_pred):.2f}")
    print(f"  Avg correct labels per sample: {np.mean(n_labels_correct):.2f}")
    
    # Per-emotion statistics
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    print(f"\nPer-emotion statistics (TEST SET):")
    print(f"{'Emotion':<12} {'True':<8} {'Pred':<8} {'Correct':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 70)
    
    for i, emotion in enumerate(class_names):
        true_count = np.sum(test_true_labels[:, i])
        pred_count = np.sum(test_pred_labels[:, i])
        correct_count = np.sum(test_pred_labels[:, i] * test_true_labels[:, i])
        
        precision = correct_count / pred_count if pred_count > 0 else 0.0
        recall = correct_count / true_count if true_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{emotion:<12} {true_count:<8} {pred_count:<8} {correct_count:<8} {precision:<10.3f} {recall:<8.3f} {f1:<8.3f}")
    
    # Multi-label classification report
    print("\n" + "-" * 50)
    print("MULTI-LABEL CLASSIFICATION SUMMARY (TEST SET)")
    print("-" * 50)
    
    # Generate individual classification reports for each label
    from sklearn.metrics import classification_report
    
    for i, emotion in enumerate(class_names):
        if np.sum(test_true_labels[:, i]) > 0:  # Only for emotions present in test set
            print(f"\n{emotion.upper()} (Binary Classification):")
            report = classification_report(
                test_true_labels[:, i], 
                test_pred_labels[:, i],
                target_names=[f'Not {emotion}', emotion],
                digits=4,
                zero_division=0
            )
            print(report)
    
    # Best epoch info
    if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
        print(f"\nBest F1-Macro achieved: {trainer.state.best_metric:.4f}")
        print(f"Best model from epoch: {trainer.state.best_model_checkpoint}")
    else:
        print(f"\nNote: Best model tracking disabled for simplified training")
    
    print("\n" + "=" * 70)

    # Save comprehensive multi-label evaluation results to files  
    save_multilabel_evaluation_results_to_file(
        val_results, test_results, label_mapping,
        val_true_labels, test_true_labels,
        val_pred_labels, test_pred_labels,
        class_names, optimized_config['output_dir'], optimized_config,
        start_time, end_time, trainer
    )

    # Save test results to CSV with tweet IDs and emojis
    save_test_results_to_csv(
        tokenized_datasets['test'], test_pred_labels, test_true_labels, 
        class_names, optimized_config['output_dir']
    )

    # Create training plots and analysis (only if training occurred)
    if optimized_config['num_epochs'] > 0:
        plot_training_history(trainer, optimized_config['output_dir'], label_mapping)
        analyze_training_recommendations(trainer, optimized_config)
    else:
        print("\n📋 Skipping training plots and analysis (baseline mode with 0 epochs)")

    # Plot multi-label performance heatmaps
    plot_multilabel_performance_heatmap(test_true_labels, test_pred_labels, class_names, optimized_config['output_dir'])

if __name__ == "__main__":
    main() 
    # this is the part where i will put the dammy of this and mke i
#!/usr/bin/env python3
"""
Multi-label Logistic Regression for Emotion Classification
Comparing performance between datasets with and without emojis
"""

import pandas as pd
import numpy as np
import ast
import re
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    multilabel_confusion_matrix, classification_report, 
    hamming_loss, jaccard_score, accuracy_score,
    precision_recall_fscore_support, f1_score
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

class MultiLabelEmotionClassifier:
    def __init__(self, max_features=10000, max_length=256):
        """
        Initialize the Multi-label Emotion Classifier
        
        Args:
            max_features: Maximum number of features for TF-IDF
            max_length: Maximum length for text preprocessing (similar to transformer approach)
        """
        self.max_features = max_features
        self.max_length = max_length
        self.emotions = ['anger', 'disgust', 'fear', 'joy', 'other', 'sadness', 'surprise']
        self.mlb = MultiLabelBinarizer(classes=self.emotions)
        self.vectorizer = None
        self.model = None
        self.best_params = None
        
    def natural_tokenizer(self, text):
        """
        Natural tokenizer that treats emojis as individual tokens
        No prior knowledge about emojis - just treats them as characters/tokens
        """
        if pd.isna(text) or not text:
            return []
        
        # Convert to string
        text = str(text).strip()
        
        # Split by whitespace first
        words = text.split()
        
        tokens = []
        for word in words:
            # For each word, we'll extract both regular characters and unicode characters (emojis)
            # This way emojis get treated as individual tokens naturally
            
            current_token = ""
            for char in word:
                # If it's a regular alphanumeric character, add to current token
                if char.isalnum() or char in ["'", "-"]:
                    current_token += char
                else:
                    # If we have a current token, add it
                    if current_token:
                        tokens.append(current_token.lower())
                        current_token = ""
                    
                    # If it's not whitespace or common punctuation, treat as individual token (emoji)
                    if char not in [' ', '.', ',', '!', '?', ';', ':', '"', '(', ')', '[', ']', '{', '}']:
                        tokens.append(char)  # This preserves emojis as individual tokens
            
            # Add any remaining token
            if current_token:
                tokens.append(current_token.lower())
        
        return tokens

    def preprocess_text(self, text):
        """
        Basic text preprocessing while preserving emojis naturally
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and truncate to max_length characters (similar to max_length in transformers)
        text = str(text)[:self.max_length]
        
        # Basic cleaning while preserving ALL unicode characters (including emojis)
        text = re.sub(r'URL', '', text)  # Remove URL placeholders
        text = re.sub(r'HASHTAG', '', text)  # Remove HASHTAG placeholders
        text = re.sub(r'&amp;', '&', text)  # Fix HTML entities
        
        return text.strip()
    
    def parse_emotions(self, emotion_str):
        """Parse emotion string to list"""
        try:
            if pd.isna(emotion_str):
                return []
            # Convert string representation of list to actual list
            emotions = ast.literal_eval(emotion_str)
            return emotions if isinstance(emotions, list) else [emotions]
        except:
            return []
    
    def load_and_prepare_data(self, data_dir):
        """Load and prepare data from the given directory"""
        train_df = pd.read_csv(Path(data_dir) / 'train.csv')
        val_df = pd.read_csv(Path(data_dir) / 'validation.csv')
        test_df = pd.read_csv(Path(data_dir) / 'test.csv')
        
        # Preprocess texts
        for df in [train_df, val_df, test_df]:
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            df['emotion_labels'] = df['aggregated_emotions'].apply(self.parse_emotions)
        
        return train_df, val_df, test_df
    
    def prepare_features_and_labels(self, train_df, val_df, test_df, use_natural_tokenizer=True):
        """Prepare features and labels for training with natural emoji discovery"""
        
        if use_natural_tokenizer:
            print("Creating TF-IDF features with natural emoji discovery...")
            # Create TF-IDF vectorizer that naturally discovers emojis as tokens
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # unigrams and bigrams
                stop_words='english',
                lowercase=False,  # Don't lowercase to preserve emoji uniqueness
                tokenizer=self.natural_tokenizer,  # Our custom tokenizer
                token_pattern=None,  # Use our custom tokenizer instead of regex
                strip_accents=None,  # Don't strip accents to preserve emojis
                min_df=2,  # Minimum document frequency (removes very rare tokens)
                max_df=0.95  # Maximum document frequency (removes very common tokens)
            )
        else:
            print("Creating standard TF-IDF features (emojis will be filtered out)...")
            # Standard TF-IDF that filters out emojis
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),  # unigrams and bigrams
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'  # This removes emojis
            )
        
        X_train = self.vectorizer.fit_transform(train_df['processed_text'])
        X_val = self.vectorizer.transform(val_df['processed_text'])
        X_test = self.vectorizer.transform(test_df['processed_text'])
        
        print(f"Total features discovered: {X_train.shape[1]}")
        
        # Let's examine what features were discovered (only if using natural tokenizer)
        if use_natural_tokenizer:
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Find potential emoji features (non-ASCII characters)
            emoji_like_features = [f for f in feature_names if any(ord(char) > 127 for char in f)]
            print(f"Potential emoji/unicode features discovered: {len(emoji_like_features)}")
            
            if emoji_like_features:
                print("Sample emoji-like features found:")
                for feat in emoji_like_features[:10]:  # Show first 10
                    print(f"  - '{feat}'")
        
        # Prepare multi-label targets
        y_train = self.mlb.fit_transform(train_df['emotion_labels'])
        y_val = self.mlb.transform(val_df['emotion_labels'])
        y_test = self.mlb.transform(test_df['emotion_labels'])
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """Perform hyperparameter tuning"""
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid (keeping it short as requested)
        param_grid = {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__solver': ['liblinear', 'lbfgs'],
            'estimator__max_iter': [1000]
        }
        
        # Create base model
        base_model = MultiOutputClassifier(
            LogisticRegression(random_state=42, class_weight='balanced')
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the model with hyperparameter tuning"""
        start_time = datetime.now()
        
        # Perform hyperparameter tuning
        self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
        
        training_time = datetime.now() - start_time
        print(f"Training completed in: {training_time}")
        
        return training_time
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        # For MultiOutputClassifier, we need to handle probabilities differently
        probabilities = []
        for i, estimator in enumerate(self.model.estimators_):
            prob = estimator.predict_proba(X)
            # Take the probability of the positive class
            if prob.shape[1] == 2:
                probabilities.append(prob[:, 1])
            else:
                probabilities.append(prob[:, 0])
        return np.column_stack(probabilities)

def calculate_multilabel_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive multi-label metrics"""
    metrics = {}
    
    # Basic multi-label metrics
    metrics['exact_match_ratio'] = accuracy_score(y_true, y_pred)
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['accuracy'] = 1 - metrics['hamming_loss']
    
    # F1 scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_samples'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Precision and Recall
    precision_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[0]
    recall_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[1]
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    
    # Jaccard score
    metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    return metrics

def calculate_per_emotion_metrics(y_true, y_pred, emotion_names):
    """Calculate per-emotion metrics"""
    per_emotion_metrics = {}
    
    for i, emotion in enumerate(emotion_names):
        y_true_emotion = y_true[:, i]
        y_pred_emotion = y_pred[:, i]
        
        # Calculate metrics for this emotion
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_emotion, y_pred_emotion, average='binary', zero_division=0
        )
        
        # Calculate actual support (number of true positive instances)
        actual_support = np.sum(y_true_emotion)
        
        per_emotion_metrics[emotion] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': actual_support
        }
    
    return per_emotion_metrics

def generate_binary_classification_reports(y_true, y_pred, emotion_names):
    """Generate binary classification reports for each emotion"""
    reports = {}
    
    for i, emotion in enumerate(emotion_names):
        y_true_emotion = y_true[:, i]
        y_pred_emotion = y_pred[:, i]
        
        # Generate classification report
        report = classification_report(
            y_true_emotion, y_pred_emotion,
            target_names=[f'Not {emotion}', emotion],
            output_dict=True,
            zero_division=0
        )
        reports[emotion] = report
    
    return reports

def calculate_dataset_statistics(y_true, emotion_names):
    """Calculate dataset statistics"""
    stats = {}
    
    total_samples = len(y_true)
    stats['total_samples'] = total_samples
    
    # Emotion frequencies
    emotion_frequencies = {}
    for i, emotion in enumerate(emotion_names):
        count = np.sum(y_true[:, i])
        emotion_frequencies[emotion] = {
            'count': count,
            'percentage': (count / total_samples) * 100
        }
    
    stats['emotion_frequencies'] = emotion_frequencies
    
    # Multi-label distribution
    labels_per_sample = np.sum(y_true, axis=1)
    label_distribution = {}
    for num_labels in range(1, max(labels_per_sample) + 1):
        count = np.sum(labels_per_sample == num_labels)
        if count > 0:
            label_distribution[num_labels] = {
                'count': count,
                'percentage': (count / total_samples) * 100
            }
    
    stats['label_distribution'] = label_distribution
    stats['avg_labels_per_sample'] = np.mean(labels_per_sample)
    
    return stats

def create_comparison_plots(results_with_emoji, results_no_emoji, output_dir):
    """Create comparison plots between emoji and no-emoji datasets"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Logistic Regression: Emoji vs No-Emoji Comparison', fontsize=16, fontweight='bold')
    
    # 1. Overall Metrics Comparison
    ax1 = axes[0, 0]
    metrics = ['F1-Macro', 'F1-Micro', 'Jaccard', 'Exact Match']
    emoji_values = [
        results_with_emoji['test_metrics']['f1_macro'],
        results_with_emoji['test_metrics']['f1_micro'],
        results_with_emoji['test_metrics']['jaccard_score'],
        results_with_emoji['test_metrics']['exact_match_ratio']
    ]
    no_emoji_values = [
        results_no_emoji['test_metrics']['f1_macro'],
        results_no_emoji['test_metrics']['f1_micro'],
        results_no_emoji['test_metrics']['jaccard_score'],
        results_no_emoji['test_metrics']['exact_match_ratio']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, emoji_values, width, label='With Emojis', alpha=0.8)
    ax1.bar(x + width/2, no_emoji_values, width, label='No Emojis', alpha=0.8)
    ax1.set_ylabel('Score')
    ax1.set_title('Overall Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Per-Emotion F1 Scores
    ax2 = axes[0, 1]
    emotions = ['anger', 'disgust', 'fear', 'joy', 'other', 'sadness', 'surprise']
    emoji_f1 = [results_with_emoji['per_emotion_test'][emotion]['f1_score'] for emotion in emotions]
    no_emoji_f1 = [results_no_emoji['per_emotion_test'][emotion]['f1_score'] for emotion in emotions]
    
    x = np.arange(len(emotions))
    ax2.bar(x - width/2, emoji_f1, width, label='With Emojis', alpha=0.8)
    ax2.bar(x + width/2, no_emoji_f1, width, label='No Emojis', alpha=0.8)
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Per-Emotion F1 Scores')
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotions, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Hamming Loss and Accuracy
    ax3 = axes[1, 0]
    categories = ['Hamming Loss', 'Accuracy']
    emoji_values = [
        results_with_emoji['test_metrics']['hamming_loss'],
        results_with_emoji['test_metrics']['accuracy']
    ]
    no_emoji_values = [
        results_no_emoji['test_metrics']['hamming_loss'],
        results_no_emoji['test_metrics']['accuracy']
    ]
    
    x = np.arange(len(categories))
    ax3.bar(x - width/2, emoji_values, width, label='With Emojis', alpha=0.8)
    ax3.bar(x + width/2, no_emoji_values, width, label='No Emojis', alpha=0.8)
    ax3.set_ylabel('Score')
    ax3.set_title('Hamming Loss vs Accuracy')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Precision vs Recall by Emotion
    ax4 = axes[1, 1]
    emoji_precision = [results_with_emoji['per_emotion_test'][emotion]['precision'] for emotion in emotions]
    emoji_recall = [results_with_emoji['per_emotion_test'][emotion]['recall'] for emotion in emotions]
    no_emoji_precision = [results_no_emoji['per_emotion_test'][emotion]['precision'] for emotion in emotions]
    no_emoji_recall = [results_no_emoji['per_emotion_test'][emotion]['recall'] for emotion in emotions]
    
    ax4.scatter(emoji_precision, emoji_recall, label='With Emojis', alpha=0.7, s=100)
    ax4.scatter(no_emoji_precision, no_emoji_recall, label='No Emojis', alpha=0.7, s=100)
    
    # Add emotion labels to points
    for i, emotion in enumerate(emotions):
        ax4.annotate(emotion, (emoji_precision[i], emoji_recall[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Precision')
    ax4.set_ylabel('Recall')
    ax4.set_title('Precision vs Recall by Emotion')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'natural_discovery_comparison_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to: {Path(output_dir) / 'natural_discovery_comparison_plots.png'}")

def save_comprehensive_results(results, dataset_name, output_file):
    """Save comprehensive results in the same format as RoBERTa output"""
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE MULTI-LABEL EVALUATION RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        # Training info
        f.write(f"Training completed in: {results['training_time']}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write("Model: Logistic Regression (Multi-label classification)\n")
        f.write(f"Best parameters: {results['best_params']}\n")
        f.write("Loss function: Logistic Regression with L2 regularization\n\n")
        
        # Overall metrics
        f.write("-" * 50 + "\n")
        f.write("OVERALL MULTI-LABEL METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<20} {'Validation':<12} {'Test':<12}\n")
        f.write("-" * 50 + "\n")
        
        val_metrics = results['val_metrics']
        test_metrics = results['test_metrics']
        
        f.write(f"{'Exact Match Ratio':<20} {val_metrics['exact_match_ratio']:<12.4f} {test_metrics['exact_match_ratio']:<12.4f}\n")
        f.write(f"{'Hamming Loss':<20} {val_metrics['hamming_loss']:<12.4f} {test_metrics['hamming_loss']:<12.4f}\n")
        f.write(f"{'Accuracy (1-Ham.)':<20} {val_metrics['accuracy']:<12.4f} {test_metrics['accuracy']:<12.4f}\n")
        f.write(f"{'F1-Macro':<20} {val_metrics['f1_macro']:<12.4f} {test_metrics['f1_macro']:<12.4f}\n")
        f.write(f"{'F1-Micro':<20} {val_metrics['f1_micro']:<12.4f} {test_metrics['f1_micro']:<12.4f}\n")
        f.write(f"{'F1-Sample-wise':<20} {val_metrics['f1_samples']:<12.4f} {test_metrics['f1_samples']:<12.4f}\n")
        f.write(f"{'Precision-Macro':<20} {val_metrics['precision_macro']:<12.4f} {test_metrics['precision_macro']:<12.4f}\n")
        f.write(f"{'Recall-Macro':<20} {val_metrics['recall_macro']:<12.4f} {test_metrics['recall_macro']:<12.4f}\n")
        f.write(f"{'Jaccard Score':<20} {val_metrics['jaccard_score']:<12.4f} {test_metrics['jaccard_score']:<12.4f}\n\n")
        
        # Per-emotion metrics for test set
        f.write("-" * 70 + "\n")
        f.write("PER-EMOTION METRICS (TEST SET)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
        f.write("-" * 70 + "\n")
        
        for emotion, metrics in results['per_emotion_test'].items():
            f.write(f"{emotion:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1_score']:<12.4f} {int(metrics['support']):<12}\n")
        
        # Per-emotion metrics for validation set
        f.write("\n" + "-" * 70 + "\n")
        f.write("PER-EMOTION METRICS (VALIDATION SET)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}\n")
        f.write("-" * 70 + "\n")
        
        for emotion, metrics in results['per_emotion_val'].items():
            f.write(f"{emotion:<12} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                   f"{metrics['f1_score']:<12.4f} {int(metrics['support']):<12}\n")
        
        # Multi-label statistics
        test_stats = results['test_stats']
        f.write("\n" + "-" * 50 + "\n")
        f.write("MULTI-LABEL STATISTICS (TEST SET)\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total samples: {test_stats['total_samples']}\n")
        exact_matches = int(test_stats['total_samples'] * test_metrics['exact_match_ratio'])
        f.write(f"Exact matches: {exact_matches} ({test_metrics['exact_match_ratio']*100:.1f}%)\n\n")
        
        f.write("Label count statistics:\n")
        f.write(f"  Avg labels per sample (true): {test_stats['avg_labels_per_sample']:.2f}\n")
        # Add predicted labels average here if needed
        f.write("\n")
        
        f.write("Emotion frequencies in test set:\n")
        for emotion, freq in test_stats['emotion_frequencies'].items():
            f.write(f"  {emotion}: {freq['count']} samples ({freq['percentage']:.1f}%)\n")
        
        f.write("\nMulti-label distribution in test set:\n")
        for num_labels, dist in test_stats['label_distribution'].items():
            f.write(f"  {num_labels} label(s): {dist['count']} samples ({dist['percentage']:.1f}%)\n")
        
        # Binary classification reports
        f.write("\n" + "-" * 50 + "\n")
        f.write("BINARY CLASSIFICATION REPORTS BY EMOTION (TEST SET)\n")
        f.write("-" * 50 + "\n")
        
        for emotion, report in results['binary_reports_test'].items():
            f.write(f"\n{emotion.upper()} (Binary Classification):\n")
            f.write("-" * 30 + "\n")
            f.write(f"              precision    recall  f1-score   support\n\n")
            
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"{class_name:>12}     {metrics['precision']:.4f}    {metrics['recall']:.4f}    "
                           f"{metrics['f1-score']:.4f}     {int(metrics['support'])}\n")
            
            f.write(f"\n    accuracy                         {report['accuracy']:.4f}     {int(report['macro avg']['support'])}\n")
            f.write(f"   macro avg     {report['macro avg']['precision']:.4f}    {report['macro avg']['recall']:.4f}    "
                   f"{report['macro avg']['f1-score']:.4f}     {int(report['macro avg']['support'])}\n")
            f.write(f"weighted avg     {report['weighted avg']['precision']:.4f}    {report['weighted avg']['recall']:.4f}    "
                   f"{report['weighted avg']['f1-score']:.4f}     {int(report['weighted avg']['support'])}\n")
        
        # Model configuration
        f.write("\n" + "-" * 50 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        f.write("model_name: Logistic Regression (Multi-Output)\n")
        f.write(f"data_dir: {dataset_name}\n")
        f.write("max_features: 10000\n")
        f.write("max_length: 256\n")
        f.write("ngram_range: (1, 2)\n")
        f.write("stop_words: english\n")
        f.write(f"best_params: {results['best_params']}\n")
        f.write("random_state: 42\n")
        f.write("class_weight: balanced\n")
        
        # Label mapping
        f.write("\n" + "-" * 50 + "\n")
        f.write("LABEL MAPPING\n")
        f.write("-" * 50 + "\n")
        emotions = ['anger', 'disgust', 'fear', 'joy', 'other', 'sadness', 'surprise']
        for i, emotion in enumerate(emotions):
            f.write(f"{i}: {emotion}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")

def main():
    """Main execution function"""
    print("Starting Multi-label Logistic Regression Analysis...")
    print("=" * 60)
    
    # Define paths
    base_dir = Path("/Users/arbelaskayo/Library/CloudStorage/OneDrive-UniversityofHaifa/information systems/S06/NLP/Project/logistic regression")
    emoji_dir = base_dir / "Full DATA" / "split with emoji"
    no_emoji_dir = base_dir / "Full DATA" / "split no emoji"
    
    results = {}
    
    # Process both datasets
    for dataset_name, data_dir in [("with_emoji", emoji_dir), ("no_emoji", no_emoji_dir)]:
        print(f"\n{'='*20} Processing {dataset_name.replace('_', ' ').title()} Dataset {'='*20}")
        
        # Initialize classifier
        classifier = MultiLabelEmotionClassifier()
        
        # Load and prepare data
        print("Loading and preparing data...")
        train_df, val_df, test_df = classifier.load_and_prepare_data(data_dir)
        
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Show a few sample texts to see what the model will see
        if dataset_name == "with_emoji":
            print("\nSample processed texts (what the model sees):")
            for i in range(3):
                print(f"{i+1}. '{train_df['processed_text'].iloc[i]}'")
                print(f"   Tokens: {classifier.natural_tokenizer(train_df['processed_text'].iloc[i])}")
        
        # Prepare features and labels
        print("Preparing features and labels...")
        # Use natural tokenizer for emoji dataset, standard for no-emoji
        use_natural = (dataset_name == "with_emoji")
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_features_and_labels(
            train_df, val_df, test_df, use_natural_tokenizer=use_natural
        )
        
        # Train model
        print("Training model with hyperparameter tuning...")
        training_time = classifier.train(X_train, y_train, X_val, y_val)
        
        # Make predictions
        print("Making predictions...")
        y_val_pred = classifier.predict(X_val)
        y_test_pred = classifier.predict(X_test)
        
        # Calculate metrics
        print("Calculating comprehensive metrics...")
        val_metrics = calculate_multilabel_metrics(y_val, y_val_pred)
        test_metrics = calculate_multilabel_metrics(y_test, y_test_pred)
        
        # Per-emotion metrics
        per_emotion_val = calculate_per_emotion_metrics(y_val, y_val_pred, classifier.emotions)
        per_emotion_test = calculate_per_emotion_metrics(y_test, y_test_pred, classifier.emotions)
        
        # Binary classification reports
        binary_reports_test = generate_binary_classification_reports(y_test, y_test_pred, classifier.emotions)
        
        # Dataset statistics
        test_stats = calculate_dataset_statistics(y_test, classifier.emotions)
        
        # Store results
        results[dataset_name] = {
            'training_time': training_time,
            'best_params': classifier.best_params,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'per_emotion_val': per_emotion_val,
            'per_emotion_test': per_emotion_test,
            'binary_reports_test': binary_reports_test,
            'test_stats': test_stats
        }
        
        # Save individual results
        output_file = base_dir / f"natural_discovery_logistic_regression_results_{dataset_name}.txt"
        save_comprehensive_results(results[dataset_name], str(data_dir), output_file)
        print(f"Results saved to: {output_file}")
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(results["with_emoji"], results["no_emoji"], base_dir)
    
    # Print summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    print(f"{'Metric':<20} {'With Emojis':<15} {'No Emojis':<15} {'Difference':<15}")
    print("-"*65)
    
    metrics_to_compare = ['f1_macro', 'f1_micro', 'jaccard_score', 'exact_match_ratio', 'hamming_loss']
    for metric in metrics_to_compare:
        emoji_val = results["with_emoji"]['test_metrics'][metric]
        no_emoji_val = results["no_emoji"]['test_metrics'][metric]
        diff = emoji_val - no_emoji_val
        print(f"{metric:<20} {emoji_val:<15.4f} {no_emoji_val:<15.4f} {diff:<15.4f}")
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

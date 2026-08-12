"""
Visualization Module
Create plots and visualizations for model training and evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

class Visualizer:
    """Create visualizations for training and evaluation"""
    
    @staticmethod
    def plot_training_history(history_phase1, history_phase2, 
                             save_path="./results/training_history.png"):
        """
        Plot training history (accuracy and loss)
        
        Args:
            history_phase1: Phase 1 training history
            history_phase2: Phase 2 training history
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Transfer Learning - Training History', 
                    fontsize=16, fontweight='bold')
        
        # Combine histories
        all_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
        all_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
        all_loss = history_phase1.history['loss'] + history_phase2.history['loss']
        all_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
        
        phase1_epochs = len(history_phase1.history['accuracy'])
        
        # Accuracy plot
        axes[0].plot(all_acc, label='Train', linewidth=2.5, marker='o')
        axes[0].plot(all_val_acc, label='Validation', linewidth=2.5, marker='s')
        axes[0].axvline(x=phase1_epochs, color='red', linestyle='--', 
                       alpha=0.5, label='Fine-tuning starts')
        axes[0].set_title('Accuracy', fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(all_loss, label='Train', linewidth=2.5, marker='o')
        axes[1].plot(all_val_loss, label='Validation', linewidth=2.5, marker='s')
        axes[1].axvline(x=phase1_epochs, color='red', linestyle='--', 
                       alpha=0.5, label='Fine-tuning starts')
        axes[1].set_title('Loss', fontweight='bold')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(true_classes, predicted_classes, 
                             class_names=None,
                             save_path="./results/confusion_matrix.png"):
        """
        Plot confusion matrix
        
        Args:
            true_classes: True class labels
            predicted_classes: Predicted class labels
            class_names (list): Class names
            save_path (str): Path to save plot
        """
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    @staticmethod
    def plot_roc_curve(predictions, true_classes,
                      save_path="./results/roc_curve.png"):
        """
        Plot ROC curve (for binary classification)
        
        Args:
            predictions: Model predictions
            true_classes: True class labels
            save_path (str): Path to save plot
        """
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontweight='bold')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" ROC curve saved to {save_path}")
        plt.close()
"""
Dogs vs Cats Classification - Evaluation Script
Comprehensive model evaluation with metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json

class Evaluator:
    def __init__(self, model_path, test_dir, img_size=(128, 128)):
        self.model = tf.keras.models.load_model(model_path)
        self.test_dir = test_dir
        self.img_size = img_size
        self.results = {}
        
    def load_test_data(self, batch_size=32):
        """Load test dataset"""
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        test_data = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode="binary",
            shuffle=False
        )
        return test_data
    
    def evaluate(self, test_data):
        """Evaluate model on test data"""
        print("\n" + "=" * 60)
        print("📊 MODEL EVALUATION")
        print("=" * 60 + "\n")
        
        # Get predictions
        test_data.reset()
        predictions = self.model.predict(test_data, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
        true_classes = test_data.classes
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(test_data, verbose=0)
        
        print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"✅ Test Loss: {test_loss:.4f}\n")
        
        # Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        print("Confusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(
            true_classes, predicted_classes,
            target_names=["Cat", "Dog"]
        ))
        
        # Save metrics
        self.results = {
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'confusion_matrix': cm.tolist()
        }
        
        return test_data, predictions, predicted_classes, true_classes
    
    def plot_confusion_matrix(self, cm, save_path="./results/confusion_matrix.png"):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Cat', 'Dog'], 
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix - Dogs vs Cats', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(self, predictions, true_classes, save_path="./results/roc_curve.png"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Dogs vs Cats')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved to {save_path}")
        plt.close()

def main():
    evaluator = Evaluator(
        model_path="./model/dogs_vs_cats_mobilenetv2.keras",
        test_dir="./data/test"
    )
    
    test_data = evaluator.load_test_data()
    test_data, predictions, predicted_classes, true_classes = evaluator.evaluate(test_data)
    
    # Generate plots
    cm = confusion_matrix(true_classes, predicted_classes)
    os.makedirs("./results", exist_ok=True)
    
    evaluator.plot_confusion_matrix(cm)
    evaluator.plot_roc_curve(predictions, true_classes)
    
    # Save results as JSON
    with open("./results/evaluation_metrics.json", 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
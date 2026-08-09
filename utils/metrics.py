"""
Metrics Module
Calculate and manage evaluation metrics
"""

import numpy as np
import json
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    def __init__(self, model, model_type="binary"):
        """
        Initialize MetricsCalculator
        
        Args:
            model: Keras model
            model_type (str): "binary" or "categorical"
        """
        self.model = model
        self.model_type = model_type
        self.results = {}
    
    def evaluate(self, test_data, class_names=None):
        """
        Evaluate model on test data
        
        Args:
            test_data: Test data generator
            class_names (list): Class names for reporting
            
        Returns:
            dict: Evaluation results
        """
        test_data.reset()
        
        # Get predictions
        predictions = self.model.predict(test_data, verbose=0)
        
        if self.model_type == "binary":
            predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
        else:
            predicted_classes = np.argmax(predictions, axis=1)
        
        true_classes = test_data.classes
        
        # Calculate loss and accuracy
        test_loss, test_accuracy = self.model.evaluate(test_data, verbose=0)
        
        # Store results
        self.results = {
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'confusion_matrix': confusion_matrix(true_classes, predicted_classes).tolist()
        }
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 MODEL EVALUATION")
        print("=" * 60 + "\n")
        
        print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"✅ Test Loss: {test_loss:.4f}\n")
        
        print("Confusion Matrix:")
        print(confusion_matrix(true_classes, predicted_classes))
        
        print("\nClassification Report:")
        print(classification_report(
            true_classes, predicted_classes,
            target_names=class_names if class_names else None
        ))
        
        return self.results, predictions, predicted_classes, true_classes
    
    def save_metrics(self, filepath):
        """
        Save metrics to JSON file
        
        Args:
            filepath (str): Path to save metrics
        """
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Metrics saved to {filepath}")
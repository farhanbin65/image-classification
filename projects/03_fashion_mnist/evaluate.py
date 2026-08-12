"""
Fashion MNIST - Evaluation Script
Detailed model evaluation and comparison
Author: Keo (Farhan Bin Hossain)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

CLASS_NAMES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress',     'Coat',
    'Sandal',  'Shirt',   'Sneaker',  'Bag', 'Ankle boot'
]

def load_test_data(test_path='./data/fashion-mnist_test.csv'):
    """Load and prepare test data"""
    
    test_df = pd.read_csv(test_path)
    X_test  = test_df.drop('label', axis=1).values
    y_test  = test_df['label'].values
    
    # Normalize and reshape
    X_test_norm = X_test.reshape(-1, 28, 28) / 255.0
    
    # CNN format
    X_test_cnn = X_test_norm.reshape(-1, 28, 28, 1)
    
    # TL format
    X_test_tl = tf.image.resize(
        X_test_norm.reshape(-1, 28, 28, 1), [96, 96])
    X_test_tl = tf.repeat(X_test_tl, 3, axis=-1)
    
    # One-hot labels
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    
    return X_test_cnn, X_test_tl, y_test_cat, y_test

def evaluate_model(model_path, X_test, y_test_cat, model_name):
    """Evaluate a single model"""
    
    model = tf.keras.models.load_model(model_path)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    
    preds = np.argmax(
        model.predict(X_test, verbose=0), axis=1
    )
    
    print(f"\n{model_name}")
    print("-" * 40)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Loss:     {loss:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(
        np.argmax(y_test_cat, axis=1),
        preds,
        target_names=CLASS_NAMES
    ))
    
    return acc, loss, preds

def main():
    print("Fashion MNIST - Model Evaluation")
    print("=" * 60)
    
    # Load test data
    X_test_cnn, X_test_tl, y_test_cat, y_test = load_test_data()
    y_true = np.argmax(y_test_cat, axis=1)
    
    # Evaluate both models
    cnn_acc, cnn_loss, cnn_preds = evaluate_model(
        './model/cnn_best.keras',
        X_test_cnn, y_test_cat,
        'Custom CNN'
    )
    
    tl_acc, tl_loss, tl_preds = evaluate_model(
        './model/tl_best.keras',
        X_test_tl, y_test_cat,
        'Transfer Learning (MobileNetV2)'
    )
    
    # Per class comparison
    cm_cnn = confusion_matrix(y_true, cnn_preds)
    cm_tl  = confusion_matrix(y_true, tl_preds)
    
    print("\n" + "=" * 55)
    print("PER CLASS ACCURACY COMPARISON")
    print("=" * 55)
    print(f"{'Class':<15} {'CNN':>10} {'TL':>10} {'Winner':>10}")
    print("-" * 55)
    
    for i, cls in enumerate(CLASS_NAMES):
        cnn_cls_acc = cm_cnn[i, i] / 1000 * 100
        tl_cls_acc  = cm_tl[i, i]  / 1000 * 100
        winner = "CNN" if cnn_cls_acc > tl_cls_acc else "TL"
        print(f"{cls:<15} {cnn_cls_acc:>9.1f}% "
              f"{tl_cls_acc:>9.1f}% {winner:>10}")
    
    print("-" * 55)
    winner = "CNN" if cnn_acc > tl_acc else "TL"
    print(f"{'OVERALL':<15} {cnn_acc*100:>9.2f}% "
          f"{tl_acc*100:>9.2f}% {winner:>10}")

if __name__ == "__main__":
    main()
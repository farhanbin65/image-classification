"""
Fast Food Classification - Evaluation Script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

def evaluate_model(model_path, test_dir, noisy_dir, img_size=(128, 128)):
    
    print("📊 FOOD CLASSIFIER - EVALUATION\n")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded!\n")
    
    datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Clean test
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )
    
    test_loss, test_acc = model.evaluate(test_data, verbose=0)
    test_data.reset()
    
    preds = model.predict(test_data, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = test_data.classes
    class_labels = list(test_data.class_indices.keys())
    
    print(f"✅ Clean Test Accuracy: {test_acc*100:.2f}%")
    print(classification_report(true_classes, pred_classes, target_names=class_labels))
    
    # Noisy test
    noisy_data = datagen.flow_from_directory(
        noisy_dir,
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )
    
    noisy_loss, noisy_acc = model.evaluate(noisy_data, verbose=0)
    print(f"✅ Noisy Test Accuracy: {noisy_acc*100:.2f}%")
    print(f"📉 Performance Drop: {(test_acc-noisy_acc)*100:.2f}%")

if __name__ == "__main__":
    evaluate_model(
        model_path="./model/food_classifier_mobilenetv2.keras",
        test_dir="./Dataset/Test",
        noisy_dir="./Dataset/Noisy Test"
    )
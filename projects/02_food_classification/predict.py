"""
Fast Food Classification - Prediction Script
Predict food class from a single image
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse
import os

CLASS_NAMES = ['Burger', 'Crispy Chicken', 'Fries', 'Pizza']

def predict_image(model_path, image_path):
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(predictions)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = predictions[pred_idx]
    
    
    # Display
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"{pred_class} - {confidence*100:.2f}% confident",
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"\nPrediction: {pred_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"\nAll probabilities:")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, predictions)):
        bar = '█' * int(prob * 20)
        print(f"  {cls:<15} {prob*100:>6.2f}% {bar}")
    
    return pred_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--model', type=str,
                       default='./model/food_classifier_mobilenetv2.keras')
    args = parser.parse_args()
    
    predict_image(args.model, args.image)
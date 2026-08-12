"""
Dogs vs Cats Classification - Prediction Script
Make predictions on single images
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (128, 128)
        
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img, img_array
    
    def predict(self, image_path):
        """Make prediction on image"""
        img, img_array = self.preprocess_image(image_path)
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        label = "Dog" if prediction > 0.5 else "Cat"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return label, confidence, img
    
    def predict_and_display(self, image_path):
        """Predict and display result"""
        label, confidence, img = self.predict(image_path)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"{label} - {confidence*100:.2f}% confident", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        return label, confidence

def main():
    parser = argparse.ArgumentParser(description='Predict dog or cat in image')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='./model/dogs_vs_cats_mobilenetv2.keras')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        return
    
    predictor = Predictor(args.model)
    label, confidence = predictor.predict_and_display(args.image)
    
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
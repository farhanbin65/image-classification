"""
Fashion MNIST - Prediction Script
Predict fashion class from a single image
Author: Keo (Farhan Bin Hossain)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse
import os

CLASS_NAMES = [
    'T-shirt/top', 'Trouser',  'Pullover', 'Dress',     'Coat',
    'Sandal',      'Shirt',    'Sneaker',  'Bag', 'Ankle boot'
]

def preprocess_image(image_path):
    """Load and preprocess image for CNN"""
    img = Image.open(image_path).convert('L')  # convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img, img_array

def predict(model_path, image_path):
    """Make prediction on single image"""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess
    img, img_array = preprocess_image(image_path)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx    = np.argmax(predictions)
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = predictions[pred_idx]
    
    # Display result
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Fashion MNIST Prediction', fontsize=14, fontweight='bold')
    
    # Show image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Prediction: {pred_class}\n'
                      f'Confidence: {confidence*100:.2f}%',
                      fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    # Show probability bar chart
    colors = ['red' if i == pred_idx else 'steelblue' 
              for i in range(10)]
    axes[1].barh(CLASS_NAMES, predictions, color=colors)
    axes[1].set_xlabel('Probability')
    axes[1].set_title('Class Probabilities', fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Predicted class: {pred_class}")
    print(f"Confidence:      {confidence*100:.2f}%")
    print("\nAll probabilities:")
    for cls, prob in zip(CLASS_NAMES, predictions):
        bar = '#' * int(prob * 30)
        print(f"  {cls:<15} {prob*100:>6.2f}% {bar}")
    
    return pred_class, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict fashion item from image'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to image file'
    )
    parser.add_argument(
        '--model', type=str,
        default='./model/cnn_best.keras',
        help='Path to model file'
    )
    args = parser.parse_args()
    
    predict(args.model, args.image)
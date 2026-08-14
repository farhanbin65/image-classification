"""
Skin Disease Classification - Prediction Script
Predict skin disease from a single image
Author: Keo (Farhan Bin Hossain)
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import argparse
import os

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_FULL  = [
    'Melanoma',
    'Melanocytic nevi',
    'Basal cell carcinoma',
    'Actinic keratoses',
    'Benign keratosis',
    'Dermatofibroma',
    'Vascular lesions'
]

# Risk levels for each class
RISK = {
    'MEL':   'HIGH - Malignant melanoma. Seek immediate medical attention!',
    'NV':    'LOW - Benign mole. Monitor for changes.',
    'BCC':   'MEDIUM - Basal cell carcinoma. Consult a dermatologist.',
    'AKIEC': 'MEDIUM - Precancerous lesion. Consult a dermatologist.',
    'BKL':   'LOW - Benign keratosis. Usually harmless.',
    'DF':    'LOW - Benign fibroma. Usually harmless.',
    'VASC':  'LOW - Vascular lesion. Usually benign.'
}

def predict(model_path, image_path):
    """Predict skin disease from image"""

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess image
    img       = Image.open(image_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    pred_idx    = np.argmax(predictions)
    pred_cls    = CLASS_NAMES[pred_idx]
    pred_full   = CLASS_FULL[pred_idx]
    confidence  = predictions[pred_idx]

    # Display
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Skin Disease Prediction', fontsize=14, fontweight='bold')

    axes[0].imshow(img)
    axes[0].set_title(
        f"Prediction: {pred_cls} - {pred_full}\n"
        f"Confidence: {confidence*100:.2f}%",
        fontweight='bold', fontsize=11
    )
    axes[0].axis('off')

    # Bar chart
    colors = ['#e74c3c' if i == pred_idx else '#3498db'
              for i in range(len(CLASS_NAMES))]
    axes[1].barh(CLASS_NAMES, predictions, color=colors)
    axes[1].set_xlabel('Probability')
    axes[1].set_title('Class Probabilities', fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Class:      {pred_cls} - {pred_full}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Risk:       {RISK[pred_cls]}")
    print("\nAll probabilities:")
    for cls, full, prob in zip(CLASS_NAMES, CLASS_FULL, predictions):
        bar = '#' * int(prob * 30)
        print(f"  {cls:<6} {prob*100:>6.2f}% {bar}")

    print("\nDISCLAIMER: This is an AI tool for educational purposes only.")
    print("Always consult a qualified dermatologist for medical diagnosis.")

    return pred_cls, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict skin disease from image'
    )
    parser.add_argument('--image', type=str, required=True,
                        help='Path to skin lesion image')
    parser.add_argument('--model', type=str,
                        default='./model/skin_disease_mobilenetv2.keras',
                        help='Path to model file')
    args = parser.parse_args()

    predict(args.model, args.image)
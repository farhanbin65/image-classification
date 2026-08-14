"""
Skin Disease Classification - Evaluation Script
Author: Keo (Farhan Bin Hossain)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_FULL  = [
    'Melanoma', 'Melanocytic nevi', 'Basal cell carcinoma',
    'Actinic keratoses', 'Benign keratosis',
    'Dermatofibroma', 'Vascular lesions'
]

def load_test_data(csv_path, img_dir, target=500, nv_cap=1500):
    """Load and balance test data"""

    df = pd.read_csv(csv_path)
    df['path']      = df['image'].apply(
        lambda x: os.path.join(img_dir, x + '.jpg')
    )
    df['label']     = df[CLASS_NAMES].values.argmax(axis=1)
    df['label_str'] = df['label'].astype(str)

    # Balance
    dfs = []
    for i, cls in enumerate(CLASS_NAMES):
        class_df = df[df['label'] == i]
        if cls == 'NV':
            dfs.append(class_df.sample(n=nv_cap, random_state=42))
        elif len(class_df) < target:
            dfs.append(class_df.sample(n=target, replace=True, random_state=42))
        else:
            dfs.append(class_df)

    balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    _, test_df = train_test_split(
        balanced, test_size=0.2, random_state=42, stratify=balanced['label']
    )

    datagen = ImageDataGenerator(rescale=1.0/255)
    test_gen = datagen.flow_from_dataframe(
        test_df.reset_index(drop=True),
        x_col='path',
        y_col='label_str',
        target_size=(128, 128),
        batch_size=32,
        class_mode='sparse',
        shuffle=False
    )

    return test_gen

def evaluate(model_path, csv_path, img_dir):
    """Full model evaluation"""

    print("Skin Disease Classifier - Evaluation")
    print("=" * 60)

    model    = tf.keras.models.load_model(model_path)
    test_gen = load_test_data(csv_path, img_dir)

    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    test_gen.reset()
    preds      = model.predict(test_gen, verbose=0)
    pred_class = np.argmax(preds, axis=1)
    true_class = np.array(test_gen.classes).astype(int)

    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")

    report = classification_report(
        true_class, pred_class,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )
    print(f"Macro F1:      {report['macro avg']['f1-score']:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        true_class, pred_class,
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(true_class, pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix ({test_acc*100:.2f}%)', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('./results/confusion_matrix_eval.png', dpi=300)
    plt.show()

    # Most dangerous confusions
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    print("\nTop confusions (potential misdiagnoses):")
    for _ in range(5):
        idx        = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
        true_cls   = CLASS_NAMES[idx[0]]
        pred_cls   = CLASS_NAMES[idx[1]]
        count      = cm_copy[idx]
        danger     = "(CRITICAL!)" if true_cls == 'MEL' else ""
        print(f"  {true_cls} mistaken as {pred_cls}: {count} times {danger}")
        cm_copy[idx] = 0

if __name__ == "__main__":
    evaluate(
        model_path='./model/skin_disease_mobilenetv2.keras',
        csv_path='./data/GroundTruth.csv',
        img_dir='./data/images'
    )
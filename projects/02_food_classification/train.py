"""
Fast Food Classification (4-class) - Training Script
Transfer Learning with MobileNetV2
Classes: Burger, Crispy Chicken, Fries, Pizza
Author: Keo (Farhan Bin Hossain)
COM672 - Computer Vision & AI
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import json

# ==================== CONFIGURATION ====================
TRAIN_DIR   = "./Dataset/Train"
VALID_DIR   = "./Dataset/Valid"
TEST_DIR    = "./Dataset/Test"
NOISY_DIR   = "./Dataset/Noisy Test"

IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
NUM_CLASSES = 4

os.makedirs('./model', exist_ok=True)
os.makedirs('./results', exist_ok=True)

print("FAST FOOD CLASSIFIER - MobileNetV2 Transfer Learning")
print("=" * 60)
print(f"Classes: Burger, Crispy Chicken, Fries, Pizza")
print(f"Image Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}\n")

# ==================== TASK 1: BASIC CLASSIFIER ====================
print("=" * 60)
print("TASK 1: Basic Classifier (No Augmentation)")
print("=" * 60 + "\n")

basic_datagen = ImageDataGenerator(rescale=1.0/255)

train_basic = basic_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_data = basic_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print(f"\n Train: {train_basic.samples} images")
print(f" Valid: {valid_data.samples} images")
print(f" Classes: {train_basic.class_indices}\n")

# Build basic model
base_m1 = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_m1.trainable = False

x = base_m1.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

basic_model = Model(inputs=base_m1.input, outputs=out)
basic_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print(" Training Basic Model...\n")
history_basic = basic_model.fit(
    train_basic,
    epochs=5,
    validation_data=valid_data,
    verbose=1
)

basic_val_acc = max(history_basic.history['val_accuracy'])
print(f"\n Task 1 Best Validation Accuracy: {basic_val_acc*100:.2f}%")

# ==================== TASK 2: IMPROVED CLASSIFIER ====================
print("\n" + "=" * 60)
print("TASK 2: Improved Classifier (Augmentation + Fine-tuning)")
print("=" * 60 + "\n")

improved_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

train_improved = improved_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Build improved model
base_m2 = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
base_m2.trainable = False

x2 = base_m2.output
x2 = GlobalAveragePooling2D()(x2)
x2 = Dense(256, activation="relu")(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.4)(x2)
x2 = Dense(128, activation="relu")(x2)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.3)(x2)
out2 = Dense(NUM_CLASSES, activation="softmax")(x2)

improved_model = Model(inputs=base_m2.input, outputs=out2)

# Phase 1
print(" Phase 1: Feature Extraction...\n")
improved_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_p1 = improved_model.fit(
    train_improved,
    epochs=5,
    validation_data=valid_data,
    verbose=1
)

# Phase 2
print("\n Phase 2: Fine-tuning...\n")
base_m2.trainable = True
for layer in base_m2.layers[:-20]:
    layer.trainable = False

improved_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_p2 = improved_model.fit(
    train_improved,
    epochs=5,
    validation_data=valid_data,
    verbose=1
)

improved_val_acc = max(
    history_p1.history['val_accuracy'] +
    history_p2.history['val_accuracy']
)
print(f"\n Task 2 Best Validation Accuracy: {improved_val_acc*100:.2f}%")

# ==================== TASK 3: CLEAN TEST ====================
print("\n" + "=" * 60)
print("TASK 3: Evaluation on Clean Test Set")
print("=" * 60 + "\n")

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_loss, test_acc = improved_model.evaluate(test_data, verbose=0)

test_data.reset()
preds = improved_model.predict(test_data, verbose=0)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

print(f" Clean Test Accuracy: {test_acc*100:.2f}%")
print(f" Clean Test Loss:     {test_loss:.4f}")
print(f"\nClassification Report (Clean):")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

# Clean Confusion Matrix
cm_clean = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix - Clean Test ({test_acc*100:.2f}%)', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('./results/food_cm_clean.png', dpi=300)
plt.close()
print(" Clean confusion matrix saved!")

# ==================== TASK 4: NOISY TEST ====================
print("\n" + "=" * 60)
print("TASK 4: Noisy Data Testing")
print("=" * 60 + "\n")

noisy_data = test_datagen.flow_from_directory(
    NOISY_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

noisy_loss, noisy_acc = improved_model.evaluate(noisy_data, verbose=0)

noisy_data.reset()
noisy_preds = improved_model.predict(noisy_data, verbose=0)
noisy_pred_classes = np.argmax(noisy_preds, axis=1)
noisy_true = noisy_data.classes

print(f" Noisy Test Accuracy: {noisy_acc*100:.2f}%")
print(f" Noisy Test Loss:     {noisy_loss:.4f}")
print(f"\nClassification Report (Noisy):")
print(classification_report(noisy_true, noisy_pred_classes, target_names=class_labels))

# Noisy Confusion Matrix
cm_noisy = confusion_matrix(noisy_true, noisy_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_noisy, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'Confusion Matrix - Noisy Test ({noisy_acc*100:.2f}%)', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('./results/food_cm_noisy.png', dpi=300)
plt.close()
print(" Noisy confusion matrix saved!")

# ==================== TRAINING HISTORY ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Food Classifier - Training History', fontsize=16, fontweight='bold')

all_acc      = history_p1.history['accuracy']     + history_p2.history['accuracy']
all_val_acc  = history_p1.history['val_accuracy'] + history_p2.history['val_accuracy']
all_loss     = history_p1.history['loss']         + history_p2.history['loss']
all_val_loss = history_p1.history['val_loss']     + history_p2.history['val_loss']

axes[0].plot(all_acc,     label='Train',      linewidth=2.5, marker='o')
axes[0].plot(all_val_acc, label='Validation', linewidth=2.5, marker='s')
axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0].set_title('Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss,     label='Train',      linewidth=2.5, marker='o')
axes[1].plot(all_val_loss, label='Validation', linewidth=2.5, marker='s')
axes[1].axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1].set_title('Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/food_training_history.png', dpi=300)
plt.close()
print(" Training history saved!")

# ==================== SAVE METRICS ====================
metrics = {
    'task1_basic_val_acc': float(basic_val_acc),
    'task2_improved_val_acc': float(improved_val_acc),
    'task3_clean_test_acc': float(test_acc),
    'task3_clean_test_loss': float(test_loss),
    'task4_noisy_test_acc': float(noisy_acc),
    'task4_noisy_test_loss': float(noisy_loss),
    'performance_drop': float(test_acc - noisy_acc),
    'improvement': float(improved_val_acc - basic_val_acc)
}

with open('./results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ==================== SAVE MODEL ====================
improved_model.save('./model/food_classifier_mobilenetv2.keras')

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("📊 FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Task 1 - Basic Model (Val):    {basic_val_acc*100:.2f}%")
print(f"  Task 2 - Improved Model (Val): {improved_val_acc*100:.2f}%")
print(f"  Task 3 - Clean Test:           {test_acc*100:.2f}%")
print(f"  Task 4 - Noisy Test:           {noisy_acc*100:.2f}%")
print(f"\n  Performance drop: {(test_acc-noisy_acc)*100:.2f}%")
print(f"\n Task 5 - Model saved to: ./model/")
print(f" Task 5 - Results saved to: ./results/")
print("\n  ALL 4 TASKS COMPLETE!")
"""
Skin Disease Classification - HAM10000
Transfer Learning with MobileNetV2
7 classes: MEL, NV, BCC, AKIEC, BKL, DF, VASC
Author: Keo (Farhan Bin Hossain)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense,
                                      Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ==================== CONFIGURATION ====================
# Update these paths to match your local setup
IMG_DIR  = "./data/images"
CSV_PATH = "./data/GroundTruth.csv"

IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
EPOCHS      = 20
PATIENCE    = 5
TARGET      = 500   # minimum samples per class after balancing
NV_CAP      = 1500  # cap dominant NV class

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
NUM_CLASSES = 7

os.makedirs('./model',   exist_ok=True)
os.makedirs('./results', exist_ok=True)

print("Skin Disease Classification - HAM10000")
print("=" * 60)
print(f"Classes:    {NUM_CLASSES}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

# ==================== LOAD DATA ====================
print("\nLoading metadata...\n")

df = pd.read_csv(CSV_PATH)
df['path']       = df['image'].apply(
    lambda x: os.path.join(IMG_DIR, x + '.jpg')
)
df['label']      = df[CLASS_NAMES].values.argmax(axis=1)
df['class_name'] = df['label'].apply(lambda x: CLASS_NAMES[x])
df['label_str']  = df['label'].astype(str)

print("Original class distribution:")
for cls, full in zip(CLASS_NAMES, CLASS_FULL):
    count = (df['class_name'] == cls).sum()
    pct   = count / len(df) * 100
    bar   = '#' * int(pct / 2)
    print(f"  {cls:<6} ({full:<25}): {count:>5} ({pct:>5.1f}%) {bar}")

# ==================== VISUALIZE DISTRIBUTION ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('HAM10000 Class Distribution', fontsize=14, fontweight='bold')

before_counts = [df[df['label'] == i].shape[0] for i in range(NUM_CLASSES)]
axes[0].bar(CLASS_NAMES, before_counts, color='#e74c3c', alpha=0.8)
axes[0].set_title('Before Balancing', fontweight='bold')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(before_counts):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=8)

# After balancing counts (estimated)
after_counts = [min(c, NV_CAP) if cls == 'NV' else max(c, TARGET)
                for cls, c in zip(CLASS_NAMES, before_counts)]
axes[1].bar(CLASS_NAMES, after_counts, color='#2ecc71', alpha=0.8)
axes[1].set_title('After Balancing', fontweight='bold')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(after_counts):
    axes[1].text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig('./results/class_distribution.png', dpi=300)
plt.close()
print("\nClass distribution plot saved!")

# ==================== BALANCE DATASET ====================
print("\nBalancing dataset...\n")

dfs = []
for i, cls in enumerate(CLASS_NAMES):
    class_df = df[df['label'] == i]
    count    = len(class_df)

    if cls == 'NV':
        sampled = class_df.sample(n=NV_CAP, random_state=42)
        dfs.append(sampled)
        print(f"  {cls:<6}: {count:>5} → {NV_CAP} (undersampled)")
    elif count < TARGET:
        oversampled = class_df.sample(n=TARGET, replace=True, random_state=42)
        dfs.append(oversampled)
        print(f"  {cls:<6}: {count:>5} → {TARGET} (oversampled)")
    else:
        dfs.append(class_df)
        print(f"  {cls:<6}: {count:>5} (kept)")

balanced_df = pd.concat(dfs).sample(frac=1, random_state=42)
balanced_df = balanced_df.reset_index(drop=True)
print(f"\nBalanced dataset: {len(balanced_df)} images")

# ==================== TRAIN TEST SPLIT ====================
train_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=42,
    stratify=balanced_df['label']
)

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ==================== VISUALIZE SAMPLES ====================
print("\nVisualizing samples...\n")

fig, axes = plt.subplots(2, 7, figsize=(18, 6))
fig.suptitle('HAM10000 - Sample Images per Class',
             fontsize=14, fontweight='bold')

for class_idx, (cls, full) in enumerate(zip(CLASS_NAMES, CLASS_FULL)):
    samples = df[df['class_name'] == cls]

    for row in range(2):
        sample = samples.iloc[row]
        img    = plt.imread(sample['path'])
        axes[row, class_idx].imshow(img)
        axes[row, class_idx].set_title(
            cls if row == 0 else full,
            fontweight='bold' if row == 0 else 'normal',
            fontsize=8
        )
        axes[row, class_idx].axis('off')

plt.tight_layout()
plt.savefig('./results/sample_images.png', dpi=300)
plt.close()
print("Sample images saved!")

# ==================== DATA GENERATORS ====================
print("\nSetting up data generators...\n")

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col='path',
    y_col='label_str',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=True
)

test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col='path',
    y_col='label_str',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# ==================== BUILD MODEL ====================
print("\nBuilding MobileNetV2 model...\n")

base_model = MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
print(f"Total parameters: {model.count_params():,}")

# ==================== CALLBACKS ====================
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='./model/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        './results/training_log.csv',
        append=True
    )
]

# ==================== PHASE 1: FEATURE EXTRACTION ====================
print("\n" + "=" * 60)
print("PHASE 1: Feature Extraction (Base Frozen)")
print("=" * 60 + "\n")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    train_gen,
    epochs=10,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1
)

p1_best = max(history1.history['val_accuracy'])
print(f"\nPhase 1 Best: {p1_best*100:.2f}%")

# ==================== PHASE 2: FINE-TUNING ====================
print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning (Last 30 Layers)")
print("=" * 60 + "\n")

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=callbacks,
    verbose=1
)

# ==================== EVALUATE ====================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60 + "\n")

test_loss, test_acc = model.evaluate(test_gen, verbose=0)

test_gen.reset()
preds      = model.predict(test_gen, verbose=0)
pred_class = np.argmax(preds, axis=1)
true_class = np.array(test_gen.classes).astype(int)

print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss:     {test_loss:.4f}")

print("\nClassification Report:")
report = classification_report(
    true_class, pred_class,
    target_names=CLASS_NAMES,
    zero_division=0,
    output_dict=True
)
print(classification_report(
    true_class, pred_class,
    target_names=CLASS_NAMES,
    zero_division=0
))

# ==================== CONFUSION MATRIX ====================
cm = confusion_matrix(true_class, pred_class)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)
plt.title(f'Confusion Matrix - Skin Disease ({test_acc*100:.2f}%)',
          fontweight='bold', fontsize=14)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('./results/confusion_matrix.png', dpi=300)
plt.close()
print("Confusion matrix saved!")

# ==================== TRAINING HISTORY ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Skin Disease Classifier - Training History',
             fontsize=16, fontweight='bold')

all_acc      = history1.history['accuracy']     + history2.history['accuracy']
all_val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss     = history1.history['loss']         + history2.history['loss']
all_val_loss = history1.history['val_loss']     + history2.history['val_loss']

axes[0].plot(all_acc,     label='Train',      linewidth=2.5, marker='o')
axes[0].plot(all_val_acc, label='Validation', linewidth=2.5, marker='s')
axes[0].axvline(x=len(history1.history['accuracy']),
                color='red', linestyle='--', alpha=0.5,
                label='Fine-tuning starts')
axes[0].set_title('Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss,     label='Train',      linewidth=2.5, marker='o')
axes[1].plot(all_val_loss, label='Validation', linewidth=2.5, marker='s')
axes[1].axvline(x=len(history1.history['loss']),
                color='red', linestyle='--', alpha=0.5,
                label='Fine-tuning starts')
axes[1].set_title('Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results/training_history.png', dpi=300)
plt.close()
print("Training history saved!")

# ==================== SAVE METRICS ====================
metrics = {
    'test_accuracy':  round(float(test_acc) * 100, 2),
    'test_loss':      round(float(test_loss), 4),
    'macro_f1':       round(report['macro avg']['f1-score'], 4),
    'train_samples':  len(train_df),
    'test_samples':   len(test_df),
    'num_classes':    NUM_CLASSES,
    'class_names':    CLASS_NAMES,
    'class_full':     CLASS_FULL,
    'balancing': {
        'target_min': TARGET,
        'nv_cap':     NV_CAP,
        'method':     'oversample rare + undersample NV'
    },
    'per_class_f1': {
        cls: round(report[cls]['f1-score'], 4)
        for cls in CLASS_NAMES
    }
}

with open('./results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ==================== SAVE MODEL ====================
model.save('./model/skin_disease_mobilenetv2.keras')

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(f"Macro F1:      {report['macro avg']['f1-score']:.4f}")
print("\nFiles saved:")
print("  model/skin_disease_mobilenetv2.keras")
print("  results/confusion_matrix.png")
print("  results/training_history.png")
print("  results/class_distribution.png")
print("  results/sample_images.png")
print("  results/metrics.json")
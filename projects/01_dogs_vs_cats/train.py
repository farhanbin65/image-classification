"""
Dogs vs Cats Classification - Training Script
Transfer Learning with MobileNetV2
Author: Keo (Farhan Bin Hossain)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import json
from datetime import datetime

# ==================== CONFIGURATION ====================
class Config:
    """Configuration for training"""
    TRAIN_DIR = "./data/train"
    TEST_DIR = "./data/test"
    
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32
    
    PHASE1_EPOCHS = 5
    PHASE2_EPOCHS = 5
    PHASE2_LR = 0.0001
    
    MODEL_SAVE_PATH = "./model/dogs_vs_cats_mobilenetv2.keras"
    RESULTS_PATH = "./results/"
    
    def __post_init__(self):
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)

# ==================== MAIN TRAINING SCRIPT ====================
def main():
    print("🚀 TRANSFER LEARNING WITH MobileNetV2\n")
    print("=" * 60)
    print("Dogs vs Cats Classification")
    print("=" * 60 + "\n")
    
    config = Config()
    
    # ==================== 1. LOAD DATA ====================
    print("📥 Loading dataset...\n")
    
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    train_data = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="binary"
    )
    
    test_data = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )
    
    print(f"✅ Training samples: {train_data.samples}")
    print(f"✅ Testing samples: {test_data.samples}\n")
    
    # ==================== 2. BUILD MODEL ====================
    print("🔧 Loading MobileNetV2 pretrained model...\n")
    
    base_model = MobileNetV2(
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    print("✅ Model built!\n")
    
    # ==================== 3. PHASE 1: FEATURE EXTRACTION ====================
    print("=" * 60)
    print("📍 PHASE 1: Feature Extraction (Base Frozen)")
    print("=" * 60 + "\n")
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    history_phase1 = model.fit(
        train_data,
        epochs=config.PHASE1_EPOCHS,
        validation_data=test_data,
        verbose=1
    )
    
    print("\n✅ Phase 1 complete!\n")
    
    # ==================== 4. PHASE 2: FINE-TUNING ====================
    print("=" * 60)
    print("🔓 PHASE 2: Fine-tuning (Last 20 Layers Unfrozen)")
    print("=" * 60 + "\n")
    
    base_model.trainable = True
    
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.PHASE2_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    history_phase2 = model.fit(
        train_data,
        epochs=config.PHASE2_EPOCHS,
        validation_data=test_data,
        verbose=1
    )
    
    print("\n✅ Phase 2 complete!\n")
    
    # ==================== 5. SAVE MODEL ====================
    print(f"💾 Saving model to {config.MODEL_SAVE_PATH}...\n")
    model.save(config.MODEL_SAVE_PATH)
    print("✅ Model saved!\n")
    
    # ==================== 6. SAVE TRAINING HISTORY ====================
    history = {
        'phase1': {
            'accuracy': history_phase1.history['accuracy'],
            'val_accuracy': history_phase1.history['val_accuracy'],
            'loss': history_phase1.history['loss'],
            'val_loss': history_phase1.history['val_loss'],
        },
        'phase2': {
            'accuracy': history_phase2.history['accuracy'],
            'val_accuracy': history_phase2.history['val_accuracy'],
            'loss': history_phase2.history['loss'],
            'val_loss': history_phase2.history['val_loss'],
        }
    }
    
    with open(f"{config.RESULTS_PATH}training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"✅ Training history saved!\n")
    
    # ==================== 7. VISUALIZE ====================
    print("📊 Generating visualizations...\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Transfer Learning - MobileNetV2 Training', fontsize=16, fontweight='bold')
    
    all_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
    all_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
    all_loss = history_phase1.history['loss'] + history_phase2.history['loss']
    all_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']
    
    # Accuracy
    axes[0].plot(all_acc, label='Train', linewidth=2.5, marker='o')
    axes[0].plot(all_val_acc, label='Validation', linewidth=2.5, marker='s')
    axes[0].axvline(x=config.PHASE1_EPOCHS, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
    axes[0].set_title('Accuracy', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(all_loss, label='Train', linewidth=2.5, marker='o')
    axes[1].plot(all_val_loss, label='Validation', linewidth=2.5, marker='s')
    axes[1].axvline(x=config.PHASE1_EPOCHS, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
    axes[1].set_title('Loss', fontweight='bold')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.RESULTS_PATH}training_history.png", dpi=300, bbox_inches='tight')
    print(f"✅ Visualizations saved!\n")
    
    print("=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
# 🍕🍔🍟🍗 Fast Food Classification

**COM672 Computer Vision & AI - CW2 Mock Assessment**

Transfer Learning with MobileNetV2 - 4 Class Food Classifier

## 📊 Results

| Task | Description | Accuracy |
|------|-------------|----------|
| Task 1 | Basic Classifier (no augmentation) | 89.00% |
| Task 2 | Improved (augmentation + fine-tuning) | 84.50% |
| Task 3 | Clean Test Set Evaluation | **88.00%** |
| Task 4 | Noisy Test Set | 78.50% |

**Performance drop (Clean → Noisy): 9.50%**

## 🏗️ Architecture
MobileNetV2 (Pretrained ImageNet)
↓
GlobalAveragePooling2D
↓
Dense(256, relu) + BatchNorm + Dropout(0.4)
↓
Dense(128, relu) + BatchNorm + Dropout(0.3)
↓
Dense(4, softmax) ← Burger | Crispy Chicken | Fries | Pizza

## 🎯 Improvements Applied (Task 2)

- ✅ Data Augmentation (rotation, flip, zoom, brightness)
- ✅ Batch Normalization (stable training)
- ✅ Dropout regularization (prevent overfitting)
- ✅ Fine-tuning (last 20 layers, LR=0.0001)

## 📈 Per-Class Performance (Clean Test)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Burger | 96% | 86% | 91% |
| Crispy Chicken | 81% | 92% | 86% |
| Fries | 89% | 80% | 84% |
| Pizza | 89% | 94% | 91% |

## 🔍 Noisy Test Analysis

The model dropped 9.5% on noisy data because:
- Fries: hardest to identify under noise (texture-based features disrupted)
- Pizza: most robust (distinctive circular shape maintained)
- Noise types: blur, brightness, rotation, occlusion

## 🚀 Quick Start

```bash
# Train model
python train.py

# Evaluate model
python evaluate.py

# Predict single image
python predict.py --image path/to/food.jpg
```

## 📂 Dataset

- **Train:** 600 images (150 per class)
- **Valid:** 200 images (50 per class)
- **Test:** 200 images (50 per class)
- **Noisy Test:** 200 images (50 per class)

Source: Fast Food Classification Dataset v2 (Kaggle)

## 📄 Results

![Training History](results/food_training_history.png)
![Confusion Matrix Clean](results/food_cm_clean.png)
![Confusion Matrix Noisy](results/food_cm_noisy.png)

---

**Author:** Farhan Bin Hossain
**Module:** COM672 - Computer Vision & AI  
**GitHub:** [@farhanbin65](https://github.com/farhanbin65)
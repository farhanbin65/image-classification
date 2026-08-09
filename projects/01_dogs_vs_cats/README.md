# 🐕🐈 Dogs vs Cats Classification

Transfer Learning with MobileNetV2

## 📊 Results

- **Test Accuracy:** 96.36%
- **Test Loss:** 0.1494
- **Training Time:** ~20 minutes (GPU)

## 🏆 Metrics

### Classification Report
precision    recall  f1-score   support

     Cat       0.95      0.98      0.96      2500
     Dog       0.98      0.94      0.96      2500

accuracy                           0.96      5000

macro avg 0.96 0.96 0.96 5000
weighted avg 0.96 0.96 0.96 5000

### Confusion Matrix
Predicted
Actual Cat Dog
Cat 2461 39
Dog 143 2357

## 🏗️ Architecture
MobileNetV2 (Pretrained - ImageNet weights)
↓
GlobalAveragePooling2D
↓
Dense(128, relu)
↓
Dense(1, sigmoid) ← Binary Classification

## 🚀 Quick Start

### Training
```bash
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Prediction on Single Image
```bash
python predict.py --image path/to/image.jpg
```

## 📦 Requirements
tensorflow>=2.13.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pillow>=9.0.0

## 📂 Dataset

- **Training:** 20,000 images (10k dogs, 10k cats)
- **Testing:** 5,000 images (2.5k dogs, 2.5k cats)
- **Image Size:** 128 × 128 pixels
- **Format:** JPEG

Source: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogsvscats)

## 🎓 Training Strategy

### Phase 1: Feature Extraction (5 epochs)
- Freeze MobileNetV2 base
- Train only custom head layers
- Learning rate: 0.001 (Adam)
- Result: 96.18% accuracy

### Phase 2: Fine-tuning (5 epochs)
- Unfreeze last 20 layers
- Train entire model
- Learning rate: 0.0001 (Adam)
- Result: 96.36% accuracy

## 📈 Training History

![Training History](../results/training_history.png)

## 🔍 Confusion Matrix

![Confusion Matrix](../results/confusion_matrix.png)

## 📝 Key Insights

✅ **High Accuracy:** 96.36% on unseen test data
✅ **Balanced Performance:** 95% precision (cats), 98% precision (dogs)
✅ **Quick Training:** Only 20 minutes on GPU
✅ **Transfer Learning Benefits:** Pre-trained weights accelerate learning

## 🛠️ Technologies

- **Framework:** TensorFlow/Keras
- **Model:** MobileNetV2 (Transfer Learning)
- **Optimization:** Adam
- **Loss:** Binary Cross-Entropy
- **Language:** Python 3.10+

## 📄 License

MIT License

---

**Author:** Keo (Farhan Bin Hossain)  
**Date:** August 2026  
**GitHub:** [@farhanbin65](https://github.com/farhanbin65)
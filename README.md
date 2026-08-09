x
x
B# Image Classification Portfolio 🎯

A comprehensive collection of **deep learning image classification projects** 
using **Transfer Learning** with MobileNetV2.

## 🎓 Overview

This portfolio demonstrates **production-ready** AI/ML solutions across multiple 
domains with **90%+ accuracy** on all projects.

**Author:** Keo (Farhan Bin Hossain)  
**GitHub:** [@farhanbin65](https://github.com/farhanbin65)  
**Portfolio:** [farhanbin.dev](https://farhanbin.dev)

---

## 📊 Projects

### 1. 🐕🐈 Dogs vs Cats Classification
**Status:** ✅ Complete  
**Accuracy:** 96.36%  
**Classes:** 2 (Binary)  
**Dataset:** 25,000 images

- Feature Extraction + Fine-tuning
- Confusion Matrix: 98% precision (Dogs), 98% recall (Cats)
- [View Project →](./projects/01_dogs_vs_cats)

  <img width="1382" height="593" alt="image" src="https://github.com/user-attachments/assets/1a544275-d167-426e-9cfe-eb002a00bcfe" />

<img width="1389" height="495" alt="image" src="https://github.com/user-attachments/assets/99761dd0-86eb-4a20-b99a-9c36aa93bf72" />


### 2. 🍕🍔🍟🍗 Food Classification
**Status:** ✅ Complete  
**Accuracy:** 88.00% (Clean) | 78.50% (Noisy)  
**Classes:** 4 (Burger, Crispy Chicken, Fries, Pizza)  
**Dataset:** 800 images (600 train, 200 test)

- Multi-class Transfer Learning (MobileNetV2)
- Two-phase training: Feature Extraction + Fine-tuning
- Data Augmentation (rotation, flip, zoom, brightness)
- Noisy data testing (blur, brightness, rotation, occlusion)
- [View Project →](./projects/02_food_classification)

<img width="1589" height="788" alt="image" src="https://github.com/user-attachments/assets/e8111bb3-e924-4578-a634-8635ce9eac2c" />

 <img width="1390" height="495" alt="image" src="https://github.com/user-attachments/assets/2e65d584-1b83-4d4c-a7e7-24fbf0df8508" />



### 3. 🐦 Bird Species Recognition
**Status:** 📋 Planned  
**Classes:** 50+

### 4. 🌾 Plant Disease Detection
**Status:** 📋 Planned  
**Classes:** 10+

---

## 🛠️ Tech Stack

- **Framework:** TensorFlow/Keras
- **Model:** MobileNetV2 (Transfer Learning)
- **Language:** Python 3.10+
- **GPU:** NVIDIA CUDA (Colab)

### Requirements
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pillow>=9.0.0

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/farhanbin65/image-classification-portfolio
cd image-classification-portfolio

# Install dependencies
pip install -r requirements.txt

# Train Dogs vs Cats model
cd projects/01_dogs_vs_cats
python train.py

# Make predictions
python predict.py --image path/to/image.jpg
```

---

## 📈 Results Summary

| Project | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Dogs vs Cats | 96.36% | 0.96 | 0.96 | 0.96 |
| Food (Coming) | - | - | - | - |
| Birds (Coming) | - | - | - | - |

---

## 🎯 Key Features

✅ **Production-Ready Code**
- Clean, modular architecture
- Comprehensive error handling
- Full documentation

✅ **Transfer Learning**
- Pre-trained MobileNetV2
- Phase 1: Feature extraction
- Phase 2: Fine-tuning

✅ **Evaluation Metrics**
- Confusion matrices
- Classification reports
- Training visualizations

✅ **Deployment Ready**
- Saved models (.keras format)
- Inference scripts
- REST API compatible

---

## 📚 Learning Resources

Each project includes:
- Jupyter notebooks with explanations
- Step-by-step training guides
- Feature visualization
- Model interpretation

---

## 🤝 Connect With Me

- **GitHub:** [@farhanbin65](https://github.com/farhanbin65)
- **Portfolio:** [farhanbin.dev](https://farhanbin.dev)
- **Email:** farhanbin65@gmail.com
- **LinkedIn:** [farhanbin](https://www.linkedin.com/in/farhanbin/)

---

## 📄 License

MIT License - Feel free to use for learning and projects!

---

## ✨ Acknowledgments

- TensorFlow/Keras team
- Kaggle datasets
- Transfer Learning techniques

**Remember:** *"Knowledge shared is knowledge multiplied."* 🤲

---

*Last Updated: August 2026*

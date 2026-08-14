# Image Classification Portfolio 

A comprehensive collection of **deep learning image classification projects** 
using **Transfer Learning** with MobileNetV2.

## Overview

This portfolio demonstrates **production-ready** AI/ML solutions across multiple 
domains with **90%+ accuracy** on all projects.

**Author:** Farhan Bin Hossain
**GitHub:** [@farhanbin65](https://github.com/farhanbin65)  
**Portfolio:** [farhanbin.dev](https://farhanbin.dev)

---
## Projects

### 1. Dogs vs Cats Classification
**Status:** Complete  
**Accuracy:** 96.74%  
**Classes:** 2 (Binary)  
**Dataset:** 25,000 images

- Transfer Learning with MobileNetV2
- Two-phase training: Feature Extraction + Fine-tuning
- Early stopping + Model checkpointing
- Confusion Matrix: 98% precision (Dogs), 98% recall (Cats)
- [View Project →](./projects/01_dogs_vs_cats)

<img width="1382" height="593" alt="image" src="https://github.com/user-attachments/assets/1a544275-d167-426e-9cfe-eb002a00bcfe" />

<img width="1389" height="495" alt="image" src="https://github.com/user-attachments/assets/99761dd0-86eb-4a20-b99a-9c36aa93bf72" />


---

### 2. Fast Food Classification
**Status:** Complete  
**Accuracy:** 88.00% (Clean) | 78.50% (Noisy)  
**Classes:** 4 (Burger, Crispy Chicken, Fries, Pizza)  
**Dataset:** 800 images (600 train, 200 test)

- Multi-class Transfer Learning (MobileNetV2)
- Two-phase training: Feature Extraction + Fine-tuning
- Data Augmentation (rotation, flip, zoom, brightness)
- Noisy data testing (blur, brightness, rotation, occlusion)
- Performance drop of 9.5% on noisy data
- [View Project →](./projects/02_food_classification)

<img width="1589" height="788" alt="image" src="https://github.com/user-attachments/assets/e8111bb3-e924-4578-a634-8635ce9eac2c" />

<img width="1390" height="495" alt="image" src="https://github.com/user-attachments/assets/2e65d584-1b83-4d4c-a7e7-24fbf0df8508" />

---

### 3. Fashion MNIST Classification
**Status:** Complete  
**Accuracy:** 94.42% (Custom CNN) | 93.24% (Transfer Learning)  
**Classes:** 10 fashion categories  
**Dataset:** 70,000 grayscale images (28x28 pixels)

- Compared Custom CNN vs Transfer Learning (MobileNetV2)
- Custom CNN won: smaller, faster, more accurate
- Key finding: Transfer learning not always better
- Hardest class: Shirt (80.2%) - visually similar to T-shirt
- Best class: Trouser (99.7%) - unique shape
- [View Project →](./projects/03_fashion_mnist)

![Sample Images](./projects/03_fashion_mnist/results/predictions.png)

![Confusion Matrix Comparison](./projects/03_fashion_mnist/results/comparison_confusion_matrix.png)

![Training History](./projects/03_fashion_mnist/results/comparison_history.png)

---

### 4. Skin Disease Classification (HAM10000)
**Status:** Complete  
**Accuracy:** 60.03% | **Macro F1:** 0.57  
**Classes:** 7 skin diseases  
**Dataset:** 10,015 dermatoscopic images  

- Medical AI with severe class imbalance (NV=67% of data)
- Oversampling rare classes + undersampling dominant class
- Accuracy dropped 68.75% → 60.03% but Macro F1 jumped 0.36 → 0.57
- Key insight: In medical AI, per-class F1 matters more than accuracy
- Missing rare cancer is far more dangerous than a false alarm
- [View Project →](./projects/04_skin_disease)

![Sample](./projects/04_skin_disease/results/sample.jpeg)

![Class Distribution](./projects/04_skin_disease/results/class_distribution.png)

![Confusion Matrix](./projects/04_skin_disease/results/confusion_matrix_balanced.png)

![Training History](./projects/04_skin_disease/results/training_history.png)

---

## Pipeline

```mermaid
flowchart TD
    A[Dataset\nLabelled images] --> C[Training pipeline\nMobileNetV2 transfer learning]
    B[Pretrained model\nImageNet weights] --> C

    C --> D[Phase 1: Feature extraction\nBase model frozen]
    C -.-> E[Callbacks\nEarly stopping · Checkpointing · LR reduction · CSV logger]

    D --> F[Phase 2: Fine-tuning\nLast 20 layers unfrozen]
    F --> G[Evaluate\nAccuracy · F1 · Confusion matrix]
    G --> H[Save model\n.keras · ready for deployment]
```


---

## Tech Stack

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

## Results Summary


| Project | Accuracy | Macro F1 | Classes | Key Technique |
|---------|----------|----------|---------|---------------|
| Dogs vs Cats | 96.74% | 0.96 | 2 | Transfer Learning |
| Food Classification | 88.00% | 0.88 | 4 | Transfer Learning + Noisy test |
| Fashion MNIST | 94.42% | 0.94 | 10 | Custom CNN vs Transfer Learning |
| Skin Disease | 60.03% | 0.57 | 7 | Class balancing + Medical AI |

---

## Key Features

**Production-Ready Code**
- Clean, modular architecture
- Comprehensive error handling
- Full documentation

**Transfer Learning**
- Pre-trained MobileNetV2
- Phase 1: Feature extraction
- Phase 2: Fine-tuning

**Evaluation Metrics**
- Confusion matrices
- Classification reports
- Training visualizations

**Deployment Ready**
- Saved models (.keras format)
- Inference scripts
- REST API compatible

---

## Learning Resources

Each project includes:
- Jupyter notebooks with explanations
- Step-by-step training guides
- Feature visualization
- Model interpretation

---

## Connect With Me

- **GitHub:** [@farhanbin65](https://github.com/farhanbin65)
- **Portfolio:** [farhanbin.dev](https://farhanbin.dev)
- **Email:** farhanbin65@gmail.com
- **LinkedIn:** [farhanbin](https://www.linkedin.com/in/farhanbin/)

---

## License

MIT License - Feel free to use for learning and projects!

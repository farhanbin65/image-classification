from google.colab import files

# Download the saved model
files.download('./models/dogs_vs_cats_mobilenetv2.keras')

# Generate and download results
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Confusion matrix plot
cm = np.array([[2461, 39], [143, 2357]])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cat', 'Dog'],
            yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix - Dogs vs Cats (96.36% Accuracy)', fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
files.download('confusion_matrix.png')

# Training history plot
all_acc = [0.9595, 0.9725, 0.9794, 0.9865, 0.9909,
           0.9647, 0.9871, 0.9916, 0.9937, 0.9962]
all_val_acc = [0.9578, 0.9636, 0.9642, 0.9666, 0.9618,
               0.9612, 0.9642, 0.9572, 0.9616, 0.9636]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MobileNetV2 Transfer Learning - Dogs vs Cats', 
             fontsize=16, fontweight='bold')

axes[0].plot(all_acc, label='Train', linewidth=2.5, marker='o')
axes[0].plot(all_val_acc, label='Validation', linewidth=2.5, marker='s')
axes[0].axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[0].set_title('Accuracy', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

all_loss = [0.1040, 0.0693, 0.0535, 0.0385, 0.0251,
            0.0935, 0.0364, 0.0245, 0.0195, 0.0110]
all_val_loss = [0.1089, 0.0924, 0.0835, 0.0873, 0.1140,
                0.1178, 0.1119, 0.1660, 0.1828, 0.1494]

axes[1].plot(all_loss, label='Train', linewidth=2.5, marker='o')
axes[1].plot(all_val_loss, label='Validation', linewidth=2.5, marker='s')
axes[1].axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Fine-tuning starts')
axes[1].set_title('Loss', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
files.download('training_history.png')

print("✅ All files downloaded!")
print("Now add them to your GitHub repo under:")
print("  projects/01_dogs_vs_cats/model/")
print("  projects/01_dogs_vs_cats/results/")
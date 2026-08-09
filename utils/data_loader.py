"""
Data Loading Module
Handles dataset loading and preprocessing for image classification
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

class DataLoader:
    """Load and preprocess image data"""
    
    def __init__(self, img_height=128, img_width=128, batch_size=32):
        """
        Initialize DataLoader
        
        Args:
            img_height (int): Image height in pixels
            img_width (int): Image width in pixels
            batch_size (int): Batch size for training
        """
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
    
    def load_binary_data(self, train_dir, test_dir, shuffle=True):
        """
        Load data for binary classification (e.g., Dogs vs Cats)
        
        Args:
            train_dir (str): Path to training directory
            test_dir (str): Path to testing directory
            shuffle (bool): Whether to shuffle data
            
        Returns:
            tuple: (train_data, test_data) generators
        """
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Testing directory not found: {test_dir}")
        
        train_datagen = ImageDataGenerator(rescale=1.0/255)
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        
        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=shuffle
        )
        
        test_data = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=False
        )
        
        return train_data, test_data
    
    def load_categorical_data(self, train_dir, test_dir, num_classes=4, shuffle=True):
        """
        Load data for multi-class classification (e.g., Food 4-class)
        
        Args:
            train_dir (str): Path to training directory
            test_dir (str): Path to testing directory
            num_classes (int): Number of classes
            shuffle (bool): Whether to shuffle data
            
        Returns:
            tuple: (train_data, test_data) generators
        """
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Testing directory not found: {test_dir}")
        
        train_datagen = ImageDataGenerator(rescale=1.0/255)
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        
        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=shuffle
        )
        
        test_data = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=False
        )
        
        return train_data, test_data
    
    def get_dataset_info(self, train_dir, test_dir):
        """
        Get information about datasets
        
        Args:
            train_dir (str): Path to training directory
            test_dir (str): Path to testing directory
            
        Returns:
            dict: Dataset information
        """
        train_path = Path(train_dir)
        test_path = Path(test_dir)
        
        train_count = len(list(train_path.rglob('*.jpg'))) + len(list(train_path.rglob('*.png')))
        test_count = len(list(test_path.rglob('*.jpg'))) + len(list(test_path.rglob('*.png')))
        
        classes = [d.name for d in train_path.iterdir() if d.is_dir()]
        
        return {
            'train_samples': train_count,
            'test_samples': test_count,
            'classes': classes,
            'num_classes': len(classes)
        }
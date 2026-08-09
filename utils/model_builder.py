"""
Model Builder Module
Build transfer learning models for image classification
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

class ModelBuilder:
    """Build transfer learning models"""
    
    def __init__(self, img_height=128, img_width=128, img_channels=3):
        """
        Initialize ModelBuilder
        
        Args:
            img_height (int): Image height
            img_width (int): Image width
            img_channels (int): Number of channels (3 for RGB)
        """
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
    
    def build_binary_model(self, freeze_base=True):
        """
        Build model for binary classification
        
        Args:
            freeze_base (bool): Freeze base model weights
            
        Returns:
            Model: Keras model
        """
        # Load pretrained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, self.img_channels),
            include_top=False,
            weights="imagenet"
        )
        
        base_model.trainable = not freeze_base
        
        # Build custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        return model, base_model
    
    def build_categorical_model(self, num_classes, freeze_base=True):
        """
        Build model for multi-class classification
        
        Args:
            num_classes (int): Number of output classes
            freeze_base (bool): Freeze base model weights
            
        Returns:
            Model: Keras model
        """
        # Load pretrained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, self.img_channels),
            include_top=False,
            weights="imagenet"
        )
        
        base_model.trainable = not freeze_base
        
        # Build custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        output = Dense(num_classes, activation="softmax")(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        return model, base_model
    
    def compile_model(self, model, learning_rate=0.001, model_type="binary"):
        """
        Compile model
        
        Args:
            model (Model): Keras model
            learning_rate (float): Learning rate for optimizer
            model_type (str): "binary" or "categorical"
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if model_type == "binary":
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
        else:
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        
        return model
    
    def unfreeze_layers(self, base_model, num_layers_to_unfreeze=20):
        """
        Unfreeze last N layers of base model
        
        Args:
            base_model (Model): Base model
            num_layers_to_unfreeze (int): Number of layers to unfreeze
        """
        base_model.trainable = True
        
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        return base_model
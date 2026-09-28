"""
This script trains a MobileNetV2 model on the Garbage Classification dataset
for 12 categories (15,150 images). It uses transfer learning and fine-tuning.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import seaborn as sns

# -- 1. Data Preparation --
# Dataset path (Kaggle: mostafaabla/garbage-classification)
DATASET_PATH = 'dataset/' 

# Image normalization and validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training Data Generator
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation Data Generator
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# -- 2. Model Architecture (Transfer Learning) --
# Base model pre-trained on ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers (as specified in the report)
base_model.trainable = False

# Add custom classification head for 12 classes
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# -- 3. Compilation --
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -- 4. Training --
print("Starting transfer learning on frozen base layers...")
history = model.fit(
    train_generator,
    epochs=15,  # Matches the epoch count in the technical report
    validation_data=validation_generator
)

# -- 5. Evaluation & Confusion Matrix --
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred)
class_names = list(train_generator.class_indices.keys())

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - MobileNetV2 Waste Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved.")

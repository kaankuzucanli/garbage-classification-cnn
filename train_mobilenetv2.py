"""
MobileNetV2 ile Garbage Classification veri seti üzerinde
transfer learning kullanarak 12 sınıflı görüntü sınıflandırma.
"""

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


# 1. Veri Hazırlama

DATASET_PATH = "dataset/"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


# 2. MobileNetV2 Modeli

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Hazır modelin katmanlarını dondur
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)

predictions = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)


# 3. Model Derleme

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# 4. Eğitim

print("MobileNetV2 eğitimi başlıyor...")

history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator
)


# 5. Confusion Matrix

validation_generator.reset()

Y_pred = model.predict(validation_generator)

y_pred = np.argmax(Y_pred, axis=1)
y_true = validation_generator.classes

class_names = list(
    validation_generator.class_indices.keys()
)

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix - MobileNetV2")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("confusion_matrix.png")

print("Confusion Matrix kaydedildi.")

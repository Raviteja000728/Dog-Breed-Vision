import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5

# Load breed labels
labels_csv = pd.read_csv("dog-breed-identification/labels.csv")
labels = np.array(labels_csv["breed"])
unique_breeds = np.unique(labels)
breed_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}

print(f"Found {len(unique_breeds)} dog breeds")

# Prepare file paths and labels
file_paths = []
label_indices = []

for idx, row in labels_csv.iterrows():
    file_id = row['id']
    breed = row['breed']
    file_path = f"dog-breed-identification/train/{file_id}.jpg"
    
    if Path(file_path).exists():
        file_paths.append(file_path)
        label_indices.append(breed_to_idx[breed])
    
    if (idx + 1) % 1000 == 0:
        print(f"Processed {idx + 1} entries...")

print(f"Found {len(file_paths)} valid image files")

# Convert to numpy arrays
file_paths = np.array(file_paths[:1000])  # Use subset for faster training
label_indices = np.array(label_indices[:1000])

# One-hot encode labels
y_onehot = tf.keras.utils.to_categorical(label_indices, num_classes=len(unique_breeds))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    file_paths, y_onehot, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} images")
print(f"Validation set: {len(X_val)} images")

# Data loading function
def load_and_process_image(image_path):
    """Load and preprocess image"""
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        return image
    except:
        # Return black image if loading fails
        return tf.zeros((IMG_SIZE, IMG_SIZE, 3))

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(100).map(
    lambda x, y: (load_and_process_image(x), y), num_parallel_calls=2
).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.map(
    lambda x, y: (load_and_process_image(x), y), num_parallel_calls=2
).batch(BATCH_SIZE)

print("\nBuilding model...")

# Create model using MobileNetV2 (lighter weight, faster)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(unique_breeds), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print(model.summary())

# Train model
print("\nTraining model...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    verbose=1
)

# Save model
print("\nSaving model...")
model.save("dog_breed_model.h5")
print("Model saved as dog_breed_model.h5")

# Save breed mapping
np.save("breed_labels.npy", unique_breeds)
print("Breed labels saved as breed_labels.npy")

print("\nTraining complete!")

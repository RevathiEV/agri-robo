# -*- coding: utf-8 -*-
"""
Tomato Disease Detection - Quick Training (10 images per class)
Uses existing train/ and val/ folders, limits to 10 images per class
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import glob

print("=" * 60)
print("Tomato Disease Detection - Quick Training")
print("Training on 10 images per class from existing folders")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(1337)

# Part 1: Build CNN architecture
print("\nBuilding CNN architecture...")

classifier = Sequential()

# First Conv Block
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Second Conv Block
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Third Conv Block
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Flatten
classifier.add(Flatten())

# Dense layers
classifier.add(Dense(units=256, activation='relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(rate=0.5))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(rate=0.3))

# Output layer
classifier.add(Dense(units=10, activation='softmax'))

# Compile model
classifier.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(classifier.summary())

# Part 2: Limit images per class in existing folders
print("\n" + "=" * 60)
print("Limiting to 10 images per class...")
print("=" * 60)

def limit_images_per_class(directory, max_images=10):
    """Limit number of images per class by keeping only first N images"""
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    for class_name in classes:
        class_path = os.path.join(directory, class_name)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(class_path, ext)))
        
        # Sort and keep only first N
        image_files.sort()
        if len(image_files) > max_images:
            # Remove excess images
            for img_path in image_files[max_images:]:
                try:
                    os.remove(img_path)
                except:
                    pass
            print(f"  {class_name}: Limited to {max_images} images (removed {len(image_files) - max_images})")
        else:
            print(f"  {class_name}: {len(image_files)} images (already ≤ {max_images})")

# Limit training images
print("\nLimiting training images...")
limit_images_per_class('train', max_images=10)

# Limit validation images
print("\nLimiting validation images...")
limit_images_per_class('val', max_images=5)  # 5 per class for validation

# Part 3: Data preparation with robust augmentation
print("\n" + "=" * 60)
print("Preparing data with robust augmentation...")
print("=" * 60)

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

# Validation data (only rescale)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
print("\nLoading training data...")
training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

label_map = training_set.class_indices
print(f"\nFound {len(label_map)} classes: {list(label_map.keys())}")

# Load validation data
print("\nLoading validation data...")
test_set = test_datagen.flow_from_directory(
    'val',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Calculate steps (will use all available images, which are now limited)
steps_per_epoch = len(training_set)
validation_steps = len(test_set)

print(f"\nTraining configuration:")
print(f"  - Training samples: {training_set.samples} (10 per class)")
print(f"  - Validation samples: {test_set.samples} (5 per class)")
print(f"  - Steps per epoch: {steps_per_epoch}")
print(f"  - Validation steps: {validation_steps}")
print(f"  - Batch size: 16")

# Part 4: Callbacks
print("\n" + "=" * 60)
print("Setting up training callbacks...")
print("=" * 60)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1,
    mode='min'
)

model_checkpoint = ModelCheckpoint(
    'tomato_disease_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

# Part 5: Train the model
print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)
print("Training on 10 images per class (100 total training images)")
print("This should complete in 15-30 minutes!")
print("=" * 60)

classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

# Part 6: Save model and class mapping
print("\n" + "=" * 60)
print("Saving model and class mapping...")
print("=" * 60)

classifier.save('tomato_disease_model.h5')
print('✓ Saved final model as tomato_disease_model.h5')

reverse_label_map = {v: k for k, v in label_map.items()}
with open('class_mapping.json', 'w') as f:
    json.dump(reverse_label_map, f, indent=4)

print('✓ Saved class mapping as class_mapping.json')
print(f'\nClass mapping ({len(reverse_label_map)} classes):')
for idx, class_name in sorted(reverse_label_map.items()):
    print(f"  {idx}: {class_name}")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)
print("Best model saved as: tomato_disease_model_best.h5")
print("Final model saved as: tomato_disease_model.h5")
print("=" * 60)
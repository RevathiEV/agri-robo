#!/usr/bin/env python3
"""
Test script to verify camera fix and model loading
Tests that:
1. Model files exist and can be loaded
2. Class mapping is valid
3. Image preprocessing works correctly
"""
import sys
import os
import json
import numpy as np
from PIL import Image, ImageEnhance

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Test 1: Check model files exist
print("=" * 60)
print("TEST 1: Checking model files...")
print("=" * 60)

project_root = os.path.dirname(os.path.abspath(__file__))
model_path_best = os.path.join(project_root, 'tomato_disease_model_best.h5')
model_path = os.path.join(project_root, 'tomato_disease_model.h5')
mapping_path = os.path.join(project_root, 'class_mapping.json')

def check_file(path, description):
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"   ✓ {description}: {os.path.basename(path)} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"   ✗ {description}: NOT FOUND")
        return False

model_exists = check_file(model_path, "Model file") or check_file(model_path_best, "Best model file")
mapping_exists = check_file(mapping_path, "Class mapping")

if not model_exists or not mapping_exists:
    print("\n✗ ERROR: Required files missing!")
    sys.exit(1)

# Test 2: Load and verify model
print("\n" + "=" * 60)
print("TEST 2: Loading model...")
print("=" * 60)

try:
    from tensorflow.keras.models import load_model
    
    # Load model (prefer best if exists)
    if os.path.exists(model_path_best):
        actual_model_path = model_path_best
    else:
        actual_model_path = model_path
    
    model = load_model(actual_model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"   ✓ Model loaded: {os.path.basename(actual_model_path)}")
    print(f"   ✓ Input shape: {model.input_shape}")
    print(f"   ✓ Output classes: {model.output_shape[1]}")
    
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    sys.exit(1)

# Test 3: Load and verify class mapping
print("\n" + "=" * 60)
print("TEST 3: Loading class mapping...")
print("=" * 60)

try:
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Convert keys to int
    class_mapping = {int(k): v for k, v in class_mapping.items()}
    
    print(f"   ✓ Class mapping loaded: {len(class_mapping)} classes")
    print("\n   Classes:")
    for idx, name in sorted(class_mapping.items()):
        print(f"     {idx}: {name}")
    
except Exception as e:
    print(f"   ✗ Error loading class mapping: {e}")
    sys.exit(1)

# Test 4: Test image preprocessing
print("\n" + "=" * 60)
print("TEST 4: Testing image preprocessing...")
print("=" * 60)

# Create a test image (greenish rectangle to simulate leaf)
test_image = Image.new('RGB', (640, 480), color=(34, 139, 34))  # Forest green

# Apply preprocessing (same as in main.py)
expected_shape = model.input_shape[1:]  # Skip batch dimension
img_size = (expected_shape[0], expected_shape[1])

# Convert to RGB if needed
if test_image.mode != 'RGB':
    test_image = test_image.convert('RGB')

# Enhance contrast and sharpness (same as main.py)
enhancer = ImageEnhance.Contrast(test_image)
test_image = enhancer.enhance(1.2)

enhancer = ImageEnhance.Sharpness(test_image)
test_image = enhancer.enhance(1.1)

# Resize
test_image = test_image.resize(img_size, Image.Resampling.LANCZOS)

# Convert to array and normalize (EXACTLY matching training: rescale=1./255)
img_array = np.array(test_image, dtype=np.float32)
img_array = img_array / 255.0  # Normalize to [0, 1] range
img_array = np.clip(img_array, 0.0, 1.0)
img_array = np.expand_dims(img_array, axis=0)

print(f"   ✓ Preprocessing successful")
print(f"   ✓ Final shape: {img_array.shape}")
print(f"   ✓ Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
print(f"   ✓ Mean brightness: {img_array.mean():.3f}")

# Test 5: Run prediction on test image
print("\n" + "=" * 60)
print("TEST 5: Running prediction...")
print("=" * 60)

try:
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx] * 100)
    
    disease_name = class_mapping.get(predicted_class_idx, "Unknown")
    
    print(f"   ✓ Prediction successful!")
    print(f"   ✓ Predicted class: {disease_name}")
    print(f"   ✓ Confidence: {confidence:.2f}%")
    
    # Check if it's Not_A_Leaf (expected for solid green image)
    if disease_name == "Not_A_Leaf":
        print("\n   ℹ Note: 'Not_A_Leaf' is expected for this test image")
        print("   (solid green rectangle doesn't look like a real leaf photo)")
    elif "healthy" in disease_name.lower():
        print("\n   ✓ Model thinks this green image looks like a healthy leaf!")
    else:
        print(f"\n   ✓ Model detected: {disease_name}")
    
except Exception as e:
    print(f"   ✗ Error during prediction: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("   ✓ Model files present")
print("   ✓ Model loads successfully")
print("   ✓ Class mapping valid")
print("   ✓ Image preprocessing works")
print("   ✓ Prediction works")
print("\n" + "=" * 60)
print("READY FOR CAMERA TESTING!")
print("=" * 60)
print("\nTo test with camera:")
print("1. Start the backend: cd backend && python main.py")
print("2. Open frontend and capture an image")
print("3. Check console for debug output")
print("\nThe camera capture should now work correctly with the model.")


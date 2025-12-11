from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.models import load_model
import json
import os
import io
from typing import Optional

app = FastAPI(title="Agri ROBO API", version="1.0.0")

# CORS middleware to allow React frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and class mapping
model = None
class_mapping = None

def load_model_and_mapping():
    """Load the disease detection model and class mapping"""
    global model, class_mapping
    
    # Get the project root directory (parent of backend folder)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    
    # Try both model files (best and regular)
    model_path_best = os.path.join(project_root, 'tomato_disease_model_best.h5')
    model_path = os.path.join(project_root, 'tomato_disease_model.h5')
    mapping_path = os.path.join(project_root, 'class_mapping.json')
    
    # Check which model file exists
    if os.path.exists(model_path_best):
        model_path = model_path_best
        print(f"Loading best model from: {model_path}")
    elif os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
    else:
        raise FileNotFoundError(
            f"Model file not found. Checked:\n"
            f"  - {model_path_best}\n"
            f"  - {model_path}\n"
            f"Please run cnn_train.py to generate the model."
        )
    
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Class mapping file not found: {mapping_path}\n"
            f"Please run cnn_train.py to generate the class mapping."
        )
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print(f"Model loaded successfully! Input shape: {model.input_shape}")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    class_mapping = {int(k): v for k, v in mapping.items()}
    
    print(f"Class mapping loaded: {len(class_mapping)} classes")
    print("Model and class mapping loaded successfully!")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    try:
        load_model_and_mapping()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but disease detection will not work until model is available.")

@app.get("/")
async def root():
    return {"message": "Agri ROBO API", "status": "running"}

@app.get("/health")
async def health_check():
    """Check API health and model status"""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    
    model_path_best = os.path.join(project_root, 'tomato_disease_model_best.h5')
    model_path = os.path.join(project_root, 'tomato_disease_model.h5')
    mapping_path = os.path.join(project_root, 'class_mapping.json')
    
    model_exists = os.path.exists(model_path) or os.path.exists(model_path_best)
    mapping_exists = os.path.exists(mapping_path)
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mapping_loaded": class_mapping is not None,
        "model_file_exists": model_exists,
        "mapping_file_exists": mapping_exists,
        "model_path_checked": [model_path, model_path_best],
        "mapping_path_checked": mapping_path,
        "num_classes": len(class_mapping) if class_mapping else 0
    }

@app.post("/api/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    """
    Detect disease from uploaded image using the trained CNN model.
    The model automatically detects the required input size (128x128 or 224x224).
    """
    if model is None or class_mapping is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files (tomato_disease_model.h5 and class_mapping.json) are available in the project root. Run cnn_train.py to generate them."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received content type: {file.content_type}"
        )
    
    try:
        # Read image file
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        image = Image.open(io.BytesIO(contents))
        
        # Validate image
        if image.size[0] == 0 or image.size[1] == 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")
        
        # Enhanced preprocessing for better accuracy
        # Get expected input size from model (supports both 128x128 and 224x224)
        expected_shape = model.input_shape[1:]  # Skip batch dimension
        img_size = (expected_shape[0], expected_shape[1])  # (height, width)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Image enhancement for better detection accuracy
        # Enhance contrast (helps with disease visibility)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        # Enhance sharpness (helps with edge detection)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Increase sharpness by 10%
        
        # Optional: Apply slight denoising (uncomment if images are noisy)
        # image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Resize image to match model input size (use high-quality resampling)
        image = image.resize(img_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize (matching training: rescale=1./255)
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1] range
        
        # Ensure values are in valid range [0, 1]
        img_array = np.clip(img_array, 0.0, 1.0)
        
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Verify shape matches model input
        if img_array.shape[1:] != expected_shape:
            raise ValueError(
                f"Image shape mismatch. Expected {expected_shape}, got {img_array.shape[1:]}"
            )
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Get disease name from mapping
        disease_name = class_mapping.get(predicted_class_idx, "Unknown")
        formatted_disease = disease_name.replace("Tomato___", "").replace("_", " ").title()
        
        # Get top 3 predictions
        top_predictions = []
        prediction_dict = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = class_mapping.get(idx, "Unknown")
            formatted_name = class_name.replace("Tomato___", "").replace("_", " ").title()
            prediction_dict[formatted_name] = float(prob * 100)
        
        sorted_predictions = sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)
        top_predictions = [{"name": name, "confidence": round(conf, 2)} for name, conf in sorted_predictions[:3]]
        
        is_healthy = "healthy" in disease_name.lower()
        
        return JSONResponse({
            "success": True,
            "disease": formatted_disease,
            "confidence": round(confidence, 2),
            "is_healthy": is_healthy,
            "top_predictions": top_predictions,
            "raw_disease_name": disease_name,
            "model_info": {
                "input_shape": str(model.input_shape),
                "num_classes": len(class_mapping)
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing image: {error_details}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing image: {str(e)}"
        )

# Placeholder endpoints for motor and servo control (for future implementation)
@app.post("/api/motor/control")
async def motor_control(direction: str):
    """
    Control robot motors (placeholder for future GPIO implementation)
    """
    valid_directions = ["front", "back", "left", "right", "stop"]
    if direction.lower() not in valid_directions:
        raise HTTPException(status_code=400, detail=f"Invalid direction. Must be one of: {valid_directions}")
    
    # TODO: Implement GPIO control when Raspberry Pi is available
    return {
        "success": True,
        "message": f"Motor command received: {direction}",
        "note": "GPIO control not yet implemented"
    }

@app.post("/api/servo/control")
async def servo_control(action: str):
    """
    Control servo motor for fertilizer (placeholder for future GPIO implementation)
    """
    valid_actions = ["start", "stop"]
    if action.lower() not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
    
    # TODO: Implement GPIO control when Raspberry Pi is available
    return {
        "success": True,
        "message": f"Servo command received: {action}",
        "note": "GPIO control not yet implemented"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


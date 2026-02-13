
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import os
import io
from typing import Optional
import sys
import threading
import time

# Try to import gpiozero for relay control (water pump) - Raspberry Pi only
try:
    from gpiozero import LED
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False
    LED = None
    print("Warning: gpiozero not available. Water pump control will be disabled.")

# GPIO pin for relay (water pump) - Pin 12 = GPIO 18
RELAY_GPIO_PIN = 18
relay = None

def initialize_relay():
    """Initialize relay at startup - relay stays OFF until disease is detected."""
    global relay
    if not GPIOZERO_AVAILABLE:
        relay = None
        return False
    try:
        # active_high=False for active-LOW relay (LOW=ON, HIGH=OFF) - common relay modules
        relay = LED(RELAY_GPIO_PIN, active_high=False)
        relay.off()
        print("✓ Water pump relay initialized - OFF at startup (will activate only when disease detected)")
        return True
    except Exception as e:
        print(f"Warning: Could not initialize relay: {e}. Water pump disabled.")
        relay = None
        return False

def activate_pump_for_duration(seconds=3):
    """Turn pump ON for specified duration, then OFF. Runs in background thread."""
    global relay
    if relay is None:
        return
    def run():
        try:
            relay.on()
            print(f"✓ Disease detected - Water pump ON for {seconds}s")
            time.sleep(seconds)
        finally:
            relay.off()
            print("✓ Water pump OFF")
    threading.Thread(target=run, daemon=True).start()

# Add system dist-packages to path for picamera2
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Try to import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Camera features will be disabled.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global relay
    
    # Startup
    try:
        load_model_and_mapping()
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("API will start but disease detection will not work until model is available.")
    
    # Initialize water pump relay - stays OFF until disease detected
    initialize_relay()
    if relay is not None:
        relay.off()
    
    yield
    
    # Shutdown - cleanup
    if relay is not None:
        try:
            relay.off()
            relay.close()
            print("✓ Water pump relay cleaned up on shutdown")
        except Exception as e:
            print(f"Warning: Error during relay cleanup: {e}")

app = FastAPI(title="Agri ROBO API", version="1.0.0", lifespan=lifespan)

# CORS middleware to allow React frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000",
        "http://10.222.54.41:3000",
        "http://10.222.54.41:5173",
        "http://10.86.141.41:3000",
        "http://10.86.141.41:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and class mapping
model = None
class_mapping = None

# Global variables for camera
camera = None
camera_streaming = False
camera_lock = threading.Lock()
current_frame = None

# Global variable for serial connection (Bluetooth)
serial_connection = None


def load_model_and_mapping():
    """Load the disease detection model and class mapping - TensorFlow 2.20.0 compatible"""
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
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Loading model from: {model_path}")
    
    # TensorFlow 2.20.0 compatible loading with compile=False
    try:
        print("Attempting to load model with compile=False...")
        model = load_model(model_path, compile=False)
        print("✓ Model loaded (compile=False)")
        
        print("Recompiling model...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✓ Model recompiled successfully")
        
    except Exception as e1:
        print(f"Warning: Could not load with compile=False: {e1}")
        print("Trying standard load method...")
        try:
            model = load_model(model_path)
            print("✓ Model loaded with standard method")
        except Exception as e2:
            raise RuntimeError(
                f"Could not load model with either method.\n"
                f"  compile=False error: {str(e1)[:200]}\n"
                f"  standard error: {str(e2)[:200]}\n"
                f"Model file may be corrupted or incompatible with TensorFlow {tf.__version__}"
            )
    
    print(f"Model loaded successfully! Input shape: {model.input_shape}")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    class_mapping = {int(k): v for k, v in mapping.items()}
    
    print(f"Class mapping loaded: {len(class_mapping)} classes")
    print("Model and class mapping loaded successfully!")


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
        "num_classes": len(class_mapping) if class_mapping else 0,
        "tensorflow_version": tf.__version__
    }


@app.post("/api/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    """
    Detect disease from uploaded image using the trained CNN model.
    """
    if model is None or class_mapping is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files (tomato_disease_model.h5 and class_mapping.json) are available in the project root."
        )


    
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
        
        # Get expected input size from model (supports both 128x128 and 224x224)
        expected_shape = model.input_shape[1:]  # Skip batch dimension
        img_size = (expected_shape[0], expected_shape[1])  # (height, width)
        
        # Convert to RGB if needed (CRITICAL for consistent detection)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get original image brightness for validation
        original_array = np.array(image)
        original_brightness = original_array.mean()
        
        # Image enhancement for better detection accuracy
        # Enhance contrast (helps with disease visibility)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)  # Increase contrast by 20%
        
        # Enhance sharpness (helps with edge detection)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)  # Increase sharpness by 10%
        
        # Resize image to match model input size (use high-quality resampling)
        image = image.resize(img_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize (EXACTLY matching training: rescale=1./255)
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0  # Normalize to [0, 1] range - THIS IS CRITICAL!
        
        # Ensure values are in valid range [0, 1]
        img_array = np.clip(img_array, 0.0, 1.0)
        
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Log preprocessing info for debugging
        print(f"DEBUG: Image preprocessing - original brightness: {original_brightness:.1f}, "
              f"final shape: {img_array.shape}, value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
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
        # Format disease name (handle Not_A_Leaf and Tomato___ classes)
        if disease_name == "Not_A_Leaf":
            formatted_disease = "Not A Leaf"
        else:
            formatted_disease = disease_name.replace("Tomato___", "").replace("_", " ").title()
        
        is_healthy = "healthy" in disease_name.lower()
        is_not_a_leaf = disease_name == "Not_A_Leaf"
        
        # Water pump: OFF for healthy/Not_A_Leaf, ON for 3s only when disease detected
        if not is_healthy and not is_not_a_leaf:
            activate_pump_for_duration(3)
        else:
            if relay is not None:
                try:
                    relay.off()
                except Exception:
                    pass
        
        return JSONResponse({
            "success": True,
            "disease": formatted_disease,
            "confidence": round(confidence, 2),
            "is_healthy": is_healthy,
            "is_not_a_leaf": is_not_a_leaf,
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


# Motor and servo control endpoints via Bluetooth serial
@app.post("/api/motor/control")
async def motor_control(direction: str):
    """
    Control robot motors via Bluetooth serial connection
    ESP32 expects single character commands: F (forward), B (back), L (left), R (right), S (stop)
    """
    global serial_connection
    valid_directions = ["front", "back", "left", "right", "stop"]
    if direction.lower() not in valid_directions:
        raise HTTPException(status_code=400, detail=f"Invalid direction. Must be one of: {valid_directions}")
    
    if serial_connection is None or not serial_connection.is_open:
        raise HTTPException(status_code=503, detail="Bluetooth serial connection not available")
    
    try:
        # Map direction to ESP32 single character command
        command_map = {
            "front": "F",
            "back": "B",
            "left": "L",
            "right": "R",
            "stop": "S"
        }
        cmd_char = command_map[direction.lower()]
        serial_connection.write(cmd_char.encode())
        return {
            "success": True,
            "message": f"Motor command sent: {direction} ({cmd_char})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending command: {str(e)}")


@app.post("/api/servo/control")
async def servo_control(action: str):
    """
    Control servo motor for fertilizer via Bluetooth serial connection
    ESP32 expects single character commands: A (start), X (stop)
    """
    global serial_connection
    valid_actions = ["start", "stop"]
    if action.lower() not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
    
    if serial_connection is None or not serial_connection.is_open:
        raise HTTPException(status_code=503, detail="Bluetooth serial connection not available")
    
    try:
        command_map = {
            "start": "A",
            "stop": "X"
        }
        cmd_char = command_map[action.lower()]
        serial_connection.write(cmd_char.encode())
        return {
            "success": True,
            "message": f"Servo command sent: {action} ({cmd_char})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending command: {str(e)}")


# ============================================
# Camera Endpoints
# ============================================

def convert_frame_to_rgb(frame):
    """Convert camera frame to RGB format - handles all cases including grayscale"""
    # Handle grayscale images (1 channel or 2D array)
    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
        if len(frame.shape) == 3:
            gray = frame[:, :, 0]
        else:
            gray = frame
        rgb_frame = np.stack([gray, gray, gray], axis=2)
        return rgb_frame
    
    # Handle RGBA/BGRA (4 channels) - convert to RGB
    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = frame[:, :, 2]  # R from channel 2
        rgb_frame[:, :, 1] = frame[:, :, 1]  # G from channel 1
        rgb_frame[:, :, 2] = frame[:, :, 0]  # B from channel 0
        return rgb_frame
    
    # Handle RGB (3 channels) - return as is
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame
    
    # Fallback - convert to RGB
    else:
        if len(frame.shape) == 3:
            frame = frame[:, :, :3]
        if len(frame.shape) == 3:
            return frame
        else:
            gray = frame.flatten()
            rgb = np.stack([gray, gray, gray], axis=1)
            return rgb.reshape(frame.shape[0], frame.shape[1], 3)


def process_frame(frame):
    """Apply minimal post-processing for natural color output"""
    rgb_frame = convert_frame_to_rgb(frame)
    mean_brightness = rgb_frame.mean()
    
    image = Image.fromarray(rgb_frame, 'RGB')
    
    # Minimal brightness correction only if very dark
    if mean_brightness < 30:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)
    
    return np.array(image)


def camera_capture_thread():
    """Background thread that continuously captures frames when streaming"""
    global camera, camera_streaming, current_frame
    
    while True:
        if camera_streaming and camera is not None:
            try:
                with camera_lock:
                    if camera is not None and camera_streaming:
                        request = camera.capture_request()
                        frame = request.make_array("main")
                        request.release()
                        
                        rgb_frame = process_frame(frame)
                        
                        image = Image.fromarray(rgb_frame, 'RGB')
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='JPEG', quality=85)
                        img_bytes.seek(0)
                        
                        frame_data = img_bytes.getvalue()
                        current_frame = frame_data
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Camera capture error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        else:
            time.sleep(0.1)


# Start camera capture thread
if PICAMERA2_AVAILABLE:
    camera_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    camera_thread.start()


@app.post("/api/camera/start")
async def start_camera():
    """Start Pi camera stream"""
    global camera, camera_streaming
    
    if not PICAMERA2_AVAILABLE:
        raise HTTPException(status_code=503, detail="picamera2 not available on this system")
    
    try:
        with camera_lock:
            if camera is None:
                try:
                    cam_info = Picamera2.global_camera_info()
                except Exception:
                    cam_info = []
                if not cam_info or len(cam_info) == 0:
                    raise HTTPException(
                        status_code=503,
                        detail="No camera detected. Connect a Pi camera or USB camera and try again."
                    )
                camera = Picamera2(camera_num=0)

                config = None
                for format_attempt in ["RGB888", "BGR888", "XRGB8888", "XBGR8888"]:
                    try:
                        config = camera.create_preview_configuration(
                            main={"size": (640, 480), "format": format_attempt},
                            colour_space="sRGB"
                        )
                        camera.configure(config)
                        print(f"   ✓ Camera configured with {format_attempt} format for leaf detection")
                        break
                    except Exception as e:
                        print(f"   Trying {format_attempt}: {e}")
                        continue
                
                if config is None:
                    config = camera.create_preview_configuration(main={"size": (640, 480)})
                    camera.configure(config)
                    print("   ⚠ Using default camera configuration")
                
                try:
                    camera.set_controls({
                        "AwbEnable": True,
                        "AeEnable": True,
                    })
                    print("   ✓ Camera controls set (auto white balance, auto exposure)")
                except:
                    print("   ⚠ Could not set camera controls, using defaults")
                
                camera.start()
                
                for i in range(5):
                    test_request = camera.capture_request()
                    test_request.release()
                    time.sleep(0.3)
            
            camera_streaming = True
            current_frame = None
            time.sleep(0.5)
        
        return {"success": True, "message": "Camera started"}
    
    except HTTPException:
        raise
    except IndexError as e:
        raise HTTPException(
            status_code=503,
            detail="No camera detected. Connect a Pi camera or USB camera and try again."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {str(e)}")


@app.get("/api/camera/stream")
async def camera_stream():
    """MJPEG stream endpoint for live video"""
    global camera_streaming, current_frame
    
    if not camera_streaming:
        raise HTTPException(status_code=400, detail="Camera not started. Call /api/camera/start first")
    
    def generate_frames():
        frame_count = 0
        while camera_streaming:
            wait_count = 0
            while current_frame is None and camera_streaming and wait_count < 100:
                time.sleep(0.01)
                wait_count += 1
            
            if current_frame:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
                    frame_count += 1
                except Exception as e:
                    print(f"Error yielding frame: {e}")
                    break
            else:
                time.sleep(0.1)
            
            time.sleep(0.033)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/camera/capture")
async def capture_image():
    """Capture fresh frame from camera and return as image - FIXED for proper leaf detection"""
    global camera, camera_streaming, current_frame
    
    if not camera_streaming or camera is None:
        raise HTTPException(status_code=400, detail="Camera not started or no frame available")
    
    try:
        with camera_lock:
            if camera is None:
                raise HTTPException(status_code=400, detail="Camera not initialized")
            
            request = camera.capture_request()
            frame = request.make_array("main")
            request.release()
            
            rgb_frame = convert_frame_to_rgb(frame)
            
            image = Image.fromarray(rgb_frame, 'RGB')
            mean_brightness = np.array(image).mean()
            
            if mean_brightness < 50:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.3)
            elif mean_brightness > 220:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(0.85)
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
            processed_frame = np.array(image)
            
            final_brightness = processed_frame.mean()
            if final_brightness < 40:
                image = Image.fromarray(processed_frame, 'RGB')
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.5)
                processed_frame = np.array(image)
            
            camera_streaming = False
            current_frame = None
            
            img_bytes = io.BytesIO()
            save_image = Image.fromarray(processed_frame, 'RGB')
            save_image.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            frame_data = img_bytes.getvalue()
            
            print(f"✓ Fresh frame captured (brightness: {final_brightness:.1f}, size: {len(frame_data)} bytes)")
            
            return Response(
                content=frame_data,
                media_type="image/jpeg",
                headers={"Content-Disposition": "inline; filename=captured.jpg"}
            )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")


@app.post("/api/camera/stop")
async def stop_camera():
    """Stop camera stream and release resources"""
    global camera, camera_streaming, current_frame
    
    try:
        camera_streaming = False
        current_frame = None
        
        with camera_lock:
            if camera is not None:
                camera.stop()
                camera.close()
                camera = None
        
        return {"success": True, "message": "Camera stopped"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {str(e)}")


if __name__ == "__main__":
    import uvicorn
@app.post("/api/camera/start")
async def start_camera():
    """Start Pi camera stream"""
    global camera, camera_streaming
    
    if not PICAMERA2_AVAILABLE:
        raise HTTPException(status_code=503, detail="picamera2 not available on this system")
    
    try:
        with camera_lock:
            if camera is None:
                # Check if any camera is detected before opening (avoids IndexError when no camera)
                try:
                    cam_info = Picamera2.global_camera_info()
                except Exception:
                    cam_info = []
                if not cam_info or len(cam_info) == 0:
                    raise HTTPException(
                        status_code=503,
                        detail="No camera detected. Connect a Pi camera or USB camera and try again."
                    )
                camera = Picamera2(camera_num=0)

                # Configure camera for COLOR output (fixes black & white issue)
                # CRITICAL: Use RGB888 format for proper leaf color detection
                config = None
                for format_attempt in ["RGB888", "BGR888", "XRGB8888", "XBGR8888"]:
                    try:
                        config = camera.create_preview_configuration(
                            main={"size": (640, 480), "format": format_attempt},
                            colour_space="sRGB"
                        )
                        camera.configure(config)
                        print(f"   ✓ Camera configured with {format_attempt} format for leaf detection")
                        break
                    except Exception as e:
                        print(f"   Trying {format_attempt}: {e}")
                        continue
                
                # If all format attempts failed, use default
                if config is None:
                    config = camera.create_preview_configuration(main={"size": (640, 480)})
                    camera.configure(config)
                    print("   ⚠ Using default camera configuration")
                
                # Set camera controls for optimal leaf imaging
                try:
                    camera.set_controls({
                        "AwbEnable": True,      # Auto white balance - critical for natural leaf colors
                        "AeEnable": True,       # Auto exposure - ensures proper brightness
                        # Don't force ExposureTime or AnalogueGain - let auto exposure handle it
                        # This matches old picamera behavior where iso=0 (auto) was default
                    })
                    print("   ✓ Camera controls set (auto white balance, auto exposure)")
                except:
                    print("   ⚠ Could not set camera controls, using defaults")
                
                camera.start()
                
                # Warm up camera
                for i in range(5):
                    test_request = camera.capture_request()
                    test_request.release()
                    time.sleep(0.3)
            
            camera_streaming = True
            # Reset frame buffer - wait a moment for first frame
            current_frame = None
            time.sleep(0.5)  # Give camera time to start capturing
        
        return {"success": True, "message": "Camera started"}
    
    except HTTPException:
        raise
    except IndexError as e:
        raise HTTPException(
            status_code=503,
            detail="No camera detected. Connect a Pi camera or USB camera and try again."
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {str(e)}")

@app.get("/api/camera/stream")
async def camera_stream():
    """MJPEG stream endpoint for live video"""
    global camera_streaming, current_frame
    
    if not camera_streaming:
        raise HTTPException(status_code=400, detail="Camera not started. Call /api/camera/start first")
    
    def generate_frames():
        frame_count = 0
        while camera_streaming:
            # Wait for frame to be available (with timeout)
            wait_count = 0
            while current_frame is None and camera_streaming and wait_count < 100:
                time.sleep(0.01)
                wait_count += 1
            
            if current_frame:
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
                    frame_count += 1
                except Exception as e:
                    print(f"Error yielding frame: {e}")
                    break
            else:
                # If no frame available, wait a bit longer
                time.sleep(0.1)
            
            time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no"  # Disable buffering for nginx if used
        }
    )

@app.post("/api/camera/capture")
async def capture_image():
    """Capture fresh frame from camera and return as image - FIXED for proper leaf detection"""
    global camera, camera_streaming, current_frame
    
    if not camera_streaming or camera is None:
        raise HTTPException(status_code=400, detail="Camera not started or no frame available")
    
    try:
        with camera_lock:
            if camera is None:
                raise HTTPException(status_code=400, detail="Camera not initialized")
            
            # CRITICAL FIX: Capture FRESH frame directly from camera
            # Don't use cached current_frame - it may be corrupted or wrong format
            request = camera.capture_request()
            frame = request.make_array("main")
            request.release()
            
            # Process frame immediately with proper color handling
            rgb_frame = convert_frame_to_rgb(frame)
            
            # Apply brightness/contrast optimization for leaf detection
            image = Image.fromarray(rgb_frame, 'RGB')
            
            # Enhance for better leaf visibility (matching training conditions)
            mean_brightness = np.array(image).mean()
            
            # If image is too dark or too bright, adjust it
            if mean_brightness < 50:
                # Boost brightness if too dark
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.3)
            elif mean_brightness > 220:
                # Reduce brightness if too washed out
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(0.85)
            
            # Enhance contrast for better leaf spot visibility
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
            # Convert back to numpy for final processing
            processed_frame = np.array(image)
            
            # Verify image quality
            final_brightness = processed_frame.mean()
            if final_brightness < 40:
                # Still too dark - apply more aggressive correction
                image = Image.fromarray(processed_frame, 'RGB')
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.5)
                processed_frame = np.array(image)
            
            # Stop streaming
            camera_streaming = False
            current_frame = None
            
            # Convert to JPEG with high quality
            img_bytes = io.BytesIO()
            save_image = Image.fromarray(processed_frame, 'RGB')
            save_image.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            frame_data = img_bytes.getvalue()
            
            print(f"✓ Fresh frame captured (brightness: {final_brightness:.1f}, size: {len(frame_data)} bytes)")
            
            # Return captured frame as JPEG
            return Response(
                content=frame_data,
                media_type="image/jpeg",
                headers={"Content-Disposition": "inline; filename=captured.jpg"}
            )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {str(e)}")

@app.post("/api/camera/stop")
async def stop_camera():
    """Stop camera stream and release resources"""
    global camera, camera_streaming, current_frame
    
    try:
        camera_streaming = False
        current_frame = None
        
        with camera_lock:
            if camera is not None:
                camera.stop()
                camera.close()
                camera = None
        
        return {"success": True, "message": "Camera stopped"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
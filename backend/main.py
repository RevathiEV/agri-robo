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
import serial

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

# Try to import gpiozero for relay control (water pump)
try:
    from gpiozero import OutputDevice
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False
    print("Warning: gpiozero not available. Motor control via GPIO will be disabled.")
    OutputDevice = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global serial_connection, relay_device
    # Startup
    try:
        load_model_and_mapping()
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("API will start but disease detection will not work until model is available.")
    
    # Initialize serial connection to Bluetooth device
    try:
        serial_connection = serial.Serial('/dev/rfcomm0', 9600, timeout=1)
        print("Bluetooth serial connection established to /dev/rfcomm0")
    except Exception as e:
        print(f"Warning: Could not connect to /dev/rfcomm0: {e}")
        print("Motor and servo control will not be available.")
        serial_connection = None
    
    # Initialize GPIO for relay control (motor/pump) using gpiozero
    if GPIOZERO_AVAILABLE:
        try:
            # Create OutputDevice for relay
            # active_high=False means LOW=ON, HIGH=OFF (for active-low relays)
            # active_high=True means HIGH=ON, LOW=OFF (for active-high relays)
            # IMPORTANT: initial_value=False ensures relay starts OFF
            # If RELAY_ACTIVE_LOW=True, then active_high=False (LOW turns relay ON)
            # If RELAY_ACTIVE_LOW=False, then active_high=True (HIGH turns relay ON)
            relay_device = OutputDevice(RELAY_GPIO_PIN, active_high=not RELAY_ACTIVE_LOW, initial_value=False)
            
            # Explicitly ensure relay is OFF at startup (multiple safety checks)
            # For active-low: we need HIGH to turn it OFF
            # For active-high: we need LOW to turn it OFF
            relay_device.off()
            time.sleep(0.3)  # Delay to ensure GPIO settles
            relay_device.off()  # Second OFF command for safety
            time.sleep(0.2)
            relay_device.off()  # Third OFF command for extra safety
            time.sleep(0.1)
            
            relay_type = "active-low" if RELAY_ACTIVE_LOW else "active-high"
            print(f"GPIO Pin {RELAY_GPIO_PIN} (Physical Pin 12) initialized - Water Pump OFF")
            print(f"Relay type: {relay_type} (Using gpiozero OutputDevice)")
            print("Water pump will ONLY activate when disease is detected via /api/detect-disease endpoint")
            print("✓ Motor is OFF and will remain OFF until disease is detected")
        except Exception as e:
            print(f"Warning: Could not initialize GPIO: {e}")
            print("Motor control via GPIO will not be available.")
            relay_device = None
    else:
        print("gpiozero not available - Motor control via GPIO disabled")
        relay_device = None
    
    # Final safety check - ensure motor is OFF after all initialization
    if relay_device is not None:
        try:
            relay_device.off()
            print("✓ Final safety check: Motor confirmed OFF")
        except:
            pass
    
    yield
    
    # Additional safety check - ensure motor is OFF when app starts serving requests
    # This runs right after the app starts serving (after yield)
    if relay_device is not None:
        try:
            relay_device.off()
            time.sleep(0.1)
            relay_device.off()  # Double check
            print("✓ App serving: Water pump confirmed OFF")
        except:
            pass
    
    # Shutdown - cleanup
    if serial_connection and serial_connection.is_open:
        serial_connection.close()
        print("Bluetooth serial connection closed")
    
    # Cleanup GPIO - ensure motor is OFF before shutdown
    if relay_device is not None:
        try:
            # Turn motor OFF before cleanup (multiple safety checks)
            relay_device.off()
            time.sleep(0.1)
            relay_device.off()  # Second OFF for safety
            relay_device.close()
            print("GPIO cleaned up - Water pump turned OFF")
        except Exception as e:
            print(f"Warning: Error during GPIO cleanup: {e}")

app = FastAPI(title="Agri ROBO API", version="1.0.0", lifespan=lifespan)

# CORS middleware to allow React frontend to access the API
# Updated to include Pi's IP address
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:5173", 
        "http://127.0.0.1:3000",
        "http://10.222.54.41:3000",  # Old Pi's IP with port 3000
        "http://10.222.54.41:5173",  # Old Pi's IP with port 5173
        "http://10.86.141.41:3000",  # New Pi's IP with port 3000
        "http://10.86.141.41:5173",  # New Pi's IP with port 5173
        "*"  # Allow all origins for development
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

# Global variable for relay device (water pump)
relay_device = None

# GPIO pin for relay control (Pin 12 = GPIO 18)
RELAY_GPIO_PIN = 18  # GPIO 18 (Physical Pin 12)

# Relay polarity configuration
# Set to True if your relay is active-low (LOW = ON, HIGH = OFF)
# Set to False if your relay is active-high (HIGH = ON, LOW = OFF) - DEFAULT
# IMPORTANT: If motor/relay turns ON automatically at startup, change this to True
RELAY_ACTIVE_LOW = True  # Changed to True because relay green light turns ON at startup


def activate_motor_for_duration(duration_seconds: float = 3.0):
    """
    Activate motor (relay) for specified duration, then turn it off automatically.
    Runs in a background thread to avoid blocking the API response.
    
    IMPORTANT: This function should ONLY be called from /api/detect-disease endpoint
    when a disease is detected (not healthy, not Not_A_Leaf).
    
    Args:
        duration_seconds: How long to keep motor ON (default: 3.0 seconds)
    
    Returns:
        bool: True if motor was activated, False if GPIO not available
    """
    global relay_device
    
    if relay_device is None:
        print("Warning: Relay device not available - cannot activate motor")
        return False
    
    def motor_control_thread():
        """Background thread to control motor timing"""
        try:
            # Turn motor ON
            relay_device.on()
            print(f"✓ Water pump activated (GPIO {RELAY_GPIO_PIN}) - Disease detected")
            
            # Wait for specified duration
            time.sleep(duration_seconds)
            
            # Turn motor OFF
            relay_device.off()
            print(f"✓ Water pump deactivated (GPIO {RELAY_GPIO_PIN}) after {duration_seconds}s")
        except Exception as e:
            print(f"ERROR controlling motor: {e}")
            import traceback
            traceback.print_exc()
            # Ensure motor is turned OFF on error
            try:
                relay_device.off()
                print(f"✓ Water pump forced OFF due to error")
            except:
                pass
    
    # Start motor control in background thread (non-blocking)
    motor_thread = threading.Thread(target=motor_control_thread, daemon=True)
    motor_thread.start()
    
    return True

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
        # Try loading with compile=False for better cross-version compatibility
        print("Attempting to load model with compile=False...")
        model = load_model(model_path, compile=False)
        print("✓ Model loaded (compile=False)")
        
        # Recompile with the same settings as training
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
        # Format disease name (handle Not_A_Leaf and Tomato___ classes)
        if disease_name == "Not_A_Leaf":
            formatted_disease = "Not A Leaf"
        else:
            formatted_disease = disease_name.replace("Tomato___", "").replace("_", " ").title()
        
        is_healthy = "healthy" in disease_name.lower()
        is_not_a_leaf = disease_name == "Not_A_Leaf"
        
        # Control motor: Turn ON ONLY if disease is detected (not healthy, not Not_A_Leaf)
        # Motor should NEVER activate for healthy or Not_A_Leaf
        # IMPORTANT: Motor only activates when frontend shows "disease detected"
        motor_activated = False
        if not is_healthy and not is_not_a_leaf:
            # Disease detected - activate motor for 3 seconds
            # This is the ONLY place where motor should turn ON
            print(f"✓ Disease detected: {disease_name} - Activating water pump for 3 seconds")
            motor_activated = activate_motor_for_duration(3.0)
        else:
            # Healthy or Not_A_Leaf - motor stays OFF
            print(f"✓ Detection: {disease_name} - Water pump stays OFF (healthy or not a leaf)")
            # Explicitly ensure motor is OFF (safety check)
            if relay_device is not None:
                try:
                    relay_device.off()
                except:
                    pass
        
        return JSONResponse({
            "success": True,
            "disease": formatted_disease,
            "confidence": round(confidence, 2),
            "is_healthy": is_healthy,
            "is_not_a_leaf": is_not_a_leaf,
            "raw_disease_name": disease_name,
            "motor_activated": motor_activated,
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
    Note: Servo control needs to be implemented in ESP32 code
    """
    global serial_connection
    valid_actions = ["start", "stop"]
    if action.lower() not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {valid_actions}")
    
    if serial_connection is None or not serial_connection.is_open:
        raise HTTPException(status_code=503, detail="Bluetooth serial connection not available")
    
    try:
        # Map action to ESP32 single character command
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
    """Convert camera frame to RGB format"""
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = frame[:, :, 2]  # R
        rgb_frame[:, :, 1] = frame[:, :, 1]  # G
        rgb_frame[:, :, 2] = frame[:, :, 0]  # B
        return rgb_frame
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame
    else:
        return frame

def process_frame(frame):
    """Apply minimal post-processing for natural look"""
    rgb_frame = convert_frame_to_rgb(frame)
    mean_brightness = rgb_frame.mean()
    
    image = Image.fromarray(rgb_frame, 'RGB')
    
    # Minimal brightness correction only if very dark
    if mean_brightness < 30:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)
    
    # Very minimal color correction
    r, g, b = image.split()
    b = ImageEnhance.Brightness(b).enhance(0.98)
    r = ImageEnhance.Brightness(r).enhance(1.01)
    g = ImageEnhance.Brightness(g).enhance(1.00)
    image = Image.merge('RGB', (r, g, b))
    
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
                        
                        # Process frame
                        rgb_frame = process_frame(frame)
                        
                        # Convert to JPEG
                        image = Image.fromarray(rgb_frame, 'RGB')
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='JPEG', quality=85)
                        img_bytes.seek(0)
                        
                        # Update frame (no lock needed for simple assignment)
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
                camera = Picamera2(camera_num=0)
                
                # Configure camera
                try:
                    config = camera.create_preview_configuration(
                        main={"size": (640, 480), "format": "RGB888"},
                        colour_space="sRGB"
                    )
                    camera.configure(config)
                except:
                    config = camera.create_preview_configuration(main={"size": (640, 480)})
                    camera.configure(config)
                
                # Set camera controls
                try:
                    camera.set_controls({
                        "AwbEnable": True,
                        "AeEnable": True,
                    })
                except:
                    pass
                
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
    """Capture current frame from camera and return as image"""
    global camera, camera_streaming, current_frame
    
    if not camera_streaming or current_frame is None:
        raise HTTPException(status_code=400, detail="Camera not streaming or no frame available")
    
    try:
        # Get current frame
        frame_data = current_frame
        
        # Stop streaming
        camera_streaming = False
        
        # Return captured frame as JPEG
        return Response(
            content=frame_data,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=captured.jpg"}
        )
    
    except Exception as e:
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
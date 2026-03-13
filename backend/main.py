from contextlib import asynccontextmanager
import io
import json
import os
import sys
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
import numpy as np
from PIL import Image, ImageEnhance
from tensorflow.keras.models import load_model
import tensorflow as tf

# Add system dist-packages to path for picamera2 on Raspberry Pi OS
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: picamera2 not available. Camera features will be disabled.")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    GPIO = None
    print("Warning: RPi.GPIO not available. Pump control will run in simulation mode.")


# Global variables for model and class mapping
model = None
class_mapping = None

# Global variables for camera
camera = None
camera_streaming = False
camera_lock = threading.Lock()
current_frame = None
current_motor_direction = "stop"

# Pump control configuration
# Physical pin 12 on Raspberry Pi = BCM GPIO 18
PUMP_GPIO_PIN = 18
SPRAY_DURATION_SECONDS = 3
RELAY_ACTIVE_LOW = True
pump_initialized = False
pump_lock = threading.Lock()
pump_running = False


def get_pump_on_state():
    return GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH


def get_pump_off_state():
    return GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW


def init_pump_gpio():
    """Initialize the relay pin in a guaranteed OFF state."""
    global pump_initialized

    if not GPIO_AVAILABLE:
        pump_initialized = False
        print("[PUMP] GPIO not available. Using simulation mode.")
        return False

    try:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        off_state = get_pump_off_state()
        GPIO.setup(PUMP_GPIO_PIN, GPIO.OUT, initial=off_state)
        GPIO.output(PUMP_GPIO_PIN, off_state)
        pump_initialized = True
        print(
            f"[PUMP] Initialized GPIO {PUMP_GPIO_PIN} in OFF state "
            f"({'active LOW' if RELAY_ACTIVE_LOW else 'active HIGH'})"
        )
        return True
    except Exception as exc:
        pump_initialized = False
        print(f"[PUMP] Initialization failed: {exc}")
        return False


def force_pump_off():
    """Force the relay output to the OFF state."""
    if not GPIO_AVAILABLE or not pump_initialized:
        print("[PUMP] OFF (simulation)")
        return

    GPIO.output(PUMP_GPIO_PIN, get_pump_off_state())


def spray_pump_for_duration(duration_seconds=SPRAY_DURATION_SECONDS):
    """Run the pump in the background for a fixed duration."""
    if duration_seconds <= 0:
        return False

    def run_spray():
        global pump_running
        with pump_lock:
            try:
                pump_running = True
                if not GPIO_AVAILABLE or not pump_initialized:
                    print(f"[PUMP] Simulated spray for {duration_seconds} seconds")
                    time.sleep(duration_seconds)
                    return

                GPIO.output(PUMP_GPIO_PIN, get_pump_on_state())
                print(f"[PUMP] ON for {duration_seconds} seconds")
                time.sleep(duration_seconds)
            except Exception as exc:
                print(f"[PUMP] Spray error: {exc}")
            finally:
                try:
                    force_pump_off()
                    pump_running = False
                    print("[PUMP] OFF")
                except Exception as exc:
                    print(f"[PUMP] Failed to turn OFF after spraying: {exc}")

    spray_thread = threading.Thread(target=run_spray, daemon=True)
    spray_thread.start()
    return True


def cleanup_gpio():
    global pump_initialized, pump_running

    if GPIO_AVAILABLE:
        try:
            if pump_initialized:
                force_pump_off()
            GPIO.cleanup()
            print("[GPIO] Cleanup complete")
        except Exception as exc:
            print(f"[GPIO] Cleanup error: {exc}")
    pump_initialized = False
    pump_running = False


def start_pump_manual():
    global pump_running

    with pump_lock:
        try:
            if not GPIO_AVAILABLE or not pump_initialized:
                pump_running = True
                print("[PUMP] Manual ON (simulation)")
                return True

            GPIO.output(PUMP_GPIO_PIN, get_pump_on_state())
            pump_running = True
            print("[PUMP] Manual ON")
            return True
        except Exception as exc:
            pump_running = False
            print(f"[PUMP] Manual start error: {exc}")
            return False


def stop_pump_manual():
    global pump_running

    with pump_lock:
        try:
            force_pump_off()
            pump_running = False
            print("[PUMP] Manual OFF")
            return True
        except Exception as exc:
            print(f"[PUMP] Manual stop error: {exc}")
            return False


def load_model_and_mapping():
    """Load the disease detection model and class mapping."""
    global model, class_mapping

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)

    model_path_best = os.path.join(project_root, "tomato_disease_model_best.h5")
    model_path = os.path.join(project_root, "tomato_disease_model.h5")
    mapping_path = os.path.join(project_root, "class_mapping.json")

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

    try:
        print("Attempting to load model with compile=False...")
        loaded_model = load_model(model_path, compile=False)
        loaded_model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model = loaded_model
        print("Model loaded and recompiled successfully")
    except Exception as first_error:
        print(f"Warning: compile=False load failed: {first_error}")
        try:
            model = load_model(model_path)
            print("Model loaded with standard method")
        except Exception as second_error:
            raise RuntimeError(
                f"Could not load model with either method.\n"
                f"  compile=False error: {str(first_error)[:200]}\n"
                f"  standard error: {str(second_error)[:200]}"
            ) from second_error

    with open(mapping_path, "r") as mapping_file:
        mapping = json.load(mapping_file)
    class_mapping = {int(key): value for key, value in mapping.items()}

    print(f"Model input shape: {model.input_shape}")
    print(f"Class mapping loaded: {len(class_mapping)} classes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_pump_gpio()
    except Exception as exc:
        print(f"[STARTUP] Pump initialization error: {exc}")

    try:
        load_model_and_mapping()
    except Exception as exc:
        print(f"Error loading model: {exc}")
        import traceback
        traceback.print_exc()
        print("API will start but disease detection will not work until model is available.")

    try:
        yield
    finally:
        try:
            cleanup_gpio()
        except Exception as exc:
            print(f"[SHUTDOWN] GPIO cleanup error: {exc}")


app = FastAPI(title="Agri ROBO API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://10.222.54.41:3000",
        "http://10.222.54.41:5173",
        "http://10.86.141.41:3000",
        "http://10.86.141.41:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_disease_name(disease_name):
    if disease_name == "Not_A_Leaf":
        return "Not A Leaf"
    return disease_name.replace("Tomato___", "").replace("_", " ").title()


def should_trigger_pump(disease_name):
    if disease_name == "Not_A_Leaf":
        return False
    if "healthy" in disease_name.lower():
        return False
    return disease_name != "Unknown"


@app.get("/")
async def root():
    return {"message": "Agri ROBO API", "status": "running"}


@app.get("/health")
async def health_check():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)

    model_path_best = os.path.join(project_root, "tomato_disease_model_best.h5")
    model_path = os.path.join(project_root, "tomato_disease_model.h5")
    mapping_path = os.path.join(project_root, "class_mapping.json")

    model_exists = os.path.exists(model_path) or os.path.exists(model_path_best)
    mapping_exists = os.path.exists(mapping_path)

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mapping_loaded": class_mapping is not None,
        "model_file_exists": model_exists,
        "mapping_file_exists": mapping_exists,
        "num_classes": len(class_mapping) if class_mapping else 0,
        "tensorflow_version": tf.__version__,
        "pump_gpio_pin": PUMP_GPIO_PIN,
        "pump_initialized": pump_initialized,
        "pump_running": pump_running,
        "current_motor_direction": current_motor_direction,
        "relay_mode": "active_low" if RELAY_ACTIVE_LOW else "active_high",
        "spray_duration_seconds": SPRAY_DURATION_SECONDS,
    }


@app.post("/api/motor/control")
async def motor_control(direction: str):
    """Motor control endpoint kept for frontend compatibility."""
    global current_motor_direction

    allowed_directions = {"front", "back", "left", "right", "stop"}
    if direction not in allowed_directions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid direction '{direction}'. Use one of {sorted(allowed_directions)}.",
        )

    current_motor_direction = direction
    print(f"[MOTOR] Direction set to: {direction}")

    return {
        "success": True,
        "direction": current_motor_direction,
        "message": f"Motor command '{direction}' received",
        "hardware_connected": False,
    }


@app.post("/api/pump/start")
async def pump_start():
    """Manual pump start endpoint kept for frontend compatibility."""
    started = start_pump_manual()
    if not started:
        raise HTTPException(status_code=500, detail="Failed to start pump")

    return {
        "success": True,
        "message": "Pump started",
        "pump_running": pump_running,
        "pump_gpio_pin": PUMP_GPIO_PIN,
    }


@app.post("/api/pump/stop")
async def pump_stop():
    """Manual pump stop endpoint kept for frontend compatibility."""
    stopped = stop_pump_manual()
    if not stopped:
        raise HTTPException(status_code=500, detail="Failed to stop pump")

    return {
        "success": True,
        "message": "Pump stopped",
        "pump_running": pump_running,
        "pump_gpio_pin": PUMP_GPIO_PIN,
    }


@app.post("/api/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    if model is None or class_mapping is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Please ensure model files "
                "(tomato_disease_model.h5 and class_mapping.json) "
                "are available in the project root."
            ),
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received content type: {file.content_type}",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        image = Image.open(io.BytesIO(contents))
        if image.size[0] == 0 or image.size[1] == 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")

        expected_shape = model.input_shape[1:]
        image_size = (expected_shape[0], expected_shape[1])

        if image.mode != "RGB":
            image = image.convert("RGB")

        original_brightness = float(np.array(image).mean())

        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(1.2)

        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(1.1)

        image = image.resize(image_size, Image.Resampling.LANCZOS)

        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.clip(img_array, 0.0, 1.0)
        img_array = np.expand_dims(img_array, axis=0)

        if img_array.shape[1:] != expected_shape:
            raise ValueError(
                f"Image shape mismatch. Expected {expected_shape}, got {img_array.shape[1:]}"
            )

        print(
            "DEBUG: Image preprocessing - "
            f"original brightness: {original_brightness:.1f}, "
            f"final shape: {img_array.shape}, "
            f"value range: [{img_array.min():.3f}, {img_array.max():.3f}]"
        )

        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx] * 100)

        disease_name = class_mapping.get(predicted_class_idx, "Unknown")
        formatted_disease = format_disease_name(disease_name)
        is_healthy = "healthy" in disease_name.lower()
        is_not_a_leaf = disease_name == "Not_A_Leaf"

        spray_triggered = False
        if should_trigger_pump(disease_name):
            spray_triggered = spray_pump_for_duration(SPRAY_DURATION_SECONDS)

        return JSONResponse(
            {
                "success": True,
                "disease": formatted_disease,
                "confidence": round(confidence, 2),
                "is_healthy": is_healthy,
                "is_not_a_leaf": is_not_a_leaf,
                "raw_disease_name": disease_name,
                "spray_triggered": spray_triggered,
                "spray_duration_seconds": SPRAY_DURATION_SECONDS if spray_triggered else 0,
                "pump_gpio_pin": PUMP_GPIO_PIN,
                "model_info": {
                    "input_shape": str(model.input_shape),
                    "num_classes": len(class_mapping),
                },
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing image: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {exc}")


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

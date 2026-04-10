from contextlib import asynccontextmanager
import io
import json
import os
from urllib.parse import urlencode
import sys
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
import numpy as np
from PIL import Image, ImageEnhance
import requests
from spray_pump_control import (
    cleanup_spray_pumps,
    get_status as get_pump_status,
    start_manual_dispense,
    stop_dispense,
    trigger_spray_by_disease,
)

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
    from tensorflow.keras.models import load_model
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    load_model = None
    tf = None
    TENSORFLOW_AVAILABLE = False
    print("Warning: tensorflow not available. Disease detection will be disabled.")

try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter

    TFLITE_RUNTIME_AVAILABLE = True
    TFLITE_RUNTIME_SOURCE = "tflite-runtime"
except ImportError:
    if TENSORFLOW_AVAILABLE:
        TFLiteInterpreter = tf.lite.Interpreter
        TFLITE_RUNTIME_AVAILABLE = True
        TFLITE_RUNTIME_SOURCE = "tensorflow-lite"
    else:
        TFLiteInterpreter = None
        TFLITE_RUNTIME_AVAILABLE = False
        TFLITE_RUNTIME_SOURCE = None


# Global variables for model and class mapping
model = None
class_mapping = None
model_backend = None
model_source_path = None
model_input_shape = None
model_input_details = None
model_output_details = None
MODEL_BACKEND_PREFERENCE = os.getenv("MODEL_BACKEND", "auto").strip().lower()

# Global variables for camera
camera = None
camera_streaming = False
camera_lock = threading.Lock()
current_frame = None
current_motor_direction = "stop"
ESP32_MOTOR_HOSTNAME = os.getenv("ESP32_MOTOR_HOSTNAME", "agri-robo-esp32.local")
ESP32_MOTOR_BASE_URL = os.getenv(
    "ESP32_MOTOR_BASE_URL",
    f"http://{ESP32_MOTOR_HOSTNAME}",
).rstrip("/")
ESP32_MOTOR_TIMEOUT = float(os.getenv("ESP32_MOTOR_TIMEOUT", "1.5"))


def motor_hardware_configured():
    return bool(ESP32_MOTOR_BASE_URL)


def send_motor_command_to_esp32(direction: str):
    """Forward a movement command to the ESP32 motor controller."""
    if not motor_hardware_configured():
        raise RuntimeError(
            "ESP32 motor controller URL is not configured. "
            "Set ESP32_MOTOR_BASE_URL, for example http://192.168.1.50"
        )

    params = urlencode({"direction": direction})
    endpoint = f"{ESP32_MOTOR_BASE_URL}/move?{params}"

    try:
        response = requests.post(endpoint, timeout=ESP32_MOTOR_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach ESP32 motor controller: {exc}") from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {
            "success": True,
            "message": response.text.strip() or "ESP32 responded without JSON payload.",
        }

    return payload


def get_model_paths():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)

    return {
        "project_root": project_root,
        "keras_best": os.path.join(project_root, "tomato_disease_model_best.h5"),
        "keras": os.path.join(project_root, "tomato_disease_model.h5"),
        "tflite": os.path.join(project_root, "tomato_disease_model.tflite"),
        "mapping": os.path.join(project_root, "class_mapping.json"),
    }


def get_model_input_shape():
    if model_input_shape is not None:
        return model_input_shape

    if model is not None and hasattr(model, "input_shape"):
        return tuple(int(dim) for dim in model.input_shape[1:])

    raise RuntimeError("Model input shape is not available because the model is not loaded.")


def predict_with_loaded_model(img_array):
    if model_backend == "keras":
        return model.predict(img_array, verbose=0)[0]

    if model_backend == "tflite":
        input_detail = model_input_details[0]
        output_detail = model_output_details[0]
        input_dtype = input_detail["dtype"]
        input_tensor = img_array.astype(input_dtype)

        if not np.issubdtype(input_dtype, np.floating):
            scale, zero_point = input_detail.get("quantization", (0.0, 0))
            if scale:
                input_tensor = np.round(img_array / scale + zero_point)
            dtype_info = np.iinfo(input_dtype)
            input_tensor = np.clip(input_tensor, dtype_info.min, dtype_info.max).astype(input_dtype)

        model.set_tensor(input_detail["index"], input_tensor)
        model.invoke()
        output_tensor = model.get_tensor(output_detail["index"])

        if not np.issubdtype(output_tensor.dtype, np.floating):
            scale, zero_point = output_detail.get("quantization", (0.0, 0))
            output_tensor = output_tensor.astype(np.float32)
            if scale:
                output_tensor = (output_tensor - zero_point) * scale

        return np.asarray(output_tensor[0], dtype=np.float32)

    raise RuntimeError("No supported model backend is currently loaded.")


def load_model_and_mapping():
    """Load the disease detection model and class mapping."""
    global model, class_mapping, model_backend, model_source_path
    global model_input_shape, model_input_details, model_output_details

    paths = get_model_paths()
    mapping_path = paths["mapping"]

    if not os.path.exists(mapping_path):
        raise FileNotFoundError(
            f"Class mapping file not found: {mapping_path}\n"
            f"Please run cnn_train.py to generate the class mapping."
        )

    with open(mapping_path, "r") as mapping_file:
        mapping = json.load(mapping_file)
    class_mapping = {int(key): value for key, value in mapping.items()}

    requested_backend = MODEL_BACKEND_PREFERENCE
    if requested_backend not in {"auto", "keras", "tflite"}:
        raise RuntimeError(
            f"Invalid MODEL_BACKEND '{requested_backend}'. Use auto, keras, or tflite."
        )

    if requested_backend in {"auto", "keras"}:
        keras_path = None
        if os.path.exists(paths["keras_best"]):
            keras_path = paths["keras_best"]
        elif os.path.exists(paths["keras"]):
            keras_path = paths["keras"]

        if keras_path is not None:
            if not TENSORFLOW_AVAILABLE and requested_backend == "keras":
                raise RuntimeError(
                    "MODEL_BACKEND=keras was requested, but TensorFlow is not installed."
                )

            if TENSORFLOW_AVAILABLE:
                print(f"TensorFlow version: {tf.__version__}")
                print(f"Loading Keras model from: {keras_path}")
                try:
                    loaded_model = load_model(keras_path, compile=False)
                    loaded_model.compile(
                        optimizer="adam",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"],
                    )
                    model = loaded_model
                except Exception as first_error:
                    print(f"Warning: compile=False load failed: {first_error}")
                    try:
                        model = load_model(keras_path)
                    except Exception as second_error:
                        raise RuntimeError(
                            f"Could not load Keras model with either method.\n"
                            f"  compile=False error: {str(first_error)[:200]}\n"
                            f"  standard error: {str(second_error)[:200]}"
                        ) from second_error

                model_backend = "keras"
                model_source_path = keras_path
                model_input_shape = tuple(int(dim) for dim in model.input_shape[1:])
                model_input_details = None
                model_output_details = None
                print(f"Model input shape: {model.input_shape}")
                print(f"Class mapping loaded: {len(class_mapping)} classes")
                return

    if requested_backend in {"auto", "tflite"} and os.path.exists(paths["tflite"]):
        if not TFLITE_RUNTIME_AVAILABLE:
            raise RuntimeError(
                "A TFLite model is available, but neither tflite-runtime nor TensorFlow Lite "
                "is installed on this system."
            )

        print(f"Loading TFLite model from: {paths['tflite']} ({TFLITE_RUNTIME_SOURCE})")
        interpreter = TFLiteInterpreter(model_path=paths["tflite"])
        interpreter.allocate_tensors()
        model = interpreter
        model_backend = "tflite"
        model_source_path = paths["tflite"]
        model_input_details = interpreter.get_input_details()
        model_output_details = interpreter.get_output_details()
        model_input_shape = tuple(int(dim) for dim in model_input_details[0]["shape"][1:])
        print(f"TFLite input shape: {model_input_shape}")
        print(f"Class mapping loaded: {len(class_mapping)} classes")
        return

    raise RuntimeError(
        "No compatible disease model could be loaded.\n"
        f"Checked Keras models: {paths['keras_best']}, {paths['keras']}\n"
        f"Checked TFLite model: {paths['tflite']}\n"
        f"TensorFlow available: {TENSORFLOW_AVAILABLE}\n"
        f"TFLite runtime available: {TFLITE_RUNTIME_AVAILABLE}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        cleanup_spray_pumps()


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


@app.get("/")
async def root():
    return {"message": "Agri ROBO API", "status": "running"}


@app.get("/health")
async def health_check():
    paths = get_model_paths()

    model_exists = any(
        os.path.exists(paths[key])
        for key in ("keras_best", "keras", "tflite")
    )
    mapping_exists = os.path.exists(paths["mapping"])

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mapping_loaded": class_mapping is not None,
        "model_file_exists": model_exists,
        "tflite_model_exists": os.path.exists(paths["tflite"]),
        "mapping_file_exists": mapping_exists,
        "num_classes": len(class_mapping) if class_mapping else 0,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "tensorflow_version": tf.__version__ if tf is not None else None,
        "tflite_runtime_available": TFLITE_RUNTIME_AVAILABLE,
        "tflite_runtime_source": TFLITE_RUNTIME_SOURCE,
        "model_backend": model_backend,
        "model_source_path": model_source_path,
        "model_input_shape": list(model_input_shape) if model_input_shape else None,
        "current_motor_direction": current_motor_direction,
        "motor_hardware_configured": motor_hardware_configured(),
        "esp32_motor_base_url": ESP32_MOTOR_BASE_URL or None,
        "pump_status": get_pump_status(),
    }


@app.post("/api/motor/control")
async def motor_control(direction: str):
    """Forward movement commands from the frontend to the ESP32 motor controller."""
    global current_motor_direction

    allowed_directions = {"front", "back", "left", "right", "stop"}
    if direction not in allowed_directions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid direction '{direction}'. Use one of {sorted(allowed_directions)}.",
        )

    current_motor_direction = direction
    print(f"[MOTOR] Direction set to: {direction}")
    try:
        hardware_response = send_motor_command_to_esp32(direction)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "success": True,
        "direction": current_motor_direction,
        "message": f"Motor command '{direction}' forwarded to ESP32",
        "hardware_connected": True,
        "hardware_response": hardware_response,
    }


@app.get("/api/pump/status")
async def pump_status():
    return {
        "success": True,
        "pump_status": get_pump_status(),
    }


@app.post("/api/pump/start")
async def pump_start():
    result = start_manual_dispense()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@app.post("/api/pump/stop")
async def pump_stop():
    result = stop_dispense()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@app.post("/api/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    if model is None or class_mapping is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Please ensure model files "
                "(tomato_disease_model.h5 or tomato_disease_model.tflite, and class_mapping.json) "
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

        expected_shape = get_model_input_shape()
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

        predictions = predict_with_loaded_model(img_array)
        predicted_class_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_class_idx] * 100)

        disease_name = class_mapping.get(predicted_class_idx, "Unknown")
        formatted_disease = format_disease_name(disease_name)
        is_healthy = "healthy" in disease_name.lower()
        is_not_a_leaf = disease_name == "Not_A_Leaf"
        pump_result = {
            "success": True,
            "auto_dispense_started": False,
            "message": "No automatic spray needed for this detection.",
            "pump_status": get_pump_status(),
        }

        if not is_healthy and not is_not_a_leaf:
            pump_result = trigger_spray_by_disease(disease_name)

        return JSONResponse(
            {
                "success": True,
                "disease": formatted_disease,
                "confidence": round(confidence, 2),
                "is_healthy": is_healthy,
                "is_not_a_leaf": is_not_a_leaf,
                "raw_disease_name": disease_name,
                "auto_dispense_started": pump_result["auto_dispense_started"],
                "pump_message": pump_result["message"],
                "pump_status": pump_result["pump_status"],
                "model_info": {
                    "backend": model_backend,
                    "source_path": model_source_path,
                    "input_shape": str(get_model_input_shape()),
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

def get_camera_info():
    """Read camera inventory without failing the whole startup path."""
    try:
        cam_info = Picamera2.global_camera_info()
        print(f"[CAMERA] global_camera_info: {cam_info}")
        return cam_info or []
    except Exception as e:
        print(f"[CAMERA] global_camera_info failed: {e}")
        return []


def create_picamera_instance():
    """
    Try a few safe ways to open the first Pi camera.
    Some Pi setups return an empty global_camera_info() even though camera 0 opens fine.
    """
    attempts = []
    cam_info = get_camera_info()
    candidate_indexes = []

    if cam_info:
        candidate_indexes.extend(range(len(cam_info)))

    if 0 not in candidate_indexes:
        candidate_indexes.append(0)

    for camera_index in candidate_indexes:
        try:
            print(f"[CAMERA] Trying Picamera2(camera_num={camera_index})")
            return Picamera2(camera_num=camera_index)
        except Exception as e:
            attempts.append(f"camera_num={camera_index}: {e}")

    try:
        print("[CAMERA] Trying Picamera2() default constructor")
        return Picamera2()
    except Exception as e:
        attempts.append(f"default constructor: {e}")

    attempt_text = "; ".join(attempts) if attempts else "no camera open attempts succeeded"
    raise HTTPException(
        status_code=503,
        detail=(
            "No camera detected or camera could not be opened. "
            f"Attempts: {attempt_text}"
        ),
    )

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
                camera = create_picamera_instance()

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
            detail=f"No camera detected. Connect a Pi camera or USB camera and try again. {str(e)}"
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
                camera = create_picamera_instance()

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
            detail=f"No camera detected. Connect a Pi camera or USB camera and try again. {str(e)}"
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

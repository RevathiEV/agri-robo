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

# Pump control configuration
# Physical pin 12 on Raspberry Pi = BCM GPIO 18
PUMP_GPIO_PIN = 18
SPRAY_DURATION_SECONDS = 3
RELAY_ACTIVE_LOW = True
pump_initialized = False
pump_lock = threading.Lock()


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
        with pump_lock:
            try:
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
                    print("[PUMP] OFF")
                except Exception as exc:
                    print(f"[PUMP] Failed to turn OFF after spraying: {exc}")

    spray_thread = threading.Thread(target=run_spray, daemon=True)
    spray_thread.start()
    return True


def cleanup_gpio():
    global pump_initialized

    if GPIO_AVAILABLE:
        try:
            if pump_initialized:
                force_pump_off()
            GPIO.cleanup()
            print("[GPIO] Cleanup complete")
        except Exception as exc:
            print(f"[GPIO] Cleanup error: {exc}")
    pump_initialized = False


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
        "relay_mode": "active_low" if RELAY_ACTIVE_LOW else "active_high",
        "spray_duration_seconds": SPRAY_DURATION_SECONDS,
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


def convert_frame_to_rgb(frame):
    """Convert camera frame to RGB format."""
    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
        gray = frame[:, :, 0] if len(frame.shape) == 3 else frame
        return np.stack([gray, gray, gray], axis=2)

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = frame[:, :, 2]
        rgb_frame[:, :, 1] = frame[:, :, 1]
        rgb_frame[:, :, 2] = frame[:, :, 0]
        return rgb_frame

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame

    if len(frame.shape) == 3:
        return frame[:, :, :3]

    gray = frame.flatten()
    rgb = np.stack([gray, gray, gray], axis=1)
    return rgb.reshape(frame.shape[0], frame.shape[1], 3)


def process_frame(frame):
    rgb_frame = convert_frame_to_rgb(frame)
    mean_brightness = rgb_frame.mean()
    image = Image.fromarray(rgb_frame, "RGB")

    if mean_brightness < 30:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)

    return np.array(image)


def camera_capture_thread():
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
                        image = Image.fromarray(rgb_frame, "RGB")
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="JPEG", quality=85)
                        img_bytes.seek(0)
                        current_frame = img_bytes.getvalue()
                time.sleep(0.033)
            except Exception as exc:
                print(f"Camera capture error: {exc}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        else:
            time.sleep(0.1)


if PICAMERA2_AVAILABLE:
    camera_thread = threading.Thread(target=camera_capture_thread, daemon=True)
    camera_thread.start()


@app.post("/api/camera/start")
async def start_camera():
    global camera, camera_streaming, current_frame

    if not PICAMERA2_AVAILABLE:
        raise HTTPException(status_code=503, detail="picamera2 not available on this system")

    try:
        with camera_lock:
            if camera is None:
                try:
                    cam_info = Picamera2.global_camera_info()
                except Exception:
                    cam_info = []

                if not cam_info:
                    raise HTTPException(
                        status_code=503,
                        detail="No camera detected. Connect a Pi camera or USB camera and try again.",
                    )

                camera = Picamera2(camera_num=0)

                config = None
                for format_attempt in ["RGB888", "BGR888", "XRGB8888", "XBGR8888"]:
                    try:
                        config = camera.create_preview_configuration(
                            main={"size": (640, 480), "format": format_attempt},
                            colour_space="sRGB",
                        )
                        camera.configure(config)
                        print(f"Camera configured with {format_attempt}")
                        break
                    except Exception as exc:
                        print(f"Trying {format_attempt}: {exc}")

                if config is None:
                    config = camera.create_preview_configuration(main={"size": (640, 480)})
                    camera.configure(config)
                    print("Using default camera configuration")

                try:
                    camera.set_controls({"AwbEnable": True, "AeEnable": True})
                except Exception:
                    print("Could not set camera controls, using defaults")

                camera.start()

                for _ in range(5):
                    test_request = camera.capture_request()
                    test_request.release()
                    time.sleep(0.3)

            camera_streaming = True
            current_frame = None
            time.sleep(0.5)

        return {"success": True, "message": "Camera started"}

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {exc}")


@app.get("/api/camera/stream")
async def camera_stream():
    global camera_streaming, current_frame

    if not camera_streaming:
        raise HTTPException(status_code=400, detail="Camera not started. Call /api/camera/start first")

    def generate_frames():
        while camera_streaming:
            wait_count = 0
            while current_frame is None and camera_streaming and wait_count < 100:
                time.sleep(0.01)
                wait_count += 1

            if current_frame:
                try:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + current_frame + b"\r\n"
                    )
                except Exception as exc:
                    print(f"Error yielding frame: {exc}")
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
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/camera/capture")
async def capture_image():
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
            image = Image.fromarray(rgb_frame, "RGB")
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
                image = Image.fromarray(processed_frame, "RGB")
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.5)
                processed_frame = np.array(image)

            camera_streaming = False
            current_frame = None

            img_bytes = io.BytesIO()
            save_image = Image.fromarray(processed_frame, "RGB")
            save_image.save(img_bytes, format="JPEG", quality=95)
            img_bytes.seek(0)
            frame_data = img_bytes.getvalue()

            print(
                f"Fresh frame captured (brightness: {final_brightness:.1f}, "
                f"size: {len(frame_data)} bytes)"
            )

            return Response(
                content=frame_data,
                media_type="image/jpeg",
                headers={"Content-Disposition": "inline; filename=captured.jpg"},
            )

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to capture image: {exc}")


@app.post("/api/camera/stop")
async def stop_camera():
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

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to stop camera: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

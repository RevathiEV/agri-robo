from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import tensorflow.lite as tflite
import json, os, io, sys, threading, time, serial

# =====================================================
# PICAMERA2
# =====================================================
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except:
    PICAMERA2_AVAILABLE = False

# =====================================================
# GPIO
# =====================================================
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except:
    GPIO_AVAILABLE = False
    class GPIO:
        BCM=None; OUT=None; LOW=0; HIGH=1
        def setmode(x): pass
        def setup(a,b,initial=None): pass
        def output(a,b): pass
        def cleanup(): pass

# =====================================================
# GLOBALS
# =====================================================
interpreter=None
input_details=None
output_details=None
class_mapping=None

camera=None
camera_streaming=False
camera_lock=threading.Lock()
current_frame=None

serial_connection=None
RELAY_GPIO_PIN=18
RELAY_ACTIVE_LOW=False

# =====================================================
# LOAD MODEL
# =====================================================
def load_model_and_mapping():
    global interpreter,input_details,output_details,class_mapping

    backend=os.path.dirname(os.path.abspath(__file__))
    root=os.path.dirname(backend)

    model_path=os.path.join(root,"model.tflite")
    mapping_path=os.path.join(root,"class_mapping.json")

    interpreter=tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details=interpreter.get_input_details()
    output_details=interpreter.get_output_details()

    with open(mapping_path) as f:
        class_mapping=json.load(f)

    print("✅ TFLite model loaded")

# =====================================================
# MOTOR
# =====================================================
def activate_motor_for_duration(sec=3):
    if not GPIO_AVAILABLE:
        return False

    def run():
        on=GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
        off=GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
        GPIO.output(RELAY_GPIO_PIN,on)
        time.sleep(sec)
        GPIO.output(RELAY_GPIO_PIN,off)

    threading.Thread(target=run,daemon=True).start()
    return True

# =====================================================
# LIFESPAN
# =====================================================
@asynccontextmanager
async def lifespan(app:FastAPI):

    global serial_connection

    load_model_and_mapping()

    try:
        serial_connection=serial.Serial("/dev/rfcomm0",9600,timeout=1)
    except:
        serial_connection=None

    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_GPIO_PIN,GPIO.OUT)

    yield

    if serial_connection:
        serial_connection.close()
    if GPIO_AVAILABLE:
        GPIO.cleanup()

# =====================================================
# APP
# =====================================================
app=FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"status":"running"}

# =====================================================
# DISEASE DETECTION
# =====================================================
@app.post("/api/detect-disease")
async def detect_disease(file:UploadFile=File(...)):

    img_bytes=await file.read()
    image=Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image=image.resize((224,224))
    img=np.array(image)/255.0
    img=np.expand_dims(img,0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'],img)
    interpreter.invoke()
    preds=interpreter.get_tensor(output_details[0]['index'])

    idx=int(np.argmax(preds))
    confidence=float(preds[0][idx])*100
    disease=class_mapping[str(idx)]

    is_healthy="healthy" in disease.lower()
    is_not_leaf="not" in disease.lower()

    motor=False
    if not is_healthy and not is_not_leaf:
        motor=activate_motor_for_duration(3)

    return {
        "disease":disease,
        "confidence":round(confidence,2),
        "motor_activated":motor
    }

# =====================================================
# ROBOT CONTROL
# =====================================================
@app.post("/api/motor/control")
async def motor_control(direction:str):

    if not serial_connection:
        raise HTTPException(503,"Bluetooth not connected")

    mp={"front":"F","back":"B","left":"L","right":"R","stop":"S"}
    cmd=mp.get(direction.lower())

    if not cmd:
        raise HTTPException(400,"Invalid")

    serial_connection.write(cmd.encode())
    return {"sent":cmd}

# =====================================================
# CAMERA THREAD
# =====================================================
def camera_thread():
    global camera,current_frame
    while True:
        if camera_streaming and camera:
            req=camera.capture_request()
            frame=req.make_array("main")
            req.release()
            img=Image.fromarray(frame)
            buf=io.BytesIO()
            img.save(buf,format="JPEG")
            current_frame=buf.getvalue()
        time.sleep(0.03)

if PICAMERA2_AVAILABLE:
    threading.Thread(target=camera_thread,daemon=True).start()

# =====================================================
# CAMERA API
# =====================================================
@app.post("/api/camera/start")
async def start_camera():
    global camera,camera_streaming

    camera=Picamera2()
    camera.configure(camera.create_preview_configuration())
    camera.start()
    camera_streaming=True
    return {"started":True}

@app.get("/api/camera/stream")
async def stream():
    def gen():
        while camera_streaming:
            if current_frame:
                yield(b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+current_frame+b"\r\n")
            time.sleep(0.03)

    return StreamingResponse(gen(),media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/camera/stop")
async def stop():
    global camera_streaming,camera
    camera_streaming=False
    if camera:
        camera.stop()
        camera.close()
        camera=None
    return {"stopped":True}

# =====================================================
# RUN
# =====================================================
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)

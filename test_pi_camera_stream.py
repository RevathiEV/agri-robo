#!/usr/bin/env python3
"""
Pi Camera Web Streamer
Streams live camera feed to web browser - accessible from laptop
Open http://<raspberry-pi-ip>:8080 in your laptop browser
Press 's' in terminal to start, 'q' to stop
"""
import sys
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

# Add system dist-packages to path
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

print("=" * 70)
print("Pi Camera Web Streamer")
print("=" * 70)
print("This will create a web server to stream camera to your laptop")
print("=" * 70)

# Check if picamera2 is available
print("\n1. Checking picamera2 availability...")
try:
    from picamera2 import Picamera2
    print("   ✓ picamera2 is available")
except ImportError as e:
    print(f"   ✗ picamera2 not available: {e}")
    sys.exit(1)

# Check for cameras
print("\n2. Checking for cameras...")
try:
    camera_info = Picamera2.global_camera_info()
    camera_count = len(camera_info)
    print(f"   Detected {camera_count} camera(s)")
    
    if camera_count == 0:
        print("   ✗ NO CAMERAS DETECTED")
        sys.exit(1)
    
    for i, info in enumerate(camera_info):
        model = info.get('Model', 'Unknown')
        print(f"   Camera {i}: {model}")
except Exception as e:
    print(f"   ✗ Error checking cameras: {e}")
    sys.exit(1)

# Global variables
camera = None
streaming = False
frame_buffer = None
frame_lock = threading.Lock()

# Image processing imports
from PIL import Image, ImageEnhance
import numpy as np

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
    """Apply minimal post-processing"""
    rgb_frame = convert_frame_to_rgb(frame)
    mean_brightness = rgb_frame.mean()
    
    image = Image.fromarray(rgb_frame, 'RGB')
    
    if mean_brightness < 30:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.3)
    
    r, g, b = image.split()
    b = ImageEnhance.Brightness(b).enhance(0.98)
    r = ImageEnhance.Brightness(r).enhance(1.01)
    g = ImageEnhance.Brightness(g).enhance(1.00)
    image = Image.merge('RGB', (r, g, b))
    
    return np.array(image)

def camera_thread():
    """Thread that captures frames from camera"""
    global camera, streaming, frame_buffer
    
    try:
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
        print("   ✓ Camera initialized")
        
        # Warm up
        for i in range(5):
            test_request = camera.capture_request()
            test_request.release()
            time.sleep(0.3)
        
        print("   ✓ Camera ready")
        
        # Capture loop
        while True:
            if streaming and camera is not None:
                try:
                    request = camera.capture_request()
                    frame = request.make_array("main")
                    request.release()
                    
                    # Process frame
                    rgb_frame = process_frame(frame)
                    
                    # Convert to JPEG
                    image = Image.fromarray(rgb_frame, 'RGB')
                    img_bytes = BytesIO()
                    image.save(img_bytes, format='JPEG', quality=85)
                    img_bytes.seek(0)
                    
                    # Update frame buffer
                    with frame_lock:
                        frame_buffer = img_bytes.getvalue()
                except Exception as e:
                    print(f"   Error capturing frame: {e}")
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    except Exception as e:
        print(f"   Camera thread error: {e}")
        import traceback
        traceback.print_exc()

class StreamingHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming"""
    
    def do_GET(self):
        global streaming, frame_buffer
        
        if self.path == '/':
            # Serve HTML page
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Pi Camera Live Stream</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        text-align: center;
                        background: #1a1a1a;
                        color: white;
                        margin: 0;
                        padding: 20px;
                    }
                    .container {
                        max-width: 800px;
                        margin: 0 auto;
                    }
                    h1 { color: #4CAF50; }
                    #video {
                        border: 3px solid #4CAF50;
                        border-radius: 10px;
                        max-width: 100%;
                        background: #000;
                    }
                    .controls {
                        margin: 20px 0;
                    }
                    button {
                        background: #4CAF50;
                        color: white;
                        border: none;
                        padding: 15px 30px;
                        font-size: 18px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin: 0 10px;
                    }
                    button:hover { background: #45a049; }
                    button:disabled {
                        background: #666;
                        cursor: not-allowed;
                    }
                    .status {
                        margin: 10px 0;
                        font-size: 16px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📷 Pi Camera Live Stream</h1>
                    <div class="status" id="status">Stopped - Click Start to begin</div>
                    <div class="controls">
                        <button onclick="startStream()" id="startBtn">▶️ START</button>
                        <button onclick="stopStream()" id="stopBtn" disabled>⏹️ STOP</button>
                    </div>
                    <img id="video" src="/stream" alt="Camera Stream" style="display:none;">
                </div>
                <script>
                    let streaming = false;
                    const video = document.getElementById('video');
                    const status = document.getElementById('status');
                    const startBtn = document.getElementById('startBtn');
                    const stopBtn = document.getElementById('stopBtn');
                    
                    function startStream() {
                        fetch('/start', {method: 'POST'})
                            .then(() => {
                                streaming = true;
                                video.style.display = 'block';
                                video.src = '/stream?t=' + Date.now();
                                status.textContent = 'Streaming...';
                                startBtn.disabled = true;
                                stopBtn.disabled = false;
                            });
                    }
                    
                    function stopStream() {
                        fetch('/stop', {method: 'POST'})
                            .then(() => {
                                streaming = false;
                                video.style.display = 'none';
                                video.src = '';
                                status.textContent = 'Stopped';
                                startBtn.disabled = false;
                                stopBtn.disabled = true;
                            });
                    }
                    
                    // Auto-reconnect if stream fails
                    video.onerror = function() {
                        if (streaming) {
                            setTimeout(() => {
                                video.src = '/stream?t=' + Date.now();
                            }, 1000);
                        }
                    };
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        elif self.path == '/stream':
            # MJPEG stream
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while streaming:
                with frame_lock:
                    if frame_buffer:
                        try:
                            self.wfile.write(b'--frame\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(frame_buffer)))
                            self.end_headers()
                            self.wfile.write(frame_buffer)
                            self.wfile.write(b'\r\n')
                        except:
                            break
                time.sleep(0.033)  # ~30 FPS
        
        elif self.path == '/start':
            streaming = True
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "started"}')
        
        elif self.path == '/stop':
            streaming = False
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "stopped"}')
    
    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass

def get_local_ip():
    """Get local IP address"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

# Initialize camera in background thread
print("\n3. Initializing camera...")
camera_thread_obj = threading.Thread(target=camera_thread, daemon=True)
camera_thread_obj.start()

# Wait for camera to initialize
time.sleep(2)

# Start HTTP server
print("\n4. Starting web server...")
port = 8080
ip_address = get_local_ip()

try:
    server = HTTPServer(('0.0.0.0', port), StreamingHandler)
    print(f"   ✓ Server started on port {port}")
    print(f"\n" + "=" * 70)
    print("🌐 Camera Stream is now available!")
    print("=" * 70)
    print(f"\nOpen this URL in your laptop browser:")
    print(f"   http://{ip_address}:{port}")
    print(f"\nOr if on the same network:")
    print(f"   http://raspberrypi.local:8080")
    print(f"\nControls:")
    print(f"  - Use START/STOP buttons in the browser")
    print(f"  - Or press 'q' in terminal to quit server")
    print("=" * 70 + "\n")
    
    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    # Wait for user to quit
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        server.shutdown()
        streaming = False
        if camera:
            try:
                camera.stop()
                camera.close()
            except:
                pass
        print("✓ Server stopped")
        print("✓ Camera cleaned up")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


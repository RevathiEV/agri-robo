#!/usr/bin/env python3
"""
Live Pi Camera Preview Script
Shows live video feed frame-by-frame with option to capture images
Press 'c' to capture, 'q' to quit
"""
import sys
import os
import time

# Add system dist-packages to path for libcamera
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

print("=" * 60)
print("Pi Camera Live Preview")
print("=" * 60)
print("Controls:")
print("  Press 'c' to capture an image")
print("  Press 'q' to quit")
print("=" * 60)

# Check if picamera2 is available
print("\n1. Checking picamera2 availability...")
try:
    from picamera2 import Picamera2
    print("   ✓ picamera2 is available")
except ImportError as e:
    print(f"   ✗ picamera2 not available: {e}")
    print("   Install with: sudo apt-get install -y python3-picamera2")
    sys.exit(1)

# Check if OpenCV is available and if display is available
CV2_AVAILABLE = False
DISPLAY_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
    # Check if we have a display
    import os
    if 'DISPLAY' in os.environ:
        DISPLAY_AVAILABLE = True
        print("   ✓ OpenCV is available for display")
    else:
        DISPLAY_AVAILABLE = False
        print("   ⚠ OpenCV available but no X11 display detected")
        print("   (Running over SSH? Use -X flag: ssh -X user@host)")
        print("   Will save frames to files instead")
except ImportError:
    CV2_AVAILABLE = False
    print("   ⚠ OpenCV not available - install with: sudo apt-get install python3-opencv")
    print("   Will save frames to files instead")
except Exception as e:
    CV2_AVAILABLE = False
    print(f"   ⚠ OpenCV error: {e}")
    print("   Will save frames to files instead")

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

# Initialize camera
print("\n3. Initializing camera...")
camera = None
try:
    camera = Picamera2(camera_num=0)
    
    # Configure camera - same settings as test script
    try:
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            colour_space="sRGB"
        )
        camera.configure(config)
        print("   ✓ Camera configured with RGB888 and sRGB color space")
    except Exception as e:
        print(f"   ⚠ RGB888 not available, using default format: {e}")
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(config)
        print("   ✓ Camera configured with default format")
    
    # Set camera controls - natural auto settings
    try:
        camera.set_controls({
            "AwbEnable": True,      # Auto white balance
            "AeEnable": True,       # Auto exposure
        })
        print("   ✓ Camera controls set (auto white balance, auto exposure)")
    except Exception as ctrl_error:
        print(f"   ⚠ Could not set camera controls: {ctrl_error}")
    
    # Start camera
    camera.start()
    print("   ✓ Camera started")
    
    # Let camera adjust for a few frames
    print("\n4. Adjusting camera (warming up)...")
    for i in range(5):
        test_request = camera.capture_request()
        test_frame = test_request.make_array("main")
        test_request.release()
        time.sleep(0.3)
    print("   ✓ Camera ready")
    
    # Image processing imports
    from PIL import Image, ImageEnhance
    import numpy as np
    
    def convert_frame_to_rgb(frame):
        """Convert camera frame to RGB format"""
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            # XBGR8888 format - convert to RGB
            rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            rgb_frame[:, :, 0] = frame[:, :, 2]  # R from channel 2
            rgb_frame[:, :, 1] = frame[:, :, 1]  # G from channel 1
            rgb_frame[:, :, 2] = frame[:, :, 0]  # B from channel 0
            return rgb_frame
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            return frame
        else:
            return frame
    
    def process_frame(frame):
        """Apply minimal post-processing (same as test script)"""
        rgb_frame = convert_frame_to_rgb(frame)
        mean_brightness = rgb_frame.mean()
        
        image = Image.fromarray(rgb_frame, 'RGB')
        
        # Apply minimal post-processing - only if very dark
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
    
    def save_image(image_array, filename):
        """Save image to file"""
        image = Image.fromarray(image_array, 'RGB')
        image.save(filename, "JPEG", quality=95, optimize=True)
        return os.path.getsize(filename)
    
    # Main loop - show live preview
    print("\n5. Starting live preview...")
    print("   Press 'c' to capture, 'q' to quit\n")
    
    frame_count = 0
    capture_count = 0
    
    if CV2_AVAILABLE and DISPLAY_AVAILABLE:
        # Use OpenCV for live display
        window_name = "Pi Camera Live Preview"
        display_works = False
        try:
            # Try to create window - this will fail if no X11 display
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            display_works = True
        except Exception as e:
            print(f"   ⚠ Cannot create display window: {e}")
            print("   Falling back to file saving mode")
            display_works = False
            DISPLAY_AVAILABLE = False
        
        if display_works:
            try:
                while True:
                    # Capture frame
                    request = camera.capture_request()
                    frame = request.make_array("main")
                    request.release()
                    
                    # Process frame
                    rgb_frame = process_frame(frame)
                    
                    # Convert RGB to BGR for OpenCV display
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    
                    # Add text overlay
                    cv2.putText(bgr_frame, f"Frame: {frame_count} | Press 'c' to capture, 'q' to quit", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display frame
                    cv2.imshow(window_name, bgr_frame)
                    
                    frame_count += 1
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        print("\n   Quitting...")
                        break
                    elif key == ord('c'):
                        # Capture image
                        capture_count += 1
                        filename = f"test_camera_capture_{capture_count:03d}.jpg"
                        file_size = save_image(rgb_frame, filename)
                        print(f"   ✓ Captured image {capture_count}: {filename} ({file_size/1024:.1f} KB)")
            
            except KeyboardInterrupt:
                print("\n   Interrupted by user")
            except Exception as e:
                print(f"\n   Display error during preview: {e}")
                print("   Falling back to file saving mode")
                DISPLAY_AVAILABLE = False
            finally:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
    
    if not CV2_AVAILABLE or not DISPLAY_AVAILABLE:
        # Fallback: Save frames to files (no display)
        print("   No display available - saving frames continuously")
        print("   Press Ctrl+C to stop")
        print("   Tip: To see live preview, connect with X11 forwarding:")
        print("        ssh -X user@raspberrypi")
        print("")
        
        try:
            start_time = time.time()
            last_capture_time = time.time()
            
            while True:
                # Capture frame
                request = camera.capture_request()
                frame = request.make_array("main")
                request.release()
                
                # Process frame
                rgb_frame = process_frame(frame)
                
                frame_count += 1
                
                # Save every frame (you can adjust this)
                capture_count += 1
                filename = f"test_camera_frame_{capture_count:05d}.jpg"
                file_size = save_image(rgb_frame, filename)
                
                # Show status every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"   Frame {frame_count:5d} | FPS: {fps:5.1f} | Saved: {filename} ({file_size/1024:.1f} KB)")
                
                time.sleep(0.033)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\n   Stopped by user")
    
    print(f"\n   Total frames processed: {frame_count}")
    print(f"   Total images captured: {capture_count}")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    # Cleanup
    if camera is not None:
        try:
            camera.stop()
            camera.close()
            print("\n✓ Camera cleaned up")
        except:
            pass

print("\n" + "=" * 60)
print("✓ Done!")
print("=" * 60)


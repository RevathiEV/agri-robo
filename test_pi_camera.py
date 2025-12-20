#!/usr/bin/env python3
"""
Simple Pi Camera Test Script
Tests if the Pi camera is working and captures one image
"""
import sys
import os

# Add system dist-packages to path for libcamera (Raspberry Pi system package)
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

print("=" * 60)
print("Pi Camera Test")
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

# Check for cameras
print("\n2. Checking for cameras...")
try:
    camera_info = Picamera2.global_camera_info()
    camera_count = len(camera_info)
    print(f"   Detected {camera_count} camera(s)")
    
    if camera_count == 0:
        print("   ✗ NO CAMERAS DETECTED")
        print("\n   Please check:")
        print("   - Camera ribbon cable is connected to J3 port (CAM/DISP 0)")
        print("   - Camera interface is enabled: sudo raspi-config")
        print("   - Reboot after enabling: sudo reboot")
        sys.exit(1)
    
    for i, info in enumerate(camera_info):
        model = info.get('Model', 'Unknown')
        print(f"   Camera {i}: {model}")
except Exception as e:
    print(f"   ✗ Error checking cameras: {e}")
    sys.exit(1)

# Initialize and capture
print("\n3. Initializing camera...")
camera = None
try:
    camera = Picamera2(camera_num=0)
    
    # Configure camera with better color settings
    # Try RGB888 first, fallback to default if not available
    try:
        config = camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            colour_space="sRGB"
        )
        camera.configure(config)
        print("   ✓ Camera configured with RGB888 and sRGB color space")
    except Exception as e:
        print(f"   ⚠ RGB888 not available, using default format: {e}")
        # Fallback to default configuration
        config = camera.create_preview_configuration(main={"size": (640, 480)})
        camera.configure(config)
        print("   ✓ Camera configured with default format")
    
    # Set camera controls - based on old picamera defaults (natural, auto settings)
    # Old picamera defaults: brightness=50, contrast=0, saturation=0, sharpness=0
    # exposure_mode='auto', awb_mode='auto', meter_mode='average', iso=0 (auto)
    # We'll use pure auto settings and let the camera adjust naturally
    try:
        camera.set_controls({
            "AwbEnable": True,      # Auto white balance (like old picamera awb_mode='auto')
            "AeEnable": True,       # Auto exposure (like old picamera exposure_mode='auto')
            # Don't force ExposureTime or AnalogueGain - let auto exposure handle it
            # This matches old picamera behavior where iso=0 (auto) was default
        })
        print("   ✓ Camera controls set (auto white balance, auto exposure - natural defaults)")
    except Exception as ctrl_error:
        print(f"   ⚠ Could not set camera controls: {ctrl_error}")
        print("   ⚠ Using default camera controls")
    
    # Start camera
    camera.start()
    print("   ✓ Camera started")
    
    # Wait and capture test frames to let camera adjust naturally
    # Old picamera would let auto exposure and auto white balance settle
    import time
    print("\n4. Adjusting white balance and exposure (capturing test frames)...")
    print("   Letting camera adjust naturally (like old picamera auto settings)...")
    
    # Let camera adjust naturally - old picamera would take a moment to settle
    # Capture enough frames for auto exposure and AWB to converge
    for i in range(10):  # More frames for natural adjustment
        test_request = camera.capture_request()
        test_frame = test_request.make_array("main")
        test_request.release()
        if i == 0 or i == 4 or i == 9:
            mean_brightness = test_frame.mean() if len(test_frame.shape) == 3 else test_frame.mean()
            print(f"   Test frame {i+1}/10 captured (avg brightness: {mean_brightness:.1f})")
        time.sleep(0.5)  # Give camera time to adjust between frames
    
    print("   ✓ Camera white balance and exposure adjusted (natural auto settings)")
    
    # Capture final image
    print("\n5. Capturing final image...")
    request = camera.capture_request()
    frame = request.make_array("main")
    request.release()
    print(f"   ✓ Frame captured (shape: {frame.shape})")
    
    # Save image
    from PIL import Image
    import numpy as np
    
    # Convert to RGB if needed - keep it natural and realistic
    from PIL import ImageEnhance
    
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        # XBGR8888 format - convert to RGB
        # On Pi 5 with picamera2, XBGR8888 is stored as BGRA (Blue, Green, Red, Alpha)
        # Extract RGB channels correctly
        rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = frame[:, :, 2]  # R from channel 2
        rgb_frame[:, :, 1] = frame[:, :, 1]  # G from channel 1
        rgb_frame[:, :, 2] = frame[:, :, 0]  # B from channel 0
        
        mean_brightness = rgb_frame.mean()
        print(f"   Image brightness: {mean_brightness:.1f} (0-255 scale)")
        
        image = Image.fromarray(rgb_frame, 'RGB')
        
        # Apply minimal post-processing - old picamera had defaults:
        # brightness=50 (0.5), contrast=0, saturation=0, sharpness=0
        # Only apply correction if image is very dark (let auto exposure do its job)
        if mean_brightness < 30:
            print("   ⚠ Image is very dark, applying minimal brightness correction...")
            enhancer = ImageEnhance.Brightness(image)
            brightness_factor = 1.3  # Moderate brightening only if very dark
            image = enhancer.enhance(brightness_factor)
            print(f"   ✓ Applied minimal brightness correction (factor: {brightness_factor:.2f})")
        else:
            print("   ✓ Image brightness is good (auto exposure working)")
        
        # Apply very minimal color correction - old picamera had saturation=0 (neutral)
        # Only subtle adjustment to reduce any blue/purple cast
        r, g, b = image.split()
        
        # Very subtle adjustments - minimal processing like old picamera defaults
        b = ImageEnhance.Brightness(b).enhance(0.98)  # Very slight reduction in blue
        r = ImageEnhance.Brightness(r).enhance(1.01)  # Very slight boost to red
        g = ImageEnhance.Brightness(g).enhance(1.00)  # Keep green neutral (saturation=0)
        
        # Merge channels back
        image = Image.merge('RGB', (r, g, b))
        
        # No saturation enhancement - old picamera default was saturation=0 (neutral)
        
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        # Already RGB
        mean_brightness = frame.mean()
        print(f"   Image brightness: {mean_brightness:.1f} (0-255 scale)")
        
        image = Image.fromarray(frame, 'RGB')
        
        # Apply minimal post-processing - old picamera defaults
        if mean_brightness < 30:
            print("   ⚠ Image is very dark, applying minimal brightness correction...")
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.3)
            print(f"   ✓ Applied minimal brightness correction")
        else:
            print("   ✓ Image brightness is good (auto exposure working)")
        
        # Apply very minimal color correction - old picamera saturation=0 (neutral)
        r, g, b = image.split()
        b = ImageEnhance.Brightness(b).enhance(0.98)
        r = ImageEnhance.Brightness(r).enhance(1.01)
        g = ImageEnhance.Brightness(g).enhance(1.00)
        image = Image.merge('RGB', (r, g, b))
    else:
        image = Image.fromarray(frame)
    
    # Optional: Adjust brightness/contrast if needed
    # enhancer = ImageEnhance.Brightness(image)
    # image = enhancer.enhance(1.05)  # Slightly brighter
    
    # Save image with better quality
    output_path = "test_camera_capture.jpg"
    image.save(output_path, "JPEG", quality=95, optimize=True)
    print(f"   ✓ Image saved to: {output_path}")
    
    # Get file size
    file_size = os.path.getsize(output_path)
    print(f"   ✓ File size: {file_size / 1024:.2f} KB")
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS! Pi camera is working!")
    print(f"✓ Image saved to: {os.path.abspath(output_path)}")
    print("=" * 60)
    
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


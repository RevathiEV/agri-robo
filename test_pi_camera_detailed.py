#!/usr/bin/env python3
"""
Detailed Pi Camera Test Script for Raspberry Pi 5
Provides more diagnostic information
"""
import sys
import os
import subprocess

# Add system dist-packages to path
if '/usr/lib/python3/dist-packages' not in sys.path:
    sys.path.insert(0, '/usr/lib/python3/dist-packages')

print("=" * 70)
print("Detailed Pi Camera Diagnostics for Raspberry Pi 5")
print("=" * 70)

# 1. System info
print("\n1. System Information:")
try:
    with open('/proc/device-tree/model', 'r') as f:
        model = f.read().strip()
    print(f"   Model: {model}")
except:
    print("   Could not read model")

# 2. Check picamera2
print("\n2. Checking picamera2...")
try:
    from picamera2 import Picamera2
    print("   ✓ picamera2 available")
except ImportError as e:
    print(f"   ✗ picamera2 not available: {e}")
    sys.exit(1)

# 3. Check camera detection
print("\n3. Camera Detection:")
try:
    camera_info = Picamera2.global_camera_info()
    print(f"   Detected cameras: {len(camera_info)}")
    
    if len(camera_info) == 0:
        print("   ✗ NO CAMERAS DETECTED")
        print("\n   Troubleshooting for Raspberry Pi 5:")
        print("   a) POWER OFF the Pi completely")
        print("   b) Check ribbon cable orientation:")
        print("      - On Pi 5: Gold contacts face DOWN (toward USB ports)")
        print("      - On camera: Gold contacts face camera lens")
        print("   c) Unlock connector (lift black tab), remove cable")
        print("   d) Reinsert cable FIRMLY until it clicks")
        print("   e) Lock connector (push black tab down)")
        print("   f) Ensure cable is fully inserted on BOTH ends")
        print("   g) Power on and test again")
        print("\n   If still not detected:")
        print("   - Try a different camera module if available")
        print("   - Check if camera module is damaged")
        print("   - Verify camera module is Pi 5 compatible")
    else:
        print("   ✓ Camera(s) detected:")
        for i, info in enumerate(camera_info):
            print(f"      Camera {i}:")
            for key, value in info.items():
                print(f"        {key}: {value}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 4. Check video devices
print("\n4. Video Devices:")
try:
    video_devices = [d for d in os.listdir('/dev') if d.startswith('video')]
    camera_devices = []
    for dev in sorted(video_devices):
        try:
            # Check if it's a camera device
            dev_path = f"/sys/class/video4linux/{dev}/name"
            if os.path.exists(dev_path):
                with open(dev_path, 'r') as f:
                    name = f.read().strip()
                if 'camera' in name.lower() or 'isp' in name.lower():
                    camera_devices.append((dev, name))
        except:
            pass
    
    if camera_devices:
        print(f"   Found {len(camera_devices)} potential camera device(s):")
        for dev, name in camera_devices:
            print(f"      /dev/{dev}: {name}")
    else:
        print("   No camera devices found in /dev/video*")
        print("   (Only codec/display devices present)")
except Exception as e:
    print(f"   Error checking devices: {e}")

# 5. Check config
print("\n5. Configuration:")
try:
    config_files = ['/boot/firmware/config.txt', '/boot/config.txt']
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                content = f.read()
                if 'camera_auto_detect=1' in content:
                    print(f"   ✓ camera_auto_detect=1 in {config_file}")
                else:
                    print(f"   ⚠ camera_auto_detect not found in {config_file}")
            break
except Exception as e:
    print(f"   Error: {e}")

# 6. Try rpicam-hello
print("\n6. Testing with rpicam-hello:")
try:
    result = subprocess.run(['rpicam-hello', '--list-cameras'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"   Error: {result.stderr.strip()}")
except FileNotFoundError:
    print("   rpicam-hello not installed")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 70)
print("Diagnostics complete!")
print("=" * 70)



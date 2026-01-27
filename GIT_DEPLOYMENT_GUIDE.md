# Git Deployment Guide - Motor Control Feature

## Overview
This guide explains how to push the motor control code changes to Git and pull them on your Raspberry Pi.

---

## 📋 What Was Changed

### Files Modified:
1. **`backend/main.py`**
   - Added GPIO control for relay (Pin 12 / GPIO 18)
   - Motor activates automatically for 3 seconds when disease is detected
   - Motor stays OFF for "healthy" and "Not_A_Leaf" detections
   - Motor is OFF at startup

2. **`backend/requirements.txt`**
   - Added `RPi.GPIO>=0.7.1` for GPIO control

---

## 🖥️ Step 1: Push Changes from Your Computer

### On Windows (Your Current Machine):

1. **Check current status:**
   ```powershell
   git status
   ```

2. **Add modified files:**
   ```powershell
   git add backend/main.py
   git add backend/requirements.txt
   ```

3. **Commit changes:**
   ```powershell
   git commit -m "Add GPIO motor control: Auto-activate pump for 3s on disease detection"
   ```

4. **Push to remote repository:**
   ```powershell
   git push origin main
   ```
   (or `git push origin master` if your branch is named `master`)

---

## 🍓 Step 2: Pull Changes on Raspberry Pi

### SSH into your Raspberry Pi:
```bash
ssh pi@<your-pi-ip-address>
# Example: ssh pi@10.86.141.41
```

### Navigate to project directory:
```bash
cd ~/tomato
# or wherever your project is located
```

### Pull latest changes:
```bash
git pull origin main
# or git pull origin master
```

### Install/Update Dependencies:

**Option 1: Using apt (Recommended on Raspberry Pi):**
```bash
sudo apt-get update
sudo apt-get install python3-rpi.gpio
```

**Option 2: Using pip (if apt doesn't work):**
```bash
# Activate your virtual environment first
source backend/venv/bin/activate

# Install RPi.GPIO
pip install RPi.GPIO>=0.7.1

# Or install all requirements
pip install -r backend/requirements.txt
```

---

## ✅ Step 3: Verify Installation

### Check if RPi.GPIO is installed:
```bash
python3 -c "import RPi.GPIO; print('RPi.GPIO installed successfully')"
```

### Test GPIO access (optional):
```bash
python3 -c "import RPi.GPIO as GPIO; GPIO.setmode(GPIO.BCM); print('GPIO access OK')"
```

---

## 🚀 Step 4: Restart Backend Service

### If running manually:
1. Stop the current backend (Ctrl+C)
2. Restart:
   ```bash
   cd backend
   source venv/bin/activate
   python main.py
   ```

### If using a service/script:
```bash
# Restart your backend service
sudo systemctl restart your-backend-service
# or
./start_backend.sh
```

---

## 🧪 Step 5: Test the Motor Control

### Expected Behavior:

1. **Startup:**
   - Backend starts → Motor should be OFF
   - Check logs: `GPIO Pin 18 (Physical Pin 12) initialized - Motor OFF`

2. **Disease Detection:**
   - Upload image with disease → Motor turns ON for 3 seconds → Motor turns OFF automatically
   - Check logs: `Motor activated (GPIO 18 HIGH) - Disease detected`
   - After 3s: `Motor deactivated (GPIO 18 LOW) after 3.0s`

3. **Healthy/Not_A_Leaf:**
   - Upload healthy leaf → Motor stays OFF
   - Upload non-leaf → Motor stays OFF

---

## 🔍 Troubleshooting

### Issue: "RPi.GPIO not available"
**Solution:**
```bash
sudo apt-get install python3-rpi.gpio
# or
pip install RPi.GPIO
```

### Issue: "Permission denied" for GPIO
**Solution:**
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
# Log out and log back in, or:
newgrp gpio
```

### Issue: Motor doesn't activate
**Check:**
1. GPIO pin is correct (Pin 12 = GPIO 18)
2. Wiring is correct (see connection diagram)
3. Check logs for GPIO errors
4. Verify relay module is working

### Issue: Git pull conflicts
**Solution:**
```bash
# If you have local changes you want to keep:
git stash
git pull origin main
git stash pop

# If you want to discard local changes:
git reset --hard origin/main
git pull origin main
```

---

## 📝 Quick Reference Commands

### On Your Computer (Windows):
```powershell
git add backend/main.py backend/requirements.txt
git commit -m "Add GPIO motor control"
git push origin main
```

### On Raspberry Pi:
```bash
cd ~/tomato
git pull origin main
sudo apt-get install python3-rpi.gpio
# Restart backend
```

---

## 🎯 Summary

1. ✅ **Push from Windows:** `git add`, `git commit`, `git push`
2. ✅ **Pull on Pi:** `git pull origin main`
3. ✅ **Install GPIO:** `sudo apt-get install python3-rpi.gpio`
4. ✅ **Restart backend:** Restart your backend service
5. ✅ **Test:** Upload disease image → Motor should activate for 3 seconds

---

## 📌 Notes

- Motor only activates for **9 diseases** (not healthy, not Not_A_Leaf)
- Motor runs for **3 seconds** then turns OFF automatically
- Motor is **OFF at startup** (safe)
- GPIO cleanup happens on backend shutdown
- Code works on non-Pi systems (with warnings, no GPIO control)

---

**Need Help?** Check the logs in your backend console for detailed error messages.

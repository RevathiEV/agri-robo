# 🔧 Motor Auto-Start Fix

## Problem
Motor turns ON automatically when running `python main.py`

## Root Causes

1. **Wrong Relay Type Setting** (Most Common)
   - Code assumes `RELAY_ACTIVE_LOW = True`
   - But your relay might be Active HIGH
   - **Solution**: Change `RELAY_ACTIVE_LOW = False` in `spray_pump_control.py`

2. **GPIO State Not Cleaned**
   - Previous GPIO state might persist
   - **Solution**: Code now cleans up GPIO before initialization

3. **Pins Floating Before Initialization**
   - GPIO pins in undefined state
   - **Solution**: Code now sets pins to OFF state before configuring as outputs

## Code Changes Made

### 1. Enhanced Initialization (`spray_pump_control.py`)
- ✅ Clean up previous GPIO state first
- ✅ Set pins to OFF state BEFORE configuring as outputs
- ✅ Multiple OFF state checks
- ✅ Detailed logging to track state

### 2. Startup Verification (`main.py`)
- ✅ Triple-check motors are OFF on startup
- ✅ Reset state variables
- ✅ Added delays to ensure state stability

## How to Fix

### Step 1: Check Your Relay Type

Run this diagnostic on your Raspberry Pi:

```bash
cd ~/agri-robo-tomato/backend
python -c "from spray_pump_control import diagnose_motor_startup; diagnose_motor_startup()"
```

### Step 2: Test Relay Type

**If motor turns ON when Pi starts:**

1. Edit `backend/spray_pump_control.py`
2. Find line: `RELAY_ACTIVE_LOW = True`
3. Change to: `RELAY_ACTIVE_LOW = False`
4. Save and restart

**If motor is OFF when Pi starts:**
- Keep `RELAY_ACTIVE_LOW = True` (current setting is correct)

### Step 3: Verify Connections

See `PROPER_CONNECTIONS.md` for complete wiring diagram.

**Key Points:**
- Relay IN → GPIO pin (NOT 5V or GND)
- Relay GND → Pi GND
- Relay COM → 9V Battery +
- Relay NO → Motor +
- Motor - → Battery -
- Pi GND ↔ Battery - (common ground)

### Step 4: Test After Fix

```bash
cd ~/agri-robo-tomato/backend
python main.py
```

**Expected Output:**
```
[STARTUP] Initializing spray pumps...
[INIT] Cleaned up previous GPIO state
[INIT] Setting up GPIO pins with OFF state...
✓ Spray pumps initialized successfully
  - Motor A: GPIO 17 (Physical Pin 11) - State: OFF
  - Motor B: GPIO 27 (Physical Pin 13) - State: OFF
✓ Motors explicitly verified OFF on startup
```

**Motor should NOT turn on!**

## Testing Motor Control

1. **Detect a disease** (capture image)
2. **Click "Start Dispenser"** → Motor should turn ON
3. **Click "Stop Dispenser"** → Motor should turn OFF

## Still Having Issues?

1. **Check relay module jumper** (if present)
   - Set to LOW trigger for Active LOW relays
   - Set to HIGH trigger for Active HIGH relays

2. **Test relay manually:**
   - Disconnect relay IN from GPIO
   - Connect IN to Pi GND → Motor ON? (Active LOW)
   - Connect IN to Pi 5V → Motor ON? (Active HIGH)

3. **Check wiring:**
   - Verify relay IN is connected to GPIO, not 5V
   - Verify all ground connections are common

4. **Check logs:**
   - Look for `[INIT]`, `[STARTUP]`, `[TURN_ON_PUMP]` messages
   - These will show exactly what's happening

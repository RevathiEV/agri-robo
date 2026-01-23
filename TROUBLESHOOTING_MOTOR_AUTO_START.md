# 🔧 Troubleshooting: Motor Auto-Start Issue

## Problem
Motor turns ON automatically when running `python main.py`, even though code is set to keep it OFF.

## Root Causes & Solutions

### 1. Relay Module Has Internal Pull-Up Resistor ⚠️ (Most Likely)

**Symptom:** Motor turns ON immediately, GPIO reads HIGH even when set to LOW

**Solution A - Add External Pull-Down Resistor:**
- Add a 10kΩ resistor between Relay IN and GND
- This will pull the pin LOW when GPIO is LOW
- **Wiring:** GPIO 17 → 10kΩ resistor → GND (parallel to Relay IN)

**Solution B - Use Different GPIO Pin:**
- Some GPIO pins have hardware pull-down resistors
- Try GPIO 2, 3, 4, 14, 15, 17, 18, 27, 22, 23
- Update `SPRAY_PUMP_A_GPIO` in `spray_pump_control.py`

### 2. Wiring Issue: Relay IN Connected to 5V Instead of GPIO

**Symptom:** Motor always ON, cannot be controlled

**Check:**
- Relay IN should connect to **GPIO 17** (Pin 11), NOT Pin 2 (5V)
- Relay IN should connect to **GPIO 27** (Pin 13), NOT Pin 4 (5V)
- Verify with multimeter: Relay IN should read 0V when GPIO is LOW

**Fix:**
- Disconnect Relay IN from 5V
- Connect Relay IN to correct GPIO pin

### 3. Relay Module Jumper Setting

**Symptom:** Motor behavior doesn't match code settings

**Check:**
- Look for a jumper on your relay module (usually labeled "JD-VCC" or "VCC-JD")
- **For Active HIGH relays:** Jumper should be set to HIGH trigger
- **For Active LOW relays:** Jumper should be set to LOW trigger

**Common Relay Module Jumpers:**
- **JD-VCC:** Optical isolation (recommended) - Use this
- **VCC-JD:** Direct connection - May cause issues

### 4. GPIO Pin Floating

**Symptom:** Motor state is unpredictable

**Solution:**
- Code now sets pins as inputs with pull-down before configuring as outputs
- This ensures pins are LOW before becoming outputs

### 5. Wrong Relay Type Setting

**Current Setting:** `RELAY_ACTIVE_LOW = False` (Active HIGH)

**Test:**
1. Run diagnostic: `python backend/test_gpio_state.py`
2. Check if GPIO state matches expected state
3. If motor is ON when GPIO is LOW → Change to `RELAY_ACTIVE_LOW = True`
4. If motor is ON when GPIO is HIGH → Keep `RELAY_ACTIVE_LOW = False`

## Diagnostic Steps

### Step 1: Run GPIO Diagnostic

```bash
cd ~/agri-robo-tomato/backend
python test_gpio_state.py
```

This will:
- Test actual GPIO state
- Verify if pins can be set to LOW/HIGH
- Identify if pull-up resistors are causing issues

### Step 2: Check Wiring with Multimeter

1. **With Pi OFF:**
   - Measure resistance between Relay IN and GND
   - Should be high resistance (not shorted)

2. **With Pi ON, GPIO set to LOW:**
   - Measure voltage at Relay IN
   - Should read ~0V (LOW)
   - If reads 3.3V or 5V → Pull-up resistor issue

3. **With Pi ON, GPIO set to HIGH:**
   - Measure voltage at Relay IN
   - Should read ~3.3V (HIGH)

### Step 3: Test Relay Manually

1. **Disconnect Relay IN from GPIO**
2. **Connect Relay IN to Pi GND:**
   - Motor should turn ON (if Active LOW)
   - Motor should turn OFF (if Active HIGH)
3. **Connect Relay IN to Pi 5V:**
   - Motor should turn OFF (if Active LOW)
   - Motor should turn ON (if Active HIGH)

### Step 4: Check Relay Module Specifications

Look for:
- **Operating voltage:** Should be 5V (not 3.3V)
- **Trigger voltage:** LOW (0-0.8V) or HIGH (2.2-5V)
- **Pull-up/Pull-down:** Check datasheet

## Hardware Solutions

### Solution 1: Add Pull-Down Resistor (Recommended)

**Components needed:**
- 10kΩ resistor (1/4W)

**Wiring:**
```
GPIO 17 (Pin 11) ──┬──→ Relay IN1
                   │
                   └──→ 10kΩ Resistor ──→ GND
```

**For Motor B:**
```
GPIO 27 (Pin 13) ──┬──→ Relay IN2
                   │
                   └──→ 10kΩ Resistor ──→ GND
```

### Solution 2: Use Optocoupler/Isolation

If relay module doesn't have proper isolation:
- Use an optocoupler between GPIO and Relay IN
- This provides electrical isolation and proper signal level

### Solution 3: Use Different Relay Module

If current module has issues:
- Get a relay module with proper LOW trigger support
- Or get one with configurable trigger level

## Code Verification

### Check Current Settings:

1. **Relay Type:**
   ```python
   # In spray_pump_control.py line 28
   RELAY_ACTIVE_LOW = False  # Should match your relay
   ```

2. **GPIO Pins:**
   ```python
   # In spray_pump_control.py
   SPRAY_PUMP_A_GPIO = 17  # Physical Pin 11
   SPRAY_PUMP_B_GPIO = 27  # Physical Pin 13
   ```

### Expected Log Output:

When running `python main.py`, you should see:
```
[INIT] Relay type: Active HIGH
[INIT] OFF state = GPIO.LOW, ON state = GPIO.HIGH
[INIT] Setting pins as inputs with pull-down first (Active HIGH relay)
[INIT] Configuring pins as outputs with initial OFF state
[INIT] GPIO pins configured and set to OFF state (GPIO=0)
✓ Motors explicitly verified OFF on startup
```

If you see GPIO=1 (HIGH) when it should be OFF, that's the problem!

## Quick Fix Checklist

- [ ] Run `python backend/test_gpio_state.py` to diagnose
- [ ] Verify wiring: Relay IN → GPIO (not 5V)
- [ ] Check relay module jumper settings
- [ ] Add external pull-down resistor if needed
- [ ] Verify `RELAY_ACTIVE_LOW = False` matches your relay
- [ ] Test with multimeter: GPIO LOW should be 0V
- [ ] Check relay module datasheet for specifications

## Still Not Working?

1. **Share diagnostic output:**
   ```bash
   python backend/test_gpio_state.py > gpio_test.txt
   ```

2. **Share startup logs:**
   ```bash
   python backend/main.py 2>&1 | tee startup.log
   ```

3. **Check relay module model number** and look up specifications

4. **Try a different GPIO pin** to rule out pin-specific issues

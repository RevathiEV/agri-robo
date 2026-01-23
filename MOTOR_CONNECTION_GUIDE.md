# 🔌 Motor Connection Guide - Spray Pump Control

## ✅ Your Current Wiring (CORRECT)

### Raspberry Pi to Relay Module:
- **Relay IN** → **GPIO 17** (Physical Pin 11) ✓
- **Relay GND** → **Pi GND** (Physical Pin 9) ✓
- **Relay VCC** → **Pi 5V** (Physical Pin 2) ✓

## ⚠️ Missing: Motor Circuit Wiring

Your relay module is correctly connected to the Raspberry Pi, but you also need to connect the **9V motor circuit** through the relay.

### Complete Motor Circuit (Motor A):

```
9V Battery/Adapter
    │
    ├─(+)─────────────────┐
    │                     │
    │                  [Relay COM]  ← Connect 9V Battery + here
    │                     │
    │                  [Relay NO]   ← Connect Motor + here
    │                     │
    │                  Motor + ────┐
    │                              │
    │                           [9V Motor]
    │                              │
    │                  Motor - ────┘
    │                     │
    │                  [Relay NC]   ← Not used (leave disconnected)
    │
    └─(-)───────────────────────────→ Connect Motor - here
```

### Step-by-Step Motor Circuit Connection:

1. **9V Battery/Adapter Positive (+)**
   - Connect to **Relay COM** (Common terminal)

2. **9V Motor Positive (+)**
   - Connect to **Relay NO** (Normally Open terminal)

3. **9V Motor Negative (-)**
   - Connect directly to **9V Battery/Adapter Negative (-)**

4. **Common Ground (Important!)**
   - Connect **Raspberry Pi GND** and **9V Battery Negative (-)** together
   - This ensures proper signal reference

### Visual Connection Diagram:

```
┌─────────────────┐
│  Raspberry Pi   │
│                 │
│  GPIO 17 ───────┼──→ Relay IN
│  GND (Pin 9) ───┼──→ Relay GND
│  5V (Pin 2) ────┼──→ Relay VCC
│                 │
└─────────────────┘
        │
        │ (Common GND)
        │
        ▼
┌─────────────────┐
│   9V Battery    │
│                 │
│  (+) ───────────┼──→ Relay COM
│  (-) ───────────┼──→ Motor (-)
│                 │      │
│                 │      │ (Common GND)
│                 │      │
└─────────────────┘      │
                         │
                         ▼
                    ┌─────────┐
                    │  Relay   │
                    │          │
                    │  COM ←───┘ (from 9V +)
                    │   │
                    │   │
                    │  NO ────→ Motor (+)
                    │          │
                    │  NC ────→ (Not used)
                    └─────────┘
```

## 🔧 Testing Your Setup

### Step 1: Test Relay Type

Run this command on your Raspberry Pi:

```bash
cd ~/agri-robo-tomato/backend
python spray_pump_control.py test-relay A
```

This will:
1. Test if your relay is **Active LOW** (LOW = ON) or **Active HIGH** (HIGH = ON)
2. Tell you which setting to use in the code

### Step 2: If Motor Doesn't Turn On

1. **Check Relay Type:**
   - Run the test above
   - If it shows "Active HIGH", edit `spray_pump_control.py`:
     ```python
     RELAY_ACTIVE_LOW = False  # Change from True to False
     ```

2. **Check Motor Circuit:**
   - Verify 9V battery is connected: Battery + → Relay COM
   - Verify motor is connected: Motor + → Relay NO, Motor - → Battery -
   - Check battery voltage with multimeter (should be ~9V)

3. **Check Relay Module:**
   - Listen for relay "click" when GPIO changes
   - Check relay module LED (if present) - should light when relay activates
   - Verify relay module jumper (if present) is set correctly

4. **Test Relay Manually:**
   - Temporarily connect Relay COM and Relay NO with a wire
   - Motor should turn on immediately
   - If motor works, relay is the issue
   - If motor doesn't work, check battery and motor connections

## 📋 Complete Connection Checklist

### Raspberry Pi Connections:
- [x] GPIO 17 (Pin 11) → Relay IN
- [x] GND (Pin 9) → Relay GND
- [x] 5V (Pin 2) → Relay VCC

### Motor Circuit (Motor A):
- [ ] 9V Battery + → Relay COM
- [ ] Motor + → Relay NO
- [ ] Motor - → 9V Battery -
- [ ] Raspberry Pi GND → 9V Battery - (common ground)

### Motor Circuit (Motor B) - Same pattern:
- [ ] GPIO 27 (Pin 13) → Relay IN2
- [ ] 9V Battery + → Relay COM2
- [ ] Motor B + → Relay NO2
- [ ] Motor B - → 9V Battery -
- [ ] Common GND connected

## 🎯 Quick Test Commands

### Test Motor A:
```bash
cd ~/agri-robo-tomato/backend
python -c "
from spray_pump_control import init_spray_pumps, turn_on_pump, turn_off_pump, cleanup_spray_pumps
import time
init_spray_pumps()
print('Turning motor ON for 3 seconds...')
turn_on_pump('A')
time.sleep(3)
turn_off_pump('A')
print('Motor should have turned OFF')
cleanup_spray_pumps()
"
```

### Test Relay Type:
```bash
python spray_pump_control.py test-relay A
```

## 🔍 Troubleshooting

### Motor doesn't turn on but relay clicks:
- ✅ Relay is working
- ❌ Motor circuit wiring issue
- **Fix:** Check battery → COM → NO → Motor connections

### Motor doesn't turn on and no relay click:
- ❌ GPIO signal issue or wrong relay type
- **Fix:** Run relay type test, change `RELAY_ACTIVE_LOW` setting

### Motor turns on but won't turn off:
- ❌ Relay stuck or wrong OFF state
- **Fix:** Check relay type, verify cleanup function

### Motor works when manually connecting COM to NO:
- ✅ Motor and battery are fine
- ❌ Relay not activating from GPIO
- **Fix:** Check relay type, GPIO connections, relay module power

## 📝 Notes

- **Relay COM/NO/NC:**
  - **COM** = Common (center terminal)
  - **NO** = Normally Open (connects to COM when relay activates)
  - **NC** = Normally Closed (disconnects from COM when relay activates)
  - We use **NO** (Normally Open) so motor is OFF when relay is OFF

- **Active LOW vs Active HIGH:**
  - **Active LOW** (most common): LOW signal = Relay ON, HIGH signal = Relay OFF
  - **Active HIGH** (less common): HIGH signal = Relay ON, LOW signal = Relay OFF
  - Your relay module may have a jumper to select this

- **Power Supply:**
  - Use a **separate 9V battery/adapter** for motors
  - **DO NOT** power motors from Raspberry Pi 5V (not enough current)
  - Ensure battery can supply enough current (check motor specs)

# 🔌 Proper Motor Connection Guide

## Your Current Connections (Verify These)

### Raspberry Pi to Relay Module:

| Raspberry Pi | Physical Pin | GPIO | Relay Module Terminal |
|--------------|--------------|------|----------------------|
| **GPIO 17** | Pin 11 | 17 | **IN1** (Motor A control) |
| **GPIO 27** | Pin 13 | 27 | **IN2** (Motor B control) |
| **GND** | Pin 6, 9, or 14 | - | **GND** |
| **5V** | Pin 2 or 4 | - | **VCC** (if relay needs power) |

### Motor/Pump Circuit:

| Component | Connection |
|-----------|------------|
| **9V Battery + (Red wire)** | → **Relay COM** (Common terminal) |
| **Pump/Motor + (Red wire)** | → **Relay NO** (Normally Open terminal) |
| **Pump/Motor - (Black wire)** | → **9V Battery - (Black wire)** |
| **9V Battery - (Black wire)** | → **Raspberry Pi GND** (Pin 6/9/14) |

### Complete Circuit Diagram:

```
┌─────────────────┐
│  Raspberry Pi   │
│                 │
│  GPIO 17 ───────┼──→ Relay IN1 ──┐
│  (Pin 11)       │                │
│                 │                │
│  GPIO 27 ───────┼──→ Relay IN2 ──┤
│  (Pin 13)       │                │
│                 │                │
│  GND (Pin 9) ───┼──→ Relay GND ───┤
│                 │                │
│  5V (Pin 2) ────┼──→ Relay VCC ───┤
│                 │                │
└─────────────────┘                │
        │                          │
        │ (Common GND)             │
        │                          │
        ▼                          │
┌─────────────────┐                │
│   9V Battery    │                │
│                 │                │
│  Red (+) ───────┼──→ Relay COM ──┘
│                 │       │
│  Black (-) ─────┼──┬────┘
│                 │  │
└─────────────────┘  │
                     │
                     ▼
                ┌─────────┐
                │  Relay   │
                │          │
                │  COM ←───┘ (from 9V +)
                │   │
                │   │
                │  NO ────→ Pump/Motor + (Red)
                │          │
                │  NC ────→ (Not used - leave disconnected)
                └─────────┘
                     │
                     │
        Pump/Motor - (Black) ──→ 9V Battery - (Black)
```

## ⚠️ Important Connection Rules:

1. **Relay IN (Signal)**: Connects to GPIO pins (17 for Motor A, 27 for Motor B)
2. **Relay GND**: MUST connect to Raspberry Pi GND
3. **Relay VCC**: Connects to Pi 5V (only if relay module needs power)
4. **Relay COM**: Connects to 9V Battery + (Red wire)
5. **Relay NO**: Connects to Motor/Pump + (Red wire)
6. **Motor/Pump -**: Connects to 9V Battery - (Black wire)
7. **Common Ground**: Pi GND and Battery - MUST be connected together

## 🔍 Troubleshooting Motor Auto-Starting:

### Problem: Motor turns ON when Pi starts

**Possible Causes:**

1. **Wrong Relay Type Setting**
   - If your relay is **Active HIGH** but code has `RELAY_ACTIVE_LOW = True`
   - **Fix**: Change `RELAY_ACTIVE_LOW = False` in `spray_pump_control.py`

2. **GPIO Pin Floating**
   - GPIO pins might be in undefined state before initialization
   - **Fix**: Code will now explicitly set pins to OFF before initialization

3. **Relay Module Jumper**
   - Some relay modules have a jumper for LOW/HIGH trigger
   - **Check**: Look for a jumper on your relay module and set it to LOW trigger

4. **Wiring Issue**
   - Relay IN might be connected incorrectly
   - **Check**: Verify IN connects to GPIO, not to 5V or GND

### How to Test Relay Type:

1. **Test 1**: With `RELAY_ACTIVE_LOW = True`:
   - Motor should be OFF when Pi starts
   - If motor is ON → Your relay is Active HIGH

2. **Test 2**: Change to `RELAY_ACTIVE_LOW = False`:
   - Motor should be OFF when Pi starts
   - If motor is ON → Your relay is Active LOW

3. **Manual Test**: 
   - Disconnect relay IN from GPIO
   - Connect relay IN to Pi GND → Motor should turn ON (if Active LOW)
   - Connect relay IN to Pi 5V → Motor should turn ON (if Active HIGH)

## 📋 Connection Checklist:

- [ ] GPIO 17 (Pin 11) → Relay IN1
- [ ] GPIO 27 (Pin 13) → Relay IN2  
- [ ] Pi GND (Pin 9) → Relay GND
- [ ] Pi 5V (Pin 2) → Relay VCC
- [ ] 9V Battery + → Relay COM
- [ ] Motor/Pump + → Relay NO
- [ ] Motor/Pump - → 9V Battery -
- [ ] Pi GND ↔ Battery - (common ground)

#!/usr/bin/env python3
"""
Standalone water pump test script.
Run this to verify the pump/relay hardware works - pump turns ON for 3 seconds, then OFF.
Uses direct RPi.GPIO control (same GPIO pin as main.py).
"""
import time

try:
    import RPi.GPIO as GPIO
except Exception:
    print("Error: RPi.GPIO not available. Run this on a Raspberry Pi.")
    exit(1)

RELAY_GPIO_PIN = 18  # GPIO 18 = Physical Pin 12 (same as main.py)
# Match backend/main.py (default)
RELAY_ACTIVE_LOW = True  # active-LOW relay: LOW=ON, HIGH=OFF

def main():
    print("Water pump test - ON for 3 seconds, then OFF")
    print(f"Using GPIO 18 (Physical Pin 12), relay is {'active-LOW' if RELAY_ACTIVE_LOW else 'active-HIGH'}")
    print()

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_GPIO_PIN, GPIO.OUT)

    off_level = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
    on_level = GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH

    try:
        GPIO.output(RELAY_GPIO_PIN, off_level)
        print("Pump OFF (start)")
        time.sleep(0.5)

        print("Pump ON...")
        GPIO.output(RELAY_GPIO_PIN, on_level)
        time.sleep(3)

        print("Pump OFF")
        GPIO.output(RELAY_GPIO_PIN, off_level)

    finally:
        try:
            GPIO.output(RELAY_GPIO_PIN, off_level)
        except Exception:
            pass
        # Don't cleanup to avoid any relay glitch on exit.

    print("Done. Pump ran for 3 seconds.")

if __name__ == "__main__":
    main()

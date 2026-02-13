#!/usr/bin/env python3
"""
Standalone water pump test script.
Run this to verify the pump/relay hardware works - pump turns ON for 3 seconds, then OFF.
Uses same GPIO 18 / active-LOW relay config as main.py.
"""
import time

try:
    from gpiozero import LED
except ImportError:
    print("Error: gpiozero not installed. Run: pip install gpiozero")
    print("(On Raspberry Pi, gpiozero is usually pre-installed)")
    exit(1)

RELAY_GPIO_PIN = 18  # GPIO 18 = Physical Pin 12 (same as main.py)
RELAY_ACTIVE_LOW = True  # active_high=False in gpiozero = active-LOW relay

def main():
    print("Water pump test - ON for 3 seconds, then OFF")
    print("Using GPIO 18 (Physical Pin 12), active-LOW relay")
    print()

    try:
        relay = LED(RELAY_GPIO_PIN, active_high=not RELAY_ACTIVE_LOW)
    except Exception as e:
        print(f"Error initializing relay: {e}")
        print("Make sure nothing else is using GPIO 18 (e.g. stop main.py)")
        exit(1)

    try:
        relay.off()
        print("Pump OFF (start)")
        time.sleep(0.5)

        print("Pump ON...")
        relay.on()
        time.sleep(3)

        print("Pump OFF")
        relay.off()

    finally:
        relay.off()
        relay.close()

    print("Done. Pump ran for 3 seconds.")

if __name__ == "__main__":
    main()

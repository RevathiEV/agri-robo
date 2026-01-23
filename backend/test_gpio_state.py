#!/usr/bin/env python3
"""
Diagnostic script to test GPIO state and relay behavior
Run this to verify GPIO pins are actually in the correct state
"""

import sys
import time

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("ERROR: RPi.GPIO not available. This script must run on Raspberry Pi.")
    sys.exit(1)

# GPIO pins
MOTOR_A_GPIO = 17
MOTOR_B_GPIO = 27

# Relay type - CHANGE THIS to match your relay
RELAY_ACTIVE_LOW = False  # Set True for Active LOW, False for Active HIGH

def test_gpio_state():
    """Test the actual GPIO state"""
    print("=" * 60)
    print("GPIO State Diagnostic Test")
    print("=" * 60)
    
    try:
        # Clean up first
        GPIO.cleanup()
        time.sleep(0.1)
        
        # Set mode
        GPIO.setmode(GPIO.BCM)
        
        # Determine states
        off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
        on_state = GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
        
        print(f"\nRelay Type: {'Active LOW' if RELAY_ACTIVE_LOW else 'Active HIGH'}")
        print(f"OFF state = GPIO.{'HIGH' if RELAY_ACTIVE_LOW else 'LOW'} ({off_state})")
        print(f"ON state = GPIO.{'LOW' if RELAY_ACTIVE_LOW else 'HIGH'} ({on_state})")
        
        # Test Motor A
        print(f"\n--- Testing Motor A (GPIO {MOTOR_A_GPIO}) ---")
        
        # Set as input with pull-down first (for Active HIGH)
        if not RELAY_ACTIVE_LOW:
            print("Setting as input with pull-down...")
            GPIO.setup(MOTOR_A_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            time.sleep(0.1)
            read_state = GPIO.input(MOTOR_A_GPIO)
            print(f"  Input state (with pull-down): {read_state} (should be 0/LOW)")
        
        # Configure as output with OFF state
        print(f"Setting as output with initial OFF state ({off_state})...")
        GPIO.setup(MOTOR_A_GPIO, GPIO.OUT, initial=off_state)
        GPIO.output(MOTOR_A_GPIO, off_state)
        time.sleep(0.1)
        
        # Try to read back
        GPIO.setup(MOTOR_A_GPIO, GPIO.IN)
        read_state = GPIO.input(MOTOR_A_GPIO)
        GPIO.setup(MOTOR_A_GPIO, GPIO.OUT, initial=off_state)
        GPIO.output(MOTOR_A_GPIO, off_state)
        
        print(f"  Read back state: {read_state} (expected: {off_state})")
        
        if read_state == off_state:
            print("  ✓ GPIO is in OFF state - Motor should be OFF")
        else:
            print(f"  ✗ GPIO is NOT in OFF state! Motor might be ON!")
            print(f"     This could be due to:")
            print(f"     1. Pull-up resistor on relay module")
            print(f"     2. Wiring issue (IN connected to 5V instead of GPIO)")
            print(f"     3. Relay module internal pull-up")
        
        # Test ON state
        print(f"\nTesting ON state...")
        GPIO.output(MOTOR_A_GPIO, on_state)
        time.sleep(0.1)
        GPIO.setup(MOTOR_A_GPIO, GPIO.IN)
        read_state = GPIO.input(MOTOR_A_GPIO)
        GPIO.setup(MOTOR_A_GPIO, GPIO.OUT, initial=off_state)
        GPIO.output(MOTOR_A_GPIO, off_state)
        
        print(f"  ON state read back: {read_state} (expected: {on_state})")
        
        if read_state == on_state:
            print("  ✓ GPIO can be set to ON state")
        else:
            print(f"  ✗ GPIO cannot be set to ON state!")
        
        # Final state
        print(f"\n--- Final State ---")
        GPIO.output(MOTOR_A_GPIO, off_state)
        time.sleep(0.2)
        print(f"Motor A GPIO set to OFF state ({off_state})")
        print("Motor should be OFF now. Is it? (y/n)")
        
        GPIO.cleanup()
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        GPIO.cleanup()

if __name__ == "__main__":
    test_gpio_state()

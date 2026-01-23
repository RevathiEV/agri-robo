"""
Spray Pump Control Module for Raspberry Pi
Controls 2 relay modules connected to 9V spray pumps
"""

import time
import threading

# Try to import RPi.GPIO
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available. Running in simulation mode.")

# GPIO Pin Configuration
SPRAY_PUMP_A_GPIO = 17  # GPIO 17 (Physical Pin 11) - Relay 1 (Motor A)
SPRAY_PUMP_B_GPIO = 27  # GPIO 27 (Physical Pin 13) - Relay 2 (Motor B)

# Spray duration in seconds
SPRAY_DURATION = 3

# Relay Configuration
# Set to True for active LOW relays (LOW = ON, HIGH = OFF) - Most common
# Set to False for active HIGH relays (HIGH = ON, LOW = OFF)
# If motor doesn't turn on, try changing this to False
RELAY_ACTIVE_LOW = True  # Change to False if your relay is active HIGH

# Global flag to track initialization
spray_pump_initialized = False

# Disease to Motor Mapping - 9 diseases divided into 3 classes
# Class 1: Motor A only (First 3 diseases)
# Class 2: Motor B only (Next 3 diseases)
# Class 3: Both Motors A & B (Last 3 diseases)
# Note: "Tomato___healthy" and "Not_A_Leaf" are NOT in this mapping (no spray)
DISEASE_MOTOR_MAPPING = {
    # Class 1: Motor A only (First 3 diseases)
    "Tomato___Bacterial_spot": "A",
    "Tomato___Early_blight": "A",
    "Tomato___Late_blight": "A",
    
    # Class 2: Motor B only (Next 3 diseases)
    "Tomato___Leaf_Mold": "B",
    "Tomato___Septoria_leaf_spot": "B",
    "Tomato___Spider_mites Two-spotted_spider_mite": "B",
    
    # Class 3: Both Motors A & B (Last 3 diseases)
    "Tomato___Target_Spot": "AB",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "AB",
    "Tomato___Tomato_mosaic_virus": "AB"
}


def init_spray_pumps():
    """
    Initialize GPIO pins for spray pumps
    Returns True if successful, False otherwise
    """
    global spray_pump_initialized
    
    if not GPIO_AVAILABLE:
        print("GPIO not available - spray pumps disabled (simulation mode)")
        spray_pump_initialized = False
        return False
    
    try:
        # Set GPIO mode to BCM (Broadcom pin numbering)
        GPIO.setmode(GPIO.BCM)
        
        # Set up GPIO pins as outputs with initial OFF state
        # For active LOW: HIGH = OFF, LOW = ON
        # For active HIGH: LOW = OFF, HIGH = ON
        initial_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
        GPIO.setup(SPRAY_PUMP_A_GPIO, GPIO.OUT, initial=initial_state)
        GPIO.setup(SPRAY_PUMP_B_GPIO, GPIO.OUT, initial=initial_state)
        
        # Explicitly set to OFF state to ensure pumps are OFF on startup
        GPIO.output(SPRAY_PUMP_A_GPIO, initial_state)
        GPIO.output(SPRAY_PUMP_B_GPIO, initial_state)
        
        spray_pump_initialized = True
        relay_type_str = "Active LOW" if RELAY_ACTIVE_LOW else "Active HIGH"
        print(f"✓ Spray pumps initialized successfully")
        print(f"  - Motor A: GPIO {SPRAY_PUMP_A_GPIO} (Physical Pin 11)")
        print(f"  - Motor B: GPIO {SPRAY_PUMP_B_GPIO} (Physical Pin 13)")
        print(f"  - Relay Type: {relay_type_str} (LOW=ON, HIGH=OFF)" if RELAY_ACTIVE_LOW else f"  - Relay Type: {relay_type_str} (HIGH=ON, LOW=OFF)")
        return True
        
    except Exception as e:
        print(f"Error initializing spray pumps: {e}")
        spray_pump_initialized = False
        return False


def turn_on_pump(motor):
    """
    Turn on a specific pump
    Args:
        motor: 'A' or 'B' or 'AB' for both
    """
    if not spray_pump_initialized:
        if GPIO_AVAILABLE:
            print("Spray pumps not initialized. Call init_spray_pumps() first.")
        else:
            print(f"[SIMULATION] Motor {motor} ON")
        return False
    
    try:
        # Support both active LOW and active HIGH relays
        on_state = GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH
        
        if motor == "A":
            GPIO.output(SPRAY_PUMP_A_GPIO, on_state)
            relay_type = "LOW" if RELAY_ACTIVE_LOW else "HIGH"
            print(f"Motor A (GPIO {SPRAY_PUMP_A_GPIO}) turned ON (GPIO={relay_type})")
        elif motor == "B":
            GPIO.output(SPRAY_PUMP_B_GPIO, on_state)
            relay_type = "LOW" if RELAY_ACTIVE_LOW else "HIGH"
            print(f"Motor B (GPIO {SPRAY_PUMP_B_GPIO}) turned ON (GPIO={relay_type})")
        elif motor == "AB":
            GPIO.output(SPRAY_PUMP_A_GPIO, on_state)
            GPIO.output(SPRAY_PUMP_B_GPIO, on_state)
            relay_type = "LOW" if RELAY_ACTIVE_LOW else "HIGH"
            print(f"Both Motors A & B turned ON (GPIO={relay_type})")
        else:
            print(f"Invalid motor selection: {motor}. Use 'A', 'B', or 'AB'")
            return False
        return True
    except Exception as e:
        print(f"Error turning on pump: {e}")
        return False


def turn_off_pump(motor):
    """
    Turn off a specific pump
    Args:
        motor: 'A' or 'B' or 'AB' for both
    """
    if not spray_pump_initialized:
        if GPIO_AVAILABLE:
            print("Spray pumps not initialized. Call init_spray_pumps() first.")
        else:
            print(f"[SIMULATION] Motor {motor} OFF")
        return False
    
    try:
        # Support both active LOW and active HIGH relays
        off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
        
        if motor == "A":
            GPIO.output(SPRAY_PUMP_A_GPIO, off_state)
            print(f"Motor A (GPIO {SPRAY_PUMP_A_GPIO}) turned OFF")
        elif motor == "B":
            GPIO.output(SPRAY_PUMP_B_GPIO, off_state)
            print(f"Motor B (GPIO {SPRAY_PUMP_B_GPIO}) turned OFF")
        elif motor == "AB":
            GPIO.output(SPRAY_PUMP_A_GPIO, off_state)
            GPIO.output(SPRAY_PUMP_B_GPIO, off_state)
            print(f"Both Motors A & B turned OFF")
        else:
            print(f"Invalid motor selection: {motor}. Use 'A', 'B', or 'AB'")
            return False
        return True
    except Exception as e:
        print(f"Error turning off pump: {e}")
        return False


def trigger_spray_pump(motor, duration=SPRAY_DURATION):
    """
    Trigger spray pump(s) for specified duration in seconds
    Runs in background thread to avoid blocking
    
    Args:
        motor: 'A' or 'B' or 'AB' for both motors simultaneously
        duration: Duration in seconds (default: 3 seconds)
    
    Returns:
        True if spray was triggered, False otherwise
    """
    if motor not in ["A", "B", "AB"]:
        print(f"Invalid motor selection: {motor}. Use 'A', 'B', or 'AB'")
        return False
    
    def run_spray():
        """Internal function to run spray in background thread"""
        try:
            # Turn on pump(s)
            turn_on_pump(motor)
            
            # Wait for specified duration
            time.sleep(duration)
            
            # Turn off pump(s)
            turn_off_pump(motor)
            
            print(f"Spray cycle completed: {motor} for {duration} seconds")
            
        except Exception as e:
            print(f"Error during spray cycle: {e}")
            # Ensure pumps are turned off in case of error
            try:
                turn_off_pump(motor)
            except:
                pass
    
    # Run in background thread to avoid blocking API response
    spray_thread = threading.Thread(target=run_spray, daemon=True)
    spray_thread.start()
    return True


def trigger_spray_by_disease(disease_name):
    """
    Trigger spray pump based on detected disease
    Does NOT spray for "healthy" or "Not_A_Leaf"
    
    Args:
        disease_name: Disease name from class mapping (e.g., "Tomato___Bacterial_spot")
    
    Returns:
        Motor(s) activated ('A', 'B', 'AB') or None if no spray needed
    """
    # Don't spray for healthy or non-leaf detections
    if disease_name == "Tomato___healthy" or disease_name == "Not_A_Leaf":
        print(f"No spray needed for: {disease_name}")
        return None
    
    # Check if disease is in mapping
    if disease_name in DISEASE_MOTOR_MAPPING:
        motor = DISEASE_MOTOR_MAPPING[disease_name]
        trigger_spray_pump(motor, duration=SPRAY_DURATION)
        print(f"Spray triggered: {motor} for disease: {disease_name}")
        return motor
    else:
        print(f"No spray mapping for disease: {disease_name}")
        return None


def cleanup_spray_pumps():
    """
    Cleanup GPIO pins on shutdown
    Turns off all pumps and cleans up GPIO
    """
    global spray_pump_initialized
    
    if spray_pump_initialized and GPIO_AVAILABLE:
        try:
            # Turn off all pumps (use OFF state, not LOW)
            off_state = GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW
            GPIO.output(SPRAY_PUMP_A_GPIO, off_state)
            GPIO.output(SPRAY_PUMP_B_GPIO, off_state)
            
            # Cleanup GPIO
            GPIO.cleanup()
            
            spray_pump_initialized = False
            print("✓ Spray pump GPIO cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    else:
        print("Spray pumps not initialized or GPIO not available")


def get_status():
    """
    Get current status of spray pump system
    
    Returns:
        Dictionary with status information
    """
    return {
        "gpio_available": GPIO_AVAILABLE,
        "initialized": spray_pump_initialized,
        "motor_a_gpio": SPRAY_PUMP_A_GPIO,
        "motor_b_gpio": SPRAY_PUMP_B_GPIO,
        "spray_duration": SPRAY_DURATION,
        "disease_mapping": DISEASE_MOTOR_MAPPING
    }


def test_relay_type(motor="A", test_duration=2):
    """
    Test function to identify relay type (active LOW vs active HIGH)
    Tests both types and asks user which one worked
    """
    if not GPIO_AVAILABLE:
        print("GPIO not available - cannot test relay type")
        return None
    
    print("=" * 60)
    print("Relay Type Test - Motor", motor)
    print("=" * 60)
    print("\nThis will test both relay types to find which one works.")
    print("Watch your motor and listen for the relay click.\n")
    
    # Initialize GPIO
    GPIO.setmode(GPIO.BCM)
    gpio_pin = SPRAY_PUMP_A_GPIO if motor == "A" else SPRAY_PUMP_B_GPIO
    GPIO.setup(gpio_pin, GPIO.OUT)
    
    try:
        # Test 1: Active LOW (LOW = ON)
        print("Test 1: Testing ACTIVE LOW (LOW = ON, HIGH = OFF)")
        print(f"Setting GPIO {gpio_pin} to LOW (should turn motor ON)...")
        GPIO.output(gpio_pin, GPIO.LOW)
        print(f"Motor should be ON now. Did it turn on? (waiting {test_duration} seconds)")
        time.sleep(test_duration)
        GPIO.output(gpio_pin, GPIO.HIGH)
        print("Setting GPIO to HIGH (should turn motor OFF)")
        time.sleep(1)
        
        response1 = input("\nDid the motor turn ON when GPIO was LOW? (y/n): ").lower().strip()
        
        # Test 2: Active HIGH (HIGH = ON)
        print("\nTest 2: Testing ACTIVE HIGH (HIGH = ON, LOW = OFF)")
        print(f"Setting GPIO {gpio_pin} to HIGH (should turn motor ON)...")
        GPIO.output(gpio_pin, GPIO.HIGH)
        print(f"Motor should be ON now. Did it turn on? (waiting {test_duration} seconds)")
        time.sleep(test_duration)
        GPIO.output(gpio_pin, GPIO.LOW)
        print("Setting GPIO to LOW (should turn motor OFF)")
        time.sleep(1)
        
        response2 = input("\nDid the motor turn ON when GPIO was HIGH? (y/n): ").lower().strip()
        
        # Determine relay type
        if response1 == 'y' and response2 != 'y':
            print("\n✓ Result: Your relay is ACTIVE LOW (LOW = ON, HIGH = OFF)")
            print("  Current setting RELAY_ACTIVE_LOW = True is CORRECT")
            return True
        elif response2 == 'y' and response1 != 'y':
            print("\n✓ Result: Your relay is ACTIVE HIGH (HIGH = ON, LOW = OFF)")
            print("  You need to change RELAY_ACTIVE_LOW = False in the code")
            return False
        elif response1 == 'y' and response2 == 'y':
            print("\n⚠ Warning: Motor turned on in both tests!")
            print("  Check your wiring - motor circuit might be always connected")
            return None
        else:
            print("\n✗ Motor didn't turn on in either test!")
            print("  Check:")
            print("  1. Motor circuit wiring (9V battery, relay COM/NO, motor)")
            print("  2. Relay module power (VCC to 5V, GND to GND)")
            print("  3. Relay module jumper settings")
            return None
            
    finally:
        # Turn off motor
        GPIO.output(gpio_pin, GPIO.HIGH)
        time.sleep(0.5)
        GPIO.output(gpio_pin, GPIO.LOW)
        time.sleep(0.5)
        GPIO.output(gpio_pin, GPIO.HIGH)
        GPIO.cleanup()


# Test function (run when executed directly)
if __name__ == "__main__":
    import sys
    
    # Check if user wants to test relay type
    if len(sys.argv) > 1 and sys.argv[1] == "test-relay":
        motor_to_test = sys.argv[2] if len(sys.argv) > 2 else "A"
        test_relay_type(motor_to_test)
    else:
        print("=" * 60)
        print("Spray Pump Control - Test Mode")
        print("=" * 60)
        print(f"Current relay setting: {'Active LOW' if RELAY_ACTIVE_LOW else 'Active HIGH'}")
        print("\nTo test relay type, run: python spray_pump_control.py test-relay [A|B]")
        print("=" * 60)
        
        # Initialize
        init_spray_pumps()
        
        if spray_pump_initialized or not GPIO_AVAILABLE:
            print("\nTesting Motor A (3 seconds)...")
            trigger_spray_pump("A", duration=3)
            time.sleep(4)  # Wait for spray to complete
            
            print("\nTesting Motor B (3 seconds)...")
            trigger_spray_pump("B", duration=3)
            time.sleep(4)  # Wait for spray to complete
            
            print("\nTesting Both Motors (3 seconds)...")
            trigger_spray_pump("AB", duration=3)
            time.sleep(4)  # Wait for spray to complete
            
            print("\nTest completed!")
            print("\nIf motor didn't turn on, try:")
            print("  1. Run: python spray_pump_control.py test-relay A")
            print("  2. Change RELAY_ACTIVE_LOW to False if test shows active HIGH")
        else:
            print("Failed to initialize spray pumps")
        
        # Cleanup
        cleanup_spray_pumps()
        print("\nStatus:", get_status())

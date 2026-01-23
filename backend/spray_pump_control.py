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
        
        # Set up GPIO pins as outputs with initial HIGH value
        # HIGH = OFF (for active LOW relays) or HIGH = ON (for active HIGH relays)
        # Setting initial=GPIO.HIGH ensures pump is OFF on startup
        GPIO.setup(SPRAY_PUMP_A_GPIO, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(SPRAY_PUMP_B_GPIO, GPIO.OUT, initial=GPIO.HIGH)
        
        # Explicitly set to HIGH to ensure pumps are OFF
        # If your relay is active LOW: HIGH = OFF, LOW = ON
        # If your relay is active HIGH: HIGH = ON, LOW = OFF
        # Adjust turn_on_pump() and turn_off_pump() if needed
        GPIO.output(SPRAY_PUMP_A_GPIO, GPIO.HIGH)
        GPIO.output(SPRAY_PUMP_B_GPIO, GPIO.HIGH)
        
        spray_pump_initialized = True
        print(f"✓ Spray pumps initialized successfully")
        print(f"  - Motor A: GPIO {SPRAY_PUMP_A_GPIO} (Physical Pin 11)")
        print(f"  - Motor B: GPIO {SPRAY_PUMP_B_GPIO} (Physical Pin 13)")
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
        # For active LOW relays: LOW = ON, HIGH = OFF
        if motor == "A":
            GPIO.output(SPRAY_PUMP_A_GPIO, GPIO.LOW)  # LOW turns ON for active LOW relay
            print(f"Motor A (GPIO {SPRAY_PUMP_A_GPIO}) turned ON")
        elif motor == "B":
            GPIO.output(SPRAY_PUMP_B_GPIO, GPIO.LOW)  # LOW turns ON for active LOW relay
            print(f"Motor B (GPIO {SPRAY_PUMP_B_GPIO}) turned ON")
        elif motor == "AB":
            GPIO.output(SPRAY_PUMP_A_GPIO, GPIO.LOW)  # LOW turns ON for active LOW relay
            GPIO.output(SPRAY_PUMP_B_GPIO, GPIO.LOW)  # LOW turns ON for active LOW relay
            print(f"Both Motors A & B turned ON")
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
        # For active LOW relays: LOW = ON, HIGH = OFF
        if motor == "A":
            GPIO.output(SPRAY_PUMP_A_GPIO, GPIO.HIGH)  # HIGH turns OFF for active LOW relay
            print(f"Motor A (GPIO {SPRAY_PUMP_A_GPIO}) turned OFF")
        elif motor == "B":
            GPIO.output(SPRAY_PUMP_B_GPIO, GPIO.HIGH)  # HIGH turns OFF for active LOW relay
            print(f"Motor B (GPIO {SPRAY_PUMP_B_GPIO}) turned OFF")
        elif motor == "AB":
            GPIO.output(SPRAY_PUMP_A_GPIO, GPIO.HIGH)  # HIGH turns OFF for active LOW relay
            GPIO.output(SPRAY_PUMP_B_GPIO, GPIO.HIGH)  # HIGH turns OFF for active LOW relay
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
            # Turn off all pumps
            GPIO.output(SPRAY_PUMP_A_GPIO, GPIO.LOW)
            GPIO.output(SPRAY_PUMP_B_GPIO, GPIO.LOW)
            
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


# Test function (run when executed directly)
if __name__ == "__main__":
    print("=" * 60)
    print("Spray Pump Control - Test Mode")
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
    else:
        print("Failed to initialize spray pumps")
    
    # Cleanup
    cleanup_spray_pumps()
    print("\nStatus:", get_status())

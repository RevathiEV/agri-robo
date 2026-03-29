"""
Single water pump control for Raspberry Pi.

Supports:
- active-low relay wiring (LOW = ON, HIGH = OFF)
- automatic 3 second spray cycles after disease detection
- manual start/stop control from the UI
- manual stop interrupting an active automatic cycle immediately
"""

import threading
import time

try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available. Running in simulation mode.")


PUMP_GPIO = 17
SPRAY_DURATION = 3
RELAY_ACTIVE_LOW = True

# Backward-compatible aliases used by older diagnostics/tests.
SPRAY_PUMP_A_GPIO = PUMP_GPIO
SPRAY_PUMP_B_GPIO = PUMP_GPIO

spray_pump_initialized = False

_state_lock = threading.RLock()
_control_run_id = 0
_auto_stop_event = None
_pump_running = False
_pump_mode = "idle"
_last_action = "idle"


def _on_state():
    return GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH


def _off_state():
    return GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW


def _ensure_initialized():
    global spray_pump_initialized

    if spray_pump_initialized:
        return True

    return init_spray_pumps()


def _write_pump_state(enabled):
    if not GPIO_AVAILABLE:
        print(f"[SIMULATION] Pump {'ON' if enabled else 'OFF'}")
        return True

    if not spray_pump_initialized:
        print("[PUMP] Pump not initialized. Call init_spray_pumps() first.")
        return False

    try:
        GPIO.output(PUMP_GPIO, _on_state() if enabled else _off_state())
        print(
            f"[PUMP] GPIO {PUMP_GPIO} set to "
            f"{'ON' if enabled else 'OFF'} "
            f"(relay is {'active LOW' if RELAY_ACTIVE_LOW else 'active HIGH'})"
        )
        return True
    except Exception as exc:
        print(f"[PUMP ERROR] Failed to write GPIO {PUMP_GPIO}: {exc}")
        return False


def _advance_run_id_locked():
    global _control_run_id

    _control_run_id += 1
    return _control_run_id


def _cancel_auto_cycle_locked():
    global _auto_stop_event

    if _auto_stop_event is not None:
        _auto_stop_event.set()
        _auto_stop_event = None


def _set_mode_locked(mode, running, action):
    global _pump_mode, _pump_running, _last_action

    _pump_mode = mode
    _pump_running = running
    _last_action = action


def init_spray_pumps():
    """Initialize the relay output pin and force the pump OFF."""
    global spray_pump_initialized

    if not GPIO_AVAILABLE:
        spray_pump_initialized = True
        print("[INIT] GPIO not available. Pump controller ready in simulation mode.")
        return True

    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(PUMP_GPIO, GPIO.OUT, initial=_off_state())
        GPIO.output(PUMP_GPIO, _off_state())
        spray_pump_initialized = True
        print(
            f"[INIT] Pump GPIO {PUMP_GPIO} initialized. "
            f"Relay mode: {'active LOW' if RELAY_ACTIVE_LOW else 'active HIGH'}."
        )
        return True
    except Exception as exc:
        spray_pump_initialized = False
        print(f"[INIT ERROR] Failed to initialize pump GPIO: {exc}")
        return False


def turn_on_pump(_motor="A"):
    """Compatibility wrapper for older code paths."""
    if not _ensure_initialized():
        return False
    return _write_pump_state(True)


def turn_off_pump(_motor="A"):
    """Compatibility wrapper for older code paths."""
    if not _ensure_initialized():
        return False
    return _write_pump_state(False)


def _run_auto_cycle(run_id, stop_event, duration):
    interrupted = stop_event.wait(duration)

    with _state_lock:
        global _auto_stop_event

        if run_id != _control_run_id:
            return

        if _auto_stop_event is stop_event:
            _auto_stop_event = None

        if interrupted:
            print("[AUTO] Automatic spray interrupted before completion.")
            return

        if _pump_mode != "auto":
            return

        turn_off_pump()
        _set_mode_locked("idle", False, "auto_complete")
        print(f"[AUTO] Automatic spray completed after {duration} seconds.")


def trigger_auto_dispense(duration=SPRAY_DURATION):
    """Run the pump automatically for a fixed duration."""
    if not _ensure_initialized():
        return {
            "success": False,
            "auto_dispense_started": False,
            "message": "Pump controller is not initialized.",
            "pump_status": get_status(),
        }

    with _state_lock:
        run_id = _advance_run_id_locked()
        _cancel_auto_cycle_locked()

        if _pump_mode == "manual":
            print("[AUTO] Manual dispensing is active. Leaving pump running manually.")
            return {
                "success": True,
                "auto_dispense_started": False,
                "message": "Pump is already running in manual mode.",
                "pump_status": get_status(),
            }

        if not turn_on_pump():
            _set_mode_locked("idle", False, "auto_failed")
            return {
                "success": False,
                "auto_dispense_started": False,
                "message": "Failed to turn the pump on for automatic spray.",
                "pump_status": get_status(),
            }

        stop_event = threading.Event()
        global _auto_stop_event
        _auto_stop_event = stop_event
        _set_mode_locked("auto", True, "auto_start")

        spray_thread = threading.Thread(
            target=_run_auto_cycle,
            args=(run_id, stop_event, duration),
            daemon=True,
        )
        spray_thread.start()

        print(f"[AUTO] Automatic spray started for {duration} seconds.")
        return {
            "success": True,
            "auto_dispense_started": True,
            "message": f"Automatic spray started for {duration} seconds.",
            "pump_status": get_status(),
        }


def start_manual_dispense():
    """Turn the pump on and keep it on until stopped manually."""
    if not _ensure_initialized():
        return {
            "success": False,
            "message": "Pump controller is not initialized.",
            "pump_status": get_status(),
        }

    with _state_lock:
        _advance_run_id_locked()
        _cancel_auto_cycle_locked()

        if not turn_on_pump():
            _set_mode_locked("idle", False, "manual_failed")
            return {
                "success": False,
                "message": "Failed to start manual dispensing.",
                "pump_status": get_status(),
            }

        _set_mode_locked("manual", True, "manual_start")

    print("[MANUAL] Pump started manually.")
    return {
        "success": True,
        "message": "Pump started manually.",
        "pump_status": get_status(),
    }


def stop_dispense():
    """Turn the pump off immediately and cancel any automatic timer."""
    if not _ensure_initialized():
        return {
            "success": False,
            "message": "Pump controller is not initialized.",
            "pump_status": get_status(),
        }

    with _state_lock:
        _advance_run_id_locked()
        _cancel_auto_cycle_locked()
        turn_off_pump()
        _set_mode_locked("idle", False, "manual_stop")

    print("[MANUAL] Pump stopped.")
    return {
        "success": True,
        "message": "Pump stopped.",
        "pump_status": get_status(),
    }


def trigger_spray_pump(_motor="A", duration=SPRAY_DURATION):
    """Compatibility wrapper for timed spray behavior."""
    result = trigger_auto_dispense(duration=duration)
    return result["success"]


def trigger_spray_by_disease(disease_name, duration=SPRAY_DURATION):
    """
    Spray only for actual diseases.

    Healthy leaves, non-leaf detections, and unknown values must not spray.
    """
    if disease_name in {"Tomato___healthy", "Not_A_Leaf", "Unknown"}:
        print(f"[AUTO] No spray needed for result: {disease_name}")
        return {
            "success": True,
            "auto_dispense_started": False,
            "message": "No automatic spray needed for this detection.",
            "pump_status": get_status(),
        }

    return trigger_auto_dispense(duration=duration)


def cleanup_spray_pumps():
    """Force the pump OFF and release GPIO resources."""
    global spray_pump_initialized

    with _state_lock:
        _advance_run_id_locked()
        _cancel_auto_cycle_locked()
        _set_mode_locked("idle", False, "cleanup")

    if not GPIO_AVAILABLE:
        spray_pump_initialized = False
        print("[CLEANUP] Pump controller cleaned up in simulation mode.")
        return

    if not spray_pump_initialized:
        print("[CLEANUP] Pump controller was not initialized.")
        return

    try:
        GPIO.output(PUMP_GPIO, _off_state())
        GPIO.cleanup(PUMP_GPIO)
        spray_pump_initialized = False
        print("[CLEANUP] Pump GPIO cleaned up.")
    except Exception as exc:
        print(f"[CLEANUP ERROR] Failed during GPIO cleanup: {exc}")


def get_status():
    """Return the current state of the single-pump controller."""
    with _state_lock:
        return {
            "gpio_available": GPIO_AVAILABLE,
            "initialized": spray_pump_initialized,
            "pump_gpio": PUMP_GPIO,
            "relay_active_low": RELAY_ACTIVE_LOW,
            "pump_running": _pump_running,
            "mode": _pump_mode,
            "last_action": _last_action,
            "spray_duration": SPRAY_DURATION,
        }


def test_relay_type(test_duration=2):
    """Simple relay test helper for the single pump wiring."""
    if not GPIO_AVAILABLE:
        print("GPIO not available - cannot test relay type")
        return None

    print("=" * 60)
    print("Relay Type Test")
    print("=" * 60)
    print(f"Testing GPIO {PUMP_GPIO} for {test_duration} seconds each mode.\n")

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PUMP_GPIO, GPIO.OUT)

    try:
        print("Test 1: LOW = ON")
        GPIO.output(PUMP_GPIO, GPIO.LOW)
        time.sleep(test_duration)
        GPIO.output(PUMP_GPIO, GPIO.HIGH)
        response_low = input("Did the pump turn on while GPIO was LOW? (y/n): ").strip().lower()

        print("\nTest 2: HIGH = ON")
        GPIO.output(PUMP_GPIO, GPIO.HIGH)
        time.sleep(test_duration)
        GPIO.output(PUMP_GPIO, GPIO.LOW)
        response_high = input("Did the pump turn on while GPIO was HIGH? (y/n): ").strip().lower()

        if response_low == "y" and response_high != "y":
            print("\nYour relay is ACTIVE LOW.")
            return True
        if response_high == "y" and response_low != "y":
            print("\nYour relay is ACTIVE HIGH.")
            return False

        print("\nCould not determine relay type clearly. Recheck wiring.")
        return None
    finally:
        GPIO.output(PUMP_GPIO, _off_state())
        GPIO.cleanup(PUMP_GPIO)


if __name__ == "__main__":
    init_spray_pumps()
    print("Status:", get_status())
    print("\nTesting 3 second automatic spray...")
    trigger_auto_dispense()
    time.sleep(SPRAY_DURATION + 1)
    cleanup_spray_pumps()

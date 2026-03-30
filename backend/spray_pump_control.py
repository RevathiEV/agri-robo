"""
Two-pump control for Raspberry Pi.

Supports:
- active-low relay wiring (LOW = ON, released input = OFF)
- automatic 3 second spray cycles after disease detection
- manual start/stop control from the UI
- manual stop interrupting an active automatic cycle immediately
- disease groups mapped to pump A, pump B, or both pumps
"""

import sys
import threading
import time

# Add Raspberry Pi OS dist-packages so GPIO libraries installed with apt
# are available inside the project's virtual environment.
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.insert(0, "/usr/lib/python3/dist-packages")

GPIO = None
GPIO_AVAILABLE = False
GPIOZERO_AVAILABLE = False
OutputDevice = None
LGPIOFactory = None
_gpio_backend = "simulation"
_pump_factory = None
_pump_devices = {}

try:
    from gpiozero import OutputDevice
    from gpiozero.pins.lgpio import LGPIOFactory

    GPIOZERO_AVAILABLE = True
except Exception as exc:
    print(f"Warning: gpiozero/lgpio not available: {exc}")

try:
    import RPi.GPIO as GPIO

    GPIO_AVAILABLE = True
except Exception as exc:
    print(f"Warning: RPi.GPIO not available: {exc}")


PUMP_A_GPIO = 17
PUMP_B_GPIO = 27
SPRAY_DURATION = 3
RELAY_ACTIVE_LOW = True

PUMP_GPIO_MAP = {
    "A": PUMP_A_GPIO,
    "B": PUMP_B_GPIO,
}

DISEASE_PUMP_MAPPING = {
    # First 3 diseases -> pump 1
    "Tomato___Bacterial_spot": "A",
    "Tomato___Early_blight": "A",
    "Tomato___Late_blight": "A",
    # Second 3 diseases -> pump 2
    "Tomato___Leaf_Mold": "B",
    "Tomato___Septoria_leaf_spot": "B",
    "Tomato___Spider_mites Two-spotted_spider_mite": "B",
    # Final 3 diseases -> both pumps
    "Tomato___Target_Spot": "AB",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "AB",
    "Tomato___Tomato_mosaic_virus": "AB",
}

# Backward-compatible aliases used elsewhere.
SPRAY_PUMP_A_GPIO = PUMP_A_GPIO
SPRAY_PUMP_B_GPIO = PUMP_B_GPIO

spray_pump_initialized = False

_state_lock = threading.RLock()
_control_run_id = 0
_auto_stop_event = None
_pump_running = False
_pump_mode = "idle"
_last_action = "idle"
_active_target = "none"


def _on_state():
    return GPIO.LOW if RELAY_ACTIVE_LOW else GPIO.HIGH


def _off_state():
    return GPIO.HIGH if RELAY_ACTIVE_LOW else GPIO.LOW


def _normalize_target(target):
    normalized = (target or "AB").upper()
    if normalized not in {"A", "B", "AB"}:
        raise ValueError(f"Invalid pump target: {target}")
    return normalized


def _channels_for_target(target):
    normalized = _normalize_target(target)
    if normalized == "AB":
        return ("A", "B")
    return (normalized,)


def _target_label(target):
    labels = {
        "A": "Pump 1",
        "B": "Pump 2",
        "AB": "Both Pumps",
        "none": "No Pumps",
    }
    return labels.get(target, target)


def _ensure_initialized():
    global spray_pump_initialized

    if spray_pump_initialized:
        return True

    return init_spray_pumps()


def _write_channel_state(channel, enabled):
    gpio_pin = PUMP_GPIO_MAP[channel]

    if _gpio_backend == "gpiozero-lgpio":
        try:
            if enabled:
                if channel not in _pump_devices:
                    _pump_devices[channel] = OutputDevice(
                        gpio_pin,
                        active_high=not RELAY_ACTIVE_LOW,
                        initial_value=True,
                        pin_factory=_pump_factory,
                    )
                else:
                    _pump_devices[channel].on()
            else:
                # Some active-low 5V relay modules turn off more reliably when the
                # control line is released instead of driven HIGH from a 3.3V GPIO.
                if channel in _pump_devices:
                    _pump_devices[channel].close()
                    del _pump_devices[channel]

            print(
                f"[PUMP {channel}] GPIO {gpio_pin} set to "
                f"{'ON' if enabled else 'OFF'} ({_gpio_backend})"
            )
            return True
        except Exception as exc:
            print(f"[PUMP {channel} ERROR] Failed to control gpiozero device: {exc}")
            return False

    if not GPIO_AVAILABLE or GPIO is None:
        print(f"[SIMULATION] Pump {channel} {'ON' if enabled else 'OFF'}")
        return True

    if not spray_pump_initialized:
        print(f"[PUMP {channel}] Pump controller not initialized.")
        return False

    try:
        if enabled:
            GPIO.setup(gpio_pin, GPIO.OUT, initial=_on_state())
            GPIO.output(gpio_pin, _on_state())
        elif RELAY_ACTIVE_LOW:
            GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
        else:
            GPIO.setup(gpio_pin, GPIO.OUT, initial=_off_state())
            GPIO.output(gpio_pin, _off_state())

        print(
            f"[PUMP {channel}] GPIO {gpio_pin} set to "
            f"{'ON' if enabled else 'OFF'} "
            f"(relay is {'active LOW' if RELAY_ACTIVE_LOW else 'active HIGH'})"
        )
        return True
    except Exception as exc:
        print(f"[PUMP {channel} ERROR] Failed to write GPIO {gpio_pin}: {exc}")
        return False


def _write_pump_state(target, enabled):
    channels = _channels_for_target(target)
    results = [_write_channel_state(channel, enabled) for channel in channels]
    return all(results)


def _advance_run_id_locked():
    global _control_run_id

    _control_run_id += 1
    return _control_run_id


def _cancel_auto_cycle_locked():
    global _auto_stop_event

    if _auto_stop_event is not None:
        _auto_stop_event.set()
        _auto_stop_event = None


def _set_mode_locked(mode, running, action, target="none"):
    global _pump_mode, _pump_running, _last_action, _active_target

    _pump_mode = mode
    _pump_running = running
    _last_action = action
    _active_target = target


def init_spray_pumps():
    """Initialize both relay pins and force them to the OFF state."""
    global _gpio_backend, _pump_factory, spray_pump_initialized

    if GPIOZERO_AVAILABLE:
        try:
            _pump_factory = LGPIOFactory()
            _pump_devices.clear()
            spray_pump_initialized = True
            _gpio_backend = "gpiozero-lgpio"
            print(
                "[INIT] Pump GPIOs initialized with gpiozero/lgpio: "
                f"A={PUMP_A_GPIO}, B={PUMP_B_GPIO}. "
                f"Relay mode: {'active LOW' if RELAY_ACTIVE_LOW else 'active HIGH'} "
                "(OFF uses released input state)."
            )
            return True
        except Exception as exc:
            _pump_devices.clear()
            _pump_factory = None
            print(f"[INIT] gpiozero/lgpio init failed, falling back: {exc}")

    if GPIO_AVAILABLE and GPIO is not None:
        try:
            GPIO.setmode(GPIO.BCM)
            for gpio_pin in PUMP_GPIO_MAP.values():
                if RELAY_ACTIVE_LOW:
                    GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
                else:
                    GPIO.setup(gpio_pin, GPIO.OUT, initial=_off_state())
                    GPIO.output(gpio_pin, _off_state())

            spray_pump_initialized = True
            _gpio_backend = "rpi-gpio"
            print(
                "[INIT] Pump GPIOs initialized with RPi.GPIO: "
                f"A={PUMP_A_GPIO}, B={PUMP_B_GPIO}. "
                f"Relay mode: {'active LOW' if RELAY_ACTIVE_LOW else 'active HIGH'} "
                "(OFF uses released input state)."
            )
            return True
        except Exception as exc:
            print(f"[INIT] RPi.GPIO init failed: {exc}")

    spray_pump_initialized = False
    _gpio_backend = "simulation"
    print("[INIT ERROR] Failed to initialize pump GPIO with any backend.")
    return False


def turn_on_pump(target="AB"):
    """Turn on one or both pumps."""
    if not _ensure_initialized():
        return False
    return _write_pump_state(target, True)


def turn_off_pump(target="AB"):
    """Turn off one or both pumps."""
    if not _ensure_initialized():
        return False
    return _write_pump_state(target, False)


def _run_auto_cycle(run_id, stop_event, duration, target):
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

        turn_off_pump(target)
        _set_mode_locked("idle", False, "auto_complete", "none")
        print(
            f"[AUTO] Automatic spray completed after {duration} seconds "
            f"for {_target_label(target)}."
        )


def trigger_auto_dispense(target="AB", duration=SPRAY_DURATION):
    """Run the selected pump target automatically for a fixed duration."""
    normalized_target = _normalize_target(target)

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
            print("[AUTO] Manual dispensing is active. Leaving pumps running manually.")
            return {
                "success": True,
                "auto_dispense_started": False,
                "message": "Pumps are already running in manual mode.",
                "pump_status": get_status(),
            }

        if not turn_on_pump(normalized_target):
            _set_mode_locked("idle", False, "auto_failed", "none")
            return {
                "success": False,
                "auto_dispense_started": False,
                "message": "Failed to turn on the pump target for automatic spray.",
                "pump_status": get_status(),
            }

        stop_event = threading.Event()
        global _auto_stop_event
        _auto_stop_event = stop_event
        _set_mode_locked("auto", True, "auto_start", normalized_target)

        spray_thread = threading.Thread(
            target=_run_auto_cycle,
            args=(run_id, stop_event, duration, normalized_target),
            daemon=True,
        )
        spray_thread.start()

        return {
            "success": True,
            "auto_dispense_started": True,
            "message": (
                f"Automatic spray started for {_target_label(normalized_target)} "
                f"for {duration} seconds."
            ),
            "pump_status": get_status(),
        }


def start_manual_dispense(target="AB"):
    """Turn on the selected pumps and keep them on until stopped manually."""
    normalized_target = _normalize_target(target)

    if not _ensure_initialized():
        return {
            "success": False,
            "message": "Pump controller is not initialized.",
            "pump_status": get_status(),
        }

    with _state_lock:
        _advance_run_id_locked()
        _cancel_auto_cycle_locked()

        if not turn_on_pump(normalized_target):
            _set_mode_locked("idle", False, "manual_failed", "none")
            return {
                "success": False,
                "message": "Failed to start manual dispensing.",
                "pump_status": get_status(),
            }

        _set_mode_locked("manual", True, "manual_start", normalized_target)

    return {
        "success": True,
        "message": f"{_target_label(normalized_target)} started manually.",
        "pump_status": get_status(),
    }


def stop_dispense():
    """Turn off all pumps immediately and cancel any automatic timer."""
    if not _ensure_initialized():
        return {
            "success": False,
            "message": "Pump controller is not initialized.",
            "pump_status": get_status(),
        }

    with _state_lock:
        _advance_run_id_locked()
        _cancel_auto_cycle_locked()
        turn_off_pump("AB")
        _set_mode_locked("idle", False, "manual_stop", "none")

    return {
        "success": True,
        "message": "All pumps stopped.",
        "pump_status": get_status(),
    }


def trigger_spray_pump(target="AB", duration=SPRAY_DURATION):
    """Compatibility wrapper for timed spray behavior."""
    result = trigger_auto_dispense(target=target, duration=duration)
    return result["success"]


def trigger_spray_by_disease(disease_name, duration=SPRAY_DURATION):
    """
    Spray only for mapped disease classes.

    Healthy leaves, non-leaf detections, and unknown values must not spray.
    """
    if disease_name in {"Tomato___healthy", "Not_A_Leaf", "Unknown"}:
        return {
            "success": True,
            "auto_dispense_started": False,
            "message": "No automatic spray needed for this detection.",
            "pump_status": get_status(),
        }

    target = DISEASE_PUMP_MAPPING.get(disease_name)
    if target is None:
        return {
            "success": True,
            "auto_dispense_started": False,
            "message": f"No pump mapping configured for {disease_name}.",
            "pump_status": get_status(),
        }

    return trigger_auto_dispense(target=target, duration=duration)


def cleanup_spray_pumps():
    """Force all pumps OFF and release GPIO resources."""
    global _gpio_backend, _pump_factory, spray_pump_initialized

    with _state_lock:
        _advance_run_id_locked()
        _cancel_auto_cycle_locked()
        _set_mode_locked("idle", False, "cleanup", "none")

    if _pump_devices:
        for channel, device in list(_pump_devices.items()):
            try:
                device.close()
            except Exception as exc:
                print(f"[CLEANUP ERROR] Failed to close pump {channel}: {exc}")
            finally:
                _pump_devices.pop(channel, None)

    if _gpio_backend == "gpiozero-lgpio":
        _pump_factory = None
        spray_pump_initialized = False
        _gpio_backend = "simulation"
        print("[CLEANUP] Pump GPIOs cleaned up.")
        return

    if not GPIO_AVAILABLE or GPIO is None:
        spray_pump_initialized = False
        print("[CLEANUP] Pump controller cleaned up in simulation mode.")
        return

    if not spray_pump_initialized:
        print("[CLEANUP] Pump controller was not initialized.")
        return

    try:
        for gpio_pin in PUMP_GPIO_MAP.values():
            if RELAY_ACTIVE_LOW:
                GPIO.setup(gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
            else:
                GPIO.output(gpio_pin, _off_state())
            GPIO.cleanup(gpio_pin)

        spray_pump_initialized = False
        print("[CLEANUP] Pump GPIOs cleaned up.")
    except Exception as exc:
        print(f"[CLEANUP ERROR] Failed during GPIO cleanup: {exc}")


def get_status():
    """Return the current state of the two-pump controller."""
    with _state_lock:
        return {
            "gpio_available": GPIO_AVAILABLE,
            "gpiozero_available": GPIOZERO_AVAILABLE,
            "initialized": spray_pump_initialized,
            "pump_a_gpio": PUMP_A_GPIO,
            "pump_b_gpio": PUMP_B_GPIO,
            "relay_active_low": RELAY_ACTIVE_LOW,
            "backend": _gpio_backend,
            "pump_running": _pump_running,
            "mode": _pump_mode,
            "last_action": _last_action,
            "active_target": _active_target,
            "active_target_label": _target_label(_active_target),
            "spray_duration": SPRAY_DURATION,
        }


def test_relay_type(channel="A", test_duration=2):
    """Simple relay test helper for one selected pump."""
    normalized_target = _normalize_target(channel if channel != "AB" else "A")
    gpio_pin = PUMP_GPIO_MAP[normalized_target]

    if not GPIO_AVAILABLE or GPIO is None:
        print("GPIO not available - cannot test relay type")
        return None

    print("=" * 60)
    print(f"Relay Type Test - Pump {normalized_target}")
    print("=" * 60)
    print(f"Testing GPIO {gpio_pin} for {test_duration} seconds each mode.\n")

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(gpio_pin, GPIO.OUT)

    try:
        print("Test 1: LOW = ON")
        GPIO.output(gpio_pin, GPIO.LOW)
        time.sleep(test_duration)
        GPIO.output(gpio_pin, GPIO.HIGH)
        response_low = input("Did the pump turn on while GPIO was LOW? (y/n): ").strip().lower()

        print("\nTest 2: HIGH = ON")
        GPIO.output(gpio_pin, GPIO.HIGH)
        time.sleep(test_duration)
        GPIO.output(gpio_pin, GPIO.LOW)
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
        GPIO.cleanup(gpio_pin)


if __name__ == "__main__":
    init_spray_pumps()
    print("Status:", get_status())
    print("\nTesting automatic spray for Pump 1...")
    trigger_auto_dispense(target="A")
    time.sleep(SPRAY_DURATION + 1)
    print("\nTesting automatic spray for Pump 2...")
    trigger_auto_dispense(target="B")
    time.sleep(SPRAY_DURATION + 1)
    print("\nTesting automatic spray for Both Pumps...")
    trigger_auto_dispense(target="AB")
    time.sleep(SPRAY_DURATION + 1)
    cleanup_spray_pumps()

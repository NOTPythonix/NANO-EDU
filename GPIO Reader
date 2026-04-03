#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import threading

# List of BCM GPIO pins commonly available on Pi 4B
ALL_PINS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

lock = threading.Lock()

def print_event(pin):
    with lock:
        val = GPIO.input(pin)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  PIN {pin:2d} -> {'HIGH' if val else 'LOW'}")

def edge_callback(pin):
    print_event(pin)

# Setup all pins as inputs with no pull (user can change)
for p in ALL_PINS:
    try:
        GPIO.setup(p, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
        GPIO.add_event_detect(p, GPIO.BOTH, callback=edge_callback, bouncetime=50)
    except Exception as e:
        print(f"Warning: pin {p} setup failed: {e}")

print("GPIO monitor started. Type 'help' for commands.")

def repl():
    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue
        parts = cmd.split()
        if parts[0] in ("exit","quit"):
            break
        if parts[0] == "help":
            print("Commands: read <pin>, set <pin> high|low, mode <pin> in|out, list, exit")
            continue
        if parts[0] == "list":
            for p in ALL_PINS:
                try:
                    m = "OUT" if GPIO.gpio_function(p)==GPIO.OUT else "IN"
                    v = GPIO.input(p)
                    print(f"PIN {p:2d}: mode={m} value={'HIGH' if v else 'LOW'}")
                except Exception as e:
                    print(f"PIN {p:2d}: error {e}")
            continue
        if parts[0] == "read" and len(parts)>=2:
            p = int(parts[1])
            print_event(p); continue
        if parts[0] == "mode" and len(parts)>=3:
            p = int(parts[1]); m = parts[2]
            if m=="in":
                GPIO.setup(p, GPIO.IN)
            else:
                GPIO.setup(p, GPIO.OUT)
            print(f"Set PIN {p} mode {m.upper()}"); continue
        if parts[0] == "set" and len(parts)>=3:
            p = int(parts[1]); v = parts[2]
            GPIO.setup(p, GPIO.OUT)
            GPIO.output(p, GPIO.HIGH if v in ("1","high","on","true") else GPIO.LOW)
            print(f"PIN {p} -> {v.upper()}"); continue
        print("Unknown command. Type 'help'.")

try:
    repl()
finally:
    GPIO.cleanup()
    print("Cleaned up GPIO and exiting.")

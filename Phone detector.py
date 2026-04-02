import cv2
import time
import RPi.GPIO as GPIO
import numpy as np
import random

# -----------------------------
# GPIO SETUP
# -----------------------------
GPIO.setmode(GPIO.BCM)

TRIG = 23
ECHO = 24

LEFT_FWD = 5
LEFT_BWD = 6
RIGHT_FWD = 13
RIGHT_BWD = 19

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

for pin in [LEFT_FWD, LEFT_BWD, RIGHT_FWD, RIGHT_BWD]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, 0)

# -----------------------------
# MOTOR CONTROL
# -----------------------------
def stop():
    for pin in [LEFT_FWD, LEFT_BWD, RIGHT_FWD, RIGHT_BWD]:
        GPIO.output(pin, 0)

def forward():
    GPIO.output(LEFT_FWD, 1)
    GPIO.output(RIGHT_FWD, 1)

def backward():
    GPIO.output(LEFT_BWD, 1)
    GPIO.output(RIGHT_BWD, 1)

def turn_left():
    GPIO.output(LEFT_BWD, 1)
    GPIO.output(RIGHT_FWD, 1)

def turn_right():
    GPIO.output(LEFT_FWD, 1)
    GPIO.output(RIGHT_BWD, 1)

# -----------------------------
# ULTRASONIC DISTANCE
# -----------------------------
def get_distance():
    GPIO.output(TRIG, 1)
    time.sleep(0.00001)
    GPIO.output(TRIG, 0)

    start = time.time()
    stop_t = time.time()

    while GPIO.input(ECHO) == 0:
        start = time.time()

    while GPIO.input(ECHO) == 1:
        stop_t = time.time()

    elapsed = stop_t - start
    distance = (elapsed * 34300) / 2
    return distance

# -----------------------------
# LOAD OBJECT DETECTION MODEL
# -----------------------------
net = cv2.dnn.readNetFromCaffe(
    "/home/pi/models/MobileNetSSD_deploy.prototxt",
    "/home/pi/models/MobileNetSSD_deploy.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "cell phone"]

# -----------------------------
# CAMERA SETUP
# -----------------------------
cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

# -----------------------------
# MAIN LOOP
# -----------------------------
try:
    while True:
        # --- Obstacle Avoidance ---
        dist = get_distance()
        print("Distance:", dist)

        if dist < 25:
            stop()
            time.sleep(0.2)
            turn_dir = random.choice(["left", "right"])
            if turn_dir == "left":
                turn_left()
            else:
                turn_right()
            time.sleep(0.5)
            stop()
        else:
            forward()

        # --- Object Detection ---
        ret, frame = cam.read()
        if not ret:
            continue

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        found_objects = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                found_objects.append(label)

        if "person" in found_objects:
            print("Detected: person")

        if "cell phone" in found_objects:
            print("Detected: phone‑shaped object")

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    stop()
    cam.release()
    GPIO.cleanup()

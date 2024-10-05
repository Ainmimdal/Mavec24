
#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import cv2
import numpy as np
import serial
import time
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
)

# Configure the serial port
ser = serial.Serial(
    port='/dev/ttyUSB0',  # Replace with the correct serial port
    baudrate=9600,        # Baud rate
    timeout=1             # Read timeout
)

class PID:
    def __init__(self, kp, ki, kd, min_output, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.previous_error = 0
        self.integral = 0

    def compute(self, error, dt):
        if abs(error) < 5:
            self.integral = 0  # Reset integral term to prevent windup
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        output = np.clip(output, self.min_output, self.max_output)
        self.previous_error = error
        return -1*output

# GStreamer pipeline for video input
def gstreamer_pipeline():
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
    )

# Lane detection and object detection integration
def process_frame(frame, hailo_detections):
    # Lane detection logic
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Hough lines to detect lanes
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        lines, lane_center = avg_lines(frame, lines)
        frame = draw_lines(frame, lines, (0, 255, 0))

    # Hailo object detection bounding boxes
    for detection in hailo_detections:
        # Draw bounding box around detected objects
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return frame

def avg_lines(image, lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    left_line = [0, 0, 0, 0]
    right_line = [0, 0, 0, 0]
    
    if left_lines:
        left_line_avg = np.average(left_lines, axis=0)
        left_line = make_coordinates(image, left_line_avg)
    
    if right_lines:
        right_line_avg = np.average(right_lines, axis=0)
        right_line = make_coordinates(image, right_line_avg)

    if left_line[2] != 0 and right_line[2] != 0:
        lane_center = (left_line[2] + right_line[2]) // 2
    elif left_line[2] != 0:
        lane_center = left_line[2] + 160
    elif right_line[2] != 0:
        lane_center = right_line[2] - 160
    else:
        lane_center = image.shape[1] // 2

    return np.array([left_line, right_line]), lane_center

def make_coordinates(image, line):   
    slope, intercept = line
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def draw_lines(image, lines, color):
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(image, (x1, y1), (x2, y2), color, 10)
    return image

def main():
    # Initialize GStreamer and Hailo inference
    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Simulate Hailo object detection for now (actual detection logic goes here)
        hailo_detections = [{'bbox': (100, 100, 200, 200)}]

        # Process frame for lane detection and object detection
        frame = process_frame(frame, hailo_detections)

        # Display frame
        cv2.imshow('Lane and Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

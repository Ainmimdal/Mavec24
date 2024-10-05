#!/usr/bin/env python3
import cv2
import numpy as np
from picamera2 import Picamera2
import serial
import time

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


def main():
    # Initialize Picamera2
    picam2 = Picamera2()
    try:
        # Create configuration for the camera
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()

        # PID controller for lane centering
        pid = PID(kp=2.5, ki=0.5, kd=1.0, min_output=-400, max_output=400)
        last_time = time.time()

        while True:
            frame = picam2.capture_array()

            if frame is None:
                print("No frame captured.")
                continue

            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 13), 0)
            edge = cv2.Canny(blur, 50, 155)

            height, width = edge.shape
            i = 155 # Adjusted to see further ahead
            j = 20  # Adjusted to see further ahead
            k = 25

            poly = np.array([
                [
                    (0, height - k),
                    (int(width / 2) - i, int(height / 2) + j),
                    (int(width / 2) + i, int(height / 2) + j),
                    (width, height - k)
                ]
            ])

            blank = np.zeros_like(edge)
            mask = blank.copy()
            mask = cv2.fillPoly(mask, pts=[poly], color=255)
            maskedImg = cv2.bitwise_and(edge, mask)

            # Process lines for lane detection
            lines = cv2.HoughLinesP(maskedImg, rho=3, theta=np.pi/45, threshold=20, lines=np.array([]), minLineLength=30, maxLineGap=5)

            final = image.copy()
            rawLinesImage = image.copy()
            color = (0, 255, 0)

            if lines is not None:
                avglines, lane_center = avg_lines(blank, lines)
                final = draw_lines(final, avglines, color)
                rawLinesImage = draw_lines(rawLinesImage, lines, (255, 0, 0))

                # Calculate error based on lane center
                error = lane_center - (width // 2)

                # Compute PID output
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                pid_output = pid.compute(error, dt)

                # Convert PID output to servo PPM value
                servo_ppm = 1500 + int(pid_output)
                servo_ppm = np.clip(servo_ppm, 1100, 1900)  # Ensure within valid range

                # Set a constant motor PPM (adjust as needed)
                motor_ppm = 1630

                # Send control signals
                send_serial_data(servo_ppm, motor_ppm)

                # Print debug information
                print(f"Error: {error}, PID Output: {pid_output}, Servo PPM: {servo_ppm}")
            else:
                # If no lines detected, stop the car
                send_serial_data(1500, 1500)
                print("No lanes detected, stopping.")

            cv2.imshow("CANNY IMAGE", edge)
            cv2.imshow("IMAGE LINES", rawLinesImage)
            cv2.imshow("FINAL", final)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

def send_serial_data(x_value, y_value):
    ser.write(f"{x_value},{y_value}\n".encode())
    print(f"Sent serial data: {x_value},{y_value}")

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

    # Calculate lane center
    if left_line[2] != 0 and right_line[2] != 0:
        lane_center = (left_line[2] + right_line[2]) // 2
    elif left_line[2] != 0:
        lane_center = left_line[2] + 160  # Assuming lane width is about 320 pixels
    elif right_line[2] != 0:
        lane_center = right_line[2] - 160
    else:
        lane_center = image.shape[1] // 2  # Default to image center if no lines detected

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
    
if __name__ == "__main__":
    main()

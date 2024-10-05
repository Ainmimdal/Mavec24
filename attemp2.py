#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import cv2
import numpy as np
import serial
import time
import setproctitle
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

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example

    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        if label == "person" or label == "parking sign":  # Added "parking sign"
            string_to_print += f"Detection: {label} {confidence:.2f}\n"
            detection_count += 1
            # Accessing bounding box and label for further processing in lane detection
            x_min, y_min, x_max, y_max = bbox
            print(f"Bounding Box - Xmin: {x_min}, Ymin: {y_min}, Xmax: {x_max}, Ymax: {y_max}")

    if user_data.use_frame:
        # Draw the detections on the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------
# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        super().__init__(args, user_data)
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        new_postprocess_path = os.path.join(self.current_path, '../resources/libyolo_hailortpp_post.so')
        if os.path.exists(new_postprocess_path):
            self.default_postprocess_so = new_postprocess_path
        else:
            self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolo_hailortpp_post.so')

        if args.hef_path is not None:
            self.hef_path = args.hef_path
        elif args.network == "yolov6n":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')
        elif args.network == "yolov8s":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
        elif args.network == "yolox_s_leaky":
            self.hef_path = os.path.join(self.current_path, '../resources/yolox_s_leaky_h8l_mz.hef')
        else:
            assert False, "Invalid network type"

        if args.labels_json is not None:
            self.labels_config = f' config-path={args.labels_json} '
        else:
            self.labels_config = ''

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        setproctitle.setproctitle("Hailo Detection App")
        self.create_pipeline()

    def get_pipeline_string(self):
         if self.source_type == "rpi":
            source_element = (
                "libcamerasrc name=src_0 ! "
                f"video/x-raw, format={self.network_format}, width=1536, height=864 ! "
                + QUEUE("queue_src_scale")
                + "videoscale ! "
                f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, framerate=30/1 ! "
            )
        elif self.source_type == "usb":
            source_element = (
                f"v4l2src device={self.video_source} name=src_0 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
            )
        else:
            source_element = (
                f"filesrc location=\"{self.video_source}\" name=src_0 ! "
                + QUEUE("queue_dec264")
                + " qtdemux ! h264parse ! avdec_h264 max-threads=2 ! "
                " video/x-raw, format=I420 ! "
            )
        source_element += QUEUE("queue_scale")
        source_element += "videoscale n-threads=2 ! "
        source_element += QUEUE("queue_src_convert")
        source_element += "videoconvert n-threads=3 name=src_convert qos=false ! "
        source_element += f"video/x-raw, format={self.network_format}, width={self.network_width}, height={self.network_height}, pixel-aspect-ratio=1/1 ! "

        pipeline_string = (
            "hailomuxer name=hmux "
            + source_element
            + "tee name=t ! "
            + QUEUE("bypass_queue", max_size_buffers=20)
            + "hmux.sink_0 "
            + "t. ! "
            + QUEUE("queue_hailonet")
            + "videoconvert n-threads=3 ! "
            f"hailonet hef-path={self.hef_path} batch-size={self.batch_size} {self.thresholds_str} force-writable=true ! "
            + QUEUE("queue_hailofilter")
            + f"hailofilter so-path={self.default_postprocess_so} {self.labels_config} qos=false ! "
            + QUEUE("queue_hmuc")
            + "hmux.sink_1 "
            + "hmux. ! "
            + QUEUE("queue_hailo_python")
            + QUEUE("queue_user_callback")
            + "identity name=identity_callback ! "
            + QUEUE("queue_hailooverlay")
            + "hailooverlay ! "
            + QUEUE("queue_videoconvert")
            + "videoconvert n-threads=3 qos=false ! "
            + QUEUE("queue_hailo_display")
            + f"fpsdisplaysink video-sink={self.video_sink} name=hailo_display sync={self.sync} text-overlay={self.options_menu.show_fps} signal-fps-measurements=true "
        )
        print(pipeline_string)
        return pipeline_string

def main():
    user_data = user_app_callback_class()
    parser = get_default_parser()
    parser.add_argument(
        "--network",
        default="yolov6n",
        choices=['yolov6n', 'yolov8s', 'yolox_s_leaky'],
        help="Which Network to use, default is yolov6n",
    )
    parser.add_argument(
        "--hef-path",
        default=None,
        help="Path to HEF file",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Path to custom labels JSON file",
    )
    args = parser.parse_args()
    app = GStreamerDetectionApp(args, user_data)
    app.run()

    try:
        pid = PID(kp=2.5, ki=0.5, kd=1.0, min_output=-400, max_output=400)
        last_time = time.time()

        while True:
            frame = user_data.get_frame()  # Get frame from Gstreamer pipeline

            if frame is None:
                print("No frame captured.")
                continue

            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 13), 0)
            edge = cv2.Canny(blur, 50, 155)

            height, width = edge.shape
            i = 155  # Adjusted to see further ahead
            j = 20   # Adjusted to see further ahead
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
:
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

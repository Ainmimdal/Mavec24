import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import serial
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
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
last_time = time.time()

# Sign handlers and state management
class SignState:
    """Enum-like class for different driving states"""
    NORMAL = "normal"
    ROUNDABOUT = "roundabout"
    LANE_CHANGE = "lane_change"
    SLIPPERY = "slippery"
    TRAFFIC_LIGHT = "traffic_light"

class ManeuverStep:
    def __init__(self, duration, servo_ppm, motor_ppm, description=""):
        self.duration = duration
        self.servo_ppm = servo_ppm
        self.motor_ppm = motor_ppm
        self.description = description

class BaseSignHandler:
    def __init__(self):
        self.is_active = False
        self.start_time = None
        self.current_step = 0
        self.steps = []
        
    def start(self, current_time):
        self.is_active = True
        self.start_time = current_time
        self.current_step = 0
        
    def stop(self):
        self.is_active = False
        self.current_step = 0
        
    def get_control_values(self, current_time):
        if not self.is_active:
            return None
            
        elapsed_time = current_time - self.start_time
        time_in_sequence = 0
        
        # Calculate which step we should be on
        for step_idx, step in enumerate(self.steps):
            if step_idx < self.current_step:
                time_in_sequence += step.duration
                
        # Check if we should move to next step
        if self.current_step < len(self.steps):
            current_step = self.steps[self.current_step]
            if elapsed_time >= time_in_sequence + current_step.duration:
                self.current_step += 1
                if self.current_step >= len(self.steps):
                    self.stop()
                    return None
                    
        if self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            return (step.servo_ppm, step.motor_ppm, step.description)
            
        return None

class RoundaboutHandler(BaseSignHandler):
    def __init__(self):
        super().__init__()
        self.steps = [
            ManeuverStep(1.0, 1500, 1630, "Approaching roundabout"),
            ManeuverStep(2.0, 1700, 1630, "Turning left"),
            ManeuverStep(1.5, 1500, 1630, "Exiting roundabout")
        ]

class LaneChangeHandler(BaseSignHandler):
    def __init__(self):
        super().__init__()
        self.steps = [
            ManeuverStep(1.0, 1500, 1630, "Preparing for lane change"),
            ManeuverStep(1.5, 1600, 1630, "Moving to left lane"),
            ManeuverStep(1.0, 1500, 1630, "Stabilizing in new lane")
        ]

class SlipperyRoadHandler(BaseSignHandler):
    def __init__(self):
        super().__init__()
        self.steps = [
            ManeuverStep(5.0, 1500, 1580, "Reducing speed in slippery zone")
        ]

class TrafficLightHandler(BaseSignHandler):
    def __init__(self):
        super().__init__()
        self.red_light = False
        self.confidence_threshold = 0.6  # Adjust this threshold as needed
        
    def handle_light(self, detections, current_time):
        """
        Handle traffic light detections
        
        Args:
            detections: List of tuples (label, confidence)
            current_time: Current timestamp
        """
        for label, confidence in detections:
            if confidence < self.confidence_threshold:
                continue
                
            if label == "red_light" and not self.red_light:
                self.steps = [ManeuverStep(999999, 1500, 1500, "Stopped at red light")]
                self.red_light = True
                self.start(current_time)
                return True
                
            elif label == "green_light" and self.red_light:
                self.steps = [ManeuverStep(2.0, 1500, 1630, "Proceeding on green")]
                self.red_light = False
                self.start(current_time)
                return True
                
        return False

class SignDetectionSystem:
    def __init__(self):
        self.current_state = SignState.NORMAL
        self.sign_confidence_threshold = 0.6
        self.detection_cooldown = 5.0
        self.last_detection_time = 0
        
        # Initialize handlers for each sign type
        self.handlers = {
            "Roundabout_sign": RoundaboutHandler(),
            "Linechange_sign": LaneChangeHandler(),
            "Slipperyroad_sign": SlipperyRoadHandler()
        }
        # Initialize traffic light handler separately
        self.traffic_light_handler = TrafficLightHandler()
        
    def handle_detections(self, detections, current_time, frame_height, frame_width):
        """Process detections and return appropriate control values"""
        if current_time - self.last_detection_time < self.detection_cooldown:
            # Check traffic light first
            if self.traffic_light_handler.is_active:
                return self.traffic_light_handler.get_control_values(current_time)
                
            # Then check other handlers
            for handler in self.handlers.values():
                if handler.is_active:
                    return handler.get_control_values(current_time)
            return None
            
        # First check for traffic lights as they have priority
        if self.traffic_light_handler.handle_light(detections, current_time):
            self.last_detection_time = current_time
            return self.traffic_light_handler.get_control_values(current_time)
            
        # Then check for other signs
        for detection in detections:
            label = detection[0]
            confidence = detection[1]
            
            if confidence < self.sign_confidence_threshold:
                continue
                
            if label in self.handlers:
                handler = self.handlers[label]
                if not handler.is_active:
                    print(f"Detected {label}! Starting corresponding maneuver.")
                    handler.start(current_time)
                    self.last_detection_time = current_time
                    return handler.get_control_values(current_time)
                    
        # Check if any handler is active and return its control values
        if self.traffic_light_handler.is_active:
            return self.traffic_light_handler.get_control_values(current_time)
            
        for handler in self.handlers.values():
            if handler.is_active:
                return handler.get_control_values(current_time)
                
        return None
# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Initialize variables for lane detection
        self.canny_low = 50
        self.canny_high = 150

         # Initialize detection storage
        self.current_detections = []
        self.sign_system = SignDetectionSystem()
        # Initialize serial connection
        self.ser = serial.Serial(
            port='/dev/ttyUSB0',  # Replace with the correct serial port
            baudrate=9600,        # Baud rate
            timeout=1             # Read timeout
        )
        
        # Initialize PID controller
        self.pid_controller = self.PID()
        self.last_time = time.time()
    
    class PID:
        def __init__(self, kp=2.5, ki=0.5, kd=1.0, min_output=-400, max_output=400):
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
            return -1 * output
    
    def process_frame(self, frame):
        # First handle the detections and draw signboard
        if hasattr(self, '_temp_detections'):  # Check if we have temporary detections stored
            intersecting_objects = self.draw_signboard(frame, self._temp_detections)
            self.current_detections = intersecting_objects  # Update current_detections
        
        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 13), 0)
        edge = cv2.Canny(blur, self.canny_low, self.canny_high)

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
        #draw ROI polygon
        cv2.polylines(frame, [poly], True, (255, 165, 0), 2)  # Orange color for ROI

        blank = np.zeros_like(edge)
        mask = cv2.fillPoly(blank.copy(), pts=[poly], color=255)
        maskedImg = cv2.bitwise_and(edge, mask)

        lines = cv2.HoughLinesP(
            maskedImg, 
            rho=3, 
            theta=np.pi/45, 
            threshold=20, 
            lines=np.array([]), 
            minLineLength=30, 
            maxLineGap=5
        )

        final = image.copy()
        rawLinesImage = image.copy()
        color = (0, 255, 0)

        # Compute PID output with proper timing
        current_time = time.time()
        
        # Check for active sign handlers
        control_values = self.sign_system.handle_detections(
            self.current_detections, 
            current_time, 
            frame.shape[0], 
            frame.shape[1]
        )
        
        if control_values:
            servo_ppm, motor_ppm, description = control_values
            self.send_serial_data(servo_ppm, motor_ppm)
            # Draw maneuver status
            cv2.putText(frame, f"Executing: {description}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return frame

        if lines is not None:
            avglines, lane_center = self.avg_lines(blank, lines)
            final = self.draw_lines(final, avglines, color)
            rawLinesImage = self.draw_lines(rawLinesImage, lines, (255, 0, 0))

            # Draw lane center
            cv2.circle(frame, (lane_center, height // 2), 5, (255, 255, 0), -1)  # Yellow dot for lane center
            # Calculate error based on lane center
            error = lane_center - (width // 2)

            # Compute PID output with proper timing
            #current_time = time.time()

            dt = current_time - self.last_time
            self.last_time = current_time
            
            pid_output = self.pid_controller.compute(error, dt)

            # Convert PID output to servo PPM value
            servo_ppm = 1500 + int(pid_output)
            servo_ppm = np.clip(servo_ppm, 1100, 1900)  # Ensure within valid range

            # Set a constant motor PPM (adjust as needed)
            motor_ppm = 1630

            # Send control signals
            self.send_serial_data(servo_ppm, motor_ppm)

            # Print debug information
            print(f"Error: {error}, PID Output: {pid_output}, Servo PPM: {servo_ppm}")
            
             # Draw PID info
            cv2.putText(frame, f"Error: {error:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Steering: {servo_ppm}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return final
        else:
            # If no lines detected, stop the car
            self.send_serial_data(1500, 1500)
            print("No lanes detected, stopping.")
            cv2.putText(frame, "No lanes detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return image
            
    def send_serial_data(self,x_value, y_value):
        self.ser.write(f"{x_value},{y_value}\n".encode())
        print(f"Sent serial data: {x_value},{y_value}")

    

    def avg_lines(self, image, lines):
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
            left_line = self.make_coordinates(image, left_line_avg)
        
        if right_lines:
            right_line_avg = np.average(right_lines, axis=0)
            right_line = self.make_coordinates(image, right_line_avg)

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

    def make_coordinates(self, image, line):   
        slope, intercept = line
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def draw_lines(self, image, lines, color):
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(image, (x1, y1), (x2, y2), color, 10)
        return image
    
    def draw_signboard(self, frame, detections):
        """
        Draw signboard region and check for intersecting objects
        """
        height, width = frame.shape[:2]
        
        # Store the detections for the next frame
        self._temp_detections = detections
        
        # Signboard region
        region_x0 = int(width * 0.65)  
        region_y0 = int(height * 0.2)
        region_x1 = width
        region_y1 = int(height * 0.7)
        
        # Draw signboard region
        cv2.rectangle(frame, (region_x0, region_y0), (region_x1, region_y1), (0, 0, 255), 2)  
        cv2.putText(frame, "Signboard Region", (region_x0 + 5, region_y0 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Check intersections with detections
        region = (region_x0, region_y0, region_x1, region_y1)
        intersecting_objects = []
        
        current_time = time.time()
        
        for detection in detections:
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            
            detection_box = (x1, y1, x2, y2)
            
            if self.check_intersection(detection_box, region):
                intersecting_objects.append((label, confidence))
                # Draw intersecting detection with different color
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        # Store current detections for the sign system
        self.current_detections = intersecting_objects
        return intersecting_objects
    
    def check_intersection(self, bbox, region):
        """
        Check if a bounding box intersects with the signboard region
        bbox: tuple of (x1, y1, x2, y2) - detection coordinates
        region: tuple of (x0, y0, x1, y1) - signboard region coordinates
        """
        box_x1, box_y1, box_x2, box_y2 = bbox
        reg_x0, reg_y0, reg_x1, reg_y1 = region
        
        # Check if one rectangle is completely to the left of the other
        if box_x2 < reg_x0 or reg_x1 < box_x1:
            return False
        
        # Check if one rectangle is completely above the other
        if box_y2 < reg_y0 or reg_y1 < box_y1:
            return False
        
        return True
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
    #TEMPORARY COMMENT
    # detection_count = 0
    # for detection in detections:
    #     label = detection.get_label()
    #     bbox = detection.get_bbox()
    #     confidence = detection.get_confidence()
    #     if label == "person":
    #         string_to_print += f"Detection: {label} {confidence:.2f}\n"
    #         detection_count += 1

    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        #cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        
        # Store detections temporarily in user_data
        user_data._temp_detections = detections

        # Process frame for lane detection
        processed_frame = user_data.process_frame(frame)
        
        if processed_frame is not None:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            user_data.set_frame(processed_frame)

        # Convert the frame to BGR
        # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        # user_data.set_frame(processed_frame)

    print(string_to_print)
    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# User Gstreamer Application
# -----------------------------------------------------------------------------------------------

# This class inherits from the hailo_rpi_common.GStreamerApp class
class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, args, user_data):
        # Call the parent class constructor
        super().__init__(args, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        # Temporary code: new postprocess will be merged to TAPPAS.
        # Check if new postprocess so file exists
        new_postprocess_path = os.path.join(self.current_path, '../resources/libyolo_hailortpp_post.so')
        if os.path.exists(new_postprocess_path):
            self.default_postprocess_so = new_postprocess_path
        else:
            self.default_postprocess_so = os.path.join(self.postprocess_dir, 'libyolo_hailortpp_post.so')

        if args.hef_path is not None:
            self.hef_path = args.hef_path
        # Set the HEF file path based on the network
        elif args.network == "yolov6n":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')
        elif args.network == "yolov8s":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
        elif args.network == "yolox_s_leaky":
            self.hef_path = os.path.join(self.current_path, '../resources/yolox_s_leaky_h8l_mz.hef')
        else:
            assert False, "Invalid network type"

        # User-defined label JSON file
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

        # Set the process title
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

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    parser = get_default_parser()
    # Add additional arguments here
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
        help="Path to costume labels JSON file",
    )
    args = parser.parse_args()
    app = GStreamerDetectionApp(args, user_data)
    app.run()

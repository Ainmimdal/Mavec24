from multiprocessing import Process, Queue
import argparse
import time

# Import the required classes and functions from detection2 and movepid2
from detection2 import GStreamerDetectionApp, user_app_callback_class
from movepid2 import car_control_loop
from hailo_rpi_common import get_default_parser  # Import the parser function

# The main entry point to run both processes
if __name__ == "__main__":
    # Step 1: Parse the command-line arguments needed for GStreamerDetectionApp
    parser = get_default_parser()  # Use the parser defined in detection2.py (hailo_rpi_common)

    # Add additional arguments specific to this script or modify them if needed
    parser.add_argument(
        "--network",
        default="yolov6n",
        choices=['yolov6n', 'yolov8s', 'yolox_s_leaky'],
        help="Which network to use. Default is yolov6n."
    )
    parser.add_argument(
        "--hef-path",
        default=None,
        help="Path to the HEF file."
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="Path to the labels JSON file."
    )
    
    args = parser.parse_args()  # Parse the arguments

    # Create a multiprocessing queue for sharing detections between processes
    detection_queue = Queue()

    # Step 2: Launch the car control loop process
    car_control_process = Process(target=car_control_loop, args=(detection_queue,))
    car_control_process.start()

    # Step 3: Launch the GStreamer detection app process
    user_data = user_app_callback_class()  # Initialize the user-defined callback class

    # Launch the GStreamer detection app
    gstreamer_detection_process = Process(
        target=lambda: GStreamerDetectionApp(args, user_data, detection_queue).run()
    )
    gstreamer_detection_process.start()

    # Step 4: Wait for both processes to complete
    car_control_process.join()
    gstreamer_detection_process.join()

from flask import Blueprint, render_template, session, jsonify, Response, request, redirect, url_for
import os
import json
import threading
import time
import cv2
import pandas as pd
import ast
from vmbpy import *
from routes.mfc import print_input_channel_values, set_mfc_values


experiment_dashboard_bp = Blueprint('experiment_dashboard', __name__, template_folder='templates')

# Global variables for experiment tracking
current_experiment_runner = None
current_experiment_recorder = None
experiment_is_running = False
current_experiment_output_path = None

def load_chamber_config(chambers_config_file):
    """
    Load the chamber configuration from the JSON file.
    If the file doesn't exist, create a new one with default values.
    """
    if os.path.exists(chambers_config_file):
        with open(chambers_config_file, 'r') as file:
            chambers = json.load(file)
    else:
        chambers = {f'chamber_{i}': {'x': 0, 'y': 0, 'width': 0, 'height': 0} for i in range(1, 21)}
        with open(chambers_config_file, 'w') as file:
            json.dump(chambers, file, indent=4)
    return chambers


@experiment_dashboard_bp.route('/fetch_mfc_data', methods=['GET'])
def fetch_mfc_data():
    # Load the MFC settings from mfc_settings.json
    file_name = "configuration_files/mfc_settings.json"
    if not os.path.exists(file_name):
        return jsonify({"error": "MFC settings file not found."}), 404

    with open(file_name, 'r') as f:
        mfc_settings = json.load(f)

    mfc_data = []
    # Iterate over MFC settings to fetch readings
    for setting in mfc_settings:
        board_num = setting['input_board_num']
        channel_num = setting['input_channel_num']
        reading = print_input_channel_values(board_num)
        mfc_data.append({
            'MFC Name': setting['MFC Name'],
            'board_num': board_num,
            'channel_num': channel_num,
            'reading': reading[channel_num]  # Get reading for the specific channel
        })

    # Maintain the voltage values based on the loaded MFC settings
    set_mfc_values(mfc_settings)
    return jsonify({"value": mfc_data})


@experiment_dashboard_bp.route('/fetch_experiment_progress', methods=['GET'])
def fetch_experiment_progress():
    # --- 1. Extract profile and experiment metadata from session ---
    selected_profile = session.get('selected_profile', {})
    experiment_metadata = session.get('experiment-metadata', {})

    profile_name = selected_profile.get('name', 'default_profile')
    protocol_name = experiment_metadata.get('protocol', 'default_protocol.csv')

    # --- 2. Build the full path to the selected protocol ---
    protocol_folder = os.path.join('protocols', profile_name)
    protocol_path = os.path.join(protocol_folder, protocol_name)

    # --- 3. Read the selected protocol CSV file ---
    protocol = pd.read_csv(protocol_path)

    # --- 4. Process the CSV to group stages (your original logic) ---
    grouped_steps = {}

    for _, row in protocol.iterrows():
        step_name = row['step_name']
        delay_seconds = int(row['delay_seconds'])
        parameters = ast.literal_eval(row['parameters'])
        duration = parameters.get('duration')

        key = (step_name, delay_seconds)

        if key not in grouped_steps:
            grouped_steps[key] = {
                "step_name": step_name,
                "start_time": delay_seconds,
                "duration": duration
            }
        else:
            grouped_steps[key]["duration"] = max(grouped_steps[key]["duration"], duration)

    experiment_steps = [
        {"step_name": step["step_name"], "start_time": step["start_time"], "duration": step["duration"]}
        for step in grouped_steps.values()
    ]

    return jsonify({"experiment_steps": experiment_steps})



@experiment_dashboard_bp.route('/fetch_mfc_settings', methods=['GET'])
def fetch_mfc_settings():
    # Load the MFC settings from mfc_settings.json
    file_name = "configuration_files/mfc_settings.json"
    if not os.path.exists(file_name):
        return jsonify({"error": "MFC settings file not found."}), 404

    with open(file_name, 'r') as f:
        mfc_settings = json.load(f)
    
    return jsonify(mfc_settings)

"""
CAMERA CONFIGURATION
The following code will deal with any configurations, functions or classes that are related to the image acquisition, handling, manipulation and delivery.
"""

import sys
from typing import Optional
from queue import Queue
import threading
from vmbpy import *
import cv2
import numpy as np
import time

class Handler:
    def __init__(self):
        self.display_queue = Queue(10)
        self.latest_frame = None
        self.lock = threading.Lock()  # To synchronize access to the latest frame

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            # print('{} acquired {}'.format(cam, frame), flush=True)

            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(opencv_display_format)

            # Store the frame in the queue and in the shared variable
            self.display_queue.put(display.as_opencv_image(), True)
            
            # Safely update the latest frame
            with self.lock:
                self.latest_frame = display.as_opencv_image()

        cam.queue_frame(frame)

def print_preamble():
    print('///////////////////////////////////////////////////')
    print('/// VmbPy Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)

def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    return None if len(args) == 0 else args[0]

def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)
            except VmbCameraError:
                abort(f"Failed to access Camera '{camera_id}'. Abort.")
        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')
            return cams[0]

def setup_camera(cam: Camera):
    with cam:
        # Disable auto exposure to manually set it
        try:
            cam.ExposureAuto.set('Off')
        except (AttributeError, VmbFeatureError):
            pass

        # Set a lower exposure time manually
        try:
            cam.ExposureTime.set(30000)  # Exposure time in microseconds (5ms)
        except (AttributeError, VmbFeatureError):
            pass

        # Enable auto white balance if available
        try:
            cam.BalanceWhiteAuto.set('Continuous')
        except (AttributeError, VmbFeatureError):
            pass

        # Check and set binning if available
        try:
            if cam.BinningHorizontal and cam.BinningVertical:
                cam.BinningHorizontal.set(2)  # Example binning factor
                cam.BinningVertical.set(2)    # Example binning factor
        except (AttributeError, VmbFeatureError):
            pass

        # Set resolution (example values; adjust according to your camera's supported resolutions)
        try:
            cam.Width.set(960)  # Example width in pixels
            cam.Height.set(540)  # Example height in pixels
        except (AttributeError, VmbFeatureError):
            pass

        try:
            stream = cam.get_streams()[0]
            stream.GVSPAdjustPacketSize.run()
            while not stream.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VmbFeatureError):
            pass

def setup_pixel_format(cam: Camera):
    # Query available pixel formats. Prefer color formats over monochrome formats
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())

    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())

    # if OpenCV compatible color format is supported directly, use that
    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)

    # else if existing color format can be converted to OpenCV format do that
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])

    # fall back to a mono format that can be converted
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])

    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')

def capture_frames(handler, cam_id=None):
    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            setup_camera(cam)
            setup_pixel_format(cam)

            try:
                cam.start_streaming(handler=handler, buffer_count=10)
                while not stop_event.is_set():
                    handler.get_image()
            finally:
                cam.stop_streaming()

class OdorColumn():
    def __init__(self) -> None:
        self.odor_delivery_status = False

    @classmethod
    def create_odor_columns(cls):
        """
        This function will create the odor column objects that are currently used in the multiplex set.
        This includes:
         - mch_right
         - oct_right
         - moil_right
         - mch_left
         - oct_left
         - moil_left
        """
        cls.mch_right = cls()
        cls.oct_right = cls()
        cls.moil_right = cls()
        cls.mch_left = cls()
        cls.oct_left = cls()
        cls.moil_left = cls()

    @classmethod
    def list_odor_columns(cls):
        """
        Lists all OdorColumn objects created by the create_odor_columns method.

        Returns:
            list: A list of names of OdorColumn objects.
        """
        odor_columns = []
        for name, obj in cls.__dict__.items():
            if isinstance(obj, cls):
                odor_columns.append(name)
        return odor_columns
    
    @classmethod
    def export_odor_status(cls):
        """
        Exports the current odor delivery status of each odor column as a dictionary.

        Returns:
            dict: A dictionary where each key is an odor column ID and the value is the current odor delivery status.
        """
        odor_columns = cls.list_odor_columns()
        odor_status = {}

        for odor_column_name in odor_columns:
            odor_column = getattr(cls, odor_column_name, None)
            if odor_column is not None:
                odor_status[odor_column_name] = odor_column.odor_delivery_status

        return odor_status


class Chamber():
    def __init__(self) -> None:
        self.fly_loc = None
        self.shock_status = False
    
    @classmethod
    def create_chambers(cls):
        """
        Creates 20 Chamber objects and assigns them to class attributes.

        Example:
            Chamber.create_chambers()
            This will create attributes like Chamber.chamber_1, Chamber.chamber_2, ..., Chamber.chamber_20.
        """
        for i in range(1, 21):
            setattr(cls, f'chamber_{i}', cls())

    @classmethod
    def list_chambers(cls):
        """
        Lists all Chamber objects created by the create_chambers method.

        Returns:
            list: A list of Chamber objects.
        """
        chambers = []
        for i in range(1, 21):
            chamber = getattr(cls, f'chamber_{i}', None)
            if chamber:
                chambers.append(chamber)
        return chambers
    
    @classmethod
    def update_flies_loc(cls, current_loc):
        """
        This function will update the last known location of the flies.
        If a fly's location is not provided in the current_loc dictionary,
        it will retain its last known value.

        Args:
            current_loc (dict): Dictionary of the flies' last location, using the assign_contours_to_chambers() function.
        """
        chambers = cls.list_chambers()
        for chamber in chambers:
            chamber_id = [name for name, obj in cls.__dict__.items() if obj == chamber][0]
            # Only update if the chamber_id is present in current_loc
            if chamber_id in current_loc:
                chamber.fly_loc = current_loc[chamber_id]

    @classmethod
    def export_flies_loc(cls):
        """
        Exports the last known location of each fly as a dictionary.

        Returns:
            dict: A dictionary where each key is a chamber ID and the value is the last known location of the fly.
        """
        chambers = cls.list_chambers()
        fly_locations = {}

        for chamber in chambers:
            chamber_id = [name for name, obj in cls.__dict__.items() if obj == chamber][0]
            fly_locations[chamber_id] = chamber.fly_loc

        return fly_locations
    
    @classmethod
    def set_shock_status(cls, chamber_id, status):
        """
        Sets the shock status of a specific chamber.

        Args:
            chamber_id (int or str): The ID of the chamber (e.g., '1' for 'chamber_1').
            status (bool): The desired shock status (True for ON, False for OFF).
        """

        # Dynamically access the chamber object using getattr
        chamber = getattr(cls, chamber_id, None)

        if chamber:
            chamber.shock_status = status
            print(f"Chamber {chamber_id} shock status set to {status}")
        else:
            print(f"Chamber {chamber_id} not found.")

    @classmethod
    def shock_all_on(cls):
        """
        Assigns shock_status = True for all chamber objects.
        """
        chambers = cls.list_chambers()
        for chamber in chambers:
            chamber.shock_status = True

    @classmethod
    def shock_all_off(cls):
        """
        Assigns shock_status = False for all chamber objects.
        """
        chambers = cls.list_chambers()
        for chamber in chambers:
            chamber.shock_status = False
    
    @classmethod
    def export_flies_shock_status(cls):
        """
        Exports the shock status of each fly as a dictionary.

        Returns:
            dict: A dictionary where each key is a chamber ID and the value is the shock status of the fly (True/False).
        """
        chambers = cls.list_chambers()
        shock_statuses = {}

        for chamber in chambers:
            chamber_id = [name for name, obj in cls.__dict__.items() if obj == chamber][0]
            shock_statuses[chamber_id] = chamber.shock_status

        return shock_statuses




# Define global variables
stop_event = threading.Event()
opencv_display_format = PixelFormat.Bgr8
handler = None
capture_thread = None
global chambers 
Chamber.create_chambers() # Create Chamber objects to store flies location
OdorColumn.create_odor_columns()
chambers = load_chamber_config('configuration_files/chambers_configuration.json')

@experiment_dashboard_bp.route('/start_camera', methods=['POST'])
def start_camera():
    global handler, capture_thread
    stop_event.clear()
    cam_id = parse_args()  # Assuming cam_id is provided
    handler = Handler()

    # Start capturing frames in the background
    capture_thread = threading.Thread(target=capture_frames, args=(handler, cam_id))
    capture_thread.start()
    return jsonify({"message": "Camera started successfully"}), 200

@experiment_dashboard_bp.route('/stop_camera', methods=['POST'])
def stop_camera():
    global stop_event, capture_thread

    # Set the event to stop the frame capture loop
    stop_event.set()

    # Wait for the background thread to finish
    if capture_thread is not None:
        capture_thread.join()
        capture_thread = None  # Reset the thread to ensure it can be restarted

    # Return a response indicating the camera has been stopped
    return jsonify({"message": "Camera stopped successfully"}), 200

def get_latest_frame():
    """
    Retrieves the latest frame from the camera handler.
    Returns a copy of the frame as a NumPy array or None if no frame is available.
    """
    global handler
    with handler.lock:
        if handler.latest_frame is not None:
            return handler.latest_frame.copy()  # Return a copy of the numpy array
        return None  # Return None if no frame is available

show_rectangles_event = threading.Event()
show_rectangles_event.set()  # Initially, show rectangles

@experiment_dashboard_bp.route('/toggle_rectangles', methods=['POST'])
def toggle_rectangles():
    if show_rectangles_event.is_set():
        show_rectangles_event.clear()  # Hide rectangles
    else:
        show_rectangles_event.set()  # Show rectangles
    return jsonify({"status": "success"}), 200

@experiment_dashboard_bp.route('/fetch_image', methods=['GET'])
def fetch_image():
    """
    This function will deliver video output to the GUI with chamber boundaries overlayed, depending on the event state,
    and draw crosses at the last known locations of the flies.
    """
    frame = get_latest_frame()

    if frame is not None:
        if show_rectangles_event.is_set():
            global chambers
            # Get the last known fly locations
            apply_detection_to_chambers()
            fly_locations = Chamber.export_flies_loc()
            
            # Draw rectangles for each chamber on the frame
            for chamber_id, config in chambers.items():
                x, y, width, height = config['x'], config['y'], config['width'], config['height']
                # Draw rectangle on the frame: (frame, top-left corner, bottom-right corner, color, thickness)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                # Optionally, add chamber ID as text on the image
                cv2.putText(frame, chamber_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Draw a cross at the last known location of the fly within the chamber
                # Draw a cross at the last known location of the fly within the chamber
                if chamber_id in fly_locations:
                    fly_x_loc = fly_locations[chamber_id]
                    # Check if fly_x_loc is a valid number before using np.isnan
                    if isinstance(fly_x_loc, (int, float)) and not np.isnan(fly_x_loc):
                        # Calculate the actual x-coordinate within the chamber
                        fly_x_pixel = int(x + (fly_x_loc + 100) * width / 200)
                        fly_y_pixel = int(y + height / 2)  # Center the cross vertically within the chamber

                        # Draw a red cross (horizontal and vertical lines)
                        cv2.line(frame, (fly_x_pixel - 5, fly_y_pixel), (fly_x_pixel + 5, fly_y_pixel), (0, 0, 255), 1)
                        cv2.line(frame, (fly_x_pixel, fly_y_pixel - 5), (fly_x_pixel, fly_y_pixel + 5), (0, 0, 255), 1)
        
        # Encode the frame with rectangles and crosses to .jpg format
        _, buffer = cv2.imencode('.jpg', frame)
        response = buffer.tobytes()
        
        return Response(response, mimetype='image/jpeg')
    else:
        return jsonify({"error": "No image available."}), 404
        
"""
This code segments handles the recording of the video and saving it to a newly created file
"""

def capture_frame():
    """
    This function will be used for back-end proccessing of the video-feed.
    """
    frame = get_latest_frame()
    if frame is not None:
        return frame
    return None

def video_recorder(control_event, fps, video_file_path):
    """
    This function will record the image obtained by the camera and save it into a file.
    It will start recording when the control_event is set and stop when the control_event is cleared.
    The video will be saved in Full HD resolution (1920x1080).
    
    Args:
        control_event (threading.Event): Event to control the start/stop of recording.
        fps (int): Frames per second to record the video.
        video_file_path (str): The full path (including file name) where the video will be saved.
    """
    video = None  # VideoWriter object

    try:
        # Initialize VideoWriter with Full HD resolution and appropriate codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for .mp4 files
        video = cv2.VideoWriter(str(video_file_path)+"/video.mp4", fourcc, fps, (1920, 1080), True)  # True indicates color

        frame_count = 0
        start_time = time.time()
        
        while control_event.is_set():  # While the control_event is set, keep recording
            # Capture the current frame and save it to the video
            frame = capture_frame()
            if frame is not None:
                # Check and ensure frame is in the correct format
                if frame.shape[1] != 1920 or frame.shape[0] != 1080:
                    frame = cv2.resize(frame, (1920, 1080))  # Resize to Full HD resolution

                if len(frame.shape) == 2:  # If frame is grayscale, convert to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                video.write(frame)  # Write the frame to the video file
                frame_count += 1
                
                # Calculate expected time for this frame
                expected_time = frame_count / fps
                actual_time = time.time() - start_time
                
                # Only sleep if we're ahead of schedule
                if expected_time > actual_time:
                    time.sleep(expected_time - actual_time)
                    
        print("Recording stopped as the control event was cleared.")
        
    finally:
        if video is not None:
            video.release()  # Release the VideoWriter to save the file
            print(f"Recording stopped and file saved at {video_file_path}.")



"""
From this point the code is made to handle the ExperimentRunner and ExperimentRecorder classes and their functions.
The goal here is to control the instruments once an experiment starts. Also, we want to measure shock activity, fly
"""

class ExperimentRunner:
    def __init__(self, recorder):
        """
        Initialize the ExperimentRunner class.
        
        Args:
            1. recorder - An instance of the ExperimentRecorder class.
        
        This class is responsible for running the experiment according to a loaded protocol and managing 
        the control flow using threading events.
        """
        self.recorder = recorder
        self.control_event = threading.Event()  # Event to start/stop the experiment
        self.finished_event = threading.Event()  # Event to signal the experiment is finished
        self.run_thread = threading.Thread(target=self.run_experiment)
        self.run_thread.daemon = True  # Ensure the thread exits when the main program does
        self.experiment_protocol = None
        self.odor_map = None
        self.chamber_map = None
        self.active_timers = []  # Track active timers for graceful stopping

    def load_protocol_file(self, protocol):
        """
        This function will load a protocol sequence from a CSV file and assign it to the self.experiment_protocol variable.

            Args:
                1. protocol - CSV file of the steps that make the protocol
            
            Returns:
                Nothing. It assigns the protocol dataframe to the self.experiment_protocol variable.
        """
        self.experiment_protocol = pd.read_csv(protocol)
    
    def run_experiment(self):
        """
        This function executes the experiment based on the given protocol CSV file.
        It runs commands as dictated by the protocol, activating devices accordingly.

        The `current_exp_step` is set to the current step when an action is running and
        reset to an empty string ("") when no action is being performed (during delays or after an action finishes).

        Args:
            Nothing
        
        Returns:
            Nothing. It executes the experiment according to the protocol.
        """
        # Load configuration for boards & channels that control valves and shock to chambers
        self.odor_map = load_odor_channel_mapping()
        self.chamber_map = load_shock_channel_mapping()
        print('Starting Experiment')

        # Load the protocol dataframe
        df = self.experiment_protocol
        print(f'Loaded protocol with {len(df)} steps')
        
        # Clear any existing timers and reset the list
        self.active_timers = []
        
        # Check control event status
        print(f'Control event is set: {self.control_event.is_set()}')
        
        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
            if not self.control_event.is_set():
                print("Experiment stopped prematurely.")
                # Reset step to empty string when the experiment stops
                self.recorder.current_exp_step = ""  
                break  # Exit if the experiment is stopped
            
            delay_seconds = int(row['delay_seconds'])
            action = row['action']
            step_name = row['step_name']
            
            print(f'Processing step: {step_name}, action: {action}, delay: {delay_seconds}s')

            # Convert the JSON string in the 'parameters' column to a Python dictionary
            parameters = json.loads(row['parameters'].replace("'", '"'))
            
            # Add the step name to the parameters
            parameters['step_name'] = step_name

            # Perform different actions based on the function name
            if action == 'apply_odor':
                print(f"Setting Timer for apply_odor")

                # Wrapper function to update the step name at the correct time
                def apply_odor_with_step(**kwargs):
                    self.recorder.current_exp_step = kwargs['step_name']  # Update the step at the right time
                    apply_odor(**kwargs)  # Call the actual function
                    self.recorder.current_exp_step = ""  # Reset step after the action completes

                parameters['loaded_odor_channel_mapping'] = self.odor_map  # Set mapping of odor valves
                timer = threading.Timer(delay_seconds, apply_odor_with_step, kwargs=parameters)
                timer.start()
                self.active_timers.append(timer)
                print(f"Timer created and started for {step_name} (delay: {delay_seconds}s)")

            elif action == 'apply_shock_to_all':
                print(f"Setting Timer for apply_shock_to_all")

                # Wrapper function to update the step name at the correct time
                def apply_shock_to_all_with_step(**kwargs):
                    self.recorder.current_exp_step = kwargs['step_name']  # Update the step at the right time
                    apply_shock_to_all(**kwargs)  # Call the actual function
                    self.recorder.current_exp_step = ""  # Reset step after the action completes

                parameters['loaded_shock_channel_mapping'] = self.chamber_map  # Set mapping of chambers
                timer = threading.Timer(delay_seconds, apply_shock_to_all_with_step, kwargs=parameters)
                timer.start()
                self.active_timers.append(timer)

            elif action == 'apply_operant_shock':
                print(f"Setting Timer for apply_operant_shock")

                # Wrapper function to update the step name at the correct time
                def apply_operant_shock_with_step(**kwargs):
                    self.recorder.current_exp_step = kwargs['step_name']  # Update the step at the right time
                    apply_operant_shock(**kwargs)  # Call the actual function
                    self.recorder.current_exp_step = ""  # Reset step after the action completes

                parameters['loaded_shock_channel_mapping'] = self.chamber_map  # Set mapping of chambers

                if 'threshold' not in parameters:
                    print(f"Error: 'threshold' parameter missing for apply_operant_shock action. Skipping this step.")
                else:
                    timer = threading.Timer(delay_seconds, apply_operant_shock_with_step, kwargs=parameters)
                    timer.start()
                    self.active_timers.append(timer)

            else:
                print(f"Unknown action: {action}. Skipping this step.")
        
        print(f'All timers created. Total active timers: {len(self.active_timers)}')
        
        # Wait for all timers to complete
        print('Waiting for all timers to complete...')
        for timer in self.active_timers:
            timer.join()
        print('All timers completed')

        # Signal that the experiment is finished
        self.finished_event.set()
        print('Finished event set')
        # Reset step to empty string after the entire experiment finishes
        self.recorder.current_exp_step = ""
        
        # Call cleanup function to reset global state
        print('Calling cleanup_experiment()')
        cleanup_experiment()



    def stop_experiment(self):
        """
        Stops the experiment by clearing the control event, cancelling all active timers,
        and immediately deactivating all channels for graceful stopping.

        This method is intended to safely stop the experiment even if it's in the middle of execution.
        """
        print("Stopping experiment...")
        
        # Clear the control event to prevent new timers from starting
        self.control_event.clear()
        
        # Cancel all active timers immediately
        for timer in self.active_timers:
            if timer.is_alive():
                timer.cancel()
                print(f"Cancelled timer: {timer}")
        
        # Clear the timers list
        self.active_timers = []
        
        # Immediately deactivate all channels for safety
        self._cleanup_channels()
        
        # Signal that the experiment is finished
        self.finished_event.set()
        print("Experiment stopped gracefully.")
    
    def _cleanup_channels(self):
        """
        Immediately deactivate all functional channels for safety.
        This is called when stopping the experiment to ensure hardware is in a safe state.
        """
        try:
            # Load channel mappings
            odor_map = load_odor_channel_mapping()
            chamber_map = load_shock_channel_mapping()
            
            # Deactivate all odor channels
            if odor_map:
                device_name = "Dev1"
                for key, (port, channel) in odor_map.items():
                    control_channels(device_name, port, [channel], activate=False)
                print("Deactivated all odor channels")
            
            # Deactivate all shock channels
            if chamber_map:
                device_name = "Dev1"
                for chamber, (port, channel) in chamber_map.items():
                    control_channels(device_name, port, [channel], activate=False)
                print("Deactivated all shock channels")
                
        except Exception as e:
            print(f"Error during channel cleanup: {e}")


@experiment_dashboard_bp.route('/start_experiment', methods=['POST'])
def start_experiment():
    """
    This is the function that will be called in order to start the experiment.
    This function will call other functions to load the selected protocol and start its executation. It will also initiate the ExperimentRecorder object to record the experiment variables
    """
    result = start_experiment_backend()

    return jsonify({'status': 'success', 'result': result})

@experiment_dashboard_bp.route('/stop_experiment', methods=['POST'])
def stop_experiment():
    """
    Stop the currently running experiment gracefully.
    """
    global current_experiment_runner, current_experiment_recorder, experiment_is_running, current_experiment_output_path
    
    if not experiment_is_running or current_experiment_runner is None:
        return jsonify({
            'status': 'error',
            'message': 'No experiment is currently running'
        }), 400
    
    try:
        # Stop the experiment using the enhanced graceful stop mechanism
        current_experiment_runner.stop_experiment()
        print("Experiment stopped gracefully")
        
        # Save experiment data to CSV file before cleanup
        if current_experiment_recorder and current_experiment_output_path:
            csv_file_path = os.path.join(current_experiment_output_path, 'fly_loc.csv')
            current_experiment_recorder.save_to_csv(csv_file_path)
            print(f'Experiment data saved to: {csv_file_path}')
        
        # Reset global variables
        experiment_is_running = False
        current_experiment_runner = None
        current_experiment_recorder = None
        current_experiment_output_path = None
        
        return jsonify({
            'status': 'success',
            'message': 'Experiment stopped successfully'
        })
        
    except Exception as e:
        print(f"Error stopping experiment: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error stopping experiment: {str(e)}'
        }), 500

@experiment_dashboard_bp.route('/experiment_status', methods=['GET'])
def experiment_status():
    """
    Get the current experiment status.
    """
    global experiment_is_running
    
    print(f'Experiment status requested: is_running={experiment_is_running}')
    return jsonify({
        'is_running': experiment_is_running
    })

def cleanup_experiment():
    """
    Clean up experiment state when experiment finishes naturally.
    This is called by the experiment thread when it completes.
"""
    global current_experiment_runner, current_experiment_recorder, experiment_is_running, current_experiment_output_path
    
    print(f'Experiment finished - cleaning up. Current state: experiment_is_running={experiment_is_running}')
    
    # Stop recording and save data to CSV
    if current_experiment_recorder:
        current_experiment_recorder.stop_recording()
        print('Recording stopped')
        
        # Save experiment data to CSV file
        if current_experiment_output_path:
            csv_file_path = os.path.join(current_experiment_output_path, 'fly_loc.csv')
            current_experiment_recorder.save_to_csv(csv_file_path)
            print(f'Experiment data saved to: {csv_file_path}')
    
    # Reset global state
    experiment_is_running = False
    current_experiment_runner = None
    current_experiment_recorder = None
    current_experiment_output_path = None
    print('Global state reset - experiment_is_running=False')

def start_experiment_backend():
    """
    Function to start the experiment, save metadata, and store experiment data.
    """

    # --- 1. Extract profile and experiment metadata from session ---
    selected_profile = session.get('selected_profile', {})
    experiment_metadata = session.get('experiment-metadata', {})

    # --- 2. Get the username and protocol file name from metadata ---
    profile_name = selected_profile.get('name', 'default_profile')  # User's folder name
    protocol_name = experiment_metadata.get('protocol', 'default_protocol.csv')  # Selected protocol file name

    # --- 3. Build the full path to the selected protocol file ---
    protocol_folder = os.path.join('protocols', profile_name)
    protocol_file = os.path.join(protocol_folder, protocol_name)

    # --- 4. Initialize recorder and runner ---
    global current_experiment_runner, current_experiment_recorder, experiment_is_running, current_experiment_output_path
    
    # Check if experiment is already running
    if experiment_is_running:
        return 'Experiment is already running'
    
    recorder = ExperimentRecorder()
    runner = ExperimentRunner(recorder)
    
    # Store global references
    current_experiment_recorder = recorder
    current_experiment_runner = runner

    # --- 5. Extract other metadata for output path ---
    project_name = experiment_metadata.get('projectName', 'default_project')
    experiment_name = experiment_metadata.get('experimentName', 'default_experiment').strip()
    date_time = experiment_metadata.get('dateTime', '00.00.00')

    try:
        # Try to parse the date
        formatted_date = datetime.strptime(date_time.split(' ')[0], "%d.%m.%y").strftime("%d.%m.%Y")
    except ValueError:
        formatted_date = '00.00.0000'

    # --- 6. Build the output path where results will be saved ---
    results_base_path = r"C:\Users\user\Documents\Results"
    base_path = os.path.join(results_base_path, profile_name, project_name, experiment_name, formatted_date)

    # --- 7. Find latest trial number ---
    trial_number = 1
    while os.path.exists(f"{base_path}\\trial_{trial_number}"):
        trial_number += 1

    output_path = os.path.join(base_path, f"trial_{trial_number}")
    os.makedirs(output_path, exist_ok=True)
    
    # Store the output path globally for later use
    current_experiment_output_path = output_path

    # --- 8. Save metadata as JSON file in the output folder ---
    metadata_file_path = os.path.join(output_path, 'experiment_metadata.json')
    with open(metadata_file_path, 'w') as metadata_file:
        json.dump(experiment_metadata, metadata_file, indent=4)

    # --- 9. Load the correct protocol file into the runner ---
    runner.load_protocol_file(protocol_file)

    # --- 10. Start recording experiment ---
    recorder.start_recording()
    recorder.record_video_to_file(output_path)

    # Start the experiment asynchronously
    runner.control_event.set()
    experiment_is_running = True
    print(f'Experiment state set to running: {experiment_is_running}')

    if not runner.run_thread.is_alive():
        runner.run_thread.start()
        print('Experiment thread started')
    else:
        print('Experiment thread was already alive')

    # Return immediately - experiment runs in background
    print('Experiment started successfully')
    return 'Experiment started successfully'


from datetime import datetime

import pandas as pd
import threading
import time
from datetime import datetime

class ExperimentRecorder:
    def __init__(self):
        """
        Initialize the ExperimentRecorder class.

        This class is responsible for recording experiment variables and video, controlled by threading events.
        """
        # Initialize a DataFrame with standard columns and one column per chamber
        # Get the list of odor column names
        odor_columns = [name for name in OdorColumn.list_odor_columns()]

        # Construct the full list of columns
        columns = (
            ['Timestamp', 'Light', 'experiment_step'] +
            [f'chamber_{i}_loc' for i in range(1, 21)] +
            [f'chamber_{i}_shock' for i in range(1, 21)] +
            [f'{column}_status' for column in odor_columns]
        )
        self.df = pd.DataFrame(columns=columns)
        self.record_event = threading.Event()  # Event to control recording
        self.recording_thread = threading.Thread(target=self.record_state)
        self.recording_thread.daemon = True  # Optional: makes the thread exit when the main program does
        self.flies_above_threshold = []
        self.current_exp_step = ""

    def start_recording(self):
        """
        Starts the recording process by setting the record event and starting the recording thread if not already started.
        """
        self.record_event.set()
        if not self.recording_thread.is_alive():
            self.recording_thread.start()

    def stop_recording(self):
        """
        Stops the recording process by clearing the record event.
        """
        self.record_event.clear()
        print("Recording stopped.")

    def record_video_to_file(self, video_file_path):
        """
        This function starts the video recording process in a separate thread.
        
        Args:
            video_file_path (str): The full path where the video will be saved.
        """
        video_thread = threading.Thread(target=video_recorder, args=(self.record_event, 30, video_file_path))
        video_thread.daemon = True  # Optional: makes the thread exit when the main program does
        video_thread.start()


    def record_state(self):
        """
        Records the state of the experiment variables and fly locations into the dataframe at 0.1-second intervals.
        """
        while self.record_event.is_set():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Assuming light state is retrieved here
            light_state = self.get_light_state()
            experiment_step = self.current_exp_step

            # Get the last known fly locations from the Chamber class
            apply_detection_to_chambers() # Locates fly and updates their last known location
            fly_locations = Chamber.export_flies_loc() # Exports last known location to fly_locations variable
            flies_shock_status = Chamber.export_flies_shock_status() # Extracts whether the flies are getting shocked or not
            odor_status = OdorColumn.export_odor_status()

            # Prepare a dictionary to store the row data
            row_data = {
                'Timestamp': timestamp,
                'Light': light_state,
                'experiment_step': experiment_step
            }

            # Add each chamber's fly location to the row data
            for chamber_id, loc in fly_locations.items():
                row_data[f'{chamber_id}_loc'] = loc

            # Add each chamber's fly shock status to the row data
            for chamber_id, shock in flies_shock_status.items():
                row_data[f'{chamber_id}_shock'] = int(shock)

            # Add each odor column's status to the row data
            for odor_column_id, status in odor_status.items():
                row_data[f'{odor_column_id}_status'] = int(status)  # Make sure the key matches the DataFrame columns

            # Convert the row data to a DataFrame and concatenate with the main DataFrame
            row_df = pd.DataFrame([row_data])
            self.df = pd.concat([self.df, row_df], ignore_index=True)

            time.sleep(0.1)  # Wait for 0.1 seconds before recording the next state

    def update_flies_above_threshold(self, threshold):
        """
        Updates the list of flies above the specified location threshold.

        Args:
            threshold (float): The location threshold for determining which flies to shock.
        """
        fly_locations = Chamber.export_flies_loc()  # Assuming it returns a dictionary of {chamber_id: loc}
        self.flies_above_threshold = [chamber_id for chamber_id, loc in fly_locations.items() if loc > threshold]

    def get_light_state(self):
        # Placeholder for getting the current light state
        return "light_state"

    def get_shock_state(self):
        # Placeholder for getting the current shock state
        return "shock_state"

    def get_valve_state(self):
        # Placeholder for getting the current valve state
        return "valve_state"
    
    def save_to_csv(self, filepath):
        """
        Saves the recorded experiment data to a CSV file.
        
        Args:
            filepath (str): The full path where the CSV file should be saved.
        """
        try:
            self.df.to_csv(filepath, index=False)
            print(f"Experiment data saved to: {filepath}")
        except Exception as e:
            print(f"Error saving experiment data to CSV: {e}")


import nidaqmx
from nidaqmx.system import System
from nidaqmx.constants import LineGrouping

# Function to control USB-6509 channels
def control_channels(device_name, port, channels, activate):
    """
    This function will deliver voltage (high or low) to NI instruments.
    This is meant to operate both the valves that control the odors and the electrical components that deliver shock to the flies.

    Args:
        device_name (str): The name of the NI device as configured within the National Instruments app.
        port (int): The desired port we want to access.
        channels (int or list): List of the desired channels we want to activate or a single channel.
        activate (bool): The state we want to apply. True for applying voltage (activate), False for stopping (deactivate).

    Returns:
        Nothing.
    """
    # Ensure channels is always a list
    if isinstance(channels, int):
        channels = [channels]

    # Create a new NI Task
    with nidaqmx.Task() as task:
        # Add each channel to the task
        for channel in channels:
            chan = f"{device_name}/port{port}/line{channel}"
            task.do_channels.add_do_chan(chan, line_grouping=LineGrouping.CHAN_PER_LINE)

        # Set the value based on the activate flag
        value = True if activate else False

        # Write the value to all channels
        task.write([value] * len(channels), auto_start=True)

        # Print the action
        action = "ACTIVATE" if activate else "DEACTIVATE"
        print(f"{action}D channels {channels} on port {port} of {device_name}")

def load_shock_channel_mapping():
    """
    This function will initialize the channel mapping.
    If a file exists, it will load it from the channel_mapping.json file and return it.
    If a file does not exist, it will create a new channel_mapping.json file, and save the default values to it.

    Args:
        Nothing.
    
    Returns:
        Dictionary of the mapping of board and channel of each chamber
    """
    # These values are based on mapping the channel and boards hardware connectivity.
    default_chambers_board_and_channel = {
        "chamber_1": [7, 4],
        "chamber_2": [7, 5],
        "chamber_3": [7, 6],
        "chamber_4": [7, 7],
        "chamber_5": [8, 0],
        "chamber_6": [8, 1],
        "chamber_7": [8, 2],
        "chamber_8": [8, 3],
        "chamber_9": [8, 4],
        "chamber_10": [8, 5],
        "chamber_11": [6, 0],
        "chamber_12": [6, 1],
        "chamber_13": [6, 2],
        "chamber_14": [6, 3],
        "chamber_15": [6, 4],
        "chamber_16": [6, 5],
        "chamber_17": [6, 6],
        "chamber_18": [6, 7],
        "chamber_19": [7, 0],
        "chamber_20": [7, 1],
    }
    
    # File path for the channel mapping file
    file_path = 'configuration_files/shock_channel_mapping.json'
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing channel mapping from the file
        with open(file_path, 'r') as f:
            shock_channel_mapping = json.load(f)
    else:
        # Create a new file with the default channel mapping
        with open(file_path, 'w') as f:
            json.dump(default_chambers_board_and_channel, f, indent=4)
        # Return the default channel mapping
        shock_channel_mapping = default_chambers_board_and_channel
    
    return shock_channel_mapping

def load_odor_channel_mapping():
    """
    This function will initialize the channel mapping.
    If a file exists, it will load it from the channel_mapping.json file and return it.
    If a file does not exist, it will create a new channel_mapping.json file, and save the default values to it.
    
    Args:
        Nothing.
    
    Returns:
        Dictionary of the mapping of board and channel of odor valves
    """
    # These values are based on mapping the channel and boards hardware connectivity.
    default_chambers_board_and_channel = {
        "right_mch_1": [0, 0],
        "right_mch_2": [0, 4],
        "right_oct_1": [0, 1],
        "right_oct_2": [0, 5],
        "right_moil_1": [0, 6],
        "right_moil_2": [0, 7],
        "right_odor_master": [0, 2],
        "left_mch_1": [2, 2],
        "left_mch_2": [2, 6],
        "left_oct_1": [2, 3],
        "left_oct_2": [2, 7],
        "left_moil_1": [1, 4],
        "left_moil_2": [1, 5],
        "left_odor_master": [2, 4],
    }
    
    # File path for the channel mapping file
    file_path = 'configuration_files/odor_channel_mapping.json'
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the existing channel mapping from the file
        with open(file_path, 'r') as f:
            odor_channel_mapping = json.load(f)
    else:
        # Create a new file with the default channel mapping
        with open(file_path, 'w') as f:
            json.dump(default_chambers_board_and_channel, f, indent=4)
        # Return the default channel mapping
        odor_channel_mapping = default_chambers_board_and_channel
    
    return odor_channel_mapping


def apply_shock_to_all(loaded_shock_channel_mapping, duration, step_name=None):
    """
    This function will deliver shock to all chambers by activating channels by port.
    It alternates between delivering a shock for 1.25 seconds and then turning it off for 3.75 seconds
    until the overall duration is completed.

    Args:
        1. duration - specify the total duration (in seconds) of the electrical shock to the chambers.

    Returns:
        Nothing.
    """
    device_name = "Dev1"  # Replace with your actual device name
    print('Applying shock to all')

    # Group channels by port
    shock_channel_mapping = {}
    for chamber, (port, channel) in loaded_shock_channel_mapping.items():
        if port not in shock_channel_mapping:
            shock_channel_mapping[port] = []
        shock_channel_mapping[port].append(channel)

    shock_on_duration = 1.25  # Duration of shock in seconds
    shock_off_duration = 3.75  # Duration of rest in seconds
    cycle_duration = shock_on_duration + shock_off_duration

    # Calculate the number of complete cycles and the remaining time
    full_cycles = int(duration // cycle_duration)
    remaining_time = duration % cycle_duration

    # Activate and deactivate channels in cycles
    for _ in range(full_cycles):
        # Activate channels by port
        for port, channels in shock_channel_mapping.items():
            control_channels(device_name, port, channels, activate=True)
        Chamber.shock_all_on()
        time.sleep(shock_on_duration)
        
        # Deactivate channels by port
        for port, channels in shock_channel_mapping.items():
            control_channels(device_name, port, channels, activate=False)
        Chamber.shock_all_off()
        time.sleep(shock_off_duration)

    # Handle any remaining time (less than a full cycle)
    if remaining_time > 0:
        # Activate channels for the remaining time if it's less than or equal to the shock on duration
        if remaining_time <= shock_on_duration:
            for port, channels in shock_channel_mapping.items():
                control_channels(device_name, port, channels, activate=True)
            Chamber.shock_all_on()
            time.sleep(remaining_time)  # Apply shock for the remaining time
            # Deactivate the channels after the remaining time
            for port, channels in shock_channel_mapping.items():
                control_channels(device_name, port, channels, activate=False)
            Chamber.shock_all_off()
        else:
            # If remaining time is more than the shock_on_duration, apply one full shock
            for port, channels in shock_channel_mapping.items():
                control_channels(device_name, port, channels, activate=True)
            Chamber.shock_all_on()
            time.sleep(shock_on_duration)
            # Deactivate channels after the shock on period
            for port, channels in shock_channel_mapping.items():
                control_channels(device_name, port, channels, activate=False)
            Chamber.shock_all_off()
            # Then wait for the remaining off period
            time.sleep(remaining_time - shock_on_duration)

    # Print completion message
    print("Shock delivered to all chambers")


import threading
import time

def apply_operant_shock(loaded_shock_channel_mapping, threshold, duration, step_name=None):
    """
    Applies shocks to flies based on their location relative to a threshold.
    Each fly is monitored in a separate thread. Shocks stop after the specified duration.
    """
    device_name = "Dev1"
    print(f"Applying individual conditional shocks to flies above location {threshold}")

    # Set the shock cycle parameters
    shock_on_duration = 0.2
    shock_off_duration = 1.25
    check_interval = 0.05  # Interval to check for stop_event during sleep

    # Dictionary to track active shock states (threads) for each fly
    active_shocks = {}

    # Create an event to signal when the experiment duration ends
    stop_event = threading.Event()

    # Helper function to start shock cycle for a specific fly
    def monitor_fly(chamber_id, port, channels):
        if isinstance(channels, int):
            channels = [channels]
        
        while not stop_event.is_set():
            fly_locations = Chamber.export_flies_loc()
            loc = fly_locations.get(chamber_id)

            if loc is None:
                continue

            if loc < threshold:
                # Shock cycle
                control_channels(device_name, port, channels, activate=True)
                Chamber.set_shock_status(chamber_id, True)

                elapsed = 0
                while elapsed < shock_on_duration and not stop_event.is_set():
                    time.sleep(min(check_interval, shock_on_duration - elapsed))
                    elapsed += check_interval

                control_channels(device_name, port, channels, activate=False)
                Chamber.set_shock_status(chamber_id, False)

                elapsed = 0
                while elapsed < shock_off_duration and not stop_event.is_set():
                    time.sleep(min(check_interval, shock_off_duration - elapsed))
                    elapsed += check_interval
            else:
                # Ensure shock is off if the fly has moved below the threshold
                control_channels(device_name, port, channels, activate=False)
                Chamber.set_shock_status(chamber_id, False)
            
            time.sleep(check_interval)

        # Ensure shocks are turned off when stopping the thread
        control_channels(device_name, port, channels, activate=False)
        Chamber.set_shock_status(chamber_id, False)

    # Start a thread for each fly
    for chamber_id, loc in Chamber.export_flies_loc().items():
        port, channels = loaded_shock_channel_mapping.get(chamber_id, (None, None))
        if port is None:
            continue

        if chamber_id not in active_shocks:
            shock_thread = threading.Thread(target=monitor_fly, args=(chamber_id, port, channels))
            shock_thread.daemon = True  # Make threads daemons so they don't block exit
            active_shocks[chamber_id] = shock_thread
            shock_thread.start()

    # Let the threads run for the specified duration
    time.sleep(duration)

    # Signal all threads to stop
    stop_event.set()

    # Wait for all threads to finish
    for thread in active_shocks.values():
        thread.join()

    # Ensure all shocks are turned off at the end of the experiment
    for chamber_id in active_shocks.keys():
        port, channels = loaded_shock_channel_mapping.get(chamber_id, (None, None))
        control_channels(device_name, port, channels, activate=False)
        Chamber.set_shock_status(chamber_id, False)

    print("Shocks have been forcefully terminated based on fly locations.")

    
def apply_odor(side, odor_type, duration, loaded_odor_channel_mapping, step_name=None):
    device_name = "Dev1"  # Replace with your actual device name
    print(f'Applying {odor_type} on side {side} for {duration}')
    
    # Group channels by port
    odor_channel_mapping = {}

    # Determine which odor columns to activate
    columns_to_activate = []

    if side in ["left", "both"]:
        columns_to_activate.append(f'{odor_type}_left')
        for key in [f"left_{odor_type}_1", f"left_{odor_type}_2"]:
            port, channel = loaded_odor_channel_mapping[key]
            if port not in odor_channel_mapping:
                odor_channel_mapping[port] = []
            odor_channel_mapping[port].append(channel)
        if odor_type != "moil":
            port, channel = loaded_odor_channel_mapping["left_odor_master"]
            if port not in odor_channel_mapping:
                odor_channel_mapping[port] = []
            odor_channel_mapping[port].append(channel)
    
    if side in ["right", "both"]:
        columns_to_activate.append(f'{odor_type}_right')
        for key in [f"right_{odor_type}_1", f"right_{odor_type}_2"]:
            port, channel = loaded_odor_channel_mapping[key]
            if port not in odor_channel_mapping:
                odor_channel_mapping[port] = []
            odor_channel_mapping[port].append(channel)
        if odor_type != "moil":
            port, channel = loaded_odor_channel_mapping["right_odor_master"]
            if port not in odor_channel_mapping:
                odor_channel_mapping[port] = []
            odor_channel_mapping[port].append(channel)
    
    # Set the odor delivery status to True for the relevant columns
    for column_name in columns_to_activate:
        odor_column = getattr(OdorColumn, column_name, None)
        if odor_column:
            odor_column.odor_delivery_status = True

    # Activate channels by port with 0.3 second delays between each port
    port_keys = list(odor_channel_mapping.keys())
    for i, port in enumerate(port_keys):
        channels = odor_channel_mapping[port]
        control_channels(device_name, port, channels, activate=True)
        print(f'Activated port {port} channels: {channels}')
        
        # Add 0.3 second delay between port activations (except for the last port)
        if i < len(port_keys) - 1:
            time.sleep(0.1)
    
    time.sleep(0.5)
    
    # Activate channels by port for a second time (to make sure it is opened) with delays
    for i, port in enumerate(port_keys):
        channels = odor_channel_mapping[port]
        control_channels(device_name, port, channels, activate=True)
        
        # Add 0.3 second delay between port activations (except for the last port)
        if i < len(port_keys) - 1:
            time.sleep(0.1)

    # Keep the odor active for the specified duration
    time.sleep(duration)
    
    # Deactivate channels by port with 0.3 second delays
    for i, port in enumerate(port_keys):
        channels = odor_channel_mapping[port]
        control_channels(device_name, port, channels, activate=False)
        print(f'Deactivated port {port} channels: {channels}')
        
        # Add 0.3 second delay between port deactivations (except for the last port)
        if i < len(port_keys) - 1:
            time.sleep(0.3)

    # Set the odor delivery status to False for the relevant columns after the odor application
    for column_name in columns_to_activate:
        odor_column = getattr(OdorColumn, column_name, None)
        if odor_column:
            odor_column.odor_delivery_status = False

    print(f'{odor_type.capitalize()} odor applied on side {side} for {duration} seconds')


def control_light():
    # Implement light control logic
    pass




"""
The following code will handle anything related to the anylsis of the fly location within the chambers
"""

@experiment_dashboard_bp.route('/define_chambers')
def define_chambers():
    global chambers # Load the current configuration
    return jsonify(chambers), 200  # Return the current chamber configuration as JSON

@experiment_dashboard_bp.route('/save_chambers', methods=['POST'])
def save_chambers():
    output_file = 'configuration_files/chambers_configuration.json'
    chambers = {}
    for i in range(1, 21):
        x = request.form.get(f'x{i}')
        y = request.form.get(f'y{i}')
        width = request.form.get(f'width{i}')
        height = request.form.get(f'height{i}')
        chambers[f'chamber_{i}'] = {
            'x': int(x),
            'y': int(y),
            'width': int(width),
            'height': int(height)
        }
    
    with open(output_file, 'w') as json_file:
        json.dump(chambers, json_file, indent=4)
    
    return jsonify({"message": "Chamber configuration saved successfully."}), 200

import cv2
import numpy as np

# Initialize global variables for the background model
background_model = None
alpha = 0.15  # Weight for the running average (adjust as needed)

def update_background_model(frame):
    """
    Updates the background model using a running average.
    """
    global background_model

    # Initialize the background model if it's the first frame
    if background_model is None:
        background_model = frame.copy().astype("float")
        return background_model

    # Update the background model with the current frame
    cv2.accumulateWeighted(frame, background_model, alpha)

    # Return the updated background model
    return background_model

def fly_detection(frame, mask):
    """
    This function processes the given frame for fly detection by identifying significant changes in the image
    within the specified mask area, and returns the processed frame with detected flies highlighted.

    Args:
        frame (ndarray): The current video frame.
        mask (ndarray): The mask specifying the area to focus on.

    Returns:
        ndarray: The processed frame with detected flies highlighted.
    """
    # Convert frame to grayscale (assuming background subtraction works best in single channel)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Apply the mask to the blurred frame
    masked_frame = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
    
    # Update the background model with the masked frame
    background_model = update_background_model(masked_frame)
    
    # Convert the background model to uint8 before subtraction
    background_model_uint8 = cv2.convertScaleAbs(background_model)
    
    # Subtract the background from the masked frame
    diff_frame = cv2.absdiff(background_model_uint8, masked_frame)
    
    # Apply a threshold to get a binary image of motion areas
    _, thresh_frame = cv2.threshold(diff_frame, 5, 255, cv2.THRESH_BINARY)  # Adjust threshold value as needed
    
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    processed_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)
    processed_frame = cv2.morphologyEx(processed_frame, cv2.MORPH_OPEN, kernel)
    
    # Find contours (which will be potential flies)
    contours, _ = cv2.findContours(processed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize a list to hold valid contours
    valid_contours = []

    # Draw bounding boxes around detected flies and filter contours
    for contour in contours:
        if 1 < cv2.contourArea(contour) < 200:  # Filter out small contours/noise, adjust value as needed
            x, y, w, h = cv2.boundingRect(contour)
            # Draw rectangle on the frame: (frame, top-left corner, bottom-right corner, color, thickness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add the valid contour to the list
            valid_contours.append(contour)

    # Return the processed frame and only the valid contours
    return processed_frame, valid_contours


def apply_detection_to_chambers():
    """
    This function retrieves the current frame, reads the chamber configurations, creates masks for each chamber,
    applies the fly_detection algorithm to each masked area, and returns the processed frame.

    Returns:
        ndarray: The processed frame with detected flies highlighted within each chamber.
    """
    # Retrieve the current frame from the video feed
    frame = capture_frame()
    
    if frame is None:
        return None

    # Load chamber configurations
    global chambers
    
    # Initialize an empty mask with the same dimensions as the frame
    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Create a mask for each chamber
    for chamber_id, config in chambers.items():
        x, y, width, height = config['x'], config['y'], config['width'], config['height']
        # Create a mask for the current chamber area
        cv2.rectangle(mask, (x, y), (x + width, y + height), 255, -1)

    # Apply fly detection to the masked areas
    processed_frame, valid_contours = fly_detection(frame, mask)
    current_loc = assign_contours_to_chambers(valid_contours, chambers)
    Chamber.update_flies_loc(current_loc)
    return processed_frame

def display_live_feed():
    """
    This function will display the processed live feed video with fly detection
    in an OpenCV window.
    """
    while True:
        # Get the processed frame with fly detection applied to the chamber masks
        processed_frame = apply_detection_to_chambers()
        
        if processed_frame is not None:
            # Display the processed frame in an OpenCV window
            cv2.imshow('Live Feed with Fly Detection', processed_frame)
            
            # Break the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("No frame captured.")

    # When everything is done, release the window
    cv2.destroyAllWindows()


def assign_contours_to_chambers(contours, chambers):
    """
    Assigns contours (flies) to their respective chambers and calculates the fly's location within each chamber's x-axis.

    Args:
        contours (list): A list of detected contours (flies).
        chambers (dict): A dictionary containing the chamber configuration, with each chamber's x, y, width, and height.

    Returns:
        dict: A dictionary where each key is a chamber ID and the value is the normalized x-axis location of the fly within that chamber.
              If no fly is detected in a chamber, the value is NaN.
    """
    chamber_fly_locations = {}

    for chamber_id, config in chambers.items():
        x, y, width, height = config['x'], config['y'], config['width'], config['height']
        chamber_center_x = x + width / 2
        
        fly_x_positions = []

        for contour in contours:
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M['m00'] != 0:
                contour_center_x = int(M['m10'] / M['m00'])
                contour_center_y = int(M['m01'] / M['m00'])

                # Check if the contour is within the current chamber's x and y bounds
                if x <= contour_center_x <= x + width and y <= contour_center_y <= y + height:
                    # Normalize the x position within the chamber
                    normalized_x = ((contour_center_x - x) / width) * 200 - 100
                    fly_x_positions.append(normalized_x)
        
        # If one or more flies are detected in the chamber, return the average position
        if fly_x_positions:
            chamber_fly_locations[chamber_id] = np.mean(fly_x_positions)

    return chamber_fly_locations


# Global valve state tracker to maintain the current state of all odor valves
# This allows us to properly handle odor master valves based on overall side state
valve_states = {
    'right_mch': False,
    'left_mch': False,
    'right_oct': False,
    'left_oct': False,
    'right_moil': False,
    'left_moil': False
}

def get_odor_master_state(side):
    """
    Determine if the odor master valve for a given side should be active.
    The odor master should be active if at least one MCH or OCT valve on that side is active.
    """
    if side == 'right':
        return valve_states['right_mch'] or valve_states['right_oct']
    elif side == 'left':
        return valve_states['left_mch'] or valve_states['left_oct']
    return False

# Manual Valve Control Routes
@experiment_dashboard_bp.route('/manual_valve_control', methods=['GET'])
def manual_valve_control():
    """Route to get the manual valve control interface data"""
    try:
        # Load odor channel mapping
        odor_mapping = load_odor_channel_mapping()
        
        # Create a list of valves with user-friendly names
        valves = []
        for valve_key, (port, channel) in odor_mapping.items():
            # Convert valve key to user-friendly name
            friendly_name = valve_key.replace('_', ' ').title()
            friendly_name = friendly_name.replace('Mch', 'MCH').replace('Oct', 'Octanol').replace('Moil', 'Mineral Oil')
            
            valves.append({
                'key': valve_key,
                'name': friendly_name,
                'port': port,
                'channel': channel,
                'active': False  # Default state
            })
        
        return jsonify({
            'status': 'success',
            'valves': valves
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@experiment_dashboard_bp.route('/activate_valve', methods=['POST'])
def activate_valve():
    """Route to activate a specific valve"""
    try:
        data = request.get_json()
        valve_key = data.get('valve_key')
        activate = data.get('activate', True)
        
        if not valve_key:
            return jsonify({'status': 'error', 'message': 'Valve key is required'}), 400
        
        # Load odor channel mapping
        odor_mapping = load_odor_channel_mapping()
        
        if valve_key not in odor_mapping:
            return jsonify({'status': 'error', 'message': 'Invalid valve key'}), 400
        
        port, channel = odor_mapping[valve_key]
        device_name = "Dev1"
        
        # Control the specific valve
        control_channels(device_name, port, [channel], activate)
        
        action = "activated" if activate else "deactivated"
        print(f"Manually {action} valve {valve_key} on port {port}, channel {channel}")
        
        return jsonify({
            'status': 'success',
            'message': f'Valve {valve_key} {action}',
            'valve_key': valve_key,
            'active': activate
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@experiment_dashboard_bp.route('/emergency_stop_valves', methods=['POST'])
def emergency_stop_valves():
    """Route to stop all valves (emergency stop)"""
    try:
        device_name = "Dev1"
        odor_mapping = load_odor_channel_mapping()
        
        # Group channels by port for efficient control
        port_channels = {}
        for valve_key, (port, channel) in odor_mapping.items():
            if port not in port_channels:
                port_channels[port] = []
            port_channels[port].append(channel)
        
        # Deactivate all valves by port
        for port, channels in port_channels.items():
            control_channels(device_name, port, channels, activate=False)
        
        # Reset all valve states to False
        for key in valve_states:
            valve_states[key] = False
        
        print("Emergency stop: All valves deactivated")
        
        return jsonify({
            'status': 'success',
            'message': 'All valves deactivated'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@experiment_dashboard_bp.route('/get_valve_states', methods=['GET'])
def get_valve_states():
    """
    Get the current state of all valve groups for the frontend to display.
    """
    return jsonify({
        "success": True,
        "valve_states": valve_states
    }), 200


@experiment_dashboard_bp.route('/control_valve_group', methods=['POST'])
def control_valve_group():
    """
    Control a group of valves (e.g., right_mch, left_oct) as a single operation.
    This mimics the behavior of apply_odor function by grouping channels by port
    and properly handling odor master valves.
    """
    try:
        data = request.get_json()
        valve_group = data.get('valve_group')
        activate = data.get('activate', False)
        
        if valve_group is None:
            return jsonify({"error": "valve_group parameter is required"}), 400
        
        # Load the odor channel mapping
        loaded_odor_channel_mapping = load_odor_channel_mapping()
        
        # Define the valve group mappings
        valve_group_mappings = {
            'right_mch': ['right_mch_1', 'right_mch_2'],
            'left_mch': ['left_mch_1', 'left_mch_2'],
            'right_oct': ['right_oct_1', 'right_oct_2'],
            'left_oct': ['left_oct_1', 'left_oct_2'],
            'right_moil': ['right_moil_1', 'right_moil_2'],
            'left_moil': ['left_moil_1', 'left_moil_2']
        }
        
        if valve_group not in valve_group_mappings:
            return jsonify({"error": f"Unknown valve group: {valve_group}"}), 400
        
        # Get the individual valve keys for this group
        valve_keys = valve_group_mappings[valve_group]
        
        # Group channels by port (like apply_odor function)
        port_channel_mapping = {}
        
        # Add the main valve channels
        for valve_key in valve_keys:
            if valve_key in loaded_odor_channel_mapping:
                port, channel = loaded_odor_channel_mapping[valve_key]
                if port not in port_channel_mapping:
                    port_channel_mapping[port] = []
                port_channel_mapping[port].append(channel)
        
        # Handle odor master valves with proper state management
        # Determine which side this group belongs to
        side = valve_group.split('_')[0]  # 'right' or 'left'
        odor_type = valve_group.split('_')[1]  # 'mch', 'oct', or 'moil'
        
        # Update the global valve state tracker for this group
        valve_states[valve_group] = activate
        
        # Handle odor master valve logic (only for MCH and OCT, not M.Oil)
        if odor_type != "moil":
            odor_master_key = f"{side}_odor_master"
            if odor_master_key in loaded_odor_channel_mapping:
                port, channel = loaded_odor_channel_mapping[odor_master_key]
                
                # Determine if odor master should be active based on overall side state
                should_activate_odor_master = get_odor_master_state(side)
                
                # Add odor master to the port mapping with the calculated state
                if port not in port_channel_mapping:
                    port_channel_mapping[port] = []
                port_channel_mapping[port].append(channel)
                
                # Update the odor master state in our tracker
                valve_states[odor_master_key] = should_activate_odor_master
                
                # If the odor master state changed, we need to control it separately
                # since it might have a different activation state than the main group
                if should_activate_odor_master != activate:
                    # Control the main group first
                    for port, channels in port_channel_mapping.items():
                        if odor_master_key not in [key for key in valve_keys if key in loaded_odor_channel_mapping]:
                            # Only control non-odor-master channels here
                            main_channels = [ch for ch in channels if ch != channel]
                            if main_channels:
                                control_channels(device_name, port, main_channels, activate)
                    
                    # Control the odor master separately with its calculated state
                    control_channels(device_name, port, [channel], should_activate_odor_master)
                    
                    # Return success with the channels that were controlled
                    return jsonify({
                        "success": True,
                        "valve_group": valve_group,
                        "activate": activate,
                        "odor_master_state": should_activate_odor_master,
                        "channels_controlled": port_channel_mapping
                    }), 200
        
        # Control all channels by port (single operation per port)
        device_name = "Dev1"
        for port, channels in port_channel_mapping.items():
            control_channels(device_name, port, channels, activate)

        # Return success with the channels that were controlled
        return jsonify({
            "success": True,
            "valve_group": valve_group,
            "activate": activate,
            "channels_controlled": port_channel_mapping
        }), 200
        
    except Exception as e:
        print(f"Error controlling valve group: {str(e)}")
        return jsonify({"error": str(e)}), 500



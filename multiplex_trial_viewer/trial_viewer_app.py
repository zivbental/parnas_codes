"""
Main application window for multiplex trial viewer.
PyQt5 application with video display, controls, and trace plots.
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, QProgressBar,
                             QFileDialog, QMessageBox, QMenuBar, QMenu, QAction,
                             QSplitter, QFrame, QGroupBox, QCheckBox, QSpinBox,
                             QDialog, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QFont
from typing import Optional

from video_processor import VideoProcessor
from trace_plotter import TracePlotter, TraceControlPanel
from chamber_config import ChamberConfig


class ChamberEditorDialog(QDialog):
    """
    Dialog for editing chamber coordinates.
    """
    
    def __init__(self, chamber_config, parent=None):
        super().__init__(parent)
        self.chamber_config = chamber_config
        self.setWindowTitle("Edit Chamber Configuration")
        self.setModal(True)
        self.resize(600, 500)
        
        self.init_ui()
        # Delay loading data to ensure table is fully initialized
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, self.load_chamber_data)
    
    def init_ui(self):
        """
        Initialize the dialog UI.
        """
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Edit chamber coordinates. Double-click cells to edit values.")
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Table widget
        self.table = QTableWidget(20, 5)
        self.table.setHorizontalHeaderLabels(["Chamber", "X", "Y", "Width", "Height"])
        
        # Set table properties
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        layout.addWidget(self.table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_changes)
        button_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def load_chamber_data(self):
        """
        Load chamber data into the table.
        """
        chambers = self.chamber_config.get_all_chambers()
        
        # Ensure we have exactly 20 chambers
        chamber_list = list(chambers.items())
        if len(chamber_list) < 20:
            # Add missing chambers with default values
            for i in range(len(chamber_list), 20):
                chamber_id = f"chamber_{i+1}"
                chambers[chamber_id] = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                chamber_list.append((chamber_id, chambers[chamber_id]))
        
        # Set table row count to match chambers
        self.table.setRowCount(len(chamber_list))
        
        # Debug: Check table dimensions
        # print(f"Table dimensions: {self.table.rowCount()} rows x {self.table.columnCount()} columns")
        
        # Clear existing items first
        self.table.clearContents()
        
        for i, (chamber_id, config) in enumerate(chamber_list):
            # Chamber ID (read-only)
            chamber_item = QTableWidgetItem(chamber_id)
            chamber_item.setFlags(chamber_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, chamber_item)
            
            # X coordinate
            x_item = QTableWidgetItem(str(config['x']))
            self.table.setItem(i, 1, x_item)
            
            # Y coordinate
            y_item = QTableWidgetItem(str(config['y']))
            self.table.setItem(i, 2, y_item)
            
            # Width
            width_item = QTableWidgetItem(str(config['width']))
            self.table.setItem(i, 3, width_item)
            
            # Height
            height_item = QTableWidgetItem(str(config['height']))
            self.table.setItem(i, 4, height_item)
            
            # Debug: Print first few chambers
            # if i < 3:
            #     print(f"Loaded chamber {i+1}: {chamber_id} = x:{config['x']}, y:{config['y']}, w:{config['width']}, h:{config['height']}")
        
        # Force table refresh
        self.table.viewport().update()
        print(f"Loaded {len(chamber_list)} chambers into table")
    
    def save_changes(self):
        """
        Save changes to chamber configuration.
        """
        try:
            chambers = self.chamber_config.get_all_chambers()
            
            # Debug: Print table state
            # print(f"Table has {self.table.rowCount()} rows and {self.table.columnCount()} columns")
            
            # Validate all inputs before saving
            for i in range(self.table.rowCount()):
                # Get chamber ID
                chamber_id_item = self.table.item(i, 0)
                if not chamber_id_item:
                    print(f"Warning: No chamber ID found for row {i}, skipping")
                    continue
                chamber_id = chamber_id_item.text()
                
                # Get values from table and validate
                try:
                    # Get text values with better null checking
                    x_item = self.table.item(i, 1)
                    y_item = self.table.item(i, 2)
                    width_item = self.table.item(i, 3)
                    height_item = self.table.item(i, 4)
                    
                    x_text = x_item.text() if x_item else ""
                    y_text = y_item.text() if y_item else ""
                    width_text = width_item.text() if width_item else ""
                    height_text = height_item.text() if height_item else ""
                    
                    # Debug: Print values for first few chambers
                    # if i < 3:
                    #     print(f"Validation - Chamber {i+1}: X='{x_text}', Y='{y_text}', W='{width_text}', H='{height_text}'")
                    
                    # Skip validation if all fields are empty (chamber not configured)
                    if not x_text.strip() and not y_text.strip() and not width_text.strip() and not height_text.strip():
                        print(f"Skipping {chamber_id} - no data")
                        continue
                    
                    # Check if any field is empty and provide specific error
                    if not x_text.strip():
                        QMessageBox.critical(self, "Error", f"Please fill in the X coordinate for {chamber_id}\n\nDebug: X='{x_text}'")
                        return
                    if not y_text.strip():
                        QMessageBox.critical(self, "Error", f"Please fill in the Y coordinate for {chamber_id}\n\nDebug: Y='{y_text}'")
                        return
                    if not width_text.strip():
                        QMessageBox.critical(self, "Error", f"Please fill in the Width for {chamber_id}\n\nDebug: Width='{width_text}'")
                        return
                    if not height_text.strip():
                        QMessageBox.critical(self, "Error", f"Please fill in the Height for {chamber_id}\n\nDebug: Height='{height_text}'")
                        return
                    
                    # Convert to integers
                    try:
                        x = int(x_text)
                        y = int(y_text)
                        width = int(width_text)
                        height = int(height_text)
                        
                        # Debug: Print converted values
                        # if i < 3:
                        #     print(f"  Converted: X={x}, Y={y}, W={width}, H={height}")
                    except ValueError as ve:
                        print(f"  Conversion error for {chamber_id}: {ve}")
                        raise
                    
                except (ValueError, AttributeError) as ve:
                    QMessageBox.critical(self, "Error", f"Invalid number format in {chamber_id}: {str(ve)}")
                    return
                
                # Validate values
                # print(f"  Validating {chamber_id}: x={x}, y={y}, width={width}, height={height}")
                if x < 0:
                    QMessageBox.critical(self, "Error", f"X coordinate must be >= 0 for {chamber_id}: x={x}")
                    return
                if y < 0:
                    QMessageBox.critical(self, "Error", f"Y coordinate must be >= 0 for {chamber_id}: y={y}")
                    return
                if width <= 0:
                    QMessageBox.critical(self, "Error", f"Width must be > 0 for {chamber_id}: width={width}")
                    return
                if height <= 0:
                    QMessageBox.critical(self, "Error", f"Height must be > 0 for {chamber_id}: height={height}")
                    return
                
                # Update chamber configuration
                if not self.chamber_config.set_chamber(chamber_id, x, y, width, height):
                    QMessageBox.critical(self, "Error", f"Failed to update {chamber_id}")
                    return
            
            # Save to file
            print(f"Attempting to save chamber config to: {self.chamber_config.config_file}")
            if self.chamber_config.save_config():
                QMessageBox.information(self, "Success", f"Chamber configuration saved successfully to:\n{self.chamber_config.config_file}")
                self.accept()
            else:
                QMessageBox.critical(self, "Error", f"Failed to save chamber configuration to:\n{self.chamber_config.config_file}\n\nPlease check file permissions and directory access.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error saving configuration: {str(e)}")
            import traceback
            traceback.print_exc()
    


class VideoProcessingThread(QThread):
    """
    Thread for video processing to avoid blocking the UI.
    """
    
    progress_updated = pyqtSignal(int, int)  # current_frame, total_frames
    processing_finished = pyqtSignal(bool)   # success
    
    def __init__(self, video_processor, video_path):
        super().__init__()
        self.video_processor = video_processor
        self.video_path = video_path
    
    def run(self):
        """
        Run video processing in background thread.
        """
        try:
            # Load video
            if not self.video_processor.load_video(self.video_path):
                self.processing_finished.emit(False)
                return
            
            # Process video with progress callback
            def progress_callback(current, total):
                self.progress_updated.emit(current, total)
            
            success = self.video_processor.process_video(progress_callback)
            self.processing_finished.emit(success)
            
        except Exception as e:
            print(f"Error in video processing thread: {e}")
            self.processing_finished.emit(False)


class TrialViewerApp(QMainWindow):
    """
    Main application window for the trial viewer.
    """
    
    def __init__(self):
        super().__init__()
        # Use absolute path for chamber configuration
        import os
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "chambers_configuration.json")
        self.video_processor = VideoProcessor(config_path)
        
        # Chamber coordinate scaling factor (to handle different video resolutions)
        self.chamber_scale_factor = 1.0
        self.processing_thread = None
        self.current_frame = 0
        self.is_playing = False
        self.show_chamber_boundaries = True  # Initialize chamber boundaries as visible
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle("Multiplex Trial Viewer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create horizontal splitter for video and controls
        video_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(video_splitter)
        
        # Left panel: Video display
        video_panel = self.create_video_panel()
        video_splitter.addWidget(video_panel)
        
        # Right panel: Controls
        controls_panel = self.create_controls_panel()
        video_splitter.addWidget(controls_panel)
        
        # Set video splitter proportions (70% video, 30% controls)
        video_splitter.setSizes([980, 420])
        
        # Bottom panel: Trace plot
        trace_panel = self.create_trace_panel()
        main_layout.addWidget(trace_panel)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
    
    def create_video_panel(self):
        """
        Create the video display panel.
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Video display
        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: black; color: white;")
        layout.addWidget(self.video_label)
        
        return panel
    
    def create_controls_panel(self):
        """
        Create the controls panel.
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Video controls
        controls_group = QGroupBox("Video Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_btn)
        
        self.prev_frame_btn = QPushButton("<<")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        playback_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QPushButton(">>")
        self.next_frame_btn.clicked.connect(self.next_frame)
        playback_layout.addWidget(self.next_frame_btn)
        
        playback_layout.addStretch()
        controls_layout.addLayout(playback_layout)
        
        # Seek slider
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.valueChanged.connect(self.seek_to_frame)
        controls_layout.addWidget(self.seek_slider)
        
        # Frame info
        self.frame_info_label = QLabel("Frame: 0 / 0")
        controls_layout.addWidget(self.frame_info_label)
        
        # Processing progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        layout.addWidget(controls_group)
        
        # Chamber selection controls
        chamber_group = QGroupBox("Chamber Selection")
        chamber_layout = QVBoxLayout(chamber_group)
        
        # Add chamber checkboxes (1-20)
        for i in range(1, 21):
            checkbox = QCheckBox(f"Chamber {i}")
            checkbox.setChecked(True)  # Default to all selected
            checkbox.stateChanged.connect(self.update_chamber_selection)
            chamber_layout.addWidget(checkbox)
        
        # Select all / Clear all buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_chambers)
        button_layout.addWidget(select_all_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_chambers)
        button_layout.addWidget(clear_all_btn)
        
        chamber_layout.addLayout(button_layout)
        layout.addWidget(chamber_group)
        
        layout.addStretch()
        return panel
    
    def create_trace_panel(self):
        """
        Create the trace plot panel.
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Trace plot
        self.trace_plotter = TracePlotter()
        layout.addWidget(self.trace_plotter)
        
        return panel
    
    def create_menu_bar(self):
        """
        Create the menu bar.
        """
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Video', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        open_with_csv_action = QAction('Open Video with CSV Data', self)
        open_with_csv_action.setShortcut('Ctrl+Shift+O')
        open_with_csv_action.triggered.connect(self.open_video_with_csv)
        file_menu.addAction(open_with_csv_action)
        
        file_menu.addSeparator()
        
        file_menu.addSeparator()
        
        export_csv_action = QAction('Export CSV', self)
        export_csv_action.triggered.connect(self.export_csv)
        file_menu.addAction(export_csv_action)
        
        export_image_action = QAction('Export Current Frame', self)
        export_image_action.triggered.connect(self.export_current_frame)
        file_menu.addAction(export_image_action)
        
        export_plot_action = QAction('Export Trace Plot', self)
        export_plot_action.triggered.connect(self.export_trace_plot)
        file_menu.addAction(export_plot_action)
        
        export_video_action = QAction('Export Annotated Video', self)
        export_video_action.triggered.connect(self.export_annotated_video)
        file_menu.addAction(export_video_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        toggle_chambers_action = QAction('Toggle Chamber Boundaries', self)
        toggle_chambers_action.setCheckable(True)
        toggle_chambers_action.setChecked(True)
        toggle_chambers_action.triggered.connect(self.toggle_chamber_boundaries)
        view_menu.addAction(toggle_chambers_action)
        
        # Config menu
        config_menu = menubar.addMenu('Config')
        
        edit_chambers_action = QAction('Edit Chamber Configuration', self)
        edit_chambers_action.triggered.connect(self.edit_chamber_config)
        config_menu.addAction(edit_chambers_action)
        
        load_chambers_action = QAction('Load Chamber Configuration...', self)
        load_chambers_action.triggered.connect(self.load_chamber_config)
        config_menu.addAction(load_chambers_action)
        
        save_chambers_action = QAction('Save Chamber Configuration As...', self)
        save_chambers_action.triggered.connect(self.save_chamber_config_as)
        config_menu.addAction(save_chambers_action)
        
        config_menu.addSeparator()
        
        reset_chambers_action = QAction('Reset to Default Configuration', self)
        reset_chambers_action.triggered.connect(self.reset_chamber_config)
        config_menu.addAction(reset_chambers_action)
    
    def create_status_bar(self):
        """
        Create the status bar.
        """
        # Show current chamber config file path
        config_file = os.path.basename(self.video_processor.chamber_config.config_file)
        self.statusBar().showMessage(f"Ready - Chamber Config: {config_file}")
    
    def setup_connections(self):
        """
        Setup signal connections.
        """
        pass
    
    def open_video(self):
        """
        Open a video file for processing.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            # First load the video without processing
            if self.video_processor.load_video(file_path):
                # Show chamber configuration selection dialog
                self.select_chamber_configuration()
            else:
                QMessageBox.critical(self, "Error", "Failed to load video file.")
    
    def open_video_with_csv(self):
        """
        Open a video file with corresponding CSV data.
        """
        # First select video file
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not video_path:
            return
        
        # Then select CSV file
        csv_path, _ = QFileDialog.getOpenFileName(
            self, "Open Fly Location CSV File", "", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not csv_path:
            return
        
        # Load video
        if not self.video_processor.load_video(video_path):
            QMessageBox.critical(self, "Error", "Failed to load video file.")
            return
        
        # Load CSV data
        if not self.load_fly_location_csv(csv_path):
            QMessageBox.critical(self, "Error", "Failed to load fly location CSV file.")
            self.video_processor.close()
            return
        
        # Auto-detect chamber scaling factor
        self.auto_detect_chamber_scaling()
        
        # Show chamber configuration selection dialog
        self.select_chamber_configuration()
    
    def auto_detect_chamber_scaling(self):
        """
        Auto-detect the appropriate scaling factor for chamber coordinates.
        """
        if not self.video_processor.video_capture:
            return
        
        # Get video dimensions
        video_width = int(self.video_processor.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.video_processor.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get chamber coordinates
        chambers = self.video_processor.chamber_config.get_all_chambers()
        
        # Find the maximum chamber coordinates
        max_x = 0
        max_y = 0
        max_width = 0
        max_height = 0
        
        for chamber_id, config in chambers.items():
            x, y, w, h = config['x'], config['y'], config['width'], config['height']
            if w > 0 and h > 0:  # Only consider configured chambers
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)
                max_width = max(max_width, w)
                max_height = max(max_height, h)
        
        # If chambers are much smaller than video, we need to scale up
        if max_x < video_width * 0.5 or max_y < video_height * 0.5:
            # Calculate scaling factor based on video dimensions
            scale_x = video_width / max_x if max_x > 0 else 1.0
            scale_y = video_height / max_y if max_y > 0 else 1.0
            
            # Use the smaller scaling factor to avoid overshooting
            self.chamber_scale_factor = min(scale_x, scale_y)
            
            print(f"Auto-detected chamber scaling factor: {self.chamber_scale_factor}")
            print(f"Video dimensions: {video_width}x{video_height}")
            print(f"Max chamber coordinates: x={max_x}, y={max_y}")
        else:
            self.chamber_scale_factor = 1.0
            print("No chamber scaling needed")
    
    def load_fly_location_csv(self, csv_path):
        """
        Load fly location data from CSV file.
        """
        try:
            import pandas as pd
            
            # Load CSV file
            self.fly_data = pd.read_csv(csv_path)
            
            # Validate CSV structure
            required_columns = ['frame', 'timestamp']
            chamber_columns = [f'chamber_{i}_loc' for i in range(1, 21)]
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns + chamber_columns if col not in self.fly_data.columns]
            if missing_columns:
                QMessageBox.critical(
                    self, "Error", 
                    f"CSV file is missing required columns: {', '.join(missing_columns)}\n\n"
                    f"Expected columns: frame, timestamp, chamber_1_loc, chamber_2_loc, ..., chamber_20_loc"
                )
                return False
            
            # Convert to dictionary format for easy access
            self.frame_data = {}
            for _, row in self.fly_data.iterrows():
                frame_num = int(row['frame'])
                chamber_locations = {}
                
                for i in range(1, 21):
                    chamber_id = f'chamber_{i}'
                    location = row[f'chamber_{i}_loc']
                    # Handle NaN values (no fly detected)
                    if pd.isna(location):
                        chamber_locations[chamber_id] = None
                    else:
                        chamber_locations[chamber_id] = float(location)
                
                self.frame_data[frame_num] = chamber_locations
            
            # Update video processor with the loaded data
            self.video_processor.frame_data = self.frame_data
            self.video_processor.is_processed = True
            
            print(f"Loaded fly location data for {len(self.frame_data)} frames from {csv_path}")
            return True
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False
    
    def select_chamber_configuration(self):
        """
        Ask user to select chamber configuration file and verify chamber locations.
        """
        from PyQt5.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
        
        # Ask user to select chamber configuration
        config_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Chamber Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not config_file:
            # User cancelled, close video
            self.video_processor.close()
            return
        
        try:
            # Load the chamber configuration
            new_config = ChamberConfig(config_file)
            
            if not new_config.validate_config():
                QMessageBox.critical(self, "Error", "Invalid chamber configuration file.")
                self.video_processor.close()
                return
            
            # Update the video processor's chamber config
            self.video_processor.chamber_config = new_config
            
            # Show chamber verification dialog
            self.verify_chamber_locations(config_file)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load chamber configuration:\n{str(e)}")
            self.video_processor.close()
    
    def verify_chamber_locations(self, config_file):
        """
        Show a dialog to verify chamber locations match the video.
        """
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap, QImage
        import cv2
        
        # Create verification dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Verify Chamber Locations")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Instructions
        instructions = QLabel(
            "Please verify that the chamber boundaries (green rectangles) match the actual chambers in your video.\n"
            "You can navigate through the video to check different frames."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Video display
        video_label = QLabel()
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        video_label.setMinimumHeight(400)
        layout.addWidget(video_label)
        
        # Frame navigation controls
        nav_layout = QHBoxLayout()
        
        prev_btn = QPushButton("← Previous Frame")
        prev_btn.clicked.connect(lambda: self.show_verification_frame(dialog, video_label, -1))
        nav_layout.addWidget(prev_btn)
        
        frame_label = QLabel("Frame: 0")
        nav_layout.addWidget(frame_label)
        
        next_btn = QPushButton("Next Frame →")
        next_btn.clicked.connect(lambda: self.show_verification_frame(dialog, video_label, 1))
        nav_layout.addWidget(next_btn)
        
        layout.addLayout(nav_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        edit_config_btn = QPushButton("Edit Chamber Configuration")
        edit_config_btn.clicked.connect(lambda: self.edit_chamber_config_from_verification(dialog))
        button_layout.addWidget(edit_config_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        proceed_btn = QPushButton("Proceed with Analysis")
        proceed_btn.clicked.connect(dialog.accept)
        proceed_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        button_layout.addWidget(proceed_btn)
        
        layout.addLayout(button_layout)
        
        # Store references for navigation
        dialog.frame_label = frame_label
        dialog.video_label = video_label
        dialog.current_frame = 0
        
        # Show first frame
        self.show_verification_frame(dialog, video_label, 0)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # User confirmed, proceed with video processing
            self.statusBar().showMessage("Processing video with chamber configuration...")
            self.process_video()
        else:
            # User cancelled, close video
            self.video_processor.close()
            self.statusBar().showMessage("Ready")
    
    def show_verification_frame(self, dialog, video_label, direction):
        """
        Show a frame with chamber boundaries for verification.
        """
        import cv2
        from PyQt5.QtGui import QPixmap, QImage
        from PyQt5.QtCore import Qt
        
        # Update frame number
        if direction == -1:
            dialog.current_frame = max(0, dialog.current_frame - 1)
        elif direction == 1:
            dialog.current_frame = min(self.video_processor.total_frames - 1, dialog.current_frame + 1)
        
        # Get the frame
        frame = self.video_processor.get_frame(dialog.current_frame)
        if frame is None:
            return
        
        # Draw chamber boundaries
        frame_with_chambers = frame.copy()
        chambers = self.video_processor.chamber_config.get_all_chambers()
        
        # Debug: Print frame dimensions and chamber coordinates
        frame_height, frame_width = frame.shape[:2]
        print(f"Verification - Frame dimensions: {frame_width}x{frame_height}")
        
        for chamber_id, config in chambers.items():
            x, y, w, h = config['x'], config['y'], config['width'], config['height']
            
            # Apply scaling factor
            scaled_x = int(x * self.chamber_scale_factor)
            scaled_y = int(y * self.chamber_scale_factor)
            scaled_w = int(w * self.chamber_scale_factor)
            scaled_h = int(h * self.chamber_scale_factor)
            
            # Debug: Print first few chamber coordinates
            if chamber_id in ['chamber_1', 'chamber_2', 'chamber_3']:
                print(f"Verification - {chamber_id}: original=({x},{y},{w},{h}) scaled=({scaled_x},{scaled_y},{scaled_w},{scaled_h})")
            
            if w > 0 and h > 0:  # Only draw if chamber is configured
                # Check if coordinates are within frame bounds
                if scaled_x >= 0 and scaled_y >= 0 and scaled_x + scaled_w <= frame_width and scaled_y + scaled_h <= frame_height:
                    cv2.rectangle(frame_with_chambers, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 2)
                    # Add chamber label
                    cv2.putText(frame_with_chambers, chamber_id.replace('chamber_', ''), 
                               (scaled_x, scaled_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    print(f"Verification - Warning: {chamber_id} coordinates outside frame bounds")
                    print(f"  Chamber: x={scaled_x}, y={scaled_y}, w={scaled_w}, h={scaled_h}")
                    print(f"  Frame: {frame_width}x{frame_height}")
        
        # Convert to QPixmap and display
        height, width, channel = frame_with_chambers.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame_with_chambers.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        video_label.setPixmap(scaled_pixmap)
        
        # Update frame label
        dialog.frame_label.setText(f"Frame: {dialog.current_frame} / {self.video_processor.total_frames - 1}")
    
    def edit_chamber_config_from_verification(self, dialog):
        """
        Edit chamber configuration from verification dialog.
        """
        # Close verification dialog temporarily
        dialog.hide()
        
        # Open chamber editor
        self.edit_chamber_config()
        
        # Show verification dialog again
        dialog.show()
        self.show_verification_frame(dialog, dialog.video_label, 0)
    
    def process_video(self):
        """
        Process the loaded video with fly detection.
        """
        self.statusBar().showMessage("Processing video with fly detection...")
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start processing thread
        self.processing_thread = VideoProcessingThread(self.video_processor, video_path)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_finished.connect(self.video_processing_finished)
        self.processing_thread.start()
    
    def update_progress(self, current_frame: int, total_frames: int):
        """
        Update progress bar during video processing.
        """
        progress = int((current_frame / total_frames) * 100) if total_frames > 0 else 0
        self.progress_bar.setValue(progress)
        self.statusBar().showMessage(f"Processing frame {current_frame}/{total_frames}")
    
    def video_processing_finished(self, success: bool):
        """
        Handle video processing completion.
        """
        self.progress_bar.setVisible(False)
        
        if success:
            self.statusBar().showMessage("Video processing completed")
            self.setup_video_playback()
        else:
            QMessageBox.critical(self, "Error", "Failed to process video")
            self.statusBar().showMessage("Video processing failed")
    
    def setup_video_playback(self):
        """
        Setup video playback after processing.
        """
        video_info = self.video_processor.get_video_info()
        self.total_frames = video_info['total_frames']
        self.fps = video_info['fps']
        
        # Update seek slider
        self.seek_slider.setMaximum(self.total_frames - 1)
        self.seek_slider.setValue(0)
        
        # Update frame info
        self.update_frame_info()
        
        # Load trace data
        if hasattr(self, 'frame_data'):
            # Use CSV data if available
            frame_data = self.frame_data
        else:
            # Use video processor data
            frame_data = self.video_processor.get_all_fly_locations()
        self.trace_plotter.load_trace_data(frame_data, self.fps)
        
        # Display first frame
        self.display_frame(0)
    
    def toggle_playback(self):
        """
        Toggle video playback.
        """
        if not self.video_processor.is_processed:
            return
        
        if self.is_playing:
            self.play_timer.stop()
            self.play_btn.setText("Play")
            self.is_playing = False
        else:
            self.play_timer.start(int(1000 / self.fps))  # Convert fps to ms
            self.play_btn.setText("Pause")
            self.is_playing = True
    
    def next_frame(self):
        """
        Go to next frame.
        """
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.display_frame(self.current_frame)
            self.seek_slider.setValue(self.current_frame)
    
    def prev_frame(self):
        """
        Go to previous frame.
        """
        if self.current_frame > 0:
            self.current_frame -= 1
            self.display_frame(self.current_frame)
            self.seek_slider.setValue(self.current_frame)
    
    def seek_to_frame(self, frame_number: int):
        """
        Seek to a specific frame.
        """
        if frame_number != self.current_frame:
            self.current_frame = frame_number
            self.display_frame(self.current_frame)
    
    def display_frame(self, frame_number: int):
        """
        Display a specific frame with chamber boundaries and fly locations.
        """
        if not self.video_processor.is_processed:
            return
        
        # Get frame
        frame = self.video_processor.get_frame(frame_number)
        if frame is None:
            return
        
        # Create a copy for annotation
        annotated_frame = frame.copy()
        
        # Draw chamber boundaries if enabled
        if hasattr(self, 'show_chamber_boundaries') and self.show_chamber_boundaries:
            chambers = self.video_processor.chamber_config.get_all_chambers()
            
            # Debug: Print frame dimensions and chamber coordinates
            frame_height, frame_width = annotated_frame.shape[:2]
            print(f"Frame dimensions: {frame_width}x{frame_height}")
            
            for chamber_id, config in chambers.items():
                x, y, width, height = config['x'], config['y'], config['width'], config['height']
                
                # Apply scaling factor
                scaled_x = int(x * self.chamber_scale_factor)
                scaled_y = int(y * self.chamber_scale_factor)
                scaled_width = int(width * self.chamber_scale_factor)
                scaled_height = int(height * self.chamber_scale_factor)
                
                # Debug: Print chamber coordinates
                if chamber_id in ['chamber_1', 'chamber_2', 'chamber_3']:
                    print(f"{chamber_id}: original=({x},{y},{width},{height}) scaled=({scaled_x},{scaled_y},{scaled_width},{scaled_height})")
                
                # Check if chamber coordinates are within frame bounds
                if scaled_x >= 0 and scaled_y >= 0 and scaled_x + scaled_width <= frame_width and scaled_y + scaled_height <= frame_height:
                    # Draw green rectangle for chamber boundary
                    cv2.rectangle(annotated_frame, (scaled_x, scaled_y), (scaled_x + scaled_width, scaled_y + scaled_height), (0, 255, 0), 2)
                    # Add chamber ID label
                    cv2.putText(annotated_frame, chamber_id, (scaled_x, scaled_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                else:
                    print(f"Warning: {chamber_id} coordinates outside frame bounds")
                    print(f"  Chamber: x={scaled_x}, y={scaled_y}, w={scaled_width}, h={scaled_height}")
                    print(f"  Frame: {frame_width}x{frame_height}")
        
        # Draw fly locations from CSV data if available
        if hasattr(self, 'frame_data') and frame_number in self.frame_data:
            fly_locations = self.frame_data[frame_number]
            chambers = self.video_processor.chamber_config.get_all_chambers()
            
            for chamber_id, location in fly_locations.items():
                if location is not None and chamber_id in chambers:
                    config = chambers[chamber_id]
                    x, y, width, height = config['x'], config['y'], config['width'], config['height']
                    
                    if width > 0 and height > 0:  # Only draw if chamber is configured
                        # Apply scaling factor
                        scaled_x = int(x * self.chamber_scale_factor)
                        scaled_y = int(y * self.chamber_scale_factor)
                        scaled_width = int(width * self.chamber_scale_factor)
                        scaled_height = int(height * self.chamber_scale_factor)
                        
                        # Calculate fly position within chamber
                        # Location -100 to +100 maps to chamber width
                        fly_x_pixel = int(scaled_x + (location + 100) * scaled_width / 200)
                        fly_y_pixel = int(scaled_y + scaled_height / 2)
                        
                        # Draw red cross for fly location
                        cv2.line(annotated_frame, (fly_x_pixel - 5, fly_y_pixel), 
                                (fly_x_pixel + 5, fly_y_pixel), (0, 0, 255), 2)
                        cv2.line(annotated_frame, (fly_x_pixel, fly_y_pixel - 5), 
                                (fly_x_pixel, fly_y_pixel + 5), (0, 0, 255), 2)
        
        # Convert frame to QImage
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update trace plot
        current_time = frame_number / self.fps if self.fps > 0 else 0
        self.trace_plotter.update_current_time(current_time)
        
        # Update frame info
        self.update_frame_info()
    
    def update_frame_info(self):
        """
        Update frame information display.
        """
        self.frame_info_label.setText(f"Frame: {self.current_frame} / {self.total_frames}")
    
    def update_chamber_selection(self):
        """
        Update chamber selection when checkboxes change.
        """
        selected_chambers = set()
        for i in range(1, 21):
            checkbox = self.findChild(QCheckBox, f"Chamber {i}")
            if checkbox and checkbox.isChecked():
                selected_chambers.add(f"chamber_{i}")
        
        self.trace_plotter.update_selected_chambers(selected_chambers)
    
    def select_all_chambers(self):
        """
        Select all chamber checkboxes.
        """
        for i in range(1, 21):
            checkbox = self.findChild(QCheckBox, f"Chamber {i}")
            if checkbox:
                checkbox.setChecked(True)
    
    def clear_all_chambers(self):
        """
        Clear all chamber checkboxes.
        """
        for i in range(1, 21):
            checkbox = self.findChild(QCheckBox, f"Chamber {i}")
            if checkbox:
                checkbox.setChecked(False)
    
    def update_trace_display(self, selected_chambers: set):
        """
        Update trace display based on selected chambers.
        """
        self.trace_plotter.update_selected_chambers(selected_chambers)
    
    def toggle_chamber_boundaries(self, checked: bool):
        """
        Toggle chamber boundary display.
        """
        self.show_chamber_boundaries = checked
        # Refresh current frame to show/hide boundaries
        if self.video_processor.is_processed:
            self.display_frame(self.current_frame)
        print(f"Chamber boundaries: {'ON' if checked else 'OFF'}")
    
    def edit_chamber_config(self):
        """
        Open chamber configuration editor.
        """
        dialog = ChamberEditorDialog(self.video_processor.chamber_config, self)
        if dialog.exec_() == QDialog.Accepted:
            # Refresh video display if video is loaded
            if self.video_processor.is_processed:
                self.display_frame(self.current_frame)
            QMessageBox.information(self, "Success", "Chamber configuration updated!")
    
    def load_chamber_config(self):
        """
        Load chamber configuration from a JSON file.
        """
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Chamber Configuration", 
            "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create a new chamber config with the selected file
                new_config = ChamberConfig(file_path)
                
                # Validate the loaded configuration
                if new_config.validate_config():
                    # Update the video processor's chamber config
                    self.video_processor.chamber_config = new_config
                    
                    # Update the current chamber config reference
                    self.chamber_config = new_config
                    
                    # Update status bar
                    config_file = os.path.basename(new_config.config_file)
                    self.statusBar().showMessage(f"Ready - Chamber Config: {config_file}")
                    
                    QMessageBox.information(
                        self, 
                        "Success", 
                        f"Chamber configuration loaded successfully from:\n{file_path}\n\n"
                        f"Loaded {new_config.get_chamber_count()} chambers."
                    )
                else:
                    QMessageBox.critical(
                        self, 
                        "Error", 
                        "The selected file does not contain a valid chamber configuration."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Failed to load chamber configuration:\n{str(e)}"
                )
    
    def save_chamber_config_as(self):
        """
        Save chamber configuration to a new JSON file.
        """
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Chamber Configuration As", 
            "chambers_configuration.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create a temporary chamber config with the new path
                temp_config = ChamberConfig(file_path)
                temp_config.chambers = self.video_processor.chamber_config.get_all_chambers()
                
                if temp_config.save_config():
                    QMessageBox.information(
                        self, 
                        "Success", 
                        f"Chamber configuration saved successfully to:\n{file_path}"
                    )
                else:
                    QMessageBox.critical(
                        self, 
                        "Error", 
                        f"Failed to save chamber configuration to:\n{file_path}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Failed to save chamber configuration:\n{str(e)}"
                )
    
    def reset_chamber_config(self):
        """
        Reset chamber configuration to default values.
        """
        from PyQt5.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self, 
            "Reset Configuration", 
            "Are you sure you want to reset the chamber configuration to default values?\n\n"
            "This will set all chambers to (0, 0, 0, 0) coordinates.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Reset to default configuration
                self.video_processor.chamber_config.reset_to_default()
                
                # Save the reset configuration
                if self.video_processor.chamber_config.save_config():
                    QMessageBox.information(
                        self, 
                        "Success", 
                        "Chamber configuration has been reset to default values."
                    )
                else:
                    QMessageBox.critical(
                        self, 
                        "Error", 
                        "Failed to save the reset configuration."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error", 
                    f"Failed to reset chamber configuration:\n{str(e)}"
                )
    
    def export_csv(self):
        """
        Export fly locations to CSV.
        """
        if not self.video_processor.is_processed:
            QMessageBox.warning(self, "Export Error", "No video processed yet")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "fly_locations.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            success = self.video_processor.export_to_csv(file_path)
            if success:
                QMessageBox.information(self, "Export Success", f"Data exported to {file_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export CSV")
    
    def export_current_frame(self):
        """
        Export current frame as annotated image (with chamber boundaries and fly crosses).
        """
        if not self.video_processor.is_processed:
            QMessageBox.warning(self, "Export Error", "No video processed yet")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Current Frame", f"frame_{self.current_frame}.png", 
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if file_path:
            # Get the raw frame
            frame = self.video_processor.get_frame(self.current_frame)
            if frame is not None:
                # Create annotated frame (same logic as display_frame)
                annotated_frame = frame.copy()
                
                # Draw chamber boundaries
                chambers = self.video_processor.chamber_config.get_all_chambers()
                for chamber_id, config in chambers.items():
                    x, y, width, height = config['x'], config['y'], config['width'], config['height']
                    # Draw green rectangle for chamber boundary
                    cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    # Add chamber ID label
                    cv2.putText(annotated_frame, chamber_id, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Draw fly locations
                fly_locations = self.video_processor.get_fly_locations(self.current_frame)
                if fly_locations:
                    for chamber_id, location in fly_locations.items():
                        if chamber_id in chambers and not np.isnan(location):
                            config = chambers[chamber_id]
                            x, y, width, height = config['x'], config['y'], config['width'], config['height']
                            
                            # Calculate fly position within chamber
                            fly_x_pixel = int(x + (location + 100) * width / 200)
                            fly_y_pixel = int(y + height / 2)
                            
                            # Draw red cross for fly location
                            cv2.line(annotated_frame, (fly_x_pixel - 5, fly_y_pixel), 
                                    (fly_x_pixel + 5, fly_y_pixel), (0, 0, 255), 2)
                            cv2.line(annotated_frame, (fly_x_pixel, fly_y_pixel - 5), 
                                    (fly_x_pixel, fly_y_pixel + 5), (0, 0, 255), 2)
                
                # Save the annotated frame
                cv2.imwrite(file_path, annotated_frame)
                QMessageBox.information(self, "Export Success", f"Annotated frame exported to {file_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export frame")
    
    def export_trace_plot(self):
        """
        Export trace plot as image.
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Trace Plot", "trace_plot.png", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            success = self.trace_plotter.export_plot(file_path)
            if success:
                QMessageBox.information(self, "Export Success", f"Plot exported to {file_path}")
            else:
                QMessageBox.critical(self, "Export Error", "Failed to export plot")
    
    def export_annotated_video(self):
        """
        Export full video with annotations (chamber boundaries and fly crosses).
        """
        if not self.video_processor.is_processed:
            QMessageBox.warning(self, "Export Error", "No video processed yet")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotated Video", "annotated_video.mp4", 
            "MP4 Files (*.mp4);;AVI Files (*.avi)"
        )
        
        if file_path:
            try:
                # Get video properties
                video_info = self.video_processor.get_video_info()
                fps = video_info['fps']
                width = video_info['width']
                height = video_info['height']
                total_frames = video_info['total_frames']
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                
                # Process each frame
                for frame_num in range(total_frames):
                    # Get raw frame
                    frame = self.video_processor.get_frame(frame_num)
                    if frame is not None:
                        # Create annotated frame (same logic as display_frame)
                        annotated_frame = frame.copy()
                        
                        # Draw chamber boundaries
                        chambers = self.video_processor.chamber_config.get_all_chambers()
                        for chamber_id, config in chambers.items():
                            x, y, width, height = config['x'], config['y'], config['width'], config['height']
                            # Draw green rectangle for chamber boundary
                            cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                            # Add chamber ID label
                            cv2.putText(annotated_frame, chamber_id, (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        # Draw fly locations
                        fly_locations = self.video_processor.get_fly_locations(frame_num)
                        if fly_locations:
                            for chamber_id, location in fly_locations.items():
                                if chamber_id in chambers and not np.isnan(location):
                                    config = chambers[chamber_id]
                                    x, y, width, height = config['x'], config['y'], config['width'], config['height']
                                    
                                    # Calculate fly position within chamber
                                    fly_x_pixel = int(x + (location + 100) * width / 200)
                                    fly_y_pixel = int(y + height / 2)
                                    
                                    # Draw red cross for fly location
                                    cv2.line(annotated_frame, (fly_x_pixel - 5, fly_y_pixel), 
                                            (fly_x_pixel + 5, fly_y_pixel), (0, 0, 255), 2)
                                    cv2.line(annotated_frame, (fly_x_pixel, fly_y_pixel - 5), 
                                            (fly_x_pixel, fly_y_pixel + 5), (0, 0, 255), 2)
                        
                        # Write frame to output video
                        out.write(annotated_frame)
                    
                    # Update progress (every 100 frames)
                    if frame_num % 100 == 0:
                        progress = int((frame_num / total_frames) * 100)
                        self.statusBar().showMessage(f"Exporting video... {progress}%")
                        QApplication.processEvents()  # Keep UI responsive
                
                # Release video writer
                out.release()
                
                QMessageBox.information(self, "Export Success", f"Annotated video exported to {file_path}")
                self.statusBar().showMessage("Video export completed")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export video: {str(e)}")
                self.statusBar().showMessage("Video export failed")
    
    def closeEvent(self, event):
        """
        Handle application close event.
        """
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        
        self.video_processor.close()
        event.accept()


def main():
    """
    Main application entry point.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Multiplex Trial Viewer")
    
    window = TrialViewerApp()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

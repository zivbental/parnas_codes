"""
Video processing module for multiplex trial viewer.
Handles video loading, fly detection processing, and frame caching.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime
from fly_detection_core import FlyDetector
from chamber_config import ChamberConfig


class VideoProcessor:
    """
    Handles video loading, processing, and frame caching for the trial viewer.
    """
    
    def __init__(self, chamber_config_file: str = "chambers_configuration.json"):
        """
        Initialize the video processor.
        
        Args:
            chamber_config_file (str): Path to chamber configuration file
        """
        self.chamber_config = ChamberConfig(chamber_config_file)
        self.fly_detector = FlyDetector()
        self.video_capture = None
        self.video_path = None
        self.fps = 0
        self.total_frames = 0
        self.duration = 0
        self.width = 0
        self.height = 0
        
        # Processed data storage
        self.frame_data = {}  # {frame_num: {chamber_id: location}}
        self.processed_frames = {}  # {frame_num: processed_frame}
        self.is_processed = False
        
    def load_video(self, video_path: str) -> bool:
        """
        Load a video file for processing.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
        
        try:
            self.video_capture = cv2.VideoCapture(video_path)
            
            if not self.video_capture.isOpened():
                print(f"Could not open video file: {video_path}")
                return False
            
            # Get video properties
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.total_frames / self.fps if self.fps > 0 else 0
            self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.video_path = video_path
            
            print(f"Loaded video: {video_path}")
            print(f"Properties: {self.total_frames} frames, {self.fps:.2f} fps, {self.duration:.2f}s")
            print(f"Resolution: {self.width}x{self.height}")
            
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def process_video(self, progress_callback=None) -> bool:
        """
        Process the entire video with fly detection.
        
        Args:
            progress_callback (callable): Optional callback for progress updates (progress_callback(current_frame, total_frames))
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.video_capture is None:
            print("No video loaded")
            return False
        
        print("Starting video processing...")
        self.frame_data = {}
        self.processed_frames = {}
        
        # Reset video to beginning
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        chambers = self.chamber_config.get_all_chambers()
        
        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                
                # Process frame with fly detection
                processed_frame, fly_locations = self.fly_detector.process_frame_with_chambers(frame, chambers)
                
                # Store results
                self.frame_data[frame_count] = fly_locations.copy()
                self.processed_frames[frame_count] = processed_frame.copy()
                
                # Update progress
                if progress_callback:
                    progress_callback(frame_count, self.total_frames)
                
                frame_count += 1
                
                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{self.total_frames} frames")
            
            self.is_processed = True
            print(f"Video processing completed. Processed {frame_count} frames")
            return True
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return False
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame from the video.
        
        Args:
            frame_number (int): Frame number to retrieve
            
        Returns:
            Optional[np.ndarray]: Frame data or None if not available
        """
        if self.video_capture is None:
            return None
        
        if frame_number < 0 or frame_number >= self.total_frames:
            return None
        
        # Set frame position
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_capture.read()
        
        if ret:
            return frame
        return None
    
    def get_processed_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a processed frame (with fly detection applied).
        
        Args:
            frame_number (int): Frame number to retrieve
            
        Returns:
            Optional[np.ndarray]: Processed frame or None if not available
        """
        if not self.is_processed:
            return None
        
        return self.processed_frames.get(frame_number)
    
    def get_fly_locations(self, frame_number: int) -> Dict[str, float]:
        """
        Get fly locations for a specific frame.
        
        Args:
            frame_number (int): Frame number
            
        Returns:
            Dict[str, float]: Dictionary of chamber_id: location pairs
        """
        if not self.is_processed:
            return {}
        
        return self.frame_data.get(frame_number, {})
    
    def get_all_fly_locations(self) -> Dict[int, Dict[str, float]]:
        """
        Get all fly locations for all frames.
        
        Returns:
            Dict[int, Dict[str, float]]: All frame data
        """
        return self.frame_data.copy()
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        Export detected fly locations to CSV file.
        
        Args:
            output_path (str): Path for the output CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_processed:
            print("Video not processed yet")
            return False
        
        try:
            # Prepare data for CSV
            data = []
            chambers = self.chamber_config.get_all_chambers()
            chamber_ids = list(chambers.keys())
            
            for frame_num, fly_locations in self.frame_data.items():
                timestamp = frame_num / self.fps if self.fps > 0 else frame_num
                
                row = {
                    'frame': frame_num,
                    'timestamp': timestamp
                }
                
                # Add location for each chamber
                for chamber_id in chamber_ids:
                    location = fly_locations.get(chamber_id, np.nan)
                    row[f'{chamber_id}_loc'] = location
                
                data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"Exported fly locations to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
    
    def get_video_info(self) -> Dict[str, Any]:
        """
        Get video information.
        
        Returns:
            Dict[str, Any]: Video information dictionary
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'width': self.width,
            'height': self.height,
            'is_processed': self.is_processed
        }
    
    def close(self):
        """
        Close the video capture and clean up resources.
        """
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        print("Video processor closed")


def test_video_processor():
    """
    Test function for video processor functionality.
    """
    print("Testing video processor...")
    
    # Create a test video processor
    processor = VideoProcessor()
    
    # Test video info (without loading a real video)
    info = processor.get_video_info()
    print(f"Initial video info: {info}")
    
    # Test chamber configuration
    chambers = processor.chamber_config.get_all_chambers()
    print(f"Loaded {len(chambers)} chambers")
    
    # Test frame data structure
    test_frame_data = {0: {'chamber_1': 50.0, 'chamber_2': -30.0}}
    processor.frame_data = test_frame_data
    processor.is_processed = True
    
    locations = processor.get_fly_locations(0)
    print(f"Test fly locations: {locations}")
    
    # Test CSV export (with test data)
    test_export_success = processor.export_to_csv("test_export.csv")
    print(f"Test CSV export: {test_export_success}")
    
    # Clean up test file
    if os.path.exists("test_export.csv"):
        os.remove("test_export.csv")
        print("Cleaned up test CSV file")
    
    processor.close()
    print("Video processor test completed successfully!")
    return True


if __name__ == "__main__":
    test_video_processor()

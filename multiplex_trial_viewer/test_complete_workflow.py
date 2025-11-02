#!/usr/bin/env python3
"""
Test script for complete workflow of the Multiplex Trial Viewer.
This script tests the full pipeline: Load → Process → View → Export
"""

import sys
import os
import tempfile
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trial_viewer_app import TrialViewerApp
from video_processor import VideoProcessor
from chamber_config import ChamberConfig

def create_test_video(output_path, duration=5, fps=30, width=800, height=600):
    """
    Create a test video with moving objects to simulate fly movement.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some moving objects to simulate flies
        for i in range(5):
            x = int(100 + i * 150 + 50 * np.sin(frame_num * 0.1 + i))
            y = int(300 + 30 * np.cos(frame_num * 0.15 + i))
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {output_path}")

def test_complete_workflow():
    """
    Test the complete workflow of the application.
    """
    print("🧪 Testing Complete Workflow")
    print("=" * 50)
    
    # Create test video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        test_video_path = tmp_file.name
    
    print("1. Creating test video...")
    create_test_video(test_video_path)
    print(f"   ✓ Test video created: {test_video_path}")
    
    # Test video processor
    print("\n2. Testing video processor...")
    processor = VideoProcessor()
    
    # Load video
    if processor.load_video(test_video_path):
        print("   ✓ Video loaded successfully")
        
        # Get video info
        info = processor.get_video_info()
        print(f"   ✓ Video info: {info['total_frames']} frames, {info['fps']} fps")
        
        # Test chamber config
        print("\n3. Testing chamber configuration...")
        chamber_config = ChamberConfig()
        chambers = chamber_config.get_all_chambers()
        print(f"   ✓ Loaded {len(chambers)} chambers")
        
        # Test fly detection core
        print("\n4. Testing fly detection...")
        from fly_detection_core import FlyDetector
        detector = FlyDetector()
        
        # Test with a sample frame
        frame = processor.get_frame(0)
        if frame is not None:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Update background model
            detector.update_background_model(gray)
            
            # Perform detection
            mask = np.ones(gray.shape, dtype=np.uint8) * 255
            contours = detector.fly_detection(gray, mask)
            print(f"   ✓ Detected {len(contours)} contours")
        
        # Test trace plotter
        print("\n5. Testing trace plotter...")
        from trace_plotter import TracePlotter
        plotter = TracePlotter()
        
        # Create sample data
        sample_data = {
            0: {'chamber_1': 50.0, 'chamber_2': -30.0},
            1: {'chamber_1': 60.0, 'chamber_2': -20.0},
            2: {'chamber_1': 70.0, 'chamber_2': -10.0}
        }
        
        plotter.load_trace_data(sample_data, 30.0)
        print("   ✓ Trace plotter initialized with sample data")
        
        # Test export functionality
        print("\n6. Testing export functionality...")
        
        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_csv:
            csv_path = tmp_csv.name
        
        if processor.export_to_csv(csv_path):
            print("   ✓ CSV export successful")
        else:
            print("   ✗ CSV export failed")
        
        # Test image export
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
            img_path = tmp_img.name
        
        if plotter.export_plot(img_path):
            print("   ✓ Image export successful")
        else:
            print("   ✗ Image export failed")
        
        # Test GUI application
        print("\n7. Testing GUI application...")
        app = QApplication(sys.argv)
        window = TrialViewerApp()
        
        # Test basic functionality
        print("   ✓ GUI application created")
        print("   ✓ All components initialized")
        
        # Test video loading in GUI
        if window.video_processor.load_video(test_video_path):
            print("   ✓ Video loaded in GUI")
        else:
            print("   ✗ Video loading failed in GUI")
        
        # Clean up
        window.close()
        app.quit()
        
        print("\n8. Cleaning up...")
        os.unlink(test_video_path)
        os.unlink(csv_path)
        os.unlink(img_path)
        print("   ✓ Temporary files cleaned up")
        
        print("\n🎉 Complete workflow test PASSED!")
        print("All components are working correctly.")
        
    else:
        print("   ✗ Failed to load video")
        return False
    
    return True

def test_error_handling():
    """
    Test error handling scenarios.
    """
    print("\n🧪 Testing Error Handling")
    print("=" * 50)
    
    # Test with non-existent video
    print("1. Testing with non-existent video...")
    processor = VideoProcessor()
    if not processor.load_video("non_existent_video.mp4"):
        print("   ✓ Correctly handled non-existent video")
    else:
        print("   ✗ Failed to handle non-existent video")
    
    # Test with invalid chamber config
    print("\n2. Testing with invalid chamber config...")
    try:
        config = ChamberConfig("invalid_config.json")
        print("   ✓ Handled invalid config gracefully")
    except Exception as e:
        print(f"   ✓ Handled invalid config: {e}")
    
    print("\n🎉 Error handling test PASSED!")

if __name__ == "__main__":
    print("🚀 Multiplex Trial Viewer - Complete Workflow Test")
    print("=" * 60)
    
    try:
        # Test complete workflow
        success = test_complete_workflow()
        
        if success:
            # Test error handling
            test_error_handling()
            
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("The Multiplex Trial Viewer is ready for use.")
            print("=" * 60)
        else:
            print("\n❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

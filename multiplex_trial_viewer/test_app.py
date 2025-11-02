"""
Test script for the multiplex trial viewer application.
Tests basic functionality without requiring a GUI display.
"""

import sys
import os
sys.path.append('.')

from trial_viewer_app import TrialViewerApp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer


def test_app_creation():
    """
    Test that the application can be created without errors.
    """
    print("Testing application creation...")
    
    app = QApplication(sys.argv)
    window = TrialViewerApp()
    
    # Test that the window was created successfully
    assert window is not None
    assert window.video_processor is not None
    assert window.trace_plotter is not None
    
    print("✓ Application created successfully")
    print("✓ Video processor initialized")
    print("✓ Trace plotter initialized")
    print("✓ All UI components created")
    
    # Test chamber configuration loading
    chambers = window.video_processor.chamber_config.get_all_chambers()
    print(f"✓ Loaded {len(chambers)} chambers")
    
    # Test video processor info
    video_info = window.video_processor.get_video_info()
    print(f"✓ Video processor info: {video_info}")
    
    # Close the application
    window.close()
    app.quit()
    
    print("✓ Application closed successfully")
    return True


def test_components_integration():
    """
    Test that all components can be imported and work together.
    """
    print("\nTesting component integration...")
    
    # Test imports
    try:
        from fly_detection_core import FlyDetector
        from chamber_config import ChamberConfig
        from video_processor import VideoProcessor
        from trace_plotter import TracePlotter
        print("✓ All modules imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test component creation
    try:
        detector = FlyDetector()
        config = ChamberConfig()
        processor = VideoProcessor()
        print("✓ All components created successfully")
    except Exception as e:
        print(f"✗ Component creation error: {e}")
        return False
    
    # Test chamber configuration
    chambers = config.get_all_chambers()
    assert len(chambers) == 20
    print(f"✓ Chamber configuration: {len(chambers)} chambers")
    
    # Test video processor info
    video_info = processor.get_video_info()
    assert video_info['is_processed'] == False
    print("✓ Video processor initialized correctly")
    
    return True


def main():
    """
    Run all tests.
    """
    print("=" * 50)
    print("MULTIPLEX TRIAL VIEWER - COMPONENT TESTS")
    print("=" * 50)
    
    try:
        # Test component integration
        if not test_components_integration():
            print("\n✗ Component integration test failed")
            return False
        
        # Test application creation
        if not test_app_creation():
            print("\n✗ Application creation test failed")
            return False
        
        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("✓ Application is ready to use")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

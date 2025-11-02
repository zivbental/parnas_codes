#!/usr/bin/env python3
"""
Test script for chamber editor dialog functionality.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trial_viewer_app import ChamberEditorDialog
from chamber_config import ChamberConfig

def test_chamber_editor():
    """
    Test the chamber editor dialog.
    """
    print("Testing Chamber Editor Dialog")
    print("=" * 40)
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create chamber config with absolute path
    config_path = os.path.join(os.path.dirname(__file__), "test_chambers.json")
    chamber_config = ChamberConfig(config_path)
    
    print(f"Chamber config file: {chamber_config.config_file}")
    print(f"Number of chambers: {chamber_config.get_chamber_count()}")
    
    # Create and show dialog
    dialog = ChamberEditorDialog(chamber_config)
    
    print("Chamber editor dialog created successfully!")
    print("Dialog should open in a new window.")
    print("You can test editing chamber coordinates and saving.")
    
    # Show dialog (this will block until closed)
    result = dialog.exec_()
    
    if result == dialog.Accepted:
        print("Dialog was accepted - changes were saved!")
    else:
        print("Dialog was cancelled - no changes were saved.")
    
    # Clean up
    app.quit()
    
    # Clean up test file
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"Cleaned up test file: {config_path}")
    
    print("Test completed!")

if __name__ == "__main__":
    test_chamber_editor()

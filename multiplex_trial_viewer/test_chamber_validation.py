#!/usr/bin/env python3
"""
Test script to verify chamber editor validation works correctly.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chamber_config import ChamberConfig

def test_table_validation():
    """
    Test the table validation logic.
    """
    print("Testing Chamber Editor Table Validation")
    print("=" * 50)
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create chamber config
    config_path = os.path.join(os.path.dirname(__file__), "test_chambers.json")
    chamber_config = ChamberConfig(config_path)
    
    # Create a test table
    table = QTableWidget(20, 5)
    table.setHorizontalHeaderLabels(["Chamber", "X", "Y", "Width", "Height"])
    
    # Load test data
    chambers = chamber_config.get_all_chambers()
    chamber_list = list(chambers.items())
    
    for i, (chamber_id, config) in enumerate(chamber_list):
        # Chamber ID
        table.setItem(i, 0, QTableWidgetItem(chamber_id))
        
        # Coordinates
        table.setItem(i, 1, QTableWidgetItem(str(config['x'])))
        table.setItem(i, 2, QTableWidgetItem(str(config['y'])))
        table.setItem(i, 3, QTableWidgetItem(str(config['width'])))
        table.setItem(i, 4, QTableWidgetItem(str(config['height'])))
    
    print(f"Table created with {table.rowCount()} rows and {table.columnCount()} columns")
    
    # Test validation logic
    print("\nTesting validation logic...")
    
    for i in range(table.rowCount()):
        chamber_id = table.item(i, 0).text()
        
        # Get text values
        x_text = table.item(i, 1).text() if table.item(i, 1) else ""
        y_text = table.item(i, 2).text() if table.item(i, 2) else ""
        width_text = table.item(i, 3).text() if table.item(i, 3) else ""
        height_text = table.item(i, 4).text() if table.item(i, 4) else ""
        
        print(f"Chamber {i+1}: X='{x_text}', Y='{y_text}', W='{width_text}', H='{height_text}'")
        
        # Check if any field is empty
        if not x_text.strip():
            print(f"  ❌ X coordinate is empty for {chamber_id}")
        elif not y_text.strip():
            print(f"  ❌ Y coordinate is empty for {chamber_id}")
        elif not width_text.strip():
            print(f"  ❌ Width is empty for {chamber_id}")
        elif not height_text.strip():
            print(f"  ❌ Height is empty for {chamber_id}")
        else:
            print(f"  ✅ All fields filled for {chamber_id}")
    
    # Test with some empty values
    print("\nTesting with empty values...")
    table.setItem(0, 1, QTableWidgetItem(""))  # Empty X
    table.setItem(1, 2, QTableWidgetItem(""))  # Empty Y
    
    for i in range(2):  # Test first 2 rows
        chamber_id = table.item(i, 0).text()
        x_text = table.item(i, 1).text() if table.item(i, 1) else ""
        y_text = table.item(i, 2).text() if table.item(i, 2) else ""
        width_text = table.item(i, 3).text() if table.item(i, 3) else ""
        height_text = table.item(i, 4).text() if table.item(i, 4) else ""
        
        if not x_text.strip():
            print(f"  ❌ X coordinate is empty for {chamber_id}")
        elif not y_text.strip():
            print(f"  ❌ Y coordinate is empty for {chamber_id}")
        elif not width_text.strip():
            print(f"  ❌ Width is empty for {chamber_id}")
        elif not height_text.strip():
            print(f"  ❌ Height is empty for {chamber_id}")
        else:
            print(f"  ✅ All fields filled for {chamber_id}")
    
    app.quit()
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"\nCleaned up test file: {config_path}")
    
    print("\n✅ Table validation test completed!")

if __name__ == "__main__":
    test_table_validation()

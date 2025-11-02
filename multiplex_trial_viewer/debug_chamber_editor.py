#!/usr/bin/env python3
"""
Debug script for chamber editor to identify the validation issue.
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chamber_config import ChamberConfig

class DebugChamberDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Debug Chamber Editor")
        self.setModal(True)
        self.resize(600, 500)
        
        # Create chamber config
        config_path = os.path.join(os.path.dirname(__file__), "chambers_configuration.json")
        self.chamber_config = ChamberConfig(config_path)
        
        self.init_ui()
        self.load_chamber_data()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Table widget
        self.table = QTableWidget(20, 5)
        self.table.setHorizontalHeaderLabels(["Chamber", "X", "Y", "Width", "Height"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table)
        
        # Debug button
        debug_btn = QPushButton("Debug Table State")
        debug_btn.clicked.connect(self.debug_table)
        layout.addWidget(debug_btn)
        
        # Save button
        save_btn = QPushButton("Test Save")
        save_btn.clicked.connect(self.test_save)
        layout.addWidget(save_btn)
    
    def load_chamber_data(self):
        chambers = self.chamber_config.get_all_chambers()
        chamber_list = list(chambers.items())
        
        # Ensure we have exactly 20 chambers
        if len(chamber_list) < 20:
            for i in range(len(chamber_list), 20):
                chamber_id = f"chamber_{i+1}"
                chambers[chamber_id] = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
                chamber_list.append((chamber_id, chambers[chamber_id]))
        
        self.table.setRowCount(len(chamber_list))
        self.table.clearContents()
        
        for i, (chamber_id, config) in enumerate(chamber_list):
            # Chamber ID
            chamber_item = QTableWidgetItem(chamber_id)
            chamber_item.setFlags(chamber_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, chamber_item)
            
            # Coordinates
            self.table.setItem(i, 1, QTableWidgetItem(str(config['x'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(config['y'])))
            self.table.setItem(i, 3, QTableWidgetItem(str(config['width'])))
            self.table.setItem(i, 4, QTableWidgetItem(str(config['height'])))
        
        print(f"Loaded {len(chamber_list)} chambers into table")
    
    def debug_table(self):
        print("\n=== DEBUG TABLE STATE ===")
        print(f"Table rows: {self.table.rowCount()}")
        print(f"Table columns: {self.table.columnCount()}")
        
        for i in range(min(5, self.table.rowCount())):  # Check first 5 rows
            chamber_id = self.table.item(i, 0).text() if self.table.item(i, 0) else "None"
            x_item = self.table.item(i, 1)
            y_item = self.table.item(i, 2)
            w_item = self.table.item(i, 3)
            h_item = self.table.item(i, 4)
            
            x_text = x_item.text() if x_item else "None"
            y_text = y_item.text() if y_item else "None"
            w_text = w_item.text() if w_item else "None"
            h_text = h_item.text() if h_item else "None"
            
            print(f"Row {i}: {chamber_id} | X='{x_text}' | Y='{y_text}' | W='{w_text}' | H='{h_text}'")
            print(f"  Items exist: X={x_item is not None}, Y={y_item is not None}, W={w_item is not None}, H={h_item is not None}")
    
    def test_save(self):
        print("\n=== TESTING SAVE VALIDATION ===")
        
        for i in range(min(3, self.table.rowCount())):  # Test first 3 rows
            chamber_id_item = self.table.item(i, 0)
            if not chamber_id_item:
                print(f"Row {i}: No chamber ID item")
                continue
            
            chamber_id = chamber_id_item.text()
            
            x_item = self.table.item(i, 1)
            y_item = self.table.item(i, 2)
            w_item = self.table.item(i, 3)
            h_item = self.table.item(i, 4)
            
            x_text = x_item.text() if x_item else ""
            y_text = y_item.text() if y_item else ""
            w_text = w_item.text() if w_item else ""
            h_text = h_item.text() if h_item else ""
            
            print(f"Chamber {i+1}: X='{x_text}', Y='{y_text}', W='{w_text}', H='{h_text}'")
            
            if not x_text.strip():
                QMessageBox.critical(self, "Error", f"X coordinate is empty for {chamber_id}")
                return
            if not y_text.strip():
                QMessageBox.critical(self, "Error", f"Y coordinate is empty for {chamber_id}")
                return
            if not w_text.strip():
                QMessageBox.critical(self, "Error", f"Width is empty for {chamber_id}")
                return
            if not h_text.strip():
                QMessageBox.critical(self, "Error", f"Height is empty for {chamber_id}")
                return
        
        QMessageBox.information(self, "Success", "All validations passed!")

def main():
    app = QApplication(sys.argv)
    dialog = DebugChamberDialog()
    dialog.show()
    app.exec_()

if __name__ == "__main__":
    main()

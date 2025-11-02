"""
Trace plotter module for multiplex trial viewer.
Provides matplotlib canvas for PyQt to display fly location traces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal
from typing import Dict, List, Optional, Tuple
import pandas as pd


class TracePlotter(FigureCanvas):
    """
    Matplotlib canvas widget for displaying fly location traces.
    """
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        """
        Initialize the trace plotter.
        
        Args:
            parent: Parent widget
            width (int): Figure width
            height (int): Figure height
            dpi (int): Figure DPI
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Initialize plot
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Chamber Position (-100 to +100)')
        self.ax.set_title('Fly Location Traces')
        self.ax.grid(True, alpha=0.3)
        
        # Data storage
        self.trace_data = {}  # {chamber_id: {'time': [], 'position': []}}
        self.selected_chambers = set()
        self.current_time = 0
        self.time_indicator = None
        
        # Color mapping for chambers
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Setup initial plot
        self.setup_plot()
    
    def setup_plot(self):
        """
        Setup the initial plot appearance.
        """
        self.ax.set_xlim(0, 10)  # Will be updated when data is loaded
        self.ax.set_ylim(-100, 100)
        self.ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add legend placeholder
        self.ax.legend(loc='upper right')
        
        self.fig.tight_layout()
    
    def load_trace_data(self, frame_data: Dict[int, Dict[str, float]], fps: float):
        """
        Load trace data from processed video frames.
        
        Args:
            frame_data (Dict[int, Dict[str, float]]): Frame data from video processor
            fps (float): Video frame rate
        """
        self.trace_data = {}
        
        # Process frame data into time series
        for frame_num, fly_locations in frame_data.items():
            time_sec = frame_num / fps if fps > 0 else frame_num
            
            for chamber_id, position in fly_locations.items():
                if chamber_id not in self.trace_data:
                    self.trace_data[chamber_id] = {'time': [], 'position': []}
                
                if not np.isnan(position):
                    self.trace_data[chamber_id]['time'].append(time_sec)
                    self.trace_data[chamber_id]['position'].append(position)
        
        # Update plot limits
        if self.trace_data:
            max_time = max(max(data['time']) for data in self.trace_data.values() if data['time'])
            self.ax.set_xlim(0, max_time)
        
        print(f"Loaded trace data for {len(self.trace_data)} chambers")
    
    def update_selected_chambers(self, selected_chambers: set):
        """
        Update which chambers are selected for display.
        
        Args:
            selected_chambers (set): Set of chamber IDs to display
        """
        self.selected_chambers = selected_chambers.copy()
        self.plot_traces()
    
    def plot_traces(self):
        """
        Plot the selected chamber traces.
        """
        self.ax.clear()
        self.setup_plot()
        
        if not self.trace_data:
            self.draw()
            return
        
        # Plot traces for selected chambers
        for i, chamber_id in enumerate(sorted(self.selected_chambers)):
            if chamber_id in self.trace_data:
                data = self.trace_data[chamber_id]
                if data['time'] and data['position']:
                    color = self.colors[i % len(self.colors)]
                    self.ax.plot(data['time'], data['position'], 
                               color=color, label=chamber_id, linewidth=1.5, alpha=0.8)
        
        # Add current time indicator
        if self.current_time > 0:
            self.time_indicator = self.ax.axvline(x=self.current_time, color='red', 
                                                 linestyle='-', linewidth=2, alpha=0.8)
        
        # Update legend
        if self.selected_chambers:
            self.ax.legend(loc='upper right', fontsize=8)
        
        self.draw()
    
    def update_current_time(self, current_time: float):
        """
        Update the current time indicator.
        
        Args:
            current_time (float): Current time in seconds
        """
        self.current_time = current_time
        self.plot_traces()
    
    def export_plot(self, filename: str, format: str = 'png'):
        """
        Export the current plot to a file.
        
        Args:
            filename (str): Output filename
            format (str): File format ('png', 'pdf', 'svg')
        """
        try:
            self.fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
            print(f"Exported plot to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting plot: {e}")
            return False
    
    def get_chamber_list(self) -> List[str]:
        """
        Get list of available chambers.
        
        Returns:
            List[str]: List of chamber IDs
        """
        return sorted(self.trace_data.keys())


class TraceControlPanel(QWidget):
    """
    Control panel for trace plotter with chamber selection checkboxes.
    """
    
    chamber_selection_changed = pyqtSignal(set)
    
    def __init__(self, parent=None):
        """
        Initialize the control panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.chamber_checkboxes = {}
        self.selected_chambers = set()
        self.setup_ui()
    
    def setup_ui(self):
        """
        Setup the user interface.
        """
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Chamber Selection")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(title)
        
        # Create checkboxes for chambers 1-20
        for i in range(1, 21):
            chamber_id = f"chamber_{i}"
            checkbox = QCheckBox(f"Chamber {i}")
            checkbox.stateChanged.connect(self.on_chamber_toggled)
            self.chamber_checkboxes[chamber_id] = checkbox
            layout.addWidget(checkbox)
        
        # Select all / Clear all buttons
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all)
        button_layout.addWidget(clear_all_btn)
        
        layout.addLayout(button_layout)
        
        # Export button
        export_btn = QPushButton("Export Plot")
        export_btn.clicked.connect(self.export_plot)
        layout.addWidget(export_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def on_chamber_toggled(self):
        """
        Handle chamber checkbox toggle.
        """
        self.selected_chambers.clear()
        
        for chamber_id, checkbox in self.chamber_checkboxes.items():
            if checkbox.isChecked():
                self.selected_chambers.add(chamber_id)
        
        self.chamber_selection_changed.emit(self.selected_chambers)
    
    def select_all(self):
        """
        Select all chambers.
        """
        for checkbox in self.chamber_checkboxes.values():
            checkbox.setChecked(True)
    
    def clear_all(self):
        """
        Clear all chamber selections.
        """
        for checkbox in self.chamber_checkboxes.values():
            checkbox.setChecked(False)
    
    def export_plot(self):
        """
        Export the current plot.
        """
        # This would be connected to the main application's export functionality
        print("Export plot requested")


def test_trace_plotter():
    """
    Test function for trace plotter functionality.
    """
    print("Testing trace plotter...")
    
    # Create mock data
    frame_data = {
        0: {'chamber_1': 50.0, 'chamber_2': -30.0},
        1: {'chamber_1': 45.0, 'chamber_2': -25.0},
        2: {'chamber_1': 40.0, 'chamber_2': -20.0},
        10: {'chamber_1': 30.0, 'chamber_2': -10.0},
        20: {'chamber_1': 20.0, 'chamber_2': 0.0},
        30: {'chamber_1': 10.0, 'chamber_2': 10.0}
    }
    
    # Test trace plotter
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Create trace plotter
    plotter = TracePlotter()
    plotter.load_trace_data(frame_data, fps=10.0)
    
    # Test chamber selection
    plotter.update_selected_chambers({'chamber_1', 'chamber_2'})
    plotter.update_current_time(1.0)
    
    # Test export
    export_success = plotter.export_plot("test_trace.png")
    print(f"Export test: {export_success}")
    
    # Clean up
    if export_success and os.path.exists("test_trace.png"):
        os.path.remove("test_trace.png")
        print("Cleaned up test plot file")
    
    print("Trace plotter test completed successfully!")
    return True


if __name__ == "__main__":
    import os
    test_trace_plotter()

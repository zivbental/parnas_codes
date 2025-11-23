"""
Plot Manager

Orchestrates all plotting operations.

This module provides a centralized plot manager that coordinates the generation
and display of all visualization figures. It delegates to specialized plotters
for different plot types (traces, summary statistics) and manages figure sizing
and display.
"""

import matplotlib.pyplot as plt
import logging
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData
    from ..models.metrics import MetricsData

from .traces import TracePlotter
from .summary import SummaryPlotter

logger = logging.getLogger(__name__)


class PlotManager:
    """
    Manages all plotting operations for the analysis pipeline.
    
    This class coordinates the generation of visualization figures by delegating
    to specialized plotters. It handles figure sizing, display, and cleanup,
    providing a unified interface for all plotting needs.
    """
    
    def __init__(self):
        """
        Initialize the plot manager with specialized plotters.
        
        The manager creates instances of trace and summary plotters, which handle
        the specific details of each plot type.
        """
        # Trace plotter: generates position trace plots with annotations
        self.trace_plotter = TracePlotter()
        # Summary plotter: generates summary statistics plots
        self.summary_plotter = SummaryPlotter()
    
    def generate_all_plots(
        self,
        trial_data: 'TrialData',
        metrics_data: 'MetricsData'
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Generate all visualization figures for the analysis.
        
        This method creates two figure windows:
        1. Trace figure: Shows position traces for all flies with digital outputs
           and annotations showing decision changes
        2. Summary figure: Shows scatter plots of behavioral metrics, before/after
           comparisons, and mean speed across epochs
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing position tracking and digital outputs.
        metrics_data : MetricsData
            Metrics data structure containing calculated behavioral metrics.
        
        Returns:
        --------
        Tuple[plt.Figure, plt.Figure]
            (trace_figure, summary_figure) containing the two generated figures.
            Figures are configured with appropriate sizes but not yet displayed.
        """
        logger.info("Generating plots...")
        
        # Generate trace plot: position trajectories with annotations
        fig1 = self.trace_plotter.plot(trial_data, metrics_data)
        # Generate summary plot: statistical summaries and comparisons
        fig2 = self.summary_plotter.plot(trial_data, metrics_data)
        
        # Configure figure sizes for optimal display
        # Trace plot: wider to show long time series
        fig1.set_size_inches(12, 8)
        # Summary plot: very wide to accommodate 4 subplots side-by-side
        fig2.set_size_inches(16, 4)
        
        return fig1, fig2
    
    def show_plots(self) -> None:
        """
        Display all generated plots.
        
        This method calls matplotlib's show() function, which displays all
        open figures and blocks execution until the user closes the figure windows.
        This is useful for interactive analysis where users want to inspect plots.
        """
        plt.show()
    
    def close_all(self) -> None:
        """
        Close all open plot figures.
        
        This method closes all matplotlib figures, freeing up memory and resources.
        Useful for batch processing or when plots are saved to files instead of displayed.
        """
        plt.close('all')


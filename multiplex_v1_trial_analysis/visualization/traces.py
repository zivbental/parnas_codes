"""
Trace Plotter

Plots position traces with annotations.

This module generates detailed position trace plots showing the trajectory
of each fly over time, along with digital output signals (odor delivery)
and shock data. The plot includes annotations showing decision preferences
before and after training, making it easy to visualize learning effects.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData
    from ..models.metrics import MetricsData

from ..utils.data_cleaning import DataCleaner

logger = logging.getLogger(__name__)


class TracePlotter:
    """
    Plots position traces with digital outputs and annotations.
    
    This class generates comprehensive trace plots that visualize:
    - Digital output signals (odor delivery control)
    - Shock delivery (EE data)
    - Position trajectories for all flies
    - Annotations showing decision preferences and changes
    """
    
    def __init__(self):
        """
        Initialize the trace plotter.
        
        Creates a data cleaner instance for preprocessing position data
        before plotting (removing discontinuities).
        """
        self.data_cleaner = DataCleaner()
    
    def plot(
        self,
        trial_data: 'TrialData',
        metrics_data: 'MetricsData'
    ) -> plt.Figure:
        """
        Generate position trace plot with annotations.
        
        This method creates a comprehensive plot showing:
        1. Digital output signals (6 lines: left/right, air/MCH/OCT)
        2. Shock delivery signals (EE data, one line per fly)
        3. Position traces (one line per fly, vertically offset)
        4. Text annotations showing decision preferences before/after/change
        
        The plot uses vertical offsetting to separate multiple flies visually,
        with each fly's trace positioned at a different vertical level.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing:
            - Time: Time vector for x-axis
            - Digital outputs: LEFTAIR, LEFTMCH, LEFTOCT, etc.
            - EE: Shock delivery data
            - cX: Position tracking data
        metrics_data : MetricsData
            Metrics data structure containing:
            - decsBefore: Decision preferences before training
            - decsAfter: Decision preferences after training
            - decisionsChanges: Change in decision preferences
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure handle for the generated plot.
        """
        # Initialize time to start at zero (relative time convention)
        trial_data.data['Time'][0, 0] = 0
        
        # Remove discontinuities from position data
        # This fills tracking gaps with previous valid positions
        cXrd, badidxs = self.data_cleaner.remove_discontinuities(trial_data.data['cX'])
        
        # Prepare position data for plotting with vertical offsetting
        # Each fly's trace is offset vertically so they don't overlap
        rows = cXrd.shape[0]
        nflies = trial_data.nflies
        # Create vertical offsets: each fly gets 2 units of vertical space
        stepper = np.arange(2, 2 * nflies + 1, 2)
        spacer = np.tile(stepper, (rows, 1))
        
        # Add offset to position data for visualization
        # First offset: base vertical spacing
        cXForPlotting1a = spacer + cXrd
        # Second offset: additional spacing to separate from digital outputs
        cXForPlotting1b = cXForPlotting1a + 14
        
        # Prepare shock (EE) data for plotting
        # Extract shock data for all flies and apply same vertical offsetting
        EEspecial = trial_data.data['EE'][:, :nflies]
        EEforPlotting1a = spacer + EEspecial
        EEforplotting1b = EEforPlotting1a + 14
        
        # Prepare digital output signals for plotting
        # Combine all 6 digital outputs (left/right, air/MCH/OCT)
        digitalOuts = np.column_stack([
            trial_data.data['LEFTAIR'],
            trial_data.data['LEFTMCH'],
            trial_data.data['LEFTOCT'],
            trial_data.data['RIGHTAIR'],
            trial_data.data['RIGHTMCH'],
            trial_data.data['RIGHTOCT']
        ])
        # Create vertical offsets for digital outputs (6 signals, spaced 2 units apart)
        stepperCont = np.arange(2, 2 * 6 + 1, 2)
        spacerCont = np.tile(stepperCont, (rows, 1))
        controlOutsforPlot = spacerCont + digitalOuts
        
        # Combine all data for plotting: digital outputs + position traces
        allForPlot1 = np.column_stack([controlOutsforPlot, cXForPlotting1b])
        
        # Create figure
        fig1 = plt.figure()
        
        # Plot digital output signals (6 lines at the bottom)
        outputlines = 6
        plt.plot(trial_data.data['Time'], allForPlot1[:, :outputlines])
        
        # Plot shock delivery signals (one line per fly, gray color)
        # These show when shocks were delivered during the experiment
        plt.plot(trial_data.data['Time'], EEforplotting1b, color=[0.8, 0.8, 0.8])
        
        # Plot position traces for each fly with annotations
        for lineidx in range(nflies):
            # Plot the position trace for this fly
            plt.plot(trial_data.data['Time'], allForPlot1[:, lineidx + outputlines])
            
            # Add text annotations showing decision metrics
            # Before: decision preference at start of experiment
            bef = str(metrics_data['decsBefore'][lineidx])
            # After: decision preference at end of experiment
            aft = str(metrics_data['decsAfter'][lineidx])
            # Change: difference between after and before (learning measure)
            change = str(metrics_data['decisionsChanges'][lineidx])
            
            # Position annotations at specific x-coordinates (time points)
            # Y-coordinate matches the fly's vertical offset
            plt.text(60, lineidx * 2 + 14, bef)
            plt.text(1080, lineidx * 2 + 14, aft)
            # Change value highlighted with green background
            plt.text(1140, lineidx * 2 + 14, change, backgroundcolor=[0.7, 0.9, 0.7])
        
        # Add mean values row at the bottom
        # Shows aggregate statistics across all flies
        meanbef = str(np.nanmean(metrics_data['decsBefore']))
        meanaft = str(np.nanmean(metrics_data['decsAfter']))
        meanchange = str(np.nanmean(metrics_data['decisionsChanges']))
        
        plt.text(60, 12, meanbef)
        plt.text(1080, 12, meanaft)
        plt.text(1140, 12, meanchange, backgroundcolor=[0.7, 0.9, 0.7])
        
        # Set axis limits and labels
        plt.xlim([0, np.max(trial_data.data['Time'])])
        plt.ylim([0, 2 * nflies + 16])  # Y-axis spans from 0 to top of all traces
        plt.xlabel('time')
        plt.ylabel('conditions and flies')
        plt.title('Position traces and digital outputs')
        
        return fig1


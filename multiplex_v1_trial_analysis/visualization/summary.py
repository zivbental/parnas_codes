"""
Summary Plotter

Creates summary plots for behavioral metrics.

This module generates summary visualization plots that provide an overview
of behavioral metrics across all flies. The plots include scatter plots
showing relationships between metrics, before/after comparisons, and
aggregate measures like mean speed across epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData
    from ..models.metrics import MetricsData

logger = logging.getLogger(__name__)


class SummaryPlotter:
    """
    Creates summary plots for behavioral analysis.
    
    This class generates a multi-panel figure with four subplots showing
    different aspects of the behavioral analysis:
    1. Scatter plot: Time change vs. decision change (learning correlation)
    2. Before/after comparison: Time ratio preferences
    3. Before/after comparison: Decision preferences
    4. Mean speed: Walking speed across epochs
    """
    
    def plot(
        self,
        trial_data: 'TrialData',
        metrics_data: 'MetricsData'
    ) -> plt.Figure:
        """
        Generate summary statistics plots.
        
        This method creates a four-panel figure visualizing key behavioral
        metrics and their relationships:
        
        1. Time vs. Decision Changes: Scatter plot showing correlation between
           time ratio changes and decision changes, with error bars and reference lines
        2. Time Ratio Before/After: Scatter plot comparing preferences at start
           and end of experiment, with reference lines showing learning thresholds
        3. Decision Ratio Before/After: Similar to time ratio plot but for decisions
        4. Mean Speed: Average walking speed across epochs
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure (used for nflies count).
        metrics_data : MetricsData
            Metrics data structure containing all calculated behavioral metrics.
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure handle for the generated summary plot.
        """
        fig2 = plt.figure()
        
        nflies = trial_data.nflies
        # Generate color map for distinguishing flies (if needed in future)
        colorcode2 = plt.cm.jet(np.linspace(0, 1, nflies))
        
        # Subplot 1: Scatter plot of time changes vs decision changes
        # This shows the correlation between two different measures of learning
        plt.subplot(1, 4, 1)
        circlesize = 55  # Marker size for scatter points
        plt.scatter(metrics_data['timeChanges'], metrics_data['decisionsChanges'], s=circlesize)
        
        plt.xlabel('Change in time ratio')
        plt.ylabel('Change in decision ratio')
        
        # Calculate mean and standard error for time changes
        # Only include flies with valid decision changes (not NaN)
        # This ensures consistent filtering across both metrics
        timChaMean = np.mean(metrics_data['timeChanges'][~np.isnan(metrics_data['decisionsChanges'])])
        timChaErr = np.nanstd(metrics_data['timeChanges']) / np.sqrt(np.sum(~np.isnan(metrics_data['decisionsChanges'])))
        
        # Create error line for plotting: three points showing mean ± error
        # This creates a horizontal line showing the confidence interval
        # Points: [mean - error, mean, mean + error]
        timChaErrLine = np.array([timChaMean - timChaErr, timChaMean, timChaMean + timChaErr])
        
        # Calculate mean decision change
        decChaMean = np.nanmean(metrics_data['decisionsChanges'])
        # Create three copies of the mean for plotting against the error line
        # This creates a vertical line at the mean decision change value
        decChaMean2 = np.repeat(decChaMean, 3)
        decChaErr = np.nanstd(metrics_data['decisionsChanges']) / np.sqrt(np.sum(~np.isnan(metrics_data['decisionsChanges'])))
        
        # Plot error bars and mean point (blue star)
        plt.errorbar(timChaMean, decChaMean, decChaErr, fmt='b*-')
        # Plot cross-hair lines showing mean ± error for both axes
        plt.plot(timChaErrLine, decChaMean2, 'b-')
        
        # Add reference lines at zero (no change)
        # Vertical line: x = 0 (no time ratio change)
        xline1 = 0
        xline2 = np.arange(-120, 121, 2)
        # Horizontal line: y = 0 (no decision change)
        yline1 = 0
        yline2 = np.arange(-120, 121, 2)
        plt.plot(np.full_like(xline2, xline1), xline2, 'k:')  # Vertical zero line
        plt.plot(yline2, np.full_like(yline2, yline1), 'k:')  # Horizontal zero line
        plt.axis([-100, 100, -100, 100])
        
        # Subplot 2: Time ratio before vs after
        # Shows how time preferences changed from first to last trial
        trialLocs = np.where(metrics_data['epochtags'] == 1)[0]
        ntrials = len(trialLocs)
        
        plt.subplot(1, 4, 2)
        if ntrials >= 2:
            # Scatter plot: each point is one fly
            # X-axis: preference before training, Y-axis: preference after training
            plt.scatter(metrics_data['trainedOdorTimePC'][trialLocs[0], :],
                       metrics_data['trainedOdorTimePC'][trialLocs[1], :],
                       s=circlesize)
            plt.axis([0, 100, 0, 100])
            plt.xlabel('Time ratio before')
            plt.ylabel('Time ratio after')
            
            # Add reference lines to visualize learning
            # Diagonal line (y=x): no change
            # Lines above diagonal: increased preference (learning)
            # Lines below diagonal: decreased preference
            xline3 = np.arange(1, 151, 50)
            yline3 = xline3  # Diagonal: no change
            yline4 = xline3 - 40  # Below diagonal: threshold for learning
            yline5 = xline3 + 40  # Above diagonal: threshold for learning
            plt.plot(xline3, yline3, 'k', xline3, yline4, 'k', xline3, yline5, 'k')
        
        # Subplot 3: Decision ratio before vs after
        # Similar to time ratio plot but for decision preferences
        plt.subplot(1, 4, 3)
        if ntrials >= 2:
            # Scatter plot: decision preferences before vs after
            plt.scatter(metrics_data['trainedOdorDecsPC'][trialLocs[0], :],
                       metrics_data['trainedOdorDecsPC'][trialLocs[1], :],
                       s=circlesize)
            plt.axis([0, 100, 0, 100])
            plt.xlabel('Decision ratio before')
            plt.ylabel('Decision ratio after')
            
            # Add same reference lines as time ratio plot
            xline3 = np.arange(1, 151, 50)
            yline3 = xline3  # Diagonal: no change
            yline4 = xline3 - 40  # Below diagonal threshold
            yline5 = xline3 + 40  # Above diagonal threshold
            plt.plot(xline3, yline3, 'k', xline3, yline4, 'k', xline3, yline5, 'k')
        
        # Subplot 4: Mean walking speed across epochs
        # Shows how activity level changes over the course of the experiment
        plt.subplot(1, 4, 4)
        speedMean = metrics_data['speedMean']
        # Plot mean speed averaged across all flies for each epoch
        plt.scatter(np.arange(1, speedMean.shape[0] + 1), np.nanmean(speedMean, axis=1), s=circlesize, color='k')
        plt.xlabel('Epoch')
        plt.ylabel('Mean walking speed (mm/s)')
        
        return fig2


"""
Statistics Calculator

Calculates statistical summaries of behavioral metrics.

This module computes aggregate statistics across all flies in the experiment,
providing summary measures of learning and behavior. It handles missing data
by filtering out invalid flies (those with NaN values) before computing
statistics, ensuring that summary measures are based only on valid data.
"""

import numpy as np
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.metrics import MetricsData

logger = logging.getLogger(__name__)


class StatisticsCalculator:
    """
    Calculates statistical summaries from metrics data.
    
    This class provides methods for computing aggregate statistics across flies,
    including means, standard errors, and other summary measures. It handles
    missing data gracefully by filtering out invalid values before calculation.
    """
    
    @staticmethod
    def calculate_summary(metrics_data: 'MetricsData') -> dict:
        """
        Calculate summary statistics across all flies.
        
        This method computes aggregate measures of learning and behavior:
        - Mean decision change: average learning across valid flies
        - Mean time ratio change: average preference change across valid flies
        - Standard errors: measures of variability in the means
        - Mean total decisions: average activity level
        
        The method filters out flies with invalid data (NaN values) before
        computing statistics. This ensures that summary measures are based
        only on flies that made sufficient decisions to be included in the
        analysis.
        
        The standard error calculation uses sample standard deviation (dividing
        by N-1 instead of N) to provide unbiased estimates of population
        variability when working with samples.
        
        Parameters:
        -----------
        metrics_data : MetricsData
            Metrics data structure containing:
            - decisionsChanges: Change in decision preferences per fly
            - timeChanges: Change in time preferences per fly
            - decisionsTotal: Total decisions per epoch/fly
        
        Returns:
        --------
        dict
            Dictionary containing:
            - dec_cha_mean: Mean decision change
            - dec_cha_sem: Standard error of mean decision change
            - tim_cha_mean: Mean time ratio change
            - tim_cha_sem: Standard error of mean time ratio change
            - mean_total_decs: Mean total decisions per valid fly
            - flies_used: Indices of flies included in statistics (1-based)
            - dec_cha_new: Filtered decision changes (NaN values removed)
            - tim_cha_new: Filtered time changes (NaN values removed)
        """
        # Filter out flies with invalid decision change data
        # NaN values indicate flies that didn't make sufficient decisions
        dec_cha_raw = metrics_data['decisionsChanges']
        dec_cha_nan = np.isnan(dec_cha_raw)
        dec_nans_locs = np.where(~dec_cha_nan)[0]
        dec_cha_new = dec_cha_raw[dec_nans_locs]
        
        logger.info("Decision changes:")
        logger.info(dec_cha_new)
        
        # Create 1-based fly indices for reporting (flies are numbered 1-20)
        flies_used = np.arange(1, 21)[dec_nans_locs]
        logger.info("Flies used:")
        logger.info(flies_used)
        
        # Calculate mean and standard error for decision changes
        # Use sample standard deviation (ddof=1) for unbiased estimation
        # Standard error = standard deviation / sqrt(sample size)
        dec_cha_mean = np.mean(dec_cha_new)
        dec_cha_sem = np.std(dec_cha_new, ddof=1) / np.sqrt(len(dec_cha_new))
        
        # Filter time changes to match decision changes
        # This ensures we only analyze time changes for flies that had valid
        # decision changes, maintaining consistency in the analysis
        tim_cha_raw = metrics_data['timeChanges']
        tim_cha_new = tim_cha_raw[dec_nans_locs]
        
        # Calculate mean and standard error for time changes
        # Same approach: sample standard deviation for unbiased estimation
        tim_cha_mean = np.mean(tim_cha_new)
        tim_cha_sem = np.std(tim_cha_new, ddof=1) / np.sqrt(len(tim_cha_new))
        
        logger.info("Time changes:")
        logger.info(tim_cha_new)
        
        # Calculate mean total decisions across all valid flies
        # This provides a measure of overall activity level in the experiment
        mean_total_decs = np.sum(metrics_data['decisionsTotal']) / np.sum(~np.isnan(metrics_data['decisionsChanges']))
        
        summary = {
            'dec_cha_mean': dec_cha_mean,
            'dec_cha_sem': dec_cha_sem,
            'tim_cha_mean': tim_cha_mean,
            'tim_cha_sem': tim_cha_sem,
            'mean_total_decs': mean_total_decs,
            'flies_used': flies_used,
            'dec_cha_new': dec_cha_new,
            'tim_cha_new': tim_cha_new
        }
        
        logger.info("\nSummary:")
        logger.info(f"Mean decision change: {dec_cha_mean}")
        logger.info(f"Decision change SEM: {dec_cha_sem}")
        logger.info(f"Mean time ratio change: {tim_cha_mean}")
        logger.info(f"Time ratio SEM: {tim_cha_sem}")
        logger.info(f"Mean # decisions: {mean_total_decs}")
        
        return summary


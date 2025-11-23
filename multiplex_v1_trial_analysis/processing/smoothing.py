"""
Smoothing Processor

Applies Savitzky-Golay smoothing filter to tracking data to reduce artifacts.

This module handles the smoothing of position tracking data to reduce noise
and artifacts introduced by video tracking systems. The Savitzky-Golay filter
is a polynomial smoothing technique that preserves local features better than
simple moving averages while still reducing high-frequency noise.
"""

import numpy as np
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData

from ..utils.filters import SavitzkyGolayFilter
from ..config import Config

logger = logging.getLogger(__name__)


class SmoothingProcessor:
    """
    Processes and smooths position tracking data.
    
    This class applies a Savitzky-Golay polynomial filter to the position
    tracking data for each fly. The filter reduces noise from video tracking
    while preserving important behavioral features like directional changes.
    Smoothing is applied independently to each fly's trajectory.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the smoothing processor with filter configuration.
        
        The filter is configured with a polynomial order (typically 2) and
        differentiation order of 0 (which means smoothing, not differentiation).
        The actual frame length (window size) is determined dynamically based
        on the data length and configured span percentage.
        
        Parameters:
        -----------
        config : Config
            Configuration object containing smoothing parameters:
            - smoothing_order: Polynomial order for the filter (default: 2)
            - span_pc: Percentage of data length to use as window size (default: 0.005)
        """
        self.config = config
        # Create filter with polynomial order from config
        # Differentiation order 0 means smoothing (not computing derivatives)
        self.filter = SavitzkyGolayFilter(
            polynomial_order=config.smoothing_order,
            differentiation_order=0  # 0 = smoothing, higher values = derivatives
        )
    
    def process(self, trial_data: 'TrialData') -> None:
        """
        Apply smoothing filter to position data for all flies.
        
        This method processes each fly's position trajectory independently,
        applying a Savitzky-Golay filter with a window size determined by
        the configured span percentage. The smoothing is applied in-place,
        modifying the cX (position) data directly.
        
        The window size is automatically adjusted to be odd (required by the
        filter algorithm), ensuring symmetric filtering around each point.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing position data in trial_data.data['cX'].
            This structure is modified in place: the cX array is replaced with
            smoothed values. The original unsmoothed data is not preserved.
        """
        logger.info("Applying smoothing filter to position data...")
        
        # Get the configured smoothing window size as a percentage of data length
        # Smaller percentages create tighter smoothing windows
        span_pc = self.config.span_pc
        
        nflies = trial_data.nflies
        
        # Process each fly's trajectory independently
        for i in range(nflies):
            # Calculate the window size (span) as a percentage of the data length
            # Use floor to ensure we get an integer number of frames
            span1 = int(np.floor(len(trial_data.data['cX'][:, i]) * span_pc))
            
            # Savitzky-Golay filter requires an odd window size for symmetric filtering
            # If span1 is even, subtract 1 to make it odd
            # Formula: span2 = span1 - (1 if span1 is even, else 0)
            span2 = span1 - (1 - (span1 % 2))
            
            # Apply the smoothing filter to this fly's position data
            # The filter processes the entire trajectory in one pass
            trial_data.data['cX'][:, i] = self.filter.apply(
                trial_data.data['cX'][:, i],
                frame_length=span2
            )
        
        logger.debug(f"Smoothed position data for {nflies} flies")


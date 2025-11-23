"""
Time Ratio Calculator

Calculates the ratios of time spent in the two odors.

This module computes the percentage of time a fly spends on the left side of
the experimental chamber. The calculation uses a weighted sum approach where
later time points contribute more to the ratio, reflecting the temporal
distribution of position preferences throughout the measurement period.
"""

import numpy as np
import logging

from ..utils.location import TernaryLocationClassifier
from ..config import Config

logger = logging.getLogger(__name__)


class TimeRatioCalculator:
    """
    Calculates time ratio metrics from position data.
    
    This class computes the percentage of time spent on the left side of the
    chamber using a weighted calculation method. Unlike simple counting, this
    method sums the frame indices (positions in time) where the fly is on the
    left, giving more weight to later time points. This approach provides a
    metric that reflects both the duration and temporal distribution of
    position preferences.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the time ratio calculator with location classification settings.
        
        The calculator uses a ternary location classifier to determine whether
        each frame shows the fly on the left, center, or right. The choicepoint
        halfwidth determines the size of the central "mixed odor" zone that
        is excluded from time ratio calculations.
        
        Parameters:
        -----------
        config : Config
            Configuration object containing:
            - choicepoint_halfwidth: Half-width of the central choicepoint zone
              (positions within [-halfwidth, +halfwidth] are considered center)
        """
        self.config = config
        # Create classifier for determining left/center/right positions
        # The classifier uses the configured choicepoint halfwidth to define zones
        self.classifier = TernaryLocationClassifier(
            choicepoint_halfwidth=config.choicepoint_halfwidth
        )
    
    def calculate(self, position_vector: np.ndarray) -> float:
        """
        Calculate the percentage of time spent on the left side of the chamber.
        
        This method uses a weighted sum approach: it finds all frames where the
        fly is on the left side, converts the 0-based frame indices to 1-based
        (to match the original calculation method), and sums these indices.
        The same is done for all frames where the fly is in a decision zone
        (left or right, excluding center). The ratio is the sum of left indices
        divided by the sum of all decision indices.
        
        This approach gives more weight to later time points, meaning that
        preferences expressed later in the trial contribute more to the final
        ratio than earlier preferences. This reflects the temporal dynamics
        of learning and preference formation.
        
        Parameters:
        -----------
        position_vector : np.ndarray
            Position vector for a single fly across time (single column array).
            Values are typically normalized to the range [-1, 1] where -1 is
            left side, +1 is right side, and 0 is center.
        
        Returns:
        --------
        float
            Percentage of weighted time spent on left side, ranging from 0-100.
            Returns NaN if there are no valid decision frames (fly never left center).
        """
        # Ensure input is in float32 format for consistent numerical precision
        # This matches the data type used throughout the analysis pipeline
        position_vector = np.asarray(position_vector, dtype=np.float32).flatten()
        
        # Classify each frame as left (-1), center (0), or right (+1)
        # The classifier uses the choicepoint_halfwidth to determine zones
        bvec = self.classifier.classify(position_vector)
        
        # Calculate weighted sum for left side positions
        # We find all frames where bvec == -1 (fly is on left side)
        indices_left = np.where(bvec == -1)[0]
        if len(indices_left) > 0:
            # Convert 0-based indices to 1-based for the weighted calculation
            # This means frame 0 becomes position 1, frame 1 becomes position 2, etc.
            # The sum of these 1-based indices gives more weight to later frames
            indices_left = indices_left + 1
            leftindex = np.sum(indices_left)
        else:
            # No frames on left side: set sum to zero
            leftindex = 0
        
        # Calculate weighted sum for all decision frames (left or right, excluding center)
        # This includes all frames where the fly is in a decision zone (abs(bvec) == 1)
        indices_total = np.where(np.abs(bvec) == 1)[0]
        if len(indices_total) > 0:
            # Convert to 1-based indices and sum them
            # This gives the total "weight" of all decision-making time
            indices_total = indices_total + 1
            total = np.sum(indices_total)
        else:
            # No decision frames: fly never left center zone
            total = 0
        
        # Calculate percentage: left weighted sum / total weighted sum * 100
        # If total is zero (no decisions), return NaN to indicate invalid result
        time_ratio_left_pc = (leftindex / total) * 100 if total > 0 else np.nan
        
        return time_ratio_left_pc


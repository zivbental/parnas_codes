"""
Decision Counter

Counts decisions (transitions and reversals) from position data using ternary logic.

This module implements a sophisticated decision detection algorithm that identifies
when flies make meaningful choices by moving between odor zones. The algorithm
detects two types of decisions:
1. Fullway transitions: Moving completely from one side to the other
2. Reversals: Entering the center zone and then exiting to the opposite side

The detection uses pattern matching on transition sequences encoded as characters,
allowing efficient identification of complex movement patterns.
"""

import numpy as np
import logging
from typing import Tuple

from ..utils.location import TernaryLocationClassifier
from ..config import Config

logger = logging.getLogger(__name__)


class DecisionCounter:
    """
    Counts behavioral decisions from position tracking data.
    
    This class analyzes position trajectories to detect when flies make decisions
    by crossing between odor zones. Decisions are classified as left or right
    choices, and the algorithm distinguishes between fullway transitions (direct
    cross-chamber movement) and reversals (center entry followed by opposite-side
    exit). The algorithm uses a filtering approach to ensure only flies with
    sufficient activity are included in the analysis.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the decision counter with classification and filtering parameters.
        
        The counter uses two location classifiers: one for precise decision detection
        and one with a wider filter zone for validation. The wider filter ensures
        that only flies with substantial movement are considered valid.
        
        Parameters:
        -----------
        config : Config
            Configuration object containing:
            - choicepoint_halfwidth: Half-width for precise location classification
            - decision_halfwidth: Half-width for filter zone (wider than choicepoint)
            - min_decisions: Minimum decisions required for valid analysis
        """
        self.config = config
        # Minimum number of decisions required in the filter zone for valid analysis
        self.min_decisions = config.min_decisions
        # Half-width for the filter zone (wider than choicepoint for validation)
        self.halfwidth = config.decision_halfwidth
        # Classifier for precise location detection (uses choicepoint_halfwidth)
        self.classifier = TernaryLocationClassifier(
            choicepoint_halfwidth=config.choicepoint_halfwidth
        )
    
    def count_decisions(
        self,
        position_vector: np.ndarray,
        gaussfwhm: float = None
    ) -> Tuple[float, float]:
        """
        Calculate decision metrics from position trajectory data.
        
        This method implements a sophisticated decision detection algorithm that:
        1. Classifies each frame as left (-2), center (0), or right (+3)
        2. Detects transitions between zones using difference calculation
        3. Encodes transitions as characters for pattern matching
        4. Identifies specific patterns that indicate decisions:
           - 'op': Left to right fullway (right decision)
           - 'jk': Right to left fullway (left decision)
           - 'ok': Left reversal (left decision)
           - 'jp': Right reversal (left decision)
        5. Validates decisions using a wider filter zone
        6. Returns NaN if the fly has insufficient activity
        
        The algorithm uses two classification zones: a precise zone for decision
        detection and a wider filter zone for validation. This ensures only
        flies with substantial movement are included in the analysis.
        
        Parameters:
        -----------
        position_vector : np.ndarray
            Position vector for a single fly across time (single column array).
            Data should already be smoothed (typically by Savitzky-Golay filter).
            Values are normalized to [-1, 1] range.
        gaussfwhm : float, optional
            Gaussian full-width at half-maximum parameter. Not currently used
            but kept for compatibility with the original interface.
        
        Returns:
        --------
        decisions_left_pc : float
            Percentage of decisions made to the left side (0-100).
            Returns NaN if the fly has fewer than min_decisions total decisions
            in the filter zone, indicating insufficient activity.
        decisions_total : float
            Total number of decisions detected (fullway + reversals).
            Returns NaN if the fly has insufficient activity (same condition as above).
        """
        position_vector = np.asarray(position_vector)
        
        # Validate input: must be a single column vector
        if position_vector.ndim > 1 and position_vector.shape[1] > 1:
            raise ValueError('Input data must be a single column vector')
        
        # Flatten to ensure 1D array
        position_vector = position_vector.flatten()
        
        # Use the smoothed position data directly (smoothing already applied upstream)
        x = position_vector
        
        # Step 1: Classify positions using precise choicepoint zone
        # Convert to ternary: -1 (left), 0 (center), 1 (right)
        ternary_position_data = self.classifier.classify(x)
        # Re-map values to create unique transition codes:
        # Right (1) -> 3, Left (-1) -> -2, Center (0) stays 0
        # This creates distinct differences for pattern matching
        ternary_position_data[ternary_position_data == 1] = 3
        ternary_position_data[ternary_position_data == -1] = -2
        
        # Step 2: Classify positions using wider filter zone (for validation)
        # This uses a larger threshold to ensure the fly moved substantially
        ternary_filter_data = self.classifier.classify_filtered(x, self.halfwidth)
        # Apply same re-mapping for consistency
        ternary_filter_data[ternary_filter_data == 1] = 3
        ternary_filter_data[ternary_filter_data == -1] = -2
        
        # Step 3: Detect transitions between zones
        # Compute differences between consecutive frames to find zone changes
        t2 = np.diff(ternary_position_data)
        # Keep only non-zero differences (actual transitions)
        transvec = t2[t2 != 0]
        
        # Same for filter zone (for validation counts)
        t2f = np.diff(ternary_filter_data)
        transvecf = t2f[t2f != 0]
        
        # Step 4: Encode transitions as characters for pattern matching
        # Convert transition values to ASCII characters using 'm' (109) as base
        # This creates a string where each character represents a transition
        # Example: transition value +2 becomes character 'o' (109+2=111)
        tr2 = ''.join([chr(int(109 + val)) for val in transvec])
        tr2f = ''.join([chr(int(109 + val)) for val in transvecf])
        
        # Step 5: Count decision patterns in precise zone
        # Pattern meanings:
        # 'op': Left(-2) to Right(+3) = difference +5, fullway transition (right decision)
        # 'jk': Right(+3) to Left(-2) = difference -5, fullway transition (left decision)
        # 'ok': Left(-2) to Center(0) then Center(0) to Right(+3) = reversal (left decision)
        # 'jp': Right(+3) to Center(0) then Center(0) to Left(-2) = reversal (left decision)
        Left2RightFull = tr2.count('op')  # Fullway right decision
        Right2LeftFull = tr2.count('jk')  # Fullway left decision
        LeftReversal = tr2.count('ok')     # Left reversal (left decision)
        RightReversal = tr2.count('jp')    # Right reversal (left decision)
        
        # Step 6: Count decision patterns in filter zone (for validation)
        # Same patterns but using the wider filter zone to ensure substantial movement
        Left2RightFullf = tr2f.count('op')
        Right2LeftFullf = tr2f.count('jk')
        LeftReversalf = tr2f.count('ok')
        RightReversalf = tr2f.count('jp')
        
        # Step 7: Aggregate decisions
        # Left decisions include: fullway right-to-left AND reversals (both types)
        LeftDecisions = Right2LeftFull + LeftReversal
        # Right decisions include: fullway left-to-right AND right reversals
        RightDecisions = Left2RightFull + RightReversal
        TotalDecisions = LeftDecisions + RightDecisions
        
        # Validation counts from filter zone
        LeftDecisionsf = Right2LeftFullf + LeftReversalf
        RightDecisionsf = Left2RightFullf + RightReversalf
        TotalDecisionsf = LeftDecisionsf + RightDecisionsf
        
        # Step 8: Validate and calculate percentage
        # Only return valid results if the fly has sufficient activity in filter zone
        # This filters out flies that barely moved or had tracking issues
        if TotalDecisionsf < self.min_decisions:
            # Insufficient activity: mark as invalid
            left_decisions_ratio_pc = np.nan
            decisions_total = np.nan
        else:
            # Sufficient activity: calculate percentage
            # Check for edge case where no decisions detected despite validation
            if TotalDecisions == 0:
                left_decisions_ratio_pc = np.nan
            else:
                # Calculate percentage: left decisions / total decisions * 100
                left_decisions_ratio_pc = (LeftDecisions / TotalDecisions) * 100
        
        # Always return total decisions (even if NaN for invalid cases)
        decisions_total = TotalDecisions
        
        return left_decisions_ratio_pc, decisions_total


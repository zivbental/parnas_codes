"""
Epoch Detector

Defines epochs in behavioral dataset that fall between odor changes and logging breaks.

This module identifies distinct experimental periods (epochs) in the behavioral data
by detecting changes in odor delivery and gaps in data logging. Epochs are then
classified based on their odor configuration and shock delivery pattern into
categories: wait periods, trial periods, and training periods (classical or operant).
"""

import numpy as np
import logging
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData

from ..config import Config

logger = logging.getLogger(__name__)


class EpochDetector:
    """
    Detects and classifies epochs in behavioral data.
    
    This class identifies distinct experimental periods by monitoring changes in
    digital output signals (which control odor delivery) and detecting breaks in
    data logging. Once epochs are identified, they are classified based on their
    odor configuration and whether shocks were delivered during that period.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the epoch detector with timing correction parameters.
        
        The detector uses a delta value to account for timing synchronization
        issues between different data streams. This offset ensures that odor state
        sampling occurs after the digital outputs have stabilized.
        
        Parameters:
        -----------
        config : Config
            Configuration object containing:
            - epoch_delta: Time offset (in frames) for sampling odor states
              (accounts for digital output settling time)
            - logging_break_time: Time threshold (in seconds) for detecting
              logging breaks that indicate new experimental periods
        """
        self.config = config
        # Time offset for sampling odor states (accounts for digital output settling)
        # Value of 30 frames = 1 second at 30 fps, allows outputs to stabilize
        self.delta = config.epoch_delta
        # Time threshold for detecting logging breaks (gaps in data collection)
        # Gaps longer than this indicate the start of a new experimental epoch
        self.break_time = config.logging_break_time
    
    def detect(
        self,
        trial_data: 'TrialData'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[int]]:
        """
        Detect and classify epochs in the behavioral dataset.
        
        This method identifies distinct experimental periods by:
        1. Detecting changes in digital output signals (odor delivery changes)
        2. Detecting large gaps in time data (logging breaks)
        3. Combining these transitions to define epoch boundaries
        4. Sampling odor states from each epoch
        5. Classifying epochs based on odor configuration and shock patterns
        
        Epochs represent periods of consistent experimental conditions. The boundaries
        are defined by transitions in odor delivery or breaks in data logging.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing:
            - Digital output signals (LEFTAIR, LEFTMCH, LEFTOCT, etc.)
            - Time data for detecting logging breaks
            - Shock data (SHOCK or EE) for epoch classification
        
        Returns:
        --------
        epochtags : np.ndarray
            Classification tags for each epoch:
            - 0: Wait period (no odors or neutral conditions)
            - 1: Trial period (asymmetric odors, no shock)
            - 2: Classical training (symmetric odors with shock)
            - 3: Operant training (asymmetric odors with shock)
        odor_states : np.ndarray
            String array encoding odor configuration for each epoch.
            Each string has 6 characters representing [LEFTAIR, LEFTMCH, LEFTOCT,
            RIGHTAIR, RIGHTMCH, RIGHTOCT] as '0' (off) or '1' (on).
        start_bins : np.ndarray
            Start frame indices for each epoch (1-based indexing).
            These indices point to the first frame of each epoch in the data array.
        end_bins : np.ndarray
            End frame indices for each epoch (1-based indexing).
            These indices point to the last frame of each epoch (inclusive).
        expt_tag : int or None
            Experiment type identifier:
            - 2: Classical conditioning experiment
            - 3: Operant conditioning experiment
            - None: Could not be determined
        """
        logger.info("Detecting epochs...")
        
        # Combine all digital control outputs into a single matrix
        # Each column represents one odor control signal (left/right, air/MCH/OCT)
        digital_outs_cat = np.column_stack([
            trial_data.data['LEFTAIR'],
            trial_data.data['LEFTMCH'],
            trial_data.data['LEFTOCT'],
            trial_data.data['RIGHTAIR'],
            trial_data.data['RIGHTMCH'],
            trial_data.data['RIGHTOCT']
        ])
        
        # Use only left-side odors for transition detection
        # Right-side ports are slower to respond, so we ignore them to avoid
        # false positives from delayed right-side transitions
        digital_outs_cat2 = digital_outs_cat[:, :3]
        
        # Detect changes in digital outputs by computing frame-to-frame differences
        # A non-zero difference indicates an odor delivery change occurred
        digital_outs_dif = np.diff(digital_outs_cat2, axis=0)
        
        # Take absolute value to capture both on->off and off->on transitions
        # This ensures we detect all changes regardless of direction
        digital_outs_dif_abs = np.abs(digital_outs_dif)
        
        # Sum across all three left-side odors to get total change magnitude
        # Any non-zero sum indicates at least one odor changed
        digital_outs_dif_abs_sum = np.sum(digital_outs_dif_abs, axis=1)
        
        # Find frame indices where transitions occurred (0-based indices)
        digital_outs_transitions_minus_one = np.where(digital_outs_dif_abs_sum > 0)[0]
        # Convert to 1-based indices for consistency with epoch boundary representation
        # The transition actually occurs between frames, so +1 gives the frame after change
        digital_outs_transitions = digital_outs_transitions_minus_one + 1
        
        # Detect logging breaks by finding large gaps in the time data
        # Large time differences indicate the data logging was paused/stopped
        time_diff = np.diff(trial_data.data['Time'].flatten())
        logging_breaks_minus_one = np.where(time_diff > self.break_time)[0]
        # Convert to 1-based: the break occurs after this frame
        logging_breaks = logging_breaks_minus_one + 1
        
        # Combine all epoch boundaries: both odor transitions and logging breaks
        # These define the edges between different experimental periods
        if len(logging_breaks) > 0:
            edges = np.concatenate([digital_outs_transitions, logging_breaks])
            # Sort to ensure chronological order
            internal_edges = np.sort(edges)
        else:
            # No logging breaks: epochs defined only by odor transitions
            internal_edges = digital_outs_transitions
        
        # Define start and end frames for each epoch
        # Epochs are defined as periods between consecutive edges
        # Start of first epoch is frame 0 (1-based: frame 1)
        # Subsequent epochs start at the previous edge
        # All indices are stored as 1-based for consistency with data representation
        start_bins = np.concatenate([[0], internal_edges[:-1]]) + 1
        # End of each epoch is the edge that marks the transition to next epoch
        end_bins = internal_edges
        
        # Sample odor states from each epoch
        # Add delta offset to account for digital output settling time
        # This ensures we sample after the outputs have stabilized
        # Use min() to prevent index out of bounds for the last epoch
        sample_indices = np.minimum(start_bins + self.delta, len(digital_outs_cat) - 1)
        # Convert 1-based indices to 0-based for array indexing
        sample_indices_0based = sample_indices - 1
        # Sample odor configuration from each epoch
        odor_states_num = digital_outs_cat[sample_indices_0based, :]
        # Convert numeric arrays to string representation
        # Each epoch gets a 6-character string: [LEFTAIR, LEFTMCH, LEFTOCT, RIGHTAIR, RIGHTMCH, RIGHTOCT]
        odor_states = []
        for row in odor_states_num:
            # Convert each value to '0' or '1' and concatenate
            odor_state = ''.join([str(int(x)) for x in row])
            odor_states.append(odor_state)
        odor_states = np.array(odor_states)
        
        nepochs = len(start_bins)
        
        # Determine which shock detection method to use
        # Older experiments use SHOCK field, newer ones use EE (operant shock) field
        shock_sum = np.sum(trial_data.data['SHOCK'])
        
        epochtags = np.zeros(nepochs)
        expt_tag = None
        
        if shock_sum > 0:
            # Use old-style shock classification method
            # This method relies on predefined odor state patterns
            for eidx in range(nepochs):
                odor_state = odor_states[eidx]
                epochtags[eidx] = self._classify_epoch_old_shock(odor_state)
        elif shock_sum == 0 and 'EE' in trial_data.data:
            # Use operant shock (EE) method for classification
            # First, sum shock data across all flies for each frame
            ee_row_sum = np.sum(trial_data.data['EE'], axis=1)
            # Then sum shocks within each epoch to get total shocks per epoch
            ee_epoch_sum = np.zeros(nepochs)
            for sIdx in range(nepochs):
                # Convert 1-based epoch boundaries to 0-based array indices
                # Epoch boundaries are inclusive: both start and end frames are included
                start_idx = int(start_bins[sIdx]) - 1  # Convert to 0-based
                end_idx = int(end_bins[sIdx])  # End is inclusive, so add 1 for Python slice
                # Sum all shocks in this epoch
                ee_epoch_sum[sIdx] = np.sum(ee_row_sum[start_idx:end_idx + 1])
            
            # Classify each epoch based on odor state and shock pattern
            for eidx in range(nepochs):
                odor_state = odor_states[eidx]
                tag, expt = self._classify_epoch_ee(odor_state, ee_epoch_sum[eidx])
                epochtags[eidx] = tag
                # Capture experiment type if determined
                if expt is not None:
                    expt_tag = expt
        else:
            # Fallback classification when shock data is unavailable
            # This is a simple heuristic: all non-wait epochs are classified as training
            logger.warning("Neither type of shock epoch definition worked, using fallback")
            for eidx in range(nepochs):
                if odor_states[eidx] == '000000':
                    # No odors present: wait period
                    epochtags[eidx] = 0
                else:
                    # Odors present: assume training
                    epochtags[eidx] = 2
            # Heuristic: mark first and last non-wait epochs as trials
            if nepochs > 1:
                epochtags[1] = 1
            epochtags[nepochs - 1] = 1
        
        logger.info(f"Detected {nepochs} epochs")
        
        return epochtags, odor_states, start_bins, end_bins, expt_tag
    
    def _classify_epoch_old_shock(self, odor_state: str) -> float:
        """
        Classify epoch using the old-style SHOCK-based method.
        
        This method uses a lookup table mapping specific odor state patterns
        to epoch types. The patterns represent different experimental conditions
        used in older experimental protocols.
        
        Parameters:
        -----------
        odor_state : str
            6-character string encoding odor configuration:
            [LEFTAIR, LEFTMCH, LEFTOCT, RIGHTAIR, RIGHTMCH, RIGHTOCT]
            Each character is '0' (off) or '1' (on).
        
        Returns:
        --------
        float
            Epoch classification tag:
            - 0: Wait period
            - 1: Trial period
            - 2: Training period
            - np.nan: Unknown/unrecognized pattern
        """
        # Lookup table mapping odor state patterns to epoch types
        # These patterns represent specific experimental conditions from older protocols
        epoch_map = {
            '000000': 0,  # No odors: wait
            '100100': 0,  # Symmetric air: wait
            '110001': 1,  # Asymmetric: trial
            '001110': 1,  # Asymmetric: trial
            '001010': 1,  # Asymmetric: trial
            '010001': 1,  # Asymmetric: trial
            '101010': 1,  # Asymmetric: trial
            '101110': 1,  # Asymmetric: trial
            '010101': 1,  # Asymmetric: trial
            '110101': 1,  # Asymmetric: trial
            '010010': 2,  # Symmetric MCH: training
            '001001': 2,  # Symmetric OCT: training
            '110110': 2,  # Symmetric: training
            '101101': 2,  # Symmetric: training
        }
        return epoch_map.get(odor_state, np.nan)
    
    def _classify_epoch_ee(
        self,
        odor_state: str,
        ee_epoch_sum: float
    ) -> Tuple[float, Optional[int]]:
        """
        Classify epoch using the operant shock (EE) method.
        
        This method classifies epochs based on two factors:
        1. Odor symmetry: whether left and right sides have the same odors
        2. Shock presence: whether shocks were delivered during this epoch
        
        The combination of these factors determines the epoch type and experiment type.
        
        Parameters:
        -----------
        odor_state : str
            6-character string encoding odor configuration.
            First 3 characters: left side [AIR, MCH, OCT]
            Last 3 characters: right side [AIR, MCH, OCT]
        ee_epoch_sum : float
            Total number of shocks delivered in this epoch (summed across all flies).
            Zero indicates no shocks, positive indicates shocks were delivered.
        
        Returns:
        --------
        Tuple[float, Optional[int]]
            (epoch_tag, expt_tag) where:
            - epoch_tag: 0=wait, 1=trial, 2=classical train, 3=operant train
            - expt_tag: 2=classical experiment, 3=operant experiment, None=not determined
        """
        if odor_state == '000000':
            # No odors present: wait period
            return 0, None
        elif ee_epoch_sum == 0 and odor_state[:3] == odor_state[3:]:
            # No shocks + symmetric odors: classical counter-training (no actual training)
            # This is a control condition where odors are balanced but no shock occurs
            return 2, None
        elif ee_epoch_sum == 0 and odor_state[:3] != odor_state[3:]:
            # No shocks + asymmetric odors: trial period
            # Fly can make choices without reinforcement
            return 1, None
        elif ee_epoch_sum > 0 and odor_state[:3] == odor_state[3:]:
            # Shocks + symmetric odors: classical conditioning training
            # Shock is delivered regardless of fly position
            return 2, 2
        elif ee_epoch_sum > 0 and odor_state[:3] != odor_state[3:]:
            # Shocks + asymmetric odors: operant conditioning training
            # Shock is delivered only when fly enters the punished side
            return 3, 3
        else:
            # Unexpected combination: log warning and return invalid
            logger.warning(f"Could not define epoch for state {odor_state}, sum={ee_epoch_sum}")
            return np.nan, None


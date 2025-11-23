"""
Trained Odor Identifier

Identifies the trained odor and calculates preference percentages.

This module determines which odor was used during training (the "trained odor")
by analyzing shock delivery patterns and digital output signals. Once identified,
it calculates preference metrics specifically for epochs where the trained odor
is present, allowing measurement of learning and memory retention.
"""

import numpy as np
import logging
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData
    from ..models.metrics import MetricsData

from ..config import Config
from ..exceptions import MissingDataError

logger = logging.getLogger(__name__)


class TrainedOdorIdentifier:
    """
    Identifies the trained odor and calculates preference metrics.
    
    This class determines which odor was associated with punishment during training
    by analyzing when and where shocks were delivered. Once identified, it computes
    preference percentages (both decision and time ratios) specifically for epochs
    where the trained odor is present, providing a measure of learned avoidance.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the trained odor identifier.
        
        The identifier doesn't require special configuration beyond the standard
        config object, as it infers the trained odor from the data itself.
        
        Parameters:
        -----------
        config : Config
            Configuration object (currently unused but kept for consistency
            with other processors)
        """
        self.config = config
    
    def identify(
        self,
        trial_data: 'TrialData',
        metrics_data: 'MetricsData'
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Identify the trained odor and calculate preference metrics.
        
        This method first determines which odor was used during training by
        examining shock delivery patterns. Then it calculates preference
        percentages for epochs where that odor is present. Preferences are
        computed as the percentage of decisions or time spent avoiding the
        trained odor (the odor associated with punishment).
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing shock data and digital outputs
            needed to identify the trained odor.
        metrics_data : MetricsData
            Metrics data structure containing epoch information, decisions,
            and time ratios that will be filtered for trained odor epochs.
        
        Returns:
        --------
        trained_odor_decs_pc : np.ndarray
            Decision preference percentages for trained odor epochs
            (nepochs x nflies). Values represent percentage of decisions
            away from the trained odor. NaN for epochs where trained odor
            is not present or conditions are invalid.
        trained_odor_time_pc : np.ndarray
            Time preference percentages for trained odor epochs
            (nepochs x nflies). Values represent percentage of time spent
            away from the trained odor. NaN for epochs where trained odor
            is not present or conditions are invalid.
        trained_odor : str
            3-character string encoding the trained odor configuration:
            [AIR, MCH, OCT] as '0' (off) or '1' (on).
            Example: '110' = AIR and MCH on, OCT off.
        """
        # Step 1: Determine which odor was used during training
        # This analyzes shock delivery patterns to infer the trained odor
        trained_odor = self._find_trained_odor(trial_data)
        
        # Step 2: Calculate preference metrics for epochs with trained odor
        # This filters the metrics to only include relevant epochs and computes
        # preferences relative to the trained odor (avoidance percentages)
        trained_odor_decs_pc, trained_odor_time_pc = self._calculate_preferences(
            trial_data, metrics_data, trained_odor
        )
        
        logger.debug(f"Identified trained odor: {trained_odor}")
        
        return trained_odor_decs_pc, trained_odor_time_pc, trained_odor
    
    def _find_trained_odor(self, trial_data: 'TrialData') -> str:
        """
        Identify the trained odor by analyzing shock delivery patterns.
        
        This method determines which odor was associated with punishment during
        training by examining when and where shocks were delivered. The trained
        odor is identified as the odor present on the side where the fly received
        the first shock. This works because in operant conditioning, shocks are
        delivered when the fly enters the punished side.
        
        The method tries two approaches:
        1. Primary: Analyze shock delivery patterns from SHOCK or EE data
        2. Fallback: Extract information from file header metadata
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing:
            - Shock data (SHOCK or EE fields)
            - Digital output signals (odor delivery states)
            - Position data (to determine which side received shock)
            - Header information (as fallback)
        
        Returns:
        --------
        str
            3-character string encoding the trained odor: [AIR, MCH, OCT]
            Example: '110' means AIR and MCH were on, OCT was off.
        
        Raises:
        -------
        MissingDataError
            If shock data is unavailable and header information is insufficient.
        """
        try:
            # Determine which shock data field is available
            # Older experiments use SHOCK (single column), newer use EE (per-fly)
            shock_sum = np.sum(trial_data.data['SHOCK'])
            
            if shock_sum > 0:
                # Use old-style SHOCK field (applies to all flies)
                shock_vector = trial_data.data['SHOCK']
            elif 'EE' in trial_data.data:
                # Use operant shock field (EE)
                # For trained odor identification, use only the first fly's data
                # This is sufficient because all flies receive the same training protocol
                shock_vector = trial_data.data['EE'][:, 0]
            else:
                raise MissingDataError("Neither kind of shocks logged")
            
            nflies = trial_data.nflies
            
            # Combine all digital output signals into a single matrix
            # This shows which odors were on at each frame
            digital_outs_cat = np.column_stack([
                trial_data.data['LEFTAIR'],
                trial_data.data['LEFTMCH'],
                trial_data.data['LEFTOCT'],
                trial_data.data['RIGHTAIR'],
                trial_data.data['RIGHTMCH'],
                trial_data.data['RIGHTOCT']
            ])
            
            # Find the first frame where a shock was delivered
            # This represents the training condition (odor + shock pairing)
            shock_locs = np.where(shock_vector == 1)[0]
            
            if len(shock_locs) > 0:
                # Get the complete odor state at the time of first shock
                # This includes all 6 odors (left and right sides)
                trained_odor_state_num = digital_outs_cat[shock_locs[0], :6]
                trained_odor_state = ''.join([str(int(x)) for x in trained_odor_state_num])
                
                # Determine which side of the chamber the fly was on when shocked
                # This tells us which side's odors were the "trained" odors
                position_snap = trial_data.data['cX'][shock_locs[0], 0]
                
                # Extract the trained odor based on fly position
                # If fly was on left (position <= 0), use left-side odors
                # If fly was on right (position > 0), use right-side odors
                if position_snap <= 0:
                    # Fly was on left side: left odors are the trained ones
                    trained_odor_num = digital_outs_cat[shock_locs[0], :3]
                else:
                    # Fly was on right side: right odors are the trained ones
                    trained_odor_num = digital_outs_cat[shock_locs[0], 3:6]
                
                # Convert to 3-character string: [AIR, MCH, OCT]
                trained_odor = ''.join([str(int(x)) for x in trained_odor_num])
            else:
                raise MissingDataError("No shock locations found")
                
        except Exception as e:
            # Fallback: try to extract trained odor from file header metadata
            # This is used when shock data is missing or unreliable
            logger.warning(f"Could not determine trained odor from shocks: {e}")
            logger.info("Attempting to use header information")
            
            trained_odor = self._find_trained_odor_from_header(trial_data)
        
        return trained_odor
    
    def _find_trained_odor_from_header(self, trial_data: 'TrialData') -> str:
        """
        Extract trained odor information from file header metadata.
        
        This fallback method reads the trained odor from the file header when
        shock data is unavailable. The header contains explicit information about
        which odor was used for training and the concentration ratios, which can
        be used to reconstruct the trained odor configuration.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing header information with fields:
            - szTrainedOdor: Name of trained odor ('MCH', 'OCT', or 'AIR')
            - odor1Percent: Concentration percentage for odor 1
            - odor2Percent: Concentration percentage for odor 2
        
        Returns:
        --------
        str
            3-character string encoding the trained odor: [AIR, MCH, OCT]
            Example: '010' means only MCH was on.
        
        Raises:
        -------
        MissingDataError
            If header is missing or doesn't contain required fields.
        """
        if trial_data.header is None:
            raise MissingDataError("No header information available and no shock data")
        
        header = trial_data.header
        
        # Validate that header contains required information
        if 'szTrainedOdor' not in header or 'odor1Percent' not in header or 'odor2Percent' not in header:
            raise MissingDataError("Insufficient header information")
        
        sz_trained_odor = header['szTrainedOdor']
        odor1_percent = header['odor1Percent']
        odor2_percent = header['odor2Percent']
        
        # Reconstruct trained odor configuration based on header information
        # The logic determines which odors are on based on the trained odor type
        # and concentration ratios between odor1 and odor2
        if sz_trained_odor == 'MCH' and odor1_percent < odor2_percent:
            # MCH trained, odor1 < odor2: AIR and MCH on
            return '110'
        elif sz_trained_odor == 'MCH' and odor1_percent >= odor2_percent:
            # MCH trained, odor1 >= odor2: Only MCH on
            return '010'
        elif sz_trained_odor == 'OCT' and odor1_percent <= odor2_percent:
            # OCT trained, odor1 <= odor2: Only OCT on
            return '001'
        elif sz_trained_odor == 'OCT' and odor1_percent > odor2_percent:
            # OCT trained, odor1 > odor2: AIR and OCT on
            return '101'
        elif sz_trained_odor == 'AIR' and odor1_percent < odor2_percent:
            # AIR trained, odor1 < odor2: No odors (all off)
            return '000'
        elif sz_trained_odor == 'AIR' and odor1_percent >= odor2_percent:
            # AIR trained, odor1 >= odor2: Only AIR on
            return '100'
        else:
            # Unexpected header values: use default (no odors)
            logger.warning("Unexpected header values, using default")
            return '000'
    
    def _calculate_preferences(
        self,
        trial_data: 'TrialData',
        metrics_data: 'MetricsData',
        trained_odor: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate preference percentages specifically for trained odor epochs.
        
        This method computes preference metrics (decisions and time ratios) for
        epochs where the trained odor is present. The preferences are calculated
        as avoidance percentages: when the trained odor is on the left, preferences
        are measured as the percentage of decisions/time spent on the right (away
        from the trained odor). When the trained odor is on the right, preferences
        are measured as the percentage spent on the left.
        
        The calculation handles three cases:
        1. Trained odor on left: use left preferences directly
        2. Trained odor on right: invert preferences (100 - left = right preference)
        3. Neutral conditions (no odors or only air): mark as invalid (NaN)
        
        Note: This method assumes asymmetric odor conditions. If the same odor
        is present on both sides, the calculation may not be meaningful.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure (used for nflies count).
        metrics_data : MetricsData
            Metrics data structure containing:
            - odorStates: Odor configuration for each epoch
            - decisionsLeftPC: Percentage of decisions to left for each epoch/fly
            - timeRatioLeftPC: Percentage of time on left for each epoch/fly
        trained_odor : str
            3-character string encoding the trained odor: [AIR, MCH, OCT]
            Example: '110' = AIR and MCH on, OCT off.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (trained_odor_decs_pc, trained_odor_time_pc) where both arrays are
            (nepochs x nflies) and contain preference percentages for trained
            odor epochs, with NaN for invalid or non-trained-odor epochs.
        """
        # Extract odor states for each epoch
        # Each state is a 6-character string, but we only need the left side (first 3 chars)
        odor_states = metrics_data['odorStates']
        
        # Extract left side odor states (first 3 characters) for comparison
        # Handle different data formats (string array vs numeric array)
        if isinstance(odor_states, np.ndarray):
            if len(odor_states) > 0 and isinstance(odor_states[0], str):
                # String array: directly extract first 3 characters
                left_odor_states = np.array([s[:3] for s in odor_states])
            else:
                # Numeric array: convert to string first
                left_odor_states = np.array([''.join([str(int(x)) for x in row[:3]]) for row in odor_states])
        else:
            # List or other iterable: extract first 3 characters
            left_odor_states = np.array([s[:3] for s in odor_states])
        
        # Get decision and time ratio data for all epochs and flies
        decL = metrics_data['decisionsLeftPC']  # Decisions to left (%)
        timL = metrics_data['timeRatioLeftPC']  # Time on left (%)
        
        nepochs = decL.shape[0]
        nflies = trial_data.nflies
        
        # Initialize output arrays with NaN (invalid by default)
        trained_odor_decs_pc = np.full((nepochs, nflies), np.nan)
        trained_odor_time_pc = np.full((nepochs, nflies), np.nan)
        
        # Calculate preferences for each epoch and fly
        # The logic depends on where the trained odor appears in each epoch
        for flyidx in range(nflies):
            for eidx in range(nepochs):
                # Get left-side odor state for this epoch (ensure it's a string)
                left_odor_state = left_odor_states[eidx] if isinstance(left_odor_states[eidx], str) else str(left_odor_states[eidx])
                
                if left_odor_state == trained_odor:
                    # Trained odor is on the left side
                    # Preference = percentage of decisions/time spent on left
                    # (This measures preference FOR the trained odor location)
                    trained_odor_decs_pc[eidx, flyidx] = decL[eidx, flyidx]
                    trained_odor_time_pc[eidx, flyidx] = timL[eidx, flyidx]
                elif left_odor_state != trained_odor:
                    # Trained odor is on the right side (or different configuration)
                    # Preference = percentage spent on right = 100 - percentage on left
                    # (This measures avoidance of the trained odor location)
                    trained_odor_decs_pc[eidx, flyidx] = 100 - decL[eidx, flyidx]
                    trained_odor_time_pc[eidx, flyidx] = 100 - timL[eidx, flyidx]
                elif left_odor_state == '000':
                    # No odors present: cannot measure preference
                    trained_odor_decs_pc[eidx, flyidx] = np.nan
                    trained_odor_time_pc[eidx, flyidx] = np.nan
                elif left_odor_state == '100':
                    # Only air present: not a meaningful odor condition
                    trained_odor_decs_pc[eidx, flyidx] = np.nan
                    trained_odor_time_pc[eidx, flyidx] = np.nan
                else:
                    # Unexpected odor state: log warning and leave as NaN
                    logger.warning(f"Problem calculating trained odor pref for epoch {eidx}, fly {flyidx}")
        
        return trained_odor_decs_pc, trained_odor_time_pc


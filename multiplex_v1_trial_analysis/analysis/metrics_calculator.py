"""
Metrics Calculator

Main metrics calculation orchestrator.

This module orchestrates the calculation of all behavioral metrics from trial data.
It coordinates epoch detection, decision counting, time ratio calculation, speed
measurement, and trained odor identification. The calculator produces a complete
metrics data structure containing all computed values needed for analysis and
statistical summarization.
"""

import numpy as np
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData
    from ..models.metrics import MetricsData

from ..processing.epochs import EpochDetector
from ..processing.decisions import DecisionCounter
from ..processing.timing import TimeRatioCalculator
from ..processing.trained_odor import TrainedOdorIdentifier
from ..config import Config

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculates all behavioral metrics from trial data.
    
    This class orchestrates the complete metrics calculation workflow by coordinating
    specialized calculators for different metric types. It manages the data flow
    between calculations, ensuring that dependent metrics (like trained odor
    preferences) are computed after their prerequisites (like epoch detection and
    basic decision/time calculations).
    """
    
    def __init__(self, config: Config):
        """
        Initialize the metrics calculator with all required sub-calculators.
        
        Each sub-calculator handles a specific aspect of metrics computation.
        They are all configured with the shared configuration object to ensure
        consistent parameter settings across all calculations.
        
        Parameters:
        -----------
        config : Config
            Configuration object containing all analysis parameters used by
            the various sub-calculators (epoch detection, decision counting, etc.)
        """
        self.config = config
        # Epoch detector: identifies experimental periods and their boundaries
        self.epoch_detector = EpochDetector(config)
        # Decision counter: counts behavioral choices (transitions and reversals)
        self.decision_counter = DecisionCounter(config)
        # Time ratio calculator: computes time spent in each odor zone
        self.time_ratio_calc = TimeRatioCalculator(config)
        # Trained odor identifier: determines which odor was used for training
        self.trained_odor_id = TrainedOdorIdentifier(config)
    
    def calculate(self, trial_data: 'TrialData') -> 'MetricsData':
        """
        Calculate all behavioral metrics from trial data.
        
        This method orchestrates the complete metrics calculation workflow:
        1. Handles special cases (UV laser data)
        2. Detects epochs (experimental periods)
        3. Calculates decisions for each epoch/fly
        4. Calculates time ratios for each epoch/fly
        5. Calculates mean speed for each epoch/fly
        6. Identifies trained odor and calculates preferences
        7. Identifies trial epochs
        8. Calculates changes between first and last trials
        9. Computes summary statistics
        10. Counts shocks (if available)
        11. Assembles complete metrics data structure
        
        The workflow is sequential, with later calculations depending on earlier
        results. For example, trained odor identification requires epoch information,
        and change calculations require trained odor preferences.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing:
            - Position tracking data (cX)
            - Digital output signals (odor delivery)
            - Shock data (SHOCK or EE)
            - Time data
            All data should already be smoothed before calling this method.
        
        Returns:
        --------
        MetricsData
            Complete metrics data structure containing:
            - Epoch information (tags, states, boundaries)
            - Decision metrics (left percentages, totals)
            - Time ratio metrics (left percentages)
            - Speed metrics (mean speed per epoch/fly)
            - Trained odor preferences (decision and time percentages)
            - Trial information (locations, counts)
            - Change metrics (before/after, changes)
            - Summary statistics (means, SEMs)
            - Shock metrics (if available)
        """
        from ..models.metrics import MetricsData
        
        logger.info("Calculating behavioral metrics...")
        
        # Handle special case: UV laser experiments
        # In some experiments, UV laser was used for punishment instead of electric shock
        # The UV data is stored in a separate field but should be treated as SHOCK data
        if 'UV' in trial_data.data:
            uv_sum = np.sum(trial_data.data['UV'])
            if uv_sum > 0:
                # Transfer UV data to SHOCK field for consistent processing
                trial_data.data['SHOCK'] = trial_data.data['UV']
        
        # Step 1: Detect epochs (experimental periods)
        # This identifies boundaries between different experimental conditions
        # and classifies each epoch by type (wait, trial, training)
        epochtags, odor_states, start_bins, end_bins, expt_tag = self.epoch_detector.detect(trial_data)
        # Combine start and end bins into a single array for easier indexing
        epoch_ends = np.column_stack([start_bins, end_bins])
        
        # Get dimensions for iteration
        nflies = trial_data.nflies
        nepochs = len(epochtags)
        
        # Step 2: Calculate decision metrics for each epoch and fly
        # Decisions are behavioral choices (crossing between odor zones)
        decisions_left_pc, decisions_total = self._calculate_decisions(
            trial_data, epoch_ends, nepochs, nflies
        )
        
        # Step 3: Calculate time ratio metrics for each epoch and fly
        # Time ratios measure how much time was spent in each odor zone
        # Only calculated if the fly made sufficient decisions (not NaN)
        time_ratio_left_pc = self._calculate_time_ratios(
            trial_data, epoch_ends, nepochs, nflies, decisions_total
        )
        
        # Step 4: Calculate mean walking speed for each epoch and fly
        # Speed is computed from position changes and converted to mm/s
        speed_mean = self._calculate_speed(trial_data, epoch_ends, nepochs, nflies)
        
        # Step 5: Identify trained odor and calculate preferences
        # This requires epoch and basic metrics, so we create a temporary structure
        temp_metrics = MetricsData()
        temp_metrics.set_epochs(epochtags, odor_states, start_bins, end_bins, expt_tag)
        temp_metrics.set_decisions(decisions_left_pc, decisions_total)
        temp_metrics.set_time_ratios(time_ratio_left_pc)
        
        # Identify which odor was used for training and calculate preferences
        trained_odor_decs_pc, trained_odor_time_pc, trained_odor = self.trained_odor_id.identify(
            trial_data, temp_metrics
        )
        
        # Step 6: Identify trial epochs (epochs where flies can make choices)
        # Trial epochs are marked with tag 1 in the epochtags array
        trial_locs = np.where(epochtags == 1)[0]
        ntrials = len(trial_locs)
        
        # Step 7: Calculate changes between first and last trial
        # This measures learning: how preferences changed from beginning to end
        decisions_changes, time_changes, decs_before, decs_after, time_before, time_after = \
            self._calculate_changes(trained_odor_decs_pc, trained_odor_time_pc, trial_locs, ntrials, nflies)
        
        # Step 8: Calculate summary statistics
        # These provide aggregate measures across all flies
        dec_cha_mean, dec_cha_sem_norm, mean_total_decs = \
            self._calculate_statistics(decisions_changes, decisions_total)
        
        # Step 9: Count shocks (if shock data is available)
        # This provides information about punishment delivery during training
        epoch_shocks, epoch_shock_time = self._calculate_shock_metrics(
            trial_data, start_bins, end_bins, nepochs, nflies
        )
        
        # Step 10: Assemble complete metrics data structure
        # Combine all calculated metrics into a single structure
        metrics_data = MetricsData()
        metrics_data.set_epochs(epochtags, odor_states, start_bins, end_bins, expt_tag)
        metrics_data.set_decisions(decisions_left_pc, decisions_total)
        metrics_data.set_time_ratios(time_ratio_left_pc)
        metrics_data.set_speed(speed_mean)
        metrics_data.set_trained_odor(trained_odor_decs_pc, trained_odor_time_pc, trained_odor)
        metrics_data.set_trial_info(trial_locs, ntrials)
        metrics_data.set_trial_changes(
            decisions_changes, time_changes,
            decs_before, decs_after, time_before, time_after
        )
        metrics_data.set_statistics(dec_cha_mean, dec_cha_sem_norm, mean_total_decs)
        
        # Add shock metrics if available (some experiments don't use shocks)
        if epoch_shocks is not None:
            metrics_data.set_shock_metrics(epoch_shocks, epoch_shock_time)
        
        logger.info(f"Calculated metrics: {nepochs} epochs, {ntrials} trials")
        
        return metrics_data
    
    def _calculate_decisions(
        self,
        trial_data: 'TrialData',
        epoch_ends: np.ndarray,
        nepochs: int,
        nflies: int
    ) -> tuple:
        """
        Calculate decision metrics for all epochs and flies.
        
        This method iterates through each epoch and fly, extracts the position
        data for that specific period, and counts the behavioral decisions made
        during that time. Decisions include both fullway transitions (direct
        cross-chamber movement) and reversals (center entry followed by
        opposite-side exit).
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing position data in cX field.
        epoch_ends : np.ndarray
            Array of epoch boundaries (nepochs x 2) with [start, end] pairs.
            Indices are 1-based (first frame is 1, not 0).
        nepochs : int
            Number of epochs detected in the data.
        nflies : int
            Number of flies tracked in the experiment.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (decisions_left_pc, decisions_total) where:
            - decisions_left_pc: Percentage of decisions to left (nepochs x nflies)
            - decisions_total: Total number of decisions (nepochs x nflies)
            Both arrays may contain NaN values for epochs/flies with insufficient activity.
        """
        decisions_left_pc = np.zeros((nepochs, nflies))
        decisions_total = np.zeros((nepochs, nflies))
        
        # Process each fly and epoch combination
        for flyidx in range(nflies):
            for epoidx in range(nepochs):
                # Extract position data for this specific epoch and fly
                # Epoch boundaries are 1-based and inclusive on both ends
                # Convert to 0-based for Python array indexing, then add 1 to end
                # to make the slice inclusive (matching 1-based inclusive range)
                start_idx = int(epoch_ends[epoidx, 0]) - 1  # Convert to 0-based
                end_idx = int(epoch_ends[epoidx, 1])  # End is inclusive, so add 1 for Python slice
                decisions_left_pc[epoidx, flyidx], decisions_total[epoidx, flyidx] = \
                    self.decision_counter.count_decisions(
                        trial_data.data['cX'][start_idx:end_idx + 1, flyidx],
                        self.config.gaussian_fwhm
                    )
        
        return decisions_left_pc, decisions_total
    
    def _calculate_time_ratios(
        self,
        trial_data: 'TrialData',
        epoch_ends: np.ndarray,
        nepochs: int,
        nflies: int,
        decisions_total: np.ndarray
    ) -> np.ndarray:
        """
        Calculate time ratio metrics for all epochs and flies.
        
        This method computes the percentage of time spent on the left side of
        the chamber for each epoch and fly. Time ratios are only calculated
        for flies that made sufficient decisions (not NaN), as time ratios
        are meaningless for inactive flies.
        
        The time ratio uses a weighted calculation method where later time points
        contribute more to the final ratio, reflecting temporal dynamics of
        preference formation.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing position data in cX field.
        epoch_ends : np.ndarray
            Array of epoch boundaries (nepochs x 2) with [start, end] pairs.
            Indices are 1-based (first frame is 1, not 0).
        nepochs : int
            Number of epochs detected in the data.
        nflies : int
            Number of flies tracked in the experiment.
        decisions_total : np.ndarray
            Total decisions for each epoch/fly (nepochs x nflies).
            Used to determine which flies are valid (not NaN).
        
        Returns:
        --------
        np.ndarray
            Time ratio left percentages (nepochs x nflies).
            Values range from 0-100, representing percentage of weighted time
            spent on left side. NaN for epochs/flies with insufficient decisions.
        """
        time_ratio_left_pc = np.zeros((nepochs, nflies))
        for flyidx in range(nflies):
            for epoidx in range(nepochs):
                # Skip calculation if fly had insufficient decisions
                # Time ratios are only meaningful when flies are active
                if np.isnan(decisions_total[epoidx, flyidx]):
                    time_ratio_left_pc[epoidx, flyidx] = np.nan
                else:
                    # Extract position data for this epoch and fly
                    # Epoch boundaries are 1-based and inclusive on both ends
                    start_idx = int(epoch_ends[epoidx, 0]) - 1  # Convert to 0-based
                    end_idx = int(epoch_ends[epoidx, 1])  # End is inclusive, so add 1 for Python slice
                    time_ratio_left_pc[epoidx, flyidx] = self.time_ratio_calc.calculate(
                        trial_data.data['cX'][start_idx:end_idx + 1, flyidx]
                    )
        return time_ratio_left_pc
    
    def _calculate_speed(
        self,
        trial_data: 'TrialData',
        epoch_ends: np.ndarray,
        nepochs: int,
        nflies: int
    ) -> np.ndarray:
        """
        Calculate mean walking speed for all epochs and flies.
        
        This method computes the average speed of movement for each fly during
        each epoch. Speed is calculated by:
        1. Computing frame-to-frame position changes (absolute differences)
        2. Averaging these changes to get mean speed per frame
        3. Converting to speed per second (multiply by frame rate)
        4. Converting to physical units (multiply by chamber dimensions)
        
        The result is in millimeters per second, representing the average
        walking speed of the fly during that epoch.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing position data in cX field.
        epoch_ends : np.ndarray
            Array of epoch boundaries (nepochs x 2) with [start, end] pairs.
            Indices are 1-based (first frame is 1, not 0).
        nepochs : int
            Number of epochs detected in the data.
        nflies : int
            Number of flies tracked in the experiment.
        
        Returns:
        --------
        np.ndarray
            Mean speed in mm/s for each epoch/fly (nepochs x nflies).
            May contain NaN values if calculation fails (e.g., empty epoch).
        """
        speed_mean = np.zeros((nepochs, nflies))
        frames_per_second = self.config.frames_per_second
        half_chamber_length = self.config.half_chamber_length_mm
        
        for flyidx in range(nflies):
            for epoidx in range(nepochs):
                try:
                    # Extract position data for this epoch and fly
                    # Epoch boundaries are 1-based and inclusive on both ends
                    start_idx = int(epoch_ends[epoidx, 0]) - 1  # Convert to 0-based
                    end_idx = int(epoch_ends[epoidx, 1])  # End is inclusive, so add 1 for Python slice
                    # Calculate frame-to-frame position changes (absolute value)
                    # This gives the distance moved between consecutive frames
                    roi_speed = np.abs(np.diff(trial_data.data['cX'][start_idx:end_idx + 1, flyidx]))
                    # Average the changes to get mean speed per frame
                    mean_roi_speed_per_frame = np.mean(roi_speed)
                    # Convert to speed per second (multiply by frame rate)
                    mean_roi_speed_per_second = mean_roi_speed_per_frame * frames_per_second
                    # Convert from normalized units to physical units (mm)
                    # Position is normalized to [-1, 1], so multiply by half chamber length
                    mean_mm_speed_per_second = mean_roi_speed_per_second * half_chamber_length
                    speed_mean[epoidx, flyidx] = mean_mm_speed_per_second
                except:
                    # If calculation fails (e.g., empty epoch), mark as invalid
                    speed_mean[epoidx, flyidx] = np.nan
        
        return speed_mean
    
    def _calculate_changes(
        self,
        trained_odor_decs_pc: np.ndarray,
        trained_odor_time_pc: np.ndarray,
        trial_locs: np.ndarray,
        ntrials: int,
        nflies: int
    ) -> tuple:
        """
        Calculate changes in preferences between first and last trial.
        
        This method measures learning by comparing preferences at the beginning
        of the experiment (first trial) to preferences at the end (last trial).
        The change is computed as the difference: after - before.
        
        Positive changes indicate increased avoidance of the trained odor
        (learning/conditioning), while negative changes indicate decreased
        avoidance (extinction or lack of learning).
        
        Parameters:
        -----------
        trained_odor_decs_pc : np.ndarray
            Decision preference percentages for trained odor epochs (nepochs x nflies).
        trained_odor_time_pc : np.ndarray
            Time preference percentages for trained odor epochs (nepochs x nflies).
        trial_locs : np.ndarray
            Indices of trial epochs (epochs where tag == 1).
        ntrials : int
            Number of trial epochs detected.
        nflies : int
            Number of flies tracked.
        
        Returns:
        --------
        Tuple containing:
        - decisions_changes: Change in decision preferences (last - first trial)
        - time_changes: Change in time preferences (last - first trial)
        - decs_before: Decision preferences in first trial
        - decs_after: Decision preferences in last trial
        - time_before: Time preferences in first trial
        - time_after: Time preferences in last trial
        
        All arrays are 1D with length nflies. Returns None/zeros if calculation fails.
        """
        decisions_changes = None
        time_changes = None
        decs_before = None
        decs_after = None
        time_before = None
        time_after = None
        
        try:
            # Extract decision preferences for trial epochs only
            trial_decs = np.zeros((ntrials, nflies))
            for triidx in range(ntrials):
                trial_decs[triidx, :] = trained_odor_decs_pc[trial_locs[triidx], :]
            
            # Get preferences from first and last trials
            decs_before = trial_decs[0, :]
            if ntrials > 0:
                decs_after = trial_decs[ntrials - 1, :]
            # Calculate changes between consecutive trials
            # diff() computes differences: row[i+1] - row[i]
            decisions_changes = np.diff(trial_decs, axis=0)
            if decisions_changes.size > 0:
                # Take the last row: change from first to last trial
                decisions_changes = decisions_changes[-1, :]
            else:
                # No trials or only one trial: no change to calculate
                decisions_changes = np.zeros(nflies)
        except Exception as e:
            logger.warning(f"No decisions changes: {e}")
        
        try:
            # Extract time preferences for trial epochs only
            trial_tims = np.zeros((ntrials, nflies))
            for triidx in range(ntrials):
                trial_tims[triidx, :] = trained_odor_time_pc[trial_locs[triidx], :]
            
            # Get preferences from first and last trials
            time_before = trial_tims[0, :]
            if ntrials > 0:
                time_after = trial_tims[ntrials - 1, :]
            # Calculate changes between consecutive trials
            time_changes = np.diff(trial_tims, axis=0)
            if time_changes.size > 0:
                # Take the last row: change from first to last trial
                time_changes = time_changes[-1, :]
            else:
                # No trials or only one trial: no change to calculate
                time_changes = np.zeros(nflies)
        except Exception as e:
            logger.warning(f"No time changes (maybe only one trial?): {e}")
        
        return decisions_changes, time_changes, decs_before, decs_after, time_before, time_after
    
    def _calculate_statistics(
        self,
        decisions_changes: np.ndarray,
        decisions_total: np.ndarray
    ) -> tuple:
        """
        Calculate summary statistics from decision change metrics.
        
        This method computes aggregate statistics across all flies:
        - Mean decision change: average learning across all flies
        - Standard error of the mean: measure of variability in learning
        - Mean total decisions: average number of decisions made per fly
        
        The standard error uses sample standard deviation (dividing by N-1)
        to account for the fact that we're estimating population parameters
        from a sample.
        
        Parameters:
        -----------
        decisions_changes : np.ndarray or None
            Change in decision preferences for each fly (1D array, length nflies).
            May contain NaN values for flies with invalid data.
        decisions_total : np.ndarray
            Total decisions for all epochs/flies (nepochs x nflies).
            Used to calculate average decisions per valid fly.
        
        Returns:
        --------
        Tuple[float, float, float]
            (dec_cha_mean, dec_cha_sem_norm, mean_total_decs) where:
            - dec_cha_mean: Mean decision change across flies
            - dec_cha_sem_norm: Standard error of the mean (normalized)
            - mean_total_decs: Average total decisions per valid fly
        """
        if decisions_changes is not None:
            # Calculate mean decision change, ignoring NaN values
            dec_cha_mean = np.nanmean(decisions_changes)
            # Calculate standard error of the mean
            # Use sample standard deviation (ddof=1) for unbiased estimation
            # Standard error = standard deviation / sqrt(sample size)
            dec_cha_sem_norm = np.nanstd(decisions_changes) / np.sqrt(np.sum(~np.isnan(decisions_changes)))
        else:
            # No valid decision changes: return invalid values
            dec_cha_mean = np.nan
            dec_cha_sem_norm = np.nan
        
        if decisions_changes is not None:
            # Calculate mean total decisions per valid fly
            # Sum all decisions and divide by number of valid flies
            mean_total_decs = np.sum(decisions_total) / np.sum(~np.isnan(decisions_changes))
        else:
            # No valid flies: set to zero
            mean_total_decs = 0
        
        return dec_cha_mean, dec_cha_sem_norm, mean_total_decs
    
    def _calculate_shock_metrics(
        self,
        trial_data: 'TrialData',
        start_bins: np.ndarray,
        end_bins: np.ndarray,
        nepochs: int,
        nflies: int
    ) -> tuple:
        """
        Calculate shock delivery metrics for each epoch and fly.
        
        This method computes two shock-related metrics:
        1. Number of shocks: Counts discrete shock events (transitions from off to on)
        2. Shock time: Total duration of shock delivery (sum of all shock frames)
        
        Shocks are detected by finding transitions in the EE (operant shock) data.
        Each transition from 0 to 1 represents the start of a shock event.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing EE (operant shock) data.
        start_bins : np.ndarray
            Start frame indices for each epoch (1-based).
        end_bins : np.ndarray
            End frame indices for each epoch (1-based, inclusive).
        nepochs : int
            Number of epochs detected.
        nflies : int
            Number of flies tracked.
        
        Returns:
        --------
        Tuple[np.ndarray or None, np.ndarray or None]
            (epoch_shocks, epoch_shock_time) where:
            - epoch_shocks: Number of shocks per epoch/fly (nepochs x nflies)
            - epoch_shock_time: Total shock time per epoch/fly in seconds (nepochs x nflies)
            Returns (None, None) if shock data is unavailable.
        """
        epoch_shocks = None
        epoch_shock_time = None
        
        try:
            # Check if operant shock data (EE) is available
            if 'EE' not in trial_data.data:
                return None, None
            
            # Detect shock events by finding transitions from off (0) to on (1)
            # Compute frame-to-frame differences to find when shock starts
            shocks_dif = np.diff(trial_data.data['EE'], axis=0)
            # Add a row of zeros at the beginning to align with original data
            # This ensures the first frame can be a shock start if it's already on
            shocks_dif1 = np.vstack([np.zeros((1, nflies)), shocks_dif])
            
            # Count number of shocks (transitions to on state) in each epoch
            epoch_shocks = np.zeros((nepochs, nflies))
            for eidx in range(nepochs):
                for flyidx in range(nflies):
                    # Extract shock transition data for this epoch
                    # Epoch boundaries are 1-based and inclusive on both ends
                    start_idx = int(start_bins[eidx]) - 1  # Convert to 0-based
                    end_idx = int(end_bins[eidx])  # End is inclusive, so add 1 for Python slice
                    # Count transitions where shock goes from off to on (> 0)
                    epoch_shocks[eidx, flyidx] = np.sum(shocks_dif1[start_idx:end_idx + 1, flyidx] > 0)
            
            # Calculate total shock time (duration) in each epoch
            log_frame_rate = self.config.log_frame_rate  # Frame rate for shock timing (29.97 fps)
            epoch_shock_time = np.zeros((nepochs, nflies))
            for eidx in range(nepochs):
                for flyidx in range(nflies):
                    # Extract shock data for this epoch
                    start_idx = int(start_bins[eidx]) - 1  # Convert to 0-based
                    end_idx = int(end_bins[eidx])  # End is inclusive, so add 1 for Python slice
                    # Sum all shock frames (1 = shock on, 0 = shock off)
                    # Divide by frame rate to convert frames to seconds
                    epoch_shock_time[eidx, flyidx] = np.sum(trial_data.data['EE'][start_idx:end_idx + 1, flyidx]) / log_frame_rate
        except Exception as e:
            # Shock metrics calculation failed: log warning and return None
            logger.warning(f"No shock metrics: {e}")
        
        return epoch_shocks, epoch_shock_time


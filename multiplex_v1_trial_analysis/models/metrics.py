"""
Metrics Data Model

Encapsulates the metrics data structure with validation and type safety.

This module defines the MetricsData class, which stores all calculated
behavioral metrics from the analysis. The metrics include epoch information,
decision counts, time ratios, speed measurements, trained odor preferences,
trial changes, and summary statistics. The class provides dictionary-style
access and serialization capabilities.
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class MetricsData:
    """
    Represents calculated behavioral metrics from trial analysis.
    
    This class encapsulates all computed metrics from the behavioral analysis,
    including epoch classifications, decision metrics, time ratios, speed,
    trained odor preferences, and statistical summaries. The class provides
    dictionary-style access for compatibility with existing code patterns
    while maintaining type safety through setter methods.
    
    The metrics are organized into logical groups:
    - Epoch information: tags, states, boundaries
    - Behavioral metrics: decisions, time ratios, speed
    - Trained odor preferences: decision and time percentages
    - Trial information: locations, changes, before/after values
    - Statistics: means, standard errors, totals
    """
    
    def __init__(self):
        """
        Initialize an empty MetricsData structure.
        
        The structure starts empty and is populated through setter methods
        that ensure proper data organization and validation.
        """
        self._data: Dict[str, Any] = {}
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment"""
        self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default"""
        return self._data.get(key, default)
    
    def set_epochs(
        self,
        epochtags: np.ndarray,
        odor_states: np.ndarray,
        start_bins: np.ndarray,
        end_bins: np.ndarray,
        expt_tag: Optional[int] = None
    ) -> None:
        """
        Set epoch information
        
        Parameters:
        -----------
        epochtags : np.ndarray
            Epoch tags: 0=wait, 1=trial, 2=classical train, 3=operant train
        odor_states : np.ndarray
            Odor states for each epoch
        start_bins : np.ndarray
            Start bin indices for each epoch
        end_bins : np.ndarray
            End bin indices for each epoch
        expt_tag : int, optional
            Experiment tag: 2=classical, 3=operant
        """
        self._data['epochtags'] = epochtags
        self._data['odorStates'] = odor_states
        self._data['startBins'] = start_bins
        self._data['endBins'] = end_bins
        if expt_tag is not None:
            self._data['exptTag'] = expt_tag
        logger.debug(f"Set epoch information: {len(epochtags)} epochs")
    
    def set_decisions(
        self,
        decisions_left_pc: np.ndarray,
        decisions_total: np.ndarray
    ) -> None:
        """
        Set decision metrics
        
        Parameters:
        -----------
        decisions_left_pc : np.ndarray
            Percentage of decisions to left (nepochs x nflies)
        decisions_total : np.ndarray
            Total number of decisions (nepochs x nflies)
        """
        self._data['decisionsLeftPC'] = decisions_left_pc
        self._data['decisionsTotal'] = decisions_total
        logger.debug(f"Set decision metrics: shape {decisions_left_pc.shape}")
    
    def set_time_ratios(self, time_ratio_left_pc: np.ndarray) -> None:
        """
        Set time ratio metrics
        
        Parameters:
        -----------
        time_ratio_left_pc : np.ndarray
            Percentage of time spent on left (nepochs x nflies)
        """
        self._data['timeRatioLeftPC'] = time_ratio_left_pc
        logger.debug(f"Set time ratio metrics: shape {time_ratio_left_pc.shape}")
    
    def set_speed(self, speed_mean: np.ndarray) -> None:
        """
        Set speed metrics
        
        Parameters:
        -----------
        speed_mean : np.ndarray
            Mean speed for each epoch/fly (nepochs x nflies)
        """
        self._data['speedMean'] = speed_mean
        logger.debug(f"Set speed metrics: shape {speed_mean.shape}")
    
    def set_trained_odor(
        self,
        trained_odor_decs_pc: np.ndarray,
        trained_odor_time_pc: np.ndarray,
        trained_odor: str
    ) -> None:
        """
        Set trained odor information
        
        Parameters:
        -----------
        trained_odor_decs_pc : np.ndarray
            Decision percentages for trained odor (nepochs x nflies)
        trained_odor_time_pc : np.ndarray
            Time percentages for trained odor (nepochs x nflies)
        trained_odor : str
            Trained odor state string (e.g., '110', '010')
        """
        self._data['trainedOdorDecsPC'] = trained_odor_decs_pc
        self._data['trainedOdorTimePC'] = trained_odor_time_pc
        self._data['trainedOdor'] = trained_odor
        logger.debug(f"Set trained odor: {trained_odor}")
    
    def set_trial_changes(
        self,
        decisions_changes: np.ndarray,
        time_changes: np.ndarray,
        decs_before: np.ndarray,
        decs_after: np.ndarray,
        time_before: np.ndarray,
        time_after: np.ndarray
    ) -> None:
        """
        Set trial change metrics
        
        Parameters:
        -----------
        decisions_changes : np.ndarray
            Changes in decisions
        time_changes : np.ndarray
            Changes in time ratios
        decs_before : np.ndarray
            Decisions before
        decs_after : np.ndarray
            Decisions after
        time_before : np.ndarray
            Time before
        time_after : np.ndarray
            Time after
        """
        self._data['decisionsChanges'] = decisions_changes
        self._data['timeChanges'] = time_changes
        self._data['decsBefore'] = decs_before
        self._data['decsAfter'] = decs_after
        self._data['timeBefore'] = time_before
        self._data['timeAfter'] = time_after
        logger.debug("Set trial change metrics")
    
    def set_statistics(
        self,
        dec_cha_mean: float,
        dec_cha_sem_norm: float,
        mean_total_decs: float
    ) -> None:
        """
        Set statistical summaries
        
        Parameters:
        -----------
        dec_cha_mean : float
            Mean decision change
        dec_cha_sem_norm : float
            Standard error of the mean (decision change)
        mean_total_decs : float
            Mean total decisions
        """
        self._data['decChaMean'] = dec_cha_mean
        self._data['decChaSEMnorm'] = dec_cha_sem_norm
        self._data['meanTotalDecs'] = mean_total_decs
        logger.debug("Set statistical summaries")
    
    def set_shock_metrics(
        self,
        epoch_shocks: np.ndarray,
        epoch_shock_time: np.ndarray
    ) -> None:
        """
        Set shock metrics
        
        Parameters:
        -----------
        epoch_shocks : np.ndarray
            Number of shocks per epoch/fly (nepochs x nflies)
        epoch_shock_time : np.ndarray
            Shock time per epoch/fly (nepochs x nflies)
        """
        self._data['epochShocks'] = epoch_shocks
        self._data['epochShockTime'] = epoch_shock_time
        logger.debug("Set shock metrics")
    
    def set_trial_info(
        self,
        trial_locs: np.ndarray,
        ntrials: int
    ) -> None:
        """
        Set trial location information
        
        Parameters:
        -----------
        trial_locs : np.ndarray
            Indices of trial epochs
        ntrials : int
            Number of trials
        """
        self._data['trialLocs'] = trial_locs
        self._data['ntrials'] = ntrials
        logger.debug(f"Set trial info: {ntrials} trials")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for serialization.
        
        This method creates a dictionary representation of all metrics data
        suitable for saving to files or passing to other systems. The dictionary
        format matches the structure expected by .mat file writers.
        
        Returns:
        --------
        dict
            Dictionary containing all metrics fields with their calculated values.
        """
        return self._data.copy()
    
    @property
    def nepochs(self) -> int:
        """Get number of epochs"""
        if 'epochtags' in self._data:
            return len(self._data['epochtags'])
        return 0
    
    @property
    def ntrials(self) -> int:
        """Get number of trials"""
        return self._data.get('ntrials', 0)


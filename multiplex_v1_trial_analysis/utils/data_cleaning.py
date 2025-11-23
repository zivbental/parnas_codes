"""
Data Cleaning Utilities

Functions for cleaning and preprocessing position data.

This module provides utilities for cleaning position tracking data by removing
discontinuities. Discontinuities are detected as zero values, which typically
indicate tracking failures or data logging issues. The cleaner replaces these
zeros with the previous valid position value, maintaining temporal continuity.
"""

import numpy as np
from typing import Tuple, Union, List


class DataCleaner:
    """
    Handles data cleaning operations for position tracking data.
    
    This class provides methods for cleaning position data by removing
    discontinuities (tracking failures). The cleaning process identifies
    zero values (which indicate tracking loss) and replaces them with the
    most recent valid position, ensuring smooth trajectories.
    """
    
    @staticmethod
    def remove_discontinuities(
        position_data: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.ndarray, List[np.ndarray]]]:
        """
        Remove discontinuities from position tracking data.
        
        This function identifies zero values in the position data, which
        typically indicate tracking failures or data logging issues. These
        discontinuities are replaced with the previous valid position value,
        maintaining temporal continuity in the trajectory.
        
        The function handles both single-fly (1D) and multi-fly (2D) data.
        For multi-fly data, each fly's trajectory is cleaned independently.
        
        Parameters:
        -----------
        position_data : np.ndarray
            Position vector to clean. Can be 1D (single fly) or 2D (multiple flies).
            Values are typically normalized to the range [-1, 1].
        
        Returns:
        --------
        corrected_data : np.ndarray
            Corrected position vector with discontinuities removed.
            Has the same shape as input data.
        bad_indices : np.ndarray or list of np.ndarray
            Indices where zeros (discontinuities) were found and corrected.
            For 2D input: returns list of arrays (one per fly).
            For 1D input: returns single array.
            Note: Valid zero-crossings (fly actually at position 0) are also
            included, but this is typically rare in practice.
        """
        position_data = np.asarray(position_data).copy()
        
        # Handle 2D case (multiple flies)
        if position_data.ndim == 2:
            bad_indices_all = []
            for col in range(position_data.shape[1]):
                # Find all zero values for this fly
                # Zeros indicate tracking failures or discontinuities
                bad_indices = np.where(position_data[:, col] == 0)[0]
                bad_indices_all.append(bad_indices)
                
                # Replace each zero with the previous valid position
                # This maintains temporal continuity by "carrying forward" the last known position
                for idx in bad_indices:
                    if idx == 0:
                        # First frame is zero: keep it (no previous value to use)
                        # In practice, this is rare as tracking usually starts valid
                        position_data[idx, col] = position_data[idx, col]
                    else:
                        # Replace zero with previous frame's value
                        # This fills the gap with the last known position
                        position_data[idx, col] = position_data[idx - 1, col]
            
            return position_data, bad_indices_all
        else:
            # 1D case (single fly)
            # Find all zero values
            bad_indices = np.where(position_data == 0)[0]
            
            # Replace each zero with the previous valid position
            for idx in bad_indices:
                if idx == 0:
                    # First frame is zero: keep it (no previous value)
                    position_data[idx] = position_data[idx]
                else:
                    # Replace zero with previous frame's value
                    position_data[idx] = position_data[idx - 1]
            
            return position_data, bad_indices


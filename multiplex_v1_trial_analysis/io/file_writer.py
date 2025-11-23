"""
File Writer Module

Handles writing of analysis results to .mat files.

This module saves the analysis results to binary .mat files that can be
loaded by other analysis tools. The files contain structured data including
trial data, calculated metrics, and summary statistics. The format is
compatible with tools that can read .mat files (e.g., MATLAB, SciPy).
"""

import scipy.io as sio
from pathlib import Path
from typing import Optional, Tuple
import logging
import numpy as np

from ..models.trial_data import TrialData
from ..models.metrics import MetricsData
from ..utils.matlab_compat import convert_to_matlab_format
from ..exceptions import FileWriteError

logger = logging.getLogger(__name__)


class FileWriter:
    """
    Writes analysis results to .mat files.
    
    This class handles the export of analysis results to binary .mat files.
    It creates two output files:
    1. Main results file: Contains complete trial data and metrics
    2. Decisions/times file: Contains simplified decision and time change data
    
    The data is converted to a format compatible with .mat file readers
    before writing, ensuring proper array dimensions and data types.
    """
    
    def __init__(self, config=None):
        """
        Initialize the file writer.
        
        The configuration object is currently unused but kept for consistency
        with other I/O classes and potential future configuration needs.
        
        Parameters:
        -----------
        config : Config, optional
            Configuration object (currently unused).
        """
        self.config = config
    
    def write_results(
        self,
        filepath: str,
        trial_data: TrialData,
        metrics_data: MetricsData
    ) -> Tuple[str, str]:
        """
        Write analysis results to .mat files
        
        Parameters:
        -----------
        filepath : str
            Original input file path
        trial_data : TrialData
            Trial data structure
        metrics_data : MetricsData
            Metrics data structure
        
        Returns:
        --------
        Tuple[str, str]
            Paths to the two output files created
        
        Raises:
        -------
        FileWriteError
            If files cannot be written
        """
        input_path = Path(filepath)
        output_dir = input_path.parent
        
        # Generate output filenames
        base_name = input_path.stem
        
        # First file: main results
        mat_file1 = output_dir / f"{base_name}.mat"
        
        # Second file: decs_times
        mat_file2 = output_dir / f"{base_name}-flies.mat"
        
        try:
            # Write main results file
            self._write_main_file(mat_file1, trial_data, metrics_data)
            logger.info(f"Saved: {mat_file1}")
            
            # Write decs_times file
            self._write_decisions_times_file(mat_file2, metrics_data)
            logger.info(f"Saved: {mat_file2}")
            
            return str(mat_file1), str(mat_file2)
            
        except Exception as e:
            raise FileWriteError(f"Error writing output files: {e}") from e
    
    def _write_main_file(
        self,
        filepath: Path,
        trial_data: TrialData,
        metrics_data: MetricsData
    ) -> None:
        """
        Write main results file
        
        Parameters:
        -----------
        filepath : Path
            Output file path
        trial_data : TrialData
            Trial data structure
        metrics_data : MetricsData
            Metrics data structure
        """
        # Convert data structures to dictionaries for serialization
        # This extracts all fields into a flat dictionary structure
        fb_dict = trial_data.to_dict()
        fbm_dict = metrics_data.to_dict()
        
        # Convert to format compatible with .mat file readers
        # This ensures arrays have correct dimensions and data types
        fb_matlab = convert_to_matlab_format(fb_dict)
        fbm_matlab = convert_to_matlab_format(fbm_dict)
        
        # Save to .mat file using SciPy's save function
        # The file contains two variables: 'fb' (trial data) and 'fbm' (metrics)
        sio.savemat(str(filepath), {'fb': fb_matlab, 'fbm': fbm_matlab})
    
    def _write_decisions_times_file(
        self,
        filepath: Path,
        metrics_data: MetricsData
    ) -> None:
        """
        Write decisions and times file
        
        Parameters:
        -----------
        filepath : Path
            Output file path
        metrics_data : MetricsData
            Metrics data structure
        """
        # Calculate decs_times
        if 'decisionsChanges' not in metrics_data or 'timeChanges' not in metrics_data:
            logger.warning("Missing decisionsChanges or timeChanges, writing empty array")
            decs_times = np.array([]).reshape(0, 2)
        else:
            valid_mask = ~np.isnan(metrics_data['decisionsChanges'])
            decs_times = np.column_stack([
                metrics_data['decisionsChanges'][valid_mask],
                metrics_data['timeChanges'][valid_mask]
            ])
        
        # Save to .mat file
        sio.savemat(str(filepath), {'decs_times': decs_times})


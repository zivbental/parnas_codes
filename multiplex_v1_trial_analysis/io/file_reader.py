"""
File Reader Module

Handles reading and parsing of trial data files.

This module reads behavioral trial data from tab-delimited text files.
The file format consists of:
- A first line (usually metadata, skipped)
- A header line with column names (tab-separated)
- Data rows with 55 columns of floating-point values

The reader parses the headers, validates the data structure, and constructs
a TrialData object containing all the experimental data organized by field.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

from ..models.trial_data import TrialData
from ..exceptions import FileReadError, DataValidationError

logger = logging.getLogger(__name__)


class FileReader:
    """
    Reads and parses trial data from tab-delimited text files.
    
    This class handles the complete file reading workflow: opening the file,
    parsing headers, reading numeric data, validating structure, and
    constructing a TrialData object with properly organized data fields.
    """
    
    EXPECTED_COLUMNS = 55  # Total number of columns expected in the data file
    
    def __init__(self, config=None):
        """
        Initialize the file reader with configuration.
        
        The configuration is used to determine the number of flies tracked
        in the experiment, which is needed for organizing the position data.
        
        Parameters:
        -----------
        config : Config, optional
            Configuration object containing nflies parameter. If None, uses
            default value from TrialData class.
        """
        self.config = config
    
    def read(self, filepath: str) -> TrialData:
        """
        Read trial data from file
        
        Parameters:
        -----------
        filepath : str
            Path to input .txt log file
        
        Returns:
        --------
        TrialData
            Parsed trial data structure
        
        Raises:
        -------
        FileReadError
            If file cannot be read
        DataValidationError
            If data format is invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileReadError(f"File not found: {filepath}")
        
        logger.info(f"Reading file: {filepath}")
        
        try:
            # Open and read the file
            with open(filepath, 'r', encoding='utf-8') as f:
                # Skip the first line (usually contains metadata or is empty)
                # This line is not part of the structured data format
                first_line = f.readline()
                
                # Read the header line containing column names
                # Headers are tab-separated and define what each column represents
                headerline = f.readline().strip()
                
                # Read the numeric data rows
                # All 55 columns contain floating-point values (float32 for consistency)
                # Data is tab-delimited with no header row (header=None)
                data = pd.read_csv(f, sep='\t', header=None, dtype=np.float32)
            
            # Parse column headers from the header line
            # Headers are tab-separated and may contain whitespace that needs trimming
            headers = self._parse_headers(headerline)
            
            # Validate
            if len(headers) != self.EXPECTED_COLUMNS:
                raise DataValidationError(
                    f"Expected {self.EXPECTED_COLUMNS} columns, got {len(headers)}"
                )
            
            if data.shape[1] != self.EXPECTED_COLUMNS:
                raise DataValidationError(
                    f"Expected {self.EXPECTED_COLUMNS} data columns, got {data.shape[1]}"
                )
            
            # Build TrialData structure
            trial_data = self._build_trial_data(headers, data)
            
            logger.info(f"Successfully read {data.shape[0]} frames for {trial_data.nflies} flies")
            
            return trial_data
            
        except pd.errors.EmptyDataError as e:
            raise FileReadError(f"File is empty or invalid: {e}") from e
        except Exception as e:
            raise FileReadError(f"Error reading file: {e}") from e
    
    def _parse_headers(self, headerline: str) -> List[str]:
        """
        Parse the header line to extract column names.
        
        The header line contains tab-separated column names. This method
        splits the line and removes any leading/trailing whitespace from
        each header name, ensuring clean field names for data organization.
        
        Parameters:
        -----------
        headerline : str
            Single line containing tab-separated column header names.
        
        Returns:
        --------
        List[str]
            List of cleaned header names, one per column.
        """
        # Split by tab character to get individual column names
        headers = headerline.split('\t')
        # Remove leading/trailing whitespace from each header
        # This handles cases where the file format has inconsistent spacing
        headers = [h.strip() for h in headers]
        return headers
    
    def _build_trial_data(self, headers: List[str], data: pd.DataFrame) -> TrialData:
        """
        Build TrialData structure from parsed data
        
        Parameters:
        -----------
        headers : List[str]
            Column headers
        data : pd.DataFrame
            Data frame with trial data
        
        Returns:
        --------
        TrialData
            Populated trial data structure
        """
        nflies = self.config.nflies if self.config else TrialData.DEFAULT_NFLIES
        trial_data = TrialData(nflies=nflies)
        
        # Assign first 13 columns to named fields
        for i in range(TrialData.NAMED_FIELDS_COUNT):
            fldname = headers[i]
            trial_data.add_named_field(fldname, data.iloc[:, i].values)
        
        # Columns 14-33 go into EE matrix
        ee_matrix = data.iloc[:, TrialData.EE_COLUMNS_START:TrialData.EE_COLUMNS_END].values
        trial_data.set_ee_data(ee_matrix)
        
        # Columns 34-35 go to named fields
        for i in range(TrialData.ADDITIONAL_FIELDS_START, TrialData.ADDITIONAL_FIELDS_END):
            fldname = headers[i]
            trial_data.add_named_field(fldname, data.iloc[:, i].values)
        
        # Columns 36-55 go into cX matrix
        cx_matrix = data.iloc[:, TrialData.CX_COLUMNS_START:TrialData.CX_COLUMNS_END].values
        trial_data.set_cx_data(cx_matrix)
        
        # Set up compatibility fields
        trial_data.setup_compatibility_fields()
        
        # Initialize Time field: set first value to zero
        # This ensures time starts at zero, which is the standard convention
        # for experimental data analysis (relative time from start)
        if 'Time' in trial_data.data:
            trial_data.data['Time'][0, 0] = 0
        
        # Validate
        if not trial_data.validate():
            logger.warning("Trial data validation failed, but continuing")
        
        return trial_data


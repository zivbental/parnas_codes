"""
Trial Data Model

Encapsulates the trial data structure with validation and type safety.

This module defines the TrialData class, which represents the complete
experimental data from a behavioral trial. The data structure organizes
information from the 55-column input file into named fields, matrices,
and metadata. The class provides validation, type safety, and dictionary
compatibility for serialization.
"""

import numpy as np
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TrialData:
    """
    Represents trial data from behavioral experiments.
    
    This class encapsulates all experimental data from a single trial,
    organizing it into a structured format with named fields. The data
    includes position tracking, digital output signals (odor delivery),
    shock data, and metadata. The structure supports both programmatic
    access and dictionary-style serialization for file output.
    
    The data organization follows the 55-column file format:
    - Columns 0-12: Named fields (Time, digital outputs, etc.)
    - Columns 13-32: EE matrix (operant shock data, one column per fly)
    - Columns 34-35: Additional named fields
    - Columns 35-54: cX matrix (position tracking, one column per fly)
    """
    
    # Expected number of flies in typical experiments
    DEFAULT_NFLIES = 20
    
    # Column structure constants for organizing 55-column file format
    # These define how the columns are mapped to different data types
    NAMED_FIELDS_COUNT = 13  # First 13 columns (indices 0-12): named fields
    EE_COLUMNS_START = 13    # Columns 13-32: EE matrix (shock data, 20 columns)
    EE_COLUMNS_END = 33      # End of EE columns (exclusive)
    ADDITIONAL_FIELDS_START = 34  # Columns 34-35: additional named fields
    ADDITIONAL_FIELDS_END = 36    # End of additional fields (exclusive)
    CX_COLUMNS_START = 35    # Columns 35-54: cX matrix (position data, 20 columns)
    CX_COLUMNS_END = 55      # End of cX columns (exclusive, total 55 columns)
    
    def __init__(self, nflies: int = DEFAULT_NFLIES):
        """
        Initialize TrialData structure
        
        Parameters:
        -----------
        nflies : int
            Number of flies in the experiment (default: 20)
        """
        self.nflies = nflies
        self.data: Dict[str, np.ndarray] = {}
        self._header: Optional[Dict[str, Any]] = None
    
    @property
    def header(self) -> Optional[Dict[str, Any]]:
        """Get header information"""
        return self._header
    
    @header.setter
    def header(self, value: Optional[Dict[str, Any]]):
        """Set header information"""
        self._header = value
    
    def add_named_field(self, name: str, values: np.ndarray) -> None:
        """
        Add a named data field
        
        Parameters:
        -----------
        name : str
            Field name
        values : np.ndarray
            Field values (should be column vector)
        """
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.data[name] = values
        logger.debug(f"Added named field: {name} with shape {values.shape}")
    
    def set_ee_data(self, ee_matrix: np.ndarray) -> None:
        """
        Set EE (shock) data matrix
        
        Parameters:
        -----------
        ee_matrix : np.ndarray
            EE data matrix (should be n_frames x n_flies)
        """
        self.data['EE'] = ee_matrix
        logger.debug(f"Set EE data with shape {ee_matrix.shape}")
    
    def set_cx_data(self, cx_matrix: np.ndarray) -> None:
        """
        Set cX (position) data matrix
        
        Parameters:
        -----------
        cx_matrix : np.ndarray
            Position data matrix (should be n_frames x n_flies)
        """
        self.data['cX'] = cx_matrix
        logger.debug(f"Set cX data with shape {cx_matrix.shape}")
    
    def setup_compatibility_fields(self) -> None:
        """
        Set up compatibility field aliases for different naming conventions.
        
        This method creates aliases for odor field names to support different
        file format versions. Some files use generic names (LEFTODOR1, LEFTODOR2)
        while others use specific names (LEFTAIR, LEFTMCH, LEFTOCT). This method
        ensures both naming conventions work by creating aliases.
        
        The mapping is:
        - LEFTAIR and LEFTMCH both point to LEFTODOR1
        - LEFTOCT points to LEFTODOR2
        - Same pattern for right side (RIGHTAIR, RIGHTMCH, RIGHTOCT)
        
        Additionally, this method initializes the SHOCK field if it doesn't exist,
        creating a zero-filled array as a placeholder for shock data.
        """
        if 'LEFTODOR1' not in self.data:
            logger.warning("LEFTODOR1 not found, cannot set compatibility fields")
            return
        
        # Create aliases for different naming conventions
        # This allows the code to work with files that use either naming scheme
        self.data['LEFTAIR'] = self.data['LEFTODOR1']
        self.data['LEFTMCH'] = self.data['LEFTODOR1']
        self.data['LEFTOCT'] = self.data['LEFTODOR2']
        self.data['RIGHTAIR'] = self.data['RIGHTODOR1']
        self.data['RIGHTMCH'] = self.data['RIGHTODOR1']
        self.data['RIGHTOCT'] = self.data['RIGHTODOR2']
        
        # Initialize SHOCK field if missing
        # Some experiments don't use shocks, so we create a zero-filled placeholder
        if 'SHOCK' not in self.data:
            n_frames = len(self.data['LEFTODOR1'])
            self.data['SHOCK'] = np.zeros((n_frames, 1))
        
        logger.debug("Compatibility fields set up")
    
    def validate(self) -> bool:
        """
        Validate the trial data structure
        
        Returns:
        --------
        bool
            True if valid, False otherwise
        """
        required_fields = ['Time', 'LEFTODOR1', 'LEFTODOR2', 'RIGHTODOR1', 'RIGHTODOR2', 'EE', 'cX']
        
        for field in required_fields:
            if field not in self.data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Check dimensions
        if 'cX' in self.data:
            if self.data['cX'].shape[1] != self.nflies:
                logger.warning(f"cX has {self.data['cX'].shape[1]} columns, expected {self.nflies}")
        
        if 'EE' in self.data:
            if self.data['EE'].shape[1] != self.nflies:
                logger.warning(f"EE has {self.data['EE'].shape[1]} columns, expected {self.nflies}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for serialization.
        
        This method creates a dictionary representation of the TrialData structure
        suitable for saving to files or passing to other systems. The dictionary
        format matches the structure expected by .mat file writers.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'nflies': Number of flies
            - 'data': Dictionary of all data fields (position, digital outputs, etc.)
            - 'header': Optional header metadata (if present)
        """
        result = {
            'nflies': self.nflies,
            'data': self.data.copy()
        }
        if self._header is not None:
            result['header'] = self._header
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        if key == 'nflies':
            return self.nflies
        elif key == 'data':
            return self.data
        elif key == 'header':
            return self._header
        else:
            raise KeyError(f"Key '{key}' not found")
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment"""
        if key == 'nflies':
            self.nflies = value
        elif key == 'data':
            self.data = value
        elif key == 'header':
            self._header = value
        else:
            raise KeyError(f"Key '{key}' not found")
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        if key == 'nflies' or key == 'data' or key == 'header':
            return True
        return False


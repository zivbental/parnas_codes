"""
Utility Functions Module

This package contains utility functions and classes used throughout the analysis pipeline.

The utility modules provide:
- Filtering: Savitzky-Golay polynomial smoothing filters
- Location classification: Position categorization (left/center/right)
- Data cleaning: Discontinuity removal and preprocessing
- Format conversion: Data structure conversion for file compatibility
"""

from .filters import SavitzkyGolayFilter
from .location import TernaryLocationClassifier
from .data_cleaning import DataCleaner
from .matlab_compat import fix, convert_to_matlab_format

__all__ = [
    'SavitzkyGolayFilter',
    'TernaryLocationClassifier',
    'DataCleaner',
    'fix',
    'convert_to_matlab_format'
]


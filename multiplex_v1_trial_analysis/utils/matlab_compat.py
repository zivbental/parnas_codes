"""
Compatibility Helpers

Utility functions for data format conversion and mathematical operations.

This module provides functions for converting Python data structures to formats
compatible with .mat file readers, and implements mathematical operations that
may behave differently between Python and other numerical computing environments.
"""

import numpy as np
from typing import Union


def fix(x: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Round toward zero (truncate fractional part).
    
    This function rounds numbers toward zero, which means:
    - Positive numbers: rounds down (e.g., 3.7 -> 3)
    - Negative numbers: rounds up (e.g., -3.7 -> -3)
    
    This differs from floor() which always rounds down (e.g., -3.7 -> -4).
    The fix() operation is equivalent to truncation: removing the fractional part.
    
    Parameters:
    -----------
    x : float or np.ndarray
        Input value(s) to round toward zero.
    
    Returns:
    --------
    int or np.ndarray
        Rounded value(s) with fractional part removed, rounded toward zero.
    """
    return np.trunc(x)


def convert_to_matlab_format(data: Union[dict, np.ndarray, list, float, int, str]) -> Union[dict, np.ndarray]:
    """
    Convert Python data structures to .mat file compatible format.
    
    .mat file readers expect specific array shapes and dimensions:
    - Arrays must be at least 2D (no 1D arrays)
    - 1D arrays should be column vectors (shape: (n, 1))
    - Scalars should be 2D arrays (shape: (1, 1))
    
    This function recursively converts nested structures (dictionaries) and
    ensures all arrays have the correct dimensions for .mat file compatibility.
    
    Parameters:
    -----------
    data : dict, np.ndarray, list, float, int, str
        Data structure to convert. Can be nested (dictionaries containing
        arrays, or arrays containing other arrays).
    
    Returns:
    --------
    dict or np.ndarray
        Converted data structure with all arrays in .mat file compatible format.
        Dictionaries are recursively converted, preserving structure.
    """
    if isinstance(data, dict):
        # Recursively convert dictionary values
        result = {}
        for key, value in data.items():
            result[key] = convert_to_matlab_format(value)
        return result
    elif isinstance(data, np.ndarray):
        # Ensure arrays have correct dimensions
        if data.ndim == 1:
            # 1D array: convert to column vector (n, 1)
            return data.reshape(-1, 1)
        elif data.ndim == 0:
            # Scalar: convert to 2D array (1, 1)
            return np.array([[data]])
        else:
            # Already 2D or higher: use as-is
            return data
    elif isinstance(data, (list, tuple)):
        # Convert list/tuple to array first, then format
        arr = np.array(data)
        return convert_to_matlab_format(arr)
    elif isinstance(data, str):
        # Strings are passed through unchanged
        return data
    elif isinstance(data, (int, float, np.integer, np.floating)):
        # Scalars: convert to 2D array (1, 1)
        return np.array([[data]])
    else:
        # Unknown type: pass through unchanged
        return data


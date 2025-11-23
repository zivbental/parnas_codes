"""
Custom Exceptions

Domain-specific exceptions for the behavioral analysis pipeline.

This module defines a hierarchy of custom exceptions that provide clear,
specific error information for different failure modes in the analysis pipeline.
Using specific exception types allows callers to handle different error
conditions appropriately (e.g., retry file operations, skip invalid data, etc.).

All exceptions inherit from TrialAnalysisError, which serves as the base
exception for the entire domain. This allows catching all analysis-related
errors with a single except clause if desired.
"""


class TrialAnalysisError(Exception):
    """
    Base exception for all trial analysis errors.
    
    This is the root exception class for the analysis pipeline. All other
    exceptions in this module inherit from this class, allowing code to catch
    all analysis-related errors with a single except clause:
    
    try:
        # analysis code
    except TrialAnalysisError as e:
        # handle any analysis error
    """
    pass


class FileReadError(TrialAnalysisError):
    """
    Raised when file reading operations fail.
    
    This exception is raised when:
    - File cannot be found at the specified path
    - File exists but cannot be opened (permissions, locks)
    - File format is invalid or corrupted
    - File structure doesn't match expected format (wrong number of columns, etc.)
    
    The exception message should indicate the specific reason for the failure.
    """
    pass


class FileWriteError(TrialAnalysisError):
    """
    Raised when file writing operations fail.
    
    This exception is raised when:
    - Output directory doesn't exist and cannot be created
    - Insufficient permissions to write to the target location
    - Disk is full
    - File is locked by another process
    
    The exception message should indicate the specific reason for the failure.
    """
    pass


class DataValidationError(TrialAnalysisError):
    """
    Raised when data validation checks fail.
    
    This exception is raised when data structure or content doesn't meet
    required criteria, such as:
    - Missing required columns or fields
    - Invalid data types (e.g., strings where numbers expected)
    - Data values outside expected ranges
    - Inconsistent data dimensions
    
    This is distinct from FileReadError because it indicates the file was
    successfully read, but the data content is invalid.
    """
    pass


class ProcessingError(TrialAnalysisError):
    """
    Raised when data processing operations fail.
    
    This exception is raised during the analysis pipeline execution when:
    - Numerical computations fail (e.g., division by zero, invalid matrix operations)
    - Algorithm-specific errors occur (e.g., epoch detection fails)
    - Data transformations produce invalid results
    
    This indicates a problem during computation, not with input data validation.
    """
    pass


class MissingDataError(TrialAnalysisError):
    """
    Raised when required data is missing from the dataset.
    
    This exception is raised when:
    - Expected data fields are absent from the trial data structure
    - Required columns are missing from the input file
    - Epochs or other structural elements are not found
    
    This is distinct from DataValidationError in that it indicates absence
    of data rather than invalid data.
    """
    pass


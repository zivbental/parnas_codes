"""
Input/Output Operations Module

This package handles all file I/O operations for the analysis pipeline.

The I/O modules provide:
- File reading: Parses tab-delimited trial data files (55-column format)
- File writing: Saves analysis results to .mat files for compatibility
- File dialog: Interactive file selection using GUI dialogs
"""

from .file_reader import FileReader
from .file_writer import FileWriter
from .file_dialog import FileDialog

__all__ = ['FileReader', 'FileWriter', 'FileDialog']


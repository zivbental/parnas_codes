"""
File Dialog Module

Handles file selection UI.

This module provides a graphical file selection dialog using tkinter.
It allows users to interactively browse and select input files for analysis,
providing a user-friendly alternative to command-line file path specification.
"""

import logging
from pathlib import Path
from typing import Optional
from tkinter import filedialog
import tkinter as tk

from ..exceptions import FileReadError

logger = logging.getLogger(__name__)


class FileDialog:
    """
    Provides file selection dialog functionality.
    
    This class wraps tkinter's file dialog to provide a simple interface
    for selecting input files. The dialog opens in a default directory
    (configurable) and allows users to browse and select .txt log files.
    """
    
    def __init__(self, config=None):
        """
        Initialize the file dialog with default directory settings.
        
        The default directory determines where the file dialog opens when
        first displayed. This can be configured via the Config object or
        uses a hardcoded default path.
        
        Parameters:
        -----------
        config : Config, optional
            Configuration object containing default_file_dialog_dir setting.
            If None, uses a hardcoded default path.
        """
        self.config = config
        # Set default directory for file dialog
        # Uses config value if available, otherwise falls back to default
        self.default_dir = (
            config.default_file_dialog_dir if config
            else r'D:\user\Desktop\Behavior Log Files'
        )
    
    def ask_open_file(
        self,
        title: str = "Please choose a TXT LOG file",
        filetypes: Optional[list] = None,
        initialdir: Optional[str] = None
    ) -> Optional[str]:
        """
        Open file dialog to select input file
        
        Parameters:
        -----------
        title : str
            Dialog title
        filetypes : list, optional
            List of (description, pattern) tuples for file types
        initialdir : str, optional
            Initial directory for dialog
        
        Returns:
        --------
        str or None
            Selected file path, or None if canceled
        """
        if filetypes is None:
            filetypes = [('TXT files', '*.txt'), ('All files', '*.*')]
        
        if initialdir is None:
            initialdir = self.default_dir
        
        # Create a hidden root window for tkinter
        # tkinter requires a root window to display dialogs, but we hide it
        root = tk.Tk()
        root.withdraw()  # Hide the main window so only dialog is visible
        root.attributes('-topmost', True)  # Bring dialog to front of other windows
        
        try:
            # Display the file selection dialog
            # Returns the selected file path or empty string if canceled
            filepath = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes,
                initialdir=initialdir
            )
            
            # Check if user canceled the dialog
            if not filepath:
                logger.info("File selection canceled")
                return None
            
            # Validate that the selected file actually exists
            # This catches cases where the path is invalid or file was deleted
            if not Path(filepath).exists():
                raise FileReadError(f"Selected file does not exist: {filepath}")
            
            logger.info(f"Selected file: {filepath}")
            return filepath
            
        except Exception as e:
            # Log and re-raise any errors during file selection
            logger.error(f"Error in file dialog: {e}")
            raise
        finally:
            # Always destroy the root window to clean up tkinter resources
            root.destroy()


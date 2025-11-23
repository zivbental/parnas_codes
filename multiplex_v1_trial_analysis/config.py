"""
Configuration Management

Handles all configuration settings for the analysis pipeline.

This module provides a centralized configuration system that defines all
parameters used throughout the behavioral analysis. Configuration values
can be set via defaults, environment variables, or programmatically,
allowing flexible adaptation to different experimental setups and analysis
requirements.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    Configuration settings for the behavioral analysis pipeline.
    
    This class encapsulates all configurable parameters needed for the analysis,
    including file I/O paths, processing parameters, behavioral thresholds,
    and output preferences. All parameters have sensible defaults based on
    typical experimental setups.
    
    The configuration can be customized through:
    - Default values (hardcoded in the class)
    - Environment variables (via from_env() class method)
    - Programmatic assignment (after instantiation)
    """
    
    # File I/O settings
    # Default directory where file dialog opens when user selects input file interactively
    default_file_dialog_dir: str = r'D:\user\Desktop\Behavior Log Files'
    
    # Processing parameters
    # Number of flies tracked in the experiment (typically 20 for standard setups)
    nflies: int = 20
    # Smoothing window size as a percentage of total data points
    # Smaller values (0.005 = 0.5%) create smoother curves but may lose detail
    span_pc: float = 0.005
    # Polynomial order for Savitzky-Golay smoothing filter
    # Higher order preserves more features but may introduce artifacts
    smoothing_order: int = 2
    # Gaussian full-width at half-maximum for decision filtering
    # Currently not actively used but kept for compatibility
    gaussian_fwhm: int = 50
    
    # Decision counting parameters
    # Minimum number of decisions a fly must make to be included in analysis
    # Flies with fewer decisions are marked as NaN (invalid)
    min_decisions: int = 1
    # Half-width of the decision zone filter (in normalized coordinates, typically -1 to 1)
    # Larger values require flies to move further from center to count as decisions
    decision_halfwidth: float = 0.3
    
    # Location classification
    # Half-width of the central choicepoint zone (in normalized coordinates)
    # Positions within [-0.2, 0.2] are considered "center" (mixed odors)
    # Positions > 0.2 are "right", positions < -0.2 are "left"
    # This accounts for the physical mixing zone in the experimental chamber
    choicepoint_halfwidth: float = 0.2
    
    # Time ratio calculation
    # Frame rate of the video recording system (frames per second)
    # Used to convert frame-based measurements to time-based metrics
    frames_per_second: float = 30.0
    # Half-length of the experimental chamber in millimeters
    # Used to convert normalized position changes to physical speed measurements
    half_chamber_length_mm: float = 25.0
    
    # Epoch detection
    # Time correction factor (in frames) to account for timing synchronization issues
    # The value 30 frames = 1 second at 30 fps, used to align digital outputs
    epoch_delta: int = 30
    # Time threshold (in seconds) for detecting logging breaks
    # Gaps longer than this indicate the start of a new experimental epoch
    logging_break_time: float = 60.0
    
    # Shock metrics
    # Frame rate used for shock time calculations (NTSC video standard: 29.97 fps)
    # Different from frames_per_second because shock timing uses a different clock
    log_frame_rate: float = 29.97
    
    # Output settings
    # Whether to automatically save analysis results as .mat files
    # These files can be loaded by other analysis tools or for archival purposes
    save_mat_files: bool = True
    # Whether to display matplotlib plots during analysis
    # Set False for batch processing or headless environments
    show_plots: bool = True
    
    # Logging
    # Verbosity level: DEBUG (most detail), INFO (normal), WARNING/ERROR (minimal)
    log_level: str = 'INFO'
    # Whether to write log messages to a file in addition to console
    log_to_file: bool = False
    # Path to log file (only used if log_to_file is True)
    log_file_path: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'Config':
        """
        Create configuration instance with environment variable overrides.
        
        This class method allows configuration to be customized via environment
        variables, which is useful for:
        - Deployment in different environments (dev/test/prod)
        - Batch processing scripts that need different settings
        - Integration with workflow management systems
        
        Environment variables checked:
        - MULTIPLEX_NFLIES: Override number of flies
        - MULTIPLEX_LOG_LEVEL: Override logging verbosity
        - MULTIPLEX_LOG_FILE: Enable file logging and set log file path
        
        Parameters:
        -----------
        (None - class method)
        
        Returns:
        --------
        Config
            Configuration instance with defaults and any environment variable overrides
        """
        config = cls()
        
        # Check for environment variables and override defaults if present
        # This allows external configuration without code changes
        if 'MULTIPLEX_NFLIES' in os.environ:
            config.nflies = int(os.environ['MULTIPLEX_NFLIES'])
        if 'MULTIPLEX_LOG_LEVEL' in os.environ:
            config.log_level = os.environ['MULTIPLEX_LOG_LEVEL']
        if 'MULTIPLEX_LOG_FILE' in os.environ:
            config.log_to_file = True
            config.log_file_path = os.environ['MULTIPLEX_LOG_FILE']
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Useful for serialization, logging configuration state, or passing
        configuration values to functions that expect dictionaries.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing all configuration parameters
        """
        return {
            'default_file_dialog_dir': self.default_file_dialog_dir,
            'nflies': self.nflies,
            'span_pc': self.span_pc,
            'smoothing_order': self.smoothing_order,
            'gaussian_fwhm': self.gaussian_fwhm,
            'min_decisions': self.min_decisions,
            'decision_halfwidth': self.decision_halfwidth,
            'choicepoint_halfwidth': self.choicepoint_halfwidth,
            'frames_per_second': self.frames_per_second,
            'half_chamber_length_mm': self.half_chamber_length_mm,
            'epoch_delta': self.epoch_delta,
            'logging_break_time': self.logging_break_time,
            'log_frame_rate': self.log_frame_rate,
            'save_mat_files': self.save_mat_files,
            'show_plots': self.show_plots,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file,
            'log_file_path': self.log_file_path,
        }


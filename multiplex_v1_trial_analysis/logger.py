"""
Logging Setup

Centralized logging configuration for the analysis pipeline.

This module provides a unified logging system that can output messages to both
console and file destinations. The logging system is configured with timestamps,
module names, and severity levels to help track the analysis workflow and
debug issues when they occur.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import Config


def setup_logging(config: Optional[Config] = None) -> logging.Logger:
    """
    Set up and configure the logging system for the analysis pipeline.
    
    This function configures the root logger with appropriate handlers for
    console output and optional file logging. The log level is set based on
    configuration, and file logging (if enabled) always captures DEBUG level
    messages for complete traceability.
    
    The logging format includes:
    - Timestamp: When the message was logged
    - Module name: Which component generated the message
    - Log level: Severity (DEBUG, INFO, WARNING, ERROR)
    - Message: The actual log content
    
    Parameters:
    -----------
    config : Config, optional
        Configuration object containing logging settings. If None, creates
        a default Config instance with default logging settings.
    
    Returns:
    --------
    logging.Logger
        Configured root logger instance that can be used throughout the application
    """
    if config is None:
        config = Config()
    
    # Get the root logger for the entire package
    # All module loggers will be children of this root logger
    logger = logging.getLogger('multiplex_v1_trial_analysis')
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # Remove any existing handlers to avoid duplicate messages
    # This is important when setup_logging is called multiple times
    logger.handlers.clear()
    
    # Configure console handler: writes log messages to stdout
    # Console output uses the configured log level (INFO by default)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.log_level.upper()))
    
    # Define log message format: timestamp, module, level, message
    # This format provides comprehensive information for debugging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Configure file handler if file logging is enabled
    # File logs capture DEBUG level regardless of console level for complete records
    if config.log_to_file and config.log_file_path:
        log_path = Path(config.log_file_path)
        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        # File handler always captures DEBUG level for complete traceability
        # This ensures we have full details even when console shows only INFO
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module or component.
    
    This function retrieves a logger that inherits configuration from the root
    logger. Module-specific loggers automatically include the module name in
    log messages, making it easy to trace which component generated each message.
    
    Usage pattern:
    - In each module: logger = get_logger(__name__)
    - This creates a logger named 'multiplex_v1_trial_analysis.module_name'
    - All log messages from that module will include the module name
    
    Parameters:
    -----------
    name : str, optional
        Logger name, typically the module's __name__. If None, returns the
        root logger. The name is automatically prefixed with the package name
        to create a hierarchical logger structure.
    
    Returns:
    --------
    logging.Logger
        Logger instance configured to inherit from the root logger
    """
    if name is None:
        # Return root logger if no specific name provided
        return logging.getLogger('multiplex_v1_trial_analysis')
    # Create child logger with hierarchical name (e.g., 'multiplex_v1_trial_analysis.processing.pipeline')
    # This allows filtering and different handling of logs from different modules
    return logging.getLogger(f'multiplex_v1_trial_analysis.{name}')


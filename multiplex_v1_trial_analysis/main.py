"""
Main Entry Point

Command-line interface for the behavioral analysis pipeline.

This module provides the primary entry point for running the behavioral analysis.
It handles command-line argument parsing, file selection (either via command line
or interactive file dialog), and orchestrates the complete analysis workflow
from data reading through processing to visualization and output generation.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# The code supports two execution modes: direct script execution and module import.
# When run directly (python main.py), we need to adjust the Python path to allow
# absolute imports. When imported as a module, relative imports work correctly.
if __name__ == '__main__':
    # Add parent directory to path so we can import from the package root
    # This allows absolute imports to work when the script is executed directly
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from multiplex_v1_trial_analysis.config import Config
    from multiplex_v1_trial_analysis.logger import setup_logging, get_logger
    from multiplex_v1_trial_analysis.io.file_reader import FileReader
    from multiplex_v1_trial_analysis.io.file_dialog import FileDialog
    from multiplex_v1_trial_analysis.processing.pipeline import Pipeline
    from multiplex_v1_trial_analysis.exceptions import TrialAnalysisError
    # Initialize logger for direct execution mode
    # We use a separate import to avoid circular dependencies
    from multiplex_v1_trial_analysis.logger import get_logger as _get_logger
    logger = _get_logger(__name__)
else:
    # When imported as a module, use relative imports which are more efficient
    # and follow Python package conventions
    from .config import Config
    from .logger import setup_logging, get_logger
    from .io.file_reader import FileReader
    from .io.file_dialog import FileDialog
    from .processing.pipeline import Pipeline
    from .exceptions import TrialAnalysisError
    logger = get_logger(__name__)


def main():
    """
    Main entry point for the behavioral analysis pipeline.
    
    This function orchestrates the complete workflow:
    1. Parses command-line arguments to configure behavior
    2. Sets up logging based on user preferences
    3. Obtains input file path (from command line or file dialog)
    4. Reads and validates the trial data file
    5. Executes the analysis pipeline (smoothing, metrics, statistics)
    6. Optionally generates plots and saves output files
    7. Handles errors gracefully with informative messages
    
    The function supports both interactive mode (with file dialog) and
    command-line mode (with file path argument), making it flexible
    for both GUI-based and script-based workflows.
    """
    parser = argparse.ArgumentParser(
        description='Behavioral Analysis Pipeline for Multiplex V1 Trial Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Interactive mode with file dialog
  %(prog)s input.txt                 # Process specific file
  %(prog)s input.txt --no-plots      # Process without displaying plots
  %(prog)s input.txt --no-save      # Process without saving .mat files
        """
    )
    
    # Input file argument: optional, allows interactive file selection if omitted
    parser.add_argument(
        'input_file',
        nargs='?',
        type=str,
        help='Path to input .txt log file (optional, opens file dialog if not provided)'
    )
    
    # Display control: when set, prevents matplotlib plots from being shown
    # Useful for batch processing or when running in headless environments
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not display plots'
    )
    
    # Output control: when set, skips writing .mat files to disk
    # Useful for quick analysis runs where only statistics are needed
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save .mat output files'
    )
    
    # Logging verbosity: controls how much detail appears in console output
    # DEBUG shows all operations, INFO shows major steps, WARNING/ERROR show only issues
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    # File logging: when provided, writes all log messages to the specified file
    # File logs always include DEBUG level for complete traceability
    parser.add_argument(
        '--log-file',
        type=str,
        help='Path to log file (enables file logging)'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration object, loading defaults and any environment variable overrides
    # Configuration contains all analysis parameters (smoothing, epochs, etc.)
    config = Config.from_env()
    # Override log settings from command line arguments
    config.log_level = args.log_level
    if args.log_file:
        config.log_to_file = True
        config.log_file_path = args.log_file
    
    # Configure logging system with the specified verbosity and output destinations
    # This sets up both console and optional file handlers
    setup_logging(config)
    
    logger.info("=" * 60)
    logger.info("Multiplex V1 Trial Analysis Pipeline")
    logger.info("=" * 60)
    
    try:
        # Determine input file source: command line argument or interactive dialog
        # If file path provided, validate it exists; otherwise launch file picker
        if args.input_file:
            input_filepath = args.input_file
            if not Path(input_filepath).exists():
                logger.error(f"Input file not found: {input_filepath}")
                sys.exit(1)
        else:
            # No file provided: open interactive file selection dialog
            # This allows users to browse and select files without typing paths
            logger.info("Opening file dialog...")
            file_dialog = FileDialog(config)
            input_filepath = file_dialog.ask_open_file()
            if input_filepath is None:
                logger.info("No file selected, exiting")
                sys.exit(0)
        
        logger.info(f"Processing file: {input_filepath}")
        
        # Step 1: Read and parse the trial data file
        # The FileReader handles all file format parsing, column mapping, and data validation
        file_reader = FileReader(config)
        trial_data = file_reader.read(input_filepath)
        
        # Step 2: Execute the complete analysis pipeline
        # This orchestrates: smoothing, epoch detection, metrics calculation,
        # statistics computation, optional plotting, and optional file saving
        pipeline = Pipeline(config)
        metrics_data = pipeline.process(
            trial_data,
            input_filepath,
            show_plots=not args.no_plots,  # Invert flag: --no-plots means show_plots=False
            save_results=not args.no_save   # Invert flag: --no-save means save_results=False
        )
        
        logger.info("=" * 60)
        logger.info("Analysis completed successfully")
        logger.info("=" * 60)
        
    except TrialAnalysisError as e:
        # Handle domain-specific errors (file reading, data validation, etc.)
        # These are expected error conditions with clear error messages
        logger.error(f"Analysis error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # User cancelled the operation (Ctrl+C)
        # Exit gracefully without error code
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        # Catch-all for unexpected errors
        # Log full traceback for debugging while showing user-friendly message
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


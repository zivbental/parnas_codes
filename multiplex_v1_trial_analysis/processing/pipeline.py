"""
Pipeline Orchestrator

Main processing pipeline that orchestrates the entire analysis workflow.

This module provides the high-level Pipeline class that coordinates all stages
of the behavioral analysis. The pipeline executes a five-step workflow:
1. Data smoothing to reduce tracking artifacts
2. Metrics calculation (decisions, time ratios, speed)
3. Statistical summarization
4. Visualization (optional)
5. Results saving (optional)

The pipeline design allows each stage to be independently configured and
tested, making the system modular and maintainable.
"""

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.trial_data import TrialData
    from ..models.metrics import MetricsData

from ..processing.smoothing import SmoothingProcessor
from ..analysis.metrics_calculator import MetricsCalculator
from ..analysis.statistics import StatisticsCalculator
from ..visualization.plot_manager import PlotManager
from ..io.file_writer import FileWriter
from ..config import Config

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main processing pipeline orchestrator.
    
    This class coordinates the complete analysis workflow by managing the
    sequence of processing steps and the data flow between them. Each stage
    is handled by a specialized processor, allowing clear separation of
    concerns and easy modification of individual steps.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the analysis pipeline with all required processors.
        
        This constructor creates instances of all the specialized processors
        needed for the analysis. Each processor is configured with the shared
        configuration object, ensuring consistent parameter settings across
        all analysis stages.
        
        Parameters:
        -----------
        config : Config
            Configuration object containing all analysis parameters (smoothing
            settings, behavioral thresholds, file paths, etc.)
        """
        self.config = config
        # Smoothing processor: reduces noise in position tracking data
        self.smoothing_processor = SmoothingProcessor(config)
        # Metrics calculator: computes behavioral metrics (decisions, time ratios, speed)
        self.metrics_calculator = MetricsCalculator(config)
        # Statistics calculator: computes summary statistics from metrics
        self.statistics_calculator = StatisticsCalculator()
        # Plot manager: handles visualization of results
        self.plot_manager = PlotManager()
        # File writer: saves analysis results to disk
        self.file_writer = FileWriter(config)
    
    def process(
        self,
        trial_data: 'TrialData',
        input_filepath: str,
        show_plots: Optional[bool] = None,
        save_results: Optional[bool] = None
    ) -> 'MetricsData':
        """
        Execute the complete analysis pipeline on trial data.
        
        This method orchestrates the five-stage analysis workflow:
        1. Smoothing: Reduces tracking noise using Savitzky-Golay filter
        2. Metrics: Calculates behavioral metrics (decisions, time ratios, speed)
        3. Statistics: Computes summary statistics across flies
        4. Visualization: Generates plots (if enabled)
        5. Output: Saves results to files (if enabled)
        
        The pipeline modifies the trial_data in place during smoothing, then
        creates new metrics_data structures for the calculated results. Each
        stage builds upon the results of previous stages.
        
        Parameters:
        -----------
        trial_data : TrialData
            Trial data structure containing raw position tracking data and
            digital output signals. This is modified in place during smoothing.
        input_filepath : str
            Path to the input file. Used for naming output files (e.g., to
            create output.mat in the same directory as input.txt).
        show_plots : bool, optional
            Whether to display matplotlib plots. If None, uses the value from
            config.show_plots. Set to False for batch processing or headless environments.
        save_results : bool, optional
            Whether to save analysis results to .mat files. If None, uses the
            value from config.save_mat_files. Set to False for quick analysis runs.
        
        Returns:
        --------
        MetricsData
            Complete metrics data structure containing all calculated behavioral
            metrics, epoch information, statistics, and trained odor preferences.
            This structure can be used for further analysis or exported to files.
        """
        logger.info("Starting analysis pipeline...")
        
        # Step 1: Smooth position tracking data
        # This reduces high-frequency noise and artifacts from video tracking
        # The smoothing is applied in-place to the cX (position) data
        logger.info("Smoothing paths...")
        self.smoothing_processor.process(trial_data)
        
        # Step 2: Calculate all behavioral metrics
        # This includes: epoch detection, decision counting, time ratio calculation,
        # speed measurement, and trained odor identification
        # Returns a complete MetricsData structure with all computed values
        logger.info("Calculating behavioral metrics...")
        metrics_data = self.metrics_calculator.calculate(trial_data)
        
        # Step 3: Calculate summary statistics
        # Computes means, standard errors, and other aggregate statistics across flies
        # The statistics are added to the metrics_data structure
        logger.info("Calculating statistics...")
        self.statistics_calculator.calculate_summary(metrics_data)
        
        # Step 4: Generate visualization plots (optional)
        # Creates two figure windows: position traces and summary statistics
        # Plots are only generated if show_plots is True
        if show_plots is None:
            show_plots = self.config.show_plots
        
        if show_plots:
            logger.info("Generating plots...")
            # Generate both plot types: traces (individual fly positions) and summary (aggregate metrics)
            fig1, fig2 = self.plot_manager.generate_all_plots(trial_data, metrics_data)
            # Display the plots (blocks until windows are closed)
            self.plot_manager.show_plots()
        
        # Step 5: Save results to files (optional)
        # Writes analysis results to .mat files that can be loaded by other tools
        # Files are saved in the same directory as the input file
        if save_results is None:
            save_results = self.config.save_mat_files
        
        if save_results:
            logger.info("Saving results...")
            # Write both the trial data (fb) and metrics data (fbm) structures
            self.file_writer.write_results(input_filepath, trial_data, metrics_data)
        
        logger.info("Pipeline completed successfully")
        
        return metrics_data


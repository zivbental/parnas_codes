"""
Processing Modules

This package contains all data processing components for the behavioral analysis pipeline.

The processing modules handle:
- Pipeline orchestration: Coordinates the complete analysis workflow
- Data smoothing: Reduces noise in position tracking data
- Epoch detection: Identifies experimental periods and boundaries
- Decision counting: Detects and counts behavioral choices
- Time ratio calculation: Computes time spent in each odor zone
- Trained odor identification: Determines which odor was used for training
"""

from .pipeline import Pipeline
from .smoothing import SmoothingProcessor
from .epochs import EpochDetector
from .decisions import DecisionCounter
from .timing import TimeRatioCalculator
from .trained_odor import TrainedOdorIdentifier

__all__ = [
    'Pipeline',
    'SmoothingProcessor',
    'EpochDetector',
    'DecisionCounter',
    'TimeRatioCalculator',
    'TrainedOdorIdentifier'
]


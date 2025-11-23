"""
Data Models Module

This package contains data structure classes that encapsulate experimental data and metrics.

The model classes provide:
- TrialData: Represents raw experimental data from a trial (position tracking, digital outputs, shocks)
- MetricsData: Represents calculated behavioral metrics (decisions, time ratios, statistics)

Both classes provide dictionary-style access for compatibility while maintaining type safety
through setter methods and validation.
"""

from .trial_data import TrialData
from .metrics import MetricsData

__all__ = ['TrialData', 'MetricsData']


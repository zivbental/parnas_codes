"""
Analysis and Metrics Module

This package contains components for calculating and summarizing behavioral metrics.

The analysis modules handle:
- Metrics calculation: Orchestrates computation of all behavioral metrics
  (decisions, time ratios, speed, trained odor preferences)
- Statistics: Computes aggregate statistics across flies (means, standard errors)
"""

from .metrics_calculator import MetricsCalculator
from .statistics import StatisticsCalculator

__all__ = ['MetricsCalculator', 'StatisticsCalculator']


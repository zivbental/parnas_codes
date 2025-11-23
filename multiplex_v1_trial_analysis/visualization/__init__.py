"""
Visualization Module

This package contains all plotting and visualization components for the analysis pipeline.

The visualization modules provide:
- Trace plotting: Position trajectories with digital outputs and annotations
- Summary plotting: Statistical summaries and before/after comparisons
- Plot management: Orchestrates plot generation and display
"""

from .traces import TracePlotter
from .summary import SummaryPlotter
from .plot_manager import PlotManager

__all__ = ['TracePlotter', 'SummaryPlotter', 'PlotManager']


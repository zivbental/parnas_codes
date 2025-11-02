"""
Multiplex Analysis Package
==========================

This package provides the core MultiplexTrial class with all analysis methods
through mixin composition.
"""

from .multiplex_core import MultiplexTrial as BaseMultiplexTrial
from .time_analysis import TimeAnalysisMixin
from .snapshot_analysis import SnapshotAnalysisMixin
from .learning_valence_analysis import LearningValenceAnalysisMixin
from .valence_habituation_analysis import ValenceHabituationAnalysisMixin


# Create the complete MultiplexTrial class by composing mixins
class MultiplexTrial(
    BaseMultiplexTrial,  # base class from multiplex_core
    TimeAnalysisMixin,
    SnapshotAnalysisMixin,
    LearningValenceAnalysisMixin,
    ValenceHabituationAnalysisMixin
):
    """
    Complete MultiplexTrial class with all analysis methods.
    
    This class combines the base MultiplexTrial with all analysis method mixins:
    - TimeAnalysisMixin: analyse_time()
    - SnapshotAnalysisMixin: analyse_snapshot()
    - LearningValenceAnalysisMixin: analyse_learning_valence()
    - ValenceHabituationAnalysisMixin: analyse_valence_habituation()
    
    Usage:
        from multiplex_analysis import MultiplexTrial
        
        trial = MultiplexTrial()
        trial.load_data('data.csv')
        trial.filter_by_num_choices(midline_borders=60, threshold=4)
        results = trial.analyse_time()
    """
    pass


__all__ = ['MultiplexTrial']

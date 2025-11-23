"""
Location Classification Functions

Functions for classifying fly position into ternary or binary categories.

This module provides position classification functionality that categorizes
fly positions into three zones: left side, center (choicepoint), and right side.
The classification accounts for the physical structure of the experimental
chamber, including the central mixing zone where odors blend.
"""

import numpy as np
from typing import Union


class TernaryLocationClassifier:
    """
    Classifies position data into ternary categories: left (-1), center (0), right (1).
    
    This classifier divides the experimental chamber into three zones based on
    position relative to the central choicepoint. The classification accounts for
    the physical structure of the chamber:
    - The chamber is 50mm long with exit vents that are 5mm long
    - In normalized coordinates (-1 to +1), the vent area spans -0.1 to +0.1
    - Flies are approximately 2.5mm long, and we track the centroid
    - An additional 1.25mm (0.05 coordinate units) is added to account for fly length
    
    The default choicepoint_halfwidth of 0.2 creates a central zone from -0.2 to +0.2,
    which represents the mixed odor region where both odors are present due to turbulent flow.
    Positions outside this zone are clearly in one odor or the other.
    """
    
    def __init__(self, choicepoint_halfwidth: float = 0.2):
        """
        Initialize the location classifier with choicepoint zone definition.
        
        The choicepoint_halfwidth defines the size of the central "mixed odor" zone.
        Positions within [-halfwidth, +halfwidth] are classified as center (0),
        positions > +halfwidth are right (1), and positions < -halfwidth are left (-1).
        
        Parameters:
        -----------
        choicepoint_halfwidth : float
            Half-width of the central choicepoint zone in normalized coordinates.
            Default 0.2 means the center zone spans from -0.2 to +0.2.
            This accounts for the physical mixing zone and fly size.
        """
        self.choicepoint_halfwidth = choicepoint_halfwidth
    
    def classify(self, position_data: np.ndarray) -> np.ndarray:
        """
        Classify position data into ternary categories.
        
        This method converts continuous position values into discrete categories:
        - Left side: positions < -choicepoint_halfwidth (classified as -1)
        - Center zone: positions between -choicepoint_halfwidth and +choicepoint_halfwidth (classified as 0)
        - Right side: positions > +choicepoint_halfwidth (classified as 1)
        
        The center zone represents the mixed odor region where both odors are present,
        making it difficult to determine which odor the fly is experiencing.
        
        Parameters:
        -----------
        position_data : np.ndarray
            Position data array, typically normalized to the range [-1, 1].
            Values represent fly position relative to chamber center.
            Can be 1D (single trajectory) or any shape (will be flattened).
        
        Returns:
        --------
        np.ndarray
            Ternary classification array with same shape as input:
            - -1: Fly is on the left side of the chamber
            - 0: Fly is in the center (mixed odor) zone
            - 1: Fly is on the right side of the chamber
        """
        position_data = np.asarray(position_data)
        # Initialize output array with zeros (center zone)
        z = np.zeros_like(position_data)
        
        cp = self.choicepoint_halfwidth
        
        # Classify right side positions (fly is clearly in right odor zone)
        # Positions greater than the choicepoint threshold are on the right
        right = np.where(position_data > cp)[0]
        z[right] = 1
        
        # Classify left side positions (fly is clearly in left odor zone)
        # Positions less than the negative choicepoint threshold are on the left
        left = np.where(position_data < -cp)[0]
        z[left] = -1
        
        # Positions between -cp and +cp remain 0 (center zone)
        
        return z
    
    def classify_filtered(
        self,
        position_data: np.ndarray,
        filter_halfwidth: float
    ) -> np.ndarray:
        """
        Classify position data with a configurable filter threshold.
        
        This method is similar to classify() but uses a different threshold value
        for the center zone. This is useful for validation purposes, where a wider
        threshold ensures that only substantial movements are classified as decisions.
        
        The filter_halfwidth is typically larger than choicepoint_halfwidth, creating
        a stricter criterion for what counts as being in a decision zone (left or right).
        
        Parameters:
        -----------
        position_data : np.ndarray
            Position data array, typically normalized to the range [-1, 1].
            Values represent fly position relative to chamber center.
        filter_halfwidth : float
            Half-width of the filter choicepoint zone. This is the threshold used
            to determine center vs. decision zones. Typically larger than the
            standard choicepoint_halfwidth to create stricter classification.
        
        Returns:
        --------
        np.ndarray
            Ternary classification array with same shape as input:
            - -1: Fly is on the left side (beyond filter threshold)
            - 0: Fly is in the center zone (within filter threshold)
            - 1: Fly is on the right side (beyond filter threshold)
        """
        position_data = np.asarray(position_data)
        # Initialize output array with zeros (center zone)
        z = np.zeros_like(position_data)
        
        cp = filter_halfwidth
        
        # Classify using the filter threshold (wider than standard choicepoint)
        # This ensures only substantial movements are classified as decisions
        right = np.where(position_data > cp)[0]
        z[right] = 1
        
        left = np.where(position_data < -cp)[0]
        z[left] = -1
        
        return z


def ternary_location_func(position_data: Union[np.ndarray, list]) -> np.ndarray:
    """
    Convert position data to ternary classification (-1, 0, 1)
    
    Convenience function maintaining compatibility with original code.
    
    Parameters:
    -----------
    position_data : array-like
        Position data (typically normalized to -1 to 1 range)
    
    Returns:
    --------
    np.ndarray
        Ternary position data: -1 (left), 0 (center), 1 (right)
    """
    classifier = TernaryLocationClassifier()
    return classifier.classify(position_data)


def ternary_location_func_filter(
    position_data: Union[np.ndarray, list],
    halfwidth: float
) -> np.ndarray:
    """
    Convert position data to ternary classification with configurable threshold
    
    Convenience function maintaining compatibility with original code.
    
    Parameters:
    -----------
    position_data : array-like
        Position data (typically normalized to -1 to 1 range)
    halfwidth : float
        Half-width of the choicepoint (threshold for center zone)
    
    Returns:
    --------
    np.ndarray
        Ternary position data: -1 (left), 0 (center), 1 (right)
    """
    classifier = TernaryLocationClassifier()
    return classifier.classify_filtered(position_data, halfwidth)


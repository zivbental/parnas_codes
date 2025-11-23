"""
Savitzky-Golay Filter Implementation

Savitzky-Golay Filtering for smoothing and differentiation.

This module implements the Savitzky-Golay filter, a polynomial-based smoothing
technique that preserves local features better than simple moving averages.
The filter fits a polynomial to a local window of data points and uses the
fitted polynomial to estimate the smoothed value at the window center.

The implementation includes both coefficient calculation and filter application,
with special handling for edge effects (transient responses at the beginning
and end of the signal).
"""

import numpy as np
from typing import Optional, Union, Tuple
from .matlab_compat import fix


def savitzky_golay(
    x: np.ndarray,
    n: int,
    dn: int,
    x0: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    flag: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Savitzky-Golay filter coefficients.
    
    This function computes the filter coefficients for a Savitzky-Golay polynomial
    filter. The filter fits a polynomial of order n to a local window of data
    points and uses the polynomial to estimate smoothed values or derivatives.
    
    The filter works by:
    1. Defining a local window around each point
    2. Fitting a polynomial to the points in that window
    3. Using the polynomial value (or its derivative) at the center point
    
    Parameters:
    -----------
    x : np.ndarray
        Array of evaluation points defining the filter window.
        Typically symmetric around zero, e.g., [-5, -4, ..., 4, 5] for
        a window of 11 points.
    n : int
        Polynomial order for fitting. Higher orders preserve more features
        but may introduce artifacts. Typical values: 2-4.
    dn : int
        Differentiation order:
        - 0: Smoothing (returns polynomial value)
        - 1: First derivative
        - 2: Second derivative
        - etc.
    x0 : np.ndarray, optional
        Points at which to evaluate the fitted polynomial.
        Default is [0] (center of window).
    W : np.ndarray, optional
        Weight vector for weighted least-squares fitting.
        If None, uses uniform weights (identity matrix).
        Must have same length as x0.
    flag : bool, optional
        Numerical (False) or symbolic (True) computation mode.
        Default False uses numerical computation.
    
    Returns:
    --------
    fc : np.ndarray
        Filter coefficients matrix. Each column contains coefficients for
        one evaluation point in x0. These coefficients can be convolved with
        the signal to apply the filter.
    df : np.ndarray
        Differentiation filter coefficients. Used when computing derivatives
        (dn > 0). Contains coefficients for derivative estimation.
    """
    x = np.asarray(x)
    n = int(n)
    dn = int(dn)
    
    if n > len(x) - 1:
        raise ValueError('The Polynomial Order must be less than the frame length.')
    if dn > n:
        raise ValueError('The Differentiation order must be less than or equal to the Polynomial order.')
    
    # Set defaults (MATLAB order: flag, W, x0, dn)
    if x0 is None:
        x0 = np.array([0])
    else:
        x0 = np.asarray(x0)
    
    if W is None or (isinstance(W, np.ndarray) and W.size == 0):
        # No weighting matrix, make W an identity
        W = np.eye(len(x0))
    else:
        W = np.asarray(W)
        # Check W is real
        if not np.isrealobj(W):
            raise ValueError('The weight vector must be real.')
        # Check for right length of W
        if W.ndim == 1:
            if len(W) != len(x0):
                raise ValueError(f'The weight vector must be of the same length as x0. Got len(W)={len(W)}, len(x0)={len(x0)}')
            # Check to see if all elements are positive
            if np.min(W) <= 0:
                raise ValueError('All the elements of the weight vector must be greater than zero')
            # Diagonalize the vector to form the weighting matrix
            W = np.diag(W)
        elif W.ndim == 2:
            if W.shape[0] != len(x0) or W.shape[1] != len(x0):
                raise ValueError(f'The weight matrix must have dimensions matching x0. Got W.shape={W.shape}, len(x0)={len(x0)}')
    
    if flag:
        # Symbolic version - not implemented in Python, would need sympy
        raise NotImplementedError('Symbolic mode not implemented in Python version')
    
    Nx = len(x)
    x = x.flatten()
    Nx0 = len(x0)
    x0 = x0.flatten()
    
    # Build Vandermonde matrix A
    A = np.ones((Nx, 1))
    for k in range(1, n + 1):
        A = np.column_stack([A, x ** k])
    
    # Solve for df (differentiation filter)
    df = np.linalg.solve(A.T @ A, A.T).T
    
    # Build hx (evaluation points)
    hx = np.zeros((Nx0, dn + 1))
    for order in range(dn):
        hx[:, order] = 0
    hx[:, dn] = np.prod(np.arange(1, dn + 1)) if dn > 0 else 1
    
    for k in range(1, n - dn + 1):
        fact = np.prod(np.arange(dn + k, k, -1))
        hx = np.column_stack([hx, x0 ** k * fact])
    
    # Filter coefficients
    fc = df @ hx.T @ W
    
    return fc, df


class SavitzkyGolayFilter:
    """
    Savitzky-Golay Filter class for smoothing and differentiation.
    
    This class provides a clean interface for applying Savitzky-Golay polynomial
    filters to signals. The filter reduces noise while preserving local features
    by fitting polynomials to local windows of data. It handles edge effects
    specially, using different calculations for the beginning and end of the signal
    where the full filter window is not available.
    """
    
    def __init__(self, polynomial_order: int = 2, differentiation_order: int = 0):
        """
        Initialize the Savitzky-Golay filter with polynomial and differentiation settings.
        
        The filter fits polynomials of the specified order to local windows of data.
        The differentiation order determines whether the filter smooths (0) or computes
        derivatives (1, 2, etc.).
        
        Parameters:
        -----------
        polynomial_order : int
            Order of polynomial to fit (typically 2-4).
            Must be less than the frame length when applying the filter.
            Higher orders preserve more features but may introduce artifacts.
        differentiation_order : int
            Order of derivative to compute:
            - 0: Smoothing (returns polynomial value)
            - 1: First derivative (rate of change)
            - 2: Second derivative (acceleration)
            - etc.
        """
        self.polynomial_order = polynomial_order
        self.differentiation_order = differentiation_order
    
    def apply(
        self,
        x: np.ndarray,
        frame_length: int,
        W: Optional[np.ndarray] = None,
        dim: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to a signal.
        
        This method applies the polynomial filter to the input signal, handling
        both steady-state (middle portion) and transient (edges) regions specially.
        The filter window size is determined by frame_length, which must be odd
        to ensure symmetric filtering around each point.
        
        The implementation processes the signal in three regions:
        1. Beginning transient: Uses partial filter windows
        2. Steady-state: Uses full filter windows (most accurate)
        3. Ending transient: Uses partial filter windows
        
        Parameters:
        -----------
        x : np.ndarray
            Input signal to filter. Can be 1D or 2D (multi-channel).
            For 2D arrays, filtering is applied along the first dimension.
        frame_length : int
            Size of the filter window (must be odd for symmetric filtering).
            Larger windows provide more smoothing but may blur sharp features.
        W : np.ndarray, optional
            Weight vector for weighted filtering (length must equal frame_length).
            If None, uses uniform weights. Useful for giving more importance to
            certain points in the window.
        dim : int, optional
            Dimension along which to apply the filter (for multi-dimensional arrays).
            If None, automatically determines based on array shape.
        
        Returns:
        --------
        y : np.ndarray
            Filtered signal with same shape as input.
            The output has the same length as input, with edge effects handled
            by special transient calculations.
        """
        x = np.asarray(x)
        F = int(frame_length)
        N = self.polynomial_order
        DN = self.differentiation_order
        
        # Validate inputs
        if int(F) != F:
            raise ValueError('Frame length must be an integer.')
        if F % 2 != 1:
            raise ValueError('Frame length must be odd.')
        if N > F - 1:
            raise ValueError('The Polynomial order must be less than the frame length.')
        if DN > N:
            raise ValueError('The Differentiation order must be less than or equal to the Polynomial order.')
        
        if W is None:
            W = np.ones(F)
        else:
            W = np.asarray(W)
            if len(W) != F:
                raise ValueError('The weight vector must be of the same length as the frame length.')
            if np.min(W) <= 0:
                raise ValueError('All the elements of the weight vector must be greater than zero.')
        
        # Handle dimensions
        if dim is not None and dim > x.ndim:
            raise ValueError('Dimension specified exceeds the dimensions of X.')
        
        # Reshape X into the right dimension
        original_shape = x.shape
        transpose_back = False
        
        if dim is None:
            # Work along the first non-singleton dimension
            if x.ndim == 1:
                x = x.reshape(-1, 1)
                transpose_back = True
        else:
            # Put DIM in the first dimension
            perm = list(range(x.ndim))
            perm.insert(0, perm.pop(dim))
            x = np.transpose(x, perm)
        
        if x.shape[0] < F:
            raise ValueError('The length of the input must be >= frame length.')
        
        # Preallocate output
        y = np.zeros_like(x)
        
        # Compute the projection matrix B (filter coefficients)
        # Create the evaluation point range: symmetric around zero
        # The range spans from -F/2 to +F/2 (rounded toward zero)
        pp_start = int(fix(-F / 2))
        pp_end = int(fix(F / 2))
        pp = np.arange(pp_start, pp_end + 1)
        # Calculate Savitzky-Golay filter coefficients
        B, _ = savitzky_golay(pp, N, DN, pp, W, False)
        
        # Compute the transient response at the beginning of the signal
        # The first few output points cannot use the full filter window because
        # there isn't enough data before them. We use a special calculation
        # that uses only the available data points.
        num_transient = (F + 1) // 2 - 1
        x_flipped = np.flipud(x[:F, :])  # Flip the first F rows for processing
        
        for i in range(num_transient):
            # Select filter coefficients for this transient point
            # Use coefficients that correspond to positions beyond the window center
            col_start = (F - 1) // 2 + 2 + i
            if col_start < F:
                B_subset = B[:, col_start:]
                B_flipped = np.fliplr(B_subset)  # Flip columns left-right
                B_trans = B_flipped.T  # Transpose for matrix multiplication
                # Apply filter to flipped input data
                result = B_trans @ x_flipped
                if result.shape[0] > 0:
                    y[i, :] = result[0, :]
                else:
                    y[i, :] = 0
        
        # Compute the steady-state output (middle portion of signal)
        # For points with full filter window available, use the center filter coefficient
        # This is the most efficient and accurate part of the filtering
        B_center = B[:, (F - 1) // 2]  # Center coefficient of the filter
        # Use numpy convolution to apply the filter
        ytemp = np.zeros_like(x)
        for col in range(x.shape[1]):
            # Convolve with reversed filter coefficients (for causal filter)
            # The reversal ensures the filter processes data in the correct temporal order
            ytemp[:, col] = np.convolve(x[:, col], B_center[::-1], mode='full')[:len(x[:, col])]
        # Extract the steady-state portion: skip initial transient and final transient
        # The indexing ensures we only use points where the filter had full window coverage
        y[(F + 1) // 2 - 1:-(F + 1) // 2 + 1, :] = ytemp[F - 1:, :]
        
        # Compute the transient response at the end of the signal
        # Similar to beginning, but for the last few points where the filter window
        # extends beyond the available data
        num_transient_off = (F + 1) // 2 - 1
        x_end_flipped = np.flipud(x[-(F):, :])  # Last F rows, flipped
        
        for i in range(num_transient_off):
            idx = y.shape[0] - num_transient_off + i
            col_end = (F - 1) // 2
            if col_end > 0:
                # Select filter coefficients for end transient points
                # Use coefficients from the beginning of the filter
                B_subset = B[:, :col_end]
                B_flipped = np.fliplr(B_subset)  # Flip columns
                B_trans = B_flipped.T  # Transpose for matrix multiplication
                # Apply filter to flipped end data
                result = B_trans @ x_end_flipped
                if result.shape[0] > 0:
                    y[idx, :] = result[0, :]
                else:
                    y[idx, :] = 0
        
        # Convert Y to the original shape of X
        if transpose_back:
            y = y[:, 0]
        
        return y


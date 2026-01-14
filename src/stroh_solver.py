"""
Real-Variable Stroh Solver Module
=================================

This module implements the core physics engine for the paper:
"Engineering Stability Band-Gaps in Graded Soft Matter" (JMPS, 2026).

It defines the RealStrohSolver class, which handles:
1. Material Constitutive Modeling (Incompressible Neo-Hookean FGM).
2. Construction of the Real-Variable Stroh Acoustic Tensor N (Eq. 52 in manuscript).
3. Transformation to the Compound Matrix Q (for CMM integration).

Author: Feng Yang
License: MIT
"""

import numpy as np

class RealStrohSolver:
    """
    A solver class for computing the spectral properties of functionally graded 
    incompressible Neo-Hookean solids using the Real-Variable Stroh Formalism.
    """

    def __init__(self, mu_func, H=1.0):
        """
        Initialize the solver with a gradient profile.

        Parameters
        ----------
        mu_func : function
            A callable that takes depth x2 (float or array) and returns 
            the shear modulus mu(x2).
            Input domain should be [-H, 0].
        H : float, optional
            The thickness of the substrate (default is 1.0).
        """
        self.mu_func = mu_func
        self.H = H

    def _get_stiffness_params(self, x2, lam):
        """
        Compute the incremental stiffness parameters alpha, gamma, K1, K2.
        
        Corresponds to Eq. (4) and Eq. (6) in the manuscript.
        
        Parameters
        ----------
        x2 : float
            Depth coordinate.
        lam : float
            Principal stretch ratio (lambda).

        Returns
        -------
        tuple : (gamma, K1, K2)
        """
        mu = self.mu_func(x2)
        
        # Prevent division by zero or negative stretch
        lam = max(lam, 1e-6)
        
        # Plane strain parameters (Eq. 43 & Eq. 59)
        # alpha = mu * lambda^2
        # gamma = mu * lambda^(-2)
        lam2_inv = 1.0 / (lam**2)
        
        alpha = mu * (lam**2)
        gamma = mu * lam2_inv
        
        # Generalized stiffness for static limit (Eq. 48)
        K1 = alpha + gamma
        K2 = alpha - gamma
        
        return gamma, K1, K2

    def get_real_stroh_matrix(self, x2, k, lam):
        """
        Construct the 4x4 Real-Variable Stroh Acoustic Tensor N.

        This matrix governs the first-order differential system:
            d(eta)/dx2 = N * eta
        where eta = [U1, U2_bar, Sigma21, Sigma22_bar]^T is the real-valued state vector.

        Ref: Eq. (52) and Appendix A (Eq. 204) in the manuscript.

        Parameters
        ----------
        x2 : float
            Current depth.
        k : float
            Wavenumber (kH).
        lam : float
            Current stretch ratio (lambda).

        Returns
        -------
        np.ndarray
            4x4 Real-valued matrix N.
        """
        gamma, K1, K2 = self._get_stiffness_params(x2, lam)
        
        # Initialize 4x4 matrix
        N = np.zeros((4, 4))
        
        # Row 1: Kinematics of U1 (Eq. 190)
        # U1' = -k * U2_bar + (1/gamma) * Sigma21
        N[0, 1] = -k
        N[0, 2] = 1.0 / gamma
        
        # Row 2: Kinematics of U2_bar (Eq. 186)
        # U2_bar' = k * U1
        N[1, 0] = k
        
        # Row 3: Horizontal Equilibrium (Eq. 197)
        # Sigma21' = k^2 * K1 * U1 - k * Sigma22_bar
        N[2, 0] = (k**2) * K1
        N[2, 3] = -k
        
        # Row 4: Vertical Equilibrium (Eq. 203)
        # Sigma22_bar' = k^2 * K2 * U2_bar + k * Sigma21
        N[3, 1] = (k**2) * K2
        N[3, 2] = k
        
        return N

    def get_compound_matrix(self, N):
        """
        Convert the 4x4 Stroh matrix N into the 6x6 Compound Matrix Q.

        The Compound Matrix Method (CMM) integrates the minors of the solution
        to eliminate numerical stiffness (parasitic growth) in deep substrates.
        
        The mapping of indices for 2-minors (exterior algebra) is:
        1:(1,2), 2:(1,3), 3:(1,4), 4:(2,3), 5:(2,4), 6:(3,4)

        Ref: Section 2.5 in the manuscript.

        Parameters
        ----------
        N : np.ndarray
            The 4x4 Stroh matrix.

        Returns
        -------
        np.ndarray
            The 6x6 Compound matrix Q.
        """
        Q = np.zeros((6, 6))
        
        # Index mapping for 6-dimensional compound vector
        # 0-based indices corresponding to the pairs (row, col) of the 4x4 system
        idx_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        
        # The formula for Q_IJ derived from N is:
        # Q_{(ik)(jl)} = N_ij * delta_kl + N_kl * delta_ij ... (simplified)
        # Explicit construction is safer and more readable:
        
        for I in range(6):
            r1, r2 = idx_pairs[I]  # Row indices of the minor corresponding to component I
            
            for J in range(6):
                c1, c2 = idx_pairs[J] # Col indices of the minor corresponding to component J
                
                # Formula for an infinitesimal generator of the compound matrix
                val = 0.0
                
                # Check derivation rule: Q_{alpha, beta} corresponds to N acting on basis wedge product
                # Logic: If minor is y = u ^ v, then y' = u' ^ v + u ^ v' = (Nu)^v + u^(Nv)
                
                if c2 == r2: val += N[r1, c1]
                if c1 == r2: val -= N[r1, c2] # Swap sign due to wedge anti-symmetry
                if c1 == r1: val += N[r2, c2]
                if c2 == r1: val -= N[r2, c1] # Swap sign
                
                Q[I, J] = val
                
        return Q

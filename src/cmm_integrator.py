"""
Compound Matrix Method (CMM) Integrator
=======================================

This module implements the numerical integration engine for the CMM.
It solves the differential equation dy/dx = Q(x) * y, where y is the 
6-dimensional compound vector (exterior algebra of the solution space).

Key Features:
1. Continuous Normalization: Prevents numerical overflow in deep substrates
   (consistent with Section 2.5 of the manuscript).
2. Clamped Boundary Condition: Initializes the integration from a rigid base.
3. Bifurcation Detection: Monitors the surface determinant (y6).

Author: Feng Yang
License: MIT
"""

import numpy as np
from .stroh_solver import RealStrohSolver

class CMMIntegrator:
    """
    Numerical integrator for the Compound Matrix Method.
    """

    def __init__(self, solver: RealStrohSolver, n_steps: int = 500):
        """
        Initialize the integrator.

        Parameters
        ----------
        solver : RealStrohSolver
            Instance of the physics engine containing material properties 
            and Stroh matrix definitions.
        n_steps : int, optional
            Number of integration steps from base to surface (default 500).
            Higher values improve accuracy for steep gradients.
        """
        self.solver = solver
        self.n_steps = n_steps
        self.H = solver.H

    def _get_initial_condition(self):
        """
        Define the boundary condition at the clamped base (x2 = -H).

        For a clamped base, displacements U1 = U2 = 0.
        The solution subspace is spanned by two independent stress states:
        Basis 1: (0, 0, 1, 0)^T  (Pure Shear Stress)
        Basis 2: (0, 0, 0, 1)^T  (Pure Normal Stress)

        The compound vector y is the wedge product of these bases:
        y(-H) = [0, 0, 0, 0, 0, 1]^T

        Ref: Section 2.5 "Numerical Implementation Details"
        """
        y0 = np.zeros(6)
        y0[5] = 1.0  # The component corresponding to the minor of the bottom 2x2 block
        return y0

    def integrate(self, k, lam):
        """
        Integrate the CMM equations from x2 = -H to x2 = 0.

        Implements an explicit RK4 scheme with continuous normalization
        to handle stiff growth modes.

        Parameters
        ----------
        k : float
            Wavenumber (kH).
        lam : float
            Stretch ratio (lambda).

        Returns
        -------
        float
            The value of the bifurcation determinant y6 at the surface (x2=0).
            If y6(0) = 0, a bifurcation exists.
        """
        # Step size
        h = self.H / self.n_steps
        
        # Initial condition at the base
        y = self._get_initial_condition()
        x = -self.H
        
        # Integration loop (Runge-Kutta 4th Order)
        for _ in range(self.n_steps):
            # RK4 Stage 1
            N1 = self.solver.get_real_stroh_matrix(x, k, lam)
            Q1 = self.solver.get_compound_matrix(N1)
            k1 = Q1 @ y
            
            # RK4 Stage 2
            x_mid = x + 0.5 * h
            N2 = self.solver.get_real_stroh_matrix(x_mid, k, lam)
            Q2 = self.solver.get_compound_matrix(N2)
            k2 = Q2 @ (y + 0.5 * h * k1)
            
            # RK4 Stage 3
            # N3 is same as N2 at midpoint
            k3 = Q2 @ (y + 0.5 * h * k2)
            
            # RK4 Stage 4
            x_end = x + h
            N4 = self.solver.get_real_stroh_matrix(x_end, k, lam)
            Q4 = self.solver.get_compound_matrix(N4)
            k4 = Q4 @ (y + h * k3)
            
            # Update state
            y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # --- Continuous Normalization ---
            # As mentioned in Manuscript Section 2.5:
            # "At each integration step, the state vector is rescaled...
            # to prevent floating-point overflow."
            norm_factor = np.max(np.abs(y))
            if norm_factor > 1e-20:
                y /= norm_factor
                
            x += h

        # Return the 6th component at the surface
        # y6 corresponds to the determinant of the traction matrix (Eq. 74)
        return y[5]

    def check_stability(self, k, lam):
        """
        Convenience wrapper to get the surface determinant.
        Same as integrate(), but conceptually clearer for external calls.
        """
        return self.integrate(k, lam)

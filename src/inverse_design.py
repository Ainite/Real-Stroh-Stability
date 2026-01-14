"""
Inverse Spectral Design Module
==============================

This module implements the optimization framework described in Section 3 of the manuscript.
It couples a Differential Evolution (DE) algorithm with the Real-CMM physics engine to 
discover gradient profiles that target specific critical stretches.

Key Components:
1. Geometric Parameterization: Log-space Cubic B-Splines (Eq. 82).
2. Global Spectral Scan: Identifies the system-level critical stretch (Eq. 85).
3. Cost Function: Balances target accuracy, material usage, and smoothness (Eq. 87).

Author: Feng Yang
License: MIT
"""

import numpy as np
from scipy.optimize import differential_evolution, brentq
from scipy.interpolate import BSpline
from .stroh_solver import RealStrohSolver
from .cmm_integrator import CMMIntegrator

class InverseDesignEngine:
    """
    Optimization engine for inverse spectral design of graded elastomers.
    """

    def __init__(self, target_lambda=0.60, n_control_points=8):
        """
        Initialize the design engine.

        Parameters
        ----------
        target_lambda : float
            The desired critical stretch (lambda_sys) for the gradient.
        n_control_points : int
            Number of control points for the B-Spline representation.
        """
        self.target_lambda = target_lambda
        self.n_control = n_control_points
        self.H = 1.0
        
        # Optimization weights (Section 3, Eq. 87)
        self.w1 = 100.0  # Weight for spectral target accuracy
        self.w2 = 0.1    # Weight for minimizing material usage (softness)
        self.w3 = 0.01   # Weight for regularization (smoothness)

        # Pre-compute B-spline knot vector (clamped uniform)
        # Degree k=3 (Cubic)
        k = 3
        self.degree = k
        # Knots: k+1 zeros, internal knots, k+1 ones
        n_internal = n_control_points - (k + 1)
        if n_internal < 0:
            raise ValueError("Too few control points for cubic spline (min 4).")
            
        knots_internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        self.knots = np.concatenate(([0]*(k+1), knots_internal, [1]*(k+1)))

    def _construct_mu_func(self, weights):
        """
        Construct the shear modulus function mu(x2) from spline weights.
        
        Implements Eq. (82): ln(mu) = Sum(w_i * B_i).
        The optimization variables are the weights in LOG SPACE.
        """
        # Create spline for ln(mu) vs normalized depth (0 to 1)
        # x2 runs from -H to 0. Normalized coordinate: s = (x2 + H) / H
        spline_log = BSpline(self.knots, weights, self.degree, extrapolate=False)

        def mu_func(x2):
            # Normalize x2 to [0, 1]
            s = (x2 + self.H) / self.H
            s = np.clip(s, 0.0, 1.0) # Safety clip
            val_log = spline_log(s)
            return np.exp(val_log)
            
        return mu_func

    def _find_critical_stretch(self, integrator, k):
        """
        Robustly find lambda_cr for a specific wavenumber k.
        
        Scans lambda from 1.0 (stable) down to 0.4 (limit).
        Uses root finding on the determinant y6.
        """
        # Coarse scan to bracket the root
        lam_scan = np.linspace(0.98, 0.45, 15)
        dets = []
        
        # Check stability at initial (lambda=0.98), usually stable
        # If unstable at 0.98, the critical stretch is essentially 1.0
        d_start = integrator.check_stability(k, 1.0)
        
        prev_d = d_start
        prev_l = 1.0
        
        for l in lam_scan:
            d = integrator.check_stability(k, l)
            
            # Check for sign change (bifurcation point)
            if d * prev_d < 0:
                try:
                    # Refine with Brent's method
                    root = brentq(lambda x: integrator.check_stability(k, x), 
                                  prev_l, l, xtol=1e-4)
                    return root
                except:
                    return l # Fallback
            
            prev_d = d
            prev_l = l
            
        return 0.4 # Lower bound (no instability found)

    def _evaluate_spectral_response(self, mu_func):
        """
        Perform a Global Spectral Scan (Eq. 86).
        
        Returns
        -------
        lambda_sys : float
            The maximum critical stretch across all wavenumbers.
        mean_stiffness : float
            Integral of mu(x) dx (for cost function).
        """
        solver = RealStrohSolver(mu_func, self.H)
        integrator = CMMIntegrator(solver, n_steps=150) # Moderate steps for speed
        
        # Wavenumber domain scan (avoiding k=0 singularity)
        # We scan typical range for Biot and Global modes
        k_vals = np.concatenate([
            np.linspace(0.5, 5.0, 10),  # Low k (Global buckling)
            np.linspace(6.0, 40.0, 8)   # High k (Surface wrinkling)
        ])
        
        lambda_crs = []
        for k in k_vals:
            l_cr = self._find_critical_stretch(integrator, k)
            lambda_crs.append(l_cr)
            
        # The system fails at the highest critical stretch
        lambda_sys = np.max(lambda_crs)
        
        # Calculate material usage (integral)
        x_grid = np.linspace(-self.H, 0, 100)
        mu_vals = mu_func(x_grid)
        mean_stiffness = np.trapz(mu_vals, x_grid) / self.H
        
        return lambda_sys, mean_stiffness

    def _cost_function(self, weights):
        """
        The Objective Function J(w).
        """
        mu_func = self._construct_mu_func(weights)
        
        # 1. Spectral Performance
        lambda_sys, mean_stiffness = self._evaluate_spectral_response(mu_func)
        term_target = (lambda_sys - self.target_lambda)**2
        
        # 2. Material Usage (Softness)
        term_material = mean_stiffness
        
        # 3. Regularization (Roughness penalty)
        # Discrete approximation of gradient
        diffs = np.diff(weights)
        term_reg = np.sum(diffs**2)
        
        # Total Cost (Eq. 87)
        cost = (self.w1 * term_target + 
                self.w2 * term_material + 
                self.w3 * term_reg)
        
        return cost

    def run_optimization(self, bounds=(np.log(0.1), np.log(10.0)), max_iter=20):
        """
        Execute the Differential Evolution algorithm.

        Parameters
        ----------
        bounds : tuple
            (min, max) bounds for the control points in Log space.
            Default corresponds to stiffness contrast ~ 0.1 to 10.0.
        max_iter : int
            Maximum number of generations.

        Returns
        -------
        res : OptimizeResult
            The result object containing the best weights and profile.
        """
        print(f"Starting Inverse Design (Target Lambda = {self.target_lambda})...")
        print("Note: This may take several minutes depending on CPU.")
        
        limit_bounds = [bounds] * self.n_control
        
        # Run DE
        result = differential_evolution(
            self._cost_function, 
            limit_bounds, 
            strategy='best1bin', 
            maxiter=max_iter, 
            popsize=10, 
            tol=0.01,
            disp=True,
            workers=1 # Set to -1 to use all cores (Parallel)
        )
        
        print("Optimization Complete.")
        print(f"Best Cost: {result.fun:.6f}")
        
        return result

if __name__ == "__main__":
    # Quick Test
    designer = InverseDesignEngine(target_lambda=0.60)
    # Run a very short optimization for demonstration
    res = designer.run_optimization(max_iter=2)
    print("Found Control Points:", res.x)

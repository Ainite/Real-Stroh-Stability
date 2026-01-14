"""
Figure 4 Reproduction Script
============================

This script reproduces the Mode Duality Phase Diagram (Figure 4) from the manuscript.
It maps the stability landscape across:
- Stiffness Contrast (beta = mu_surf / mu_sub)
- Gradient Decay Length (xi = L / H)

Key Physics:
- Region I (beta > 1): Surface Wrinkling (Short wave, Biot-like).
- Region II (beta < 1): Macroscopic Shear Buckling (Long wave, Structural).

Usage:
    python examples/fig4_phase_diagram.py
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stroh_solver import RealStrohSolver
from src.cmm_integrator import CMMIntegrator

# ==============================================================================
# 1. Plotting Style
# ==============================================================================
try:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
        "savefig.dpi": 600,
    })
except:
    pass

# ==============================================================================
# 2. Helper Functions
# ==============================================================================

def get_critical_stretch(integrator, k):
    """
    Find lambda_cr for a fixed wavenumber k.
    Scans from stable (1.0) to unstable (0.3).
    """
    l_scan = np.linspace(0.98, 0.3, 15)
    
    # Check initial stability
    val_prev = integrator.check_stability(k, 1.0)
    
    for l in l_scan:
        val = integrator.check_stability(k, l)
        if val * val_prev < 0:
            try:
                # Found bracket, solve root
                root = brentq(lambda x: integrator.check_stability(k, x), 
                              l, l + (l_scan[0]-l_scan[1]), xtol=1e-3)
                return root
            except:
                return l
        val_prev = val
    return 0.0 # No instability found in range

def find_system_instability(beta, L, H=1.0):
    """
    Find the most critical mode (highest lambda_cr) for a specific gradient.
    Returns: (max_lambda, dominant_k)
    """
    # Define the material profile for this specific case
    # mu(x) = mu_sub + (mu_surf - mu_sub) * exp(x/L)
    # mu_sub is fixed at 1.0
    def mu_func(x2):
        return 1.0 + (beta - 1.0) * np.exp(x2 / L)

    solver = RealStrohSolver(mu_func, H)
    integrator = CMMIntegrator(solver, n_steps=80) # Faster steps for phase diagram

    # We check a discrete set of wavenumbers covering both regimes
    # Low k: Global Shear Buckling
    # High k: Surface Wrinkling
    k_candidates = [2.0, 4.0, 8.0, 15.0, 25.0]
    
    max_lam = 0.0
    dom_k = 0.0
    
    for k in k_candidates:
        lam = get_critical_stretch(integrator, k)
        if lam > max_lam:
            max_lam = lam
            dom_k = k
            
    return max_lam, dom_k

# ==============================================================================
# 3. Main Calculation Loop
# ==============================================================================

def main():
    print("--- Starting Phase Diagram Calculation ---")
    print("Simulating gradient designs across the parameter space...")
    print("Note: This scans a grid of simulations. It might take 1-2 minutes.")

    # Grid Resolution (Increase to 30+ for publication quality smoothness)
    resolution = 15 
    
    # Parameter Space
    # Beta: Stiffness Contrast (0.1 to 10) - Log scale
    beta_vals = np.logspace(-1, 1, resolution) 
    # Decay: Gradient Depth (0.1H to 1.0H) - Linear scale
    decay_vals = np.linspace(0.1, 1.0, resolution)

    # Result Matrices
    Lambda_Map = np.zeros((resolution, resolution))
    Mode_Map = np.zeros((resolution, resolution)) # 1: Global, 2: Surface

    start_time = time.time()

    for i, beta in enumerate(beta_vals):
        for j, L in enumerate(decay_vals):
            
            lam_sys, k_sys = find_system_instability(beta, L)
            
            Lambda_Map[i, j] = lam_sys
            
            # Classification Criterion (as discussed in manuscript)
            # kH < 5.0 -> Dominated by finite thickness effects (Global)
            # kH >= 5.0 -> Dominated by surface skin (Biot-like)
            if lam_sys > 0.0:
                if k_sys < 5.0:
                    Mode_Map[i, j] = 1 # Global
                else:
                    Mode_Map[i, j] = 2 # Surface
        
        # Simple progress bar
        sys.stdout.write(f"\rProgress: {(i+1)/resolution*100:.1f}%")
        sys.stdout.flush()

    print(f"\nCalculation finished in {time.time() - start_time:.1f}s")

    # ==============================================================================
    # 4. Visualization
    # ==============================================================================
    print("Generating Plot...")
    
    plt.figure(figsize=(8, 6))
    
    X, Y = np.meshgrid(decay_vals, beta_vals)
    
    # A. Contour Plot of Critical Stretch
    # Levels: More stable (blue/low lambda) vs Less stable (yellow/high lambda)
    levels = np.linspace(0.4, 0.95, 20)
    cp = plt.contourf(X, Y, Lambda_Map, levels=levels, cmap='viridis', extend='both')
    cbar = plt.colorbar(cp)
    cbar.set_label(r'Critical Stretch $\lambda_{cr}$', fontsize=12)
    
    # B. Mode Boundary Line
    # Separating Region I (Mode=2) and Region II (Mode=1)
    # We use a contour at 1.5 because Mode_Map is 1 or 2
    try:
        cs = plt.contour(X, Y, Mode_Map, levels=[1.5], colors='white', 
                        linewidths=3, linestyles='--')
    except:
        pass # Handle case where grid might be too coarse to find boundary
        
    # Annotations
    plt.text(0.55, 5.0, 'Region I: Surface Wrinkling\n(Bending Dominated)', 
             color='white', fontweight='bold', ha='center', fontsize=10,
             path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])
    
    plt.text(0.55, 0.2, 'Region II: Macroscopic Shear Buckling\n(Bulk Shear Dominated)', 
             color='white', fontweight='bold', ha='center', fontsize=10,
             path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])

    plt.axhline(1.0, color='white', linestyle=':', alpha=0.8, lw=1.5)
    plt.text(0.12, 1.1, 'Homogeneous', color='white', fontsize=9, rotation=0)

    # Styling
    plt.yscale('log')
    plt.xlabel(r'Gradient Decay Length $\xi = L/H$', fontsize=12)
    plt.ylabel(r'Stiffness Ratio $\beta = \mu_{surf}/\mu_{sub}$', fontsize=12)
    plt.title(r'Stability Phase Diagram & Mode Duality', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    outfile = 'Figure4_PhaseDiagram.pdf'
    plt.savefig(outfile, format='pdf')
    print(f"Figure saved to {outfile}")

if __name__ == "__main__":
    main()

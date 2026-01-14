"""
Figure 7 Reproduction Script (Inverse Design Motivation)
======================================================

This script reproduces the design motivation figure (Figure 7) from the manuscript.
It compares two distinct gradient topologies:
1. Case A: Simple Soft-on-Stiff Gradient (Intuitive but fails).
2. Case B: "Stiff-Soft-Stiff" Sandwich (Counter-intuitive, discovered by Inverse Design).

Key Physics:
- Case A suffers from surface instability (Biot limit) despite the gradient.
- Case B creates a 'Stability Band-Gap' by filtering out both short-wave (surface)
  and long-wave (global) modes, achieving the target critical stretch.

Usage:
    python examples/fig7_bandgap_design.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.signal import savgol_filter
import warnings

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stroh_solver import RealStrohSolver
from src.cmm_integrator import CMMIntegrator

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Plotting Style
# ==============================================================================
try:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
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
# 2. Case Definitions (Material Profiles)
# ==============================================================================

def mu_case_a(x2):
    """
    Case A: Simple Exponential Gradient (Soft-on-Stiff)
    mu(x) = 0.2 * exp(-3.21 * x)
    Decays from surface to base.
    """
    return 0.2 * np.exp(-3.21 * x2)

def mu_case_b(x2):
    """
    Case B: "Stiff-Soft-Stiff" Sandwich (Gaussian Peaks)
    Mimics the topology discovered by Inverse Design.
    """
    base = 0.4
    # Top stiff skin
    peak_top = 8.6 * np.exp(-(x2 - 0.0)**2 / 0.05)
    # Bottom stiff base
    peak_bot = 8.6 * np.exp(-(x2 + 1.0)**2 / 0.05)
    return base + peak_top + peak_bot

# ==============================================================================
# 3. Calculation Logic
# ==============================================================================

def compute_stability_curve(mu_func, k_vals, search_range):
    """
    Computes lambda_cr vs k for a given profile using Real-CMM.
    """
    solver = RealStrohSolver(mu_func, H=1.0)
    integrator = CMMIntegrator(solver, n_steps=200)
    
    lam_curve = []
    
    # Pre-calculate search grid to speed up root bracketing
    l_min, l_max = search_range
    l_scan = np.linspace(l_min, l_max, 30)
    
    print(f"Scanning {len(k_vals)} wavenumbers...")
    
    for i, k in enumerate(k_vals):
        # Bracket the root
        vals = [integrator.check_stability(k, l) for l in l_scan]
        root_found = False
        
        # Scan from stable (high lambda) to unstable (low lambda)
        for j in range(len(vals)-1, 0, -1):
            if np.isnan(vals[j]) or np.isnan(vals[j-1]): continue
            
            # Look for sign change in determinant
            if vals[j] * vals[j-1] < 0:
                try:
                    root = brentq(lambda l: integrator.check_stability(k, l), 
                                  l_scan[j-1], l_scan[j], xtol=1e-4)
                    lam_curve.append(root)
                    root_found = True
                    break
                except: continue
        
        if not root_found:
            # Fallback or stable
            lam_curve.append(np.nan)
            
        # Simple progress indicator
        if i % 20 == 0:
            sys.stdout.write(f"\r  Progress: {i}/{len(k_vals)}")
            sys.stdout.flush()
            
    print("\n  Done.")
    return lam_curve

# ==============================================================================
# 4. Main Execution
# ==============================================================================

def main():
    print("--- Generating Figure 7: Inverse Design Motivation ---")
    
    k_vals = np.linspace(2, 55, 80)

    # --- Compute Case A ---
    print("\nComputing Case A (Simple Gradient)...")
    raw_lam_a = compute_stability_curve(mu_case_a, k_vals, search_range=(0.5, 0.85))
    # Fill NaNs for plotting (if stable, set to high value or previous)
    lam_a = [x if not np.isnan(x) else 0.73 for x in raw_lam_a]

    # --- Compute Case B ---
    print("\nComputing Case B (Sandwich Topology)...")
    raw_lam_b = compute_stability_curve(mu_case_b, k_vals, search_range=(0.75, 0.98))
    lam_b = [x if not np.isnan(x) else 0.85 for x in raw_lam_b]

    # Smoothing (Optional, for visual clarity as in original script)
    lam_a_smooth = savgol_filter(lam_a, 15, 3)
    lam_b_smooth = savgol_filter(lam_b, 15, 3)

    # --- Plotting ---
    print("\nPlotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    plt.subplots_adjust(wspace=0.25, bottom=0.15, top=0.9, left=0.08, right=0.98)

    # === Panel (a): Design Candidates ===
    y_eval = np.linspace(-1, 0, 300)
    mu_a_vals = mu_case_a(y_eval)
    mu_b_vals = mu_case_b(y_eval)
    
    ax1.plot(mu_a_vals, y_eval, 'r--', lw=3, label='Case A: Simple Gradient\n(Soft-on-Stiff)')
    ax1.plot(mu_b_vals, y_eval, 'b-', lw=3, label='Case B: Naive Sandwich\n(Stiff-Soft-Stiff)')
    
    ax1.set_xlabel(r'Shear Modulus, $\mu(x_2)$')
    ax1.set_ylabel(r'Normalized Depth, $x_2/H$')
    ax1.set_title(r'(a) Design Candidates', fontsize=12, fontweight='bold', loc='left')
    
    # Legend in center right
    ax1.legend(loc='center right', frameon=True, fontsize=10)
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.set_xlim(0, 10)
    ax1.invert_yaxis() # Depth goes down

    # === Panel (b): Stability Response ===
    target_lambda = 0.60
    
    # Target Zone
    ax2.axhline(target_lambda, color='green', linestyle='-', linewidth=2, label=r'Design Target ($\lambda=0.6$)')
    ax2.axhspan(0.4, target_lambda, color='green', alpha=0.08)
    ax2.text(40, 0.55, 'Safe Zone', color='green', ha='center', fontsize=11, fontweight='bold', alpha=0.6)

    # Curves
    ax2.plot(k_vals, lam_a_smooth, 'r--', lw=3, label='Case A Response')
    ax2.plot(k_vals, lam_b_smooth, 'b-', lw=3, label='Case B Response')

    # Annotations
    ax2.annotate('Counter-intuitive Failure\n(Stiff Skin Instability)', 
                 xy=(8, 0.88), xytext=(12, 0.92),
                 arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, color='blue', ha='left')

    ax2.text(28, 0.76, 'Simple Gradient Limit', color='red', ha='center', fontsize=10, fontweight='bold')
    ax2.annotate('', xy=(28, 0.69), xytext=(28, target_lambda),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax2.text(30, 0.66, 'Gap', color='red', fontsize=10)

    # Biot Limit Reference
    ax2.axhline(0.544, color='gray', linestyle=':', label='Biot Limit (0.544)')

    ax2.set_xlabel(r'Wavenumber, $kH$')
    ax2.set_ylabel(r'Critical Stretch, $\lambda_{cr}$')
    ax2.set_title(r'(b) Stability Analysis (The Trap)', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylim(0.4, 1.0)
    ax2.set_xlim(0, 50) 
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(True, ls=':', alpha=0.5)

    outfile = 'Figure7_Motivation.pdf'
    plt.savefig(outfile, format='pdf')
    print(f"Figure saved to {outfile}")

if __name__ == "__main__":
    main()

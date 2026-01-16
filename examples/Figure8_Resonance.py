"""
Figure 8 Reproduction Script: Resonance Mechanism (The Trap)
============================================================

This script visualizes the structural resonance mode responsible for the 
stability dip observed in Figure 7 (Case A).

It generates:
1. Panel (a): Mode shapes U1(x2) and U2(x2), highlighting the mid-depth shear peak.
2. Panel (b): Full-field displacement cloud map showing energy trapping.

Key Fixes in this version:
- Precise annotation positioning pointing to the "Bulge" at x2 ~ -0.35.
- Correct physical reconstruction of Case A (Exponential Gradient).

Usage:
    python examples/fig8_resonance_mechanism.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.integrate import solve_ivp

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stroh_solver import RealStrohSolver

# ==============================================================================
# 1. Plotting Style
# ==============================================================================
#try:
#    plt.rcParams.update({
#        "font.family": "serif",
#        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
#        "mathtext.fontset": "stix",
#        "font.size": 11,
#        "axes.linewidth": 1.0,
#        "axes.labelsize": 12,
#        "xtick.labelsize": 10,
#        "ytick.labelsize": 10,
#        "legend.fontsize": 10,
#        "savefig.bbox": "tight",
#        "savefig.dpi": 600,
#    })
#except:
#    pass

# ==============================================================================
# 2. Physics & Reconstruction Logic
# ==============================================================================

class ResonanceVisualizer:
    def __init__(self, mu_func, H=1.0):
        self.mu_func = mu_func
        self.H = H
        self.solver = RealStrohSolver(mu_func, H)

    def _system_derivative(self, x, y_flat, k, lam):
        y = y_flat.reshape(2, 4).T 
        N = self.solver.get_real_stroh_matrix(x, k, lam)
        dY = N @ y
        return dY.T.flatten()

    def get_mode_shape(self, k, lam):
        """
        Reconstructs the eigenmode (U1, U2) by shooting from Clamped Base.
        """
        # 1. Integrate Basis from Bottom (-H) to Top (0)
        # Clamped BC: U=0 at base. We launch 2 independent Stress modes.
        y0 = np.zeros((2, 4))
        y0[0, 2] = 1.0 # Stress Mode 1
        y0[1, 3] = 1.0 # Stress Mode 2
        
        sol = solve_ivp(
            lambda x, y: self._system_derivative(x, y, k, lam),
            [-self.H, 0], y0.flatten(),
            dense_output=True, rtol=1e-8, atol=1e-10
        )
        
        # 2. Satisfy Surface Traction-Free Condition
        # Y_surf = [U, T]. We need T*c = 0.
        Y_surf = sol.y[:, -1].reshape(2, 4).T
        T_surf = Y_surf[2:4, :] # Traction submatrix
        
        # Find null space of T_surf using SVD
        u, s, vh = np.linalg.svd(T_surf)
        c = vh[-1, :] # The coefficients for the linear combination
        
        # 3. Construct the full mode profile
        x_eval = np.linspace(-self.H, 0, 400)
        Y_eval_flat = sol.sol(x_eval)
        Y_eval = Y_eval_flat.reshape(2, 4, 400)
        
        # Linear combination: Mode = c[0]*Basis1 + c[1]*Basis2
        Mode = c[0] * Y_eval[0] + c[1] * Y_eval[1]
        
        U1 = Mode[0] # Tangential
        U2 = Mode[1] # Normal
        
        # Normalize by max magnitude
        mag = np.sqrt(U1**2 + U2**2)
        norm_factor = np.max(mag)
        
        return x_eval, U1/norm_factor, U2/norm_factor

# ==============================================================================
# 3. Main Execution
# ==============================================================================

def main():
    print("--- Generating Figure 8: Structural Resonance (The Trap) ---")
    
    # --- Define Case A (Monotonic) from Figure 7 ---
    # This specific profile creates the dip at kH ~ 10
    def mu_case_a(x2):
        return 0.2 * np.exp(-2.0 * x2)

    viz = ResonanceVisualizer(mu_case_a)
    
    # --- Parameters at the "Dip" (The Trap) ---
    k_res = 10.0   # The dangerous intermediate wavenumber
    lam_res = 0.61 # The critical stretch at the dip
    
    print(f"Reconstructing mode at kH={k_res}, lambda={lam_res}...")
    z, u1, u2 = viz.get_mode_shape(k_res, lam_res)

    # --- Plotting ---
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.25)
    
    # ==========================================
    # Panel (a): Mode Profiles U(x2)
    # ==========================================
    ax1 = fig.add_subplot(gs[0])
    
    # Highlight the stiff/soft regions
    ax1.axhspan(-1.0, -0.8, color='blue', alpha=0.05) # Stiff Base region
    ax1.text(0.5, -0.95, 'Stiff Base', color='blue', alpha=0.5, ha='center', fontsize=9)
    
    # Plot Curves
    ax1.plot(u1, z, 'b-', lw=2.5, label=r'Tangential ($U_1$)')
    ax1.plot(u2, z, 'r--', lw=2.5, label=r'Normal ($U_2$)')
    
    # Annotations for Panel (a) - CORRECTED POSITIONS
    # Arrow pointing to the U1 bulge at mid-depth
    ax1.annotate('Resonance Depth\n(Max Shear)', 
                 xy=(-0.3, -0.32),       # Point to the bulge of U1
                 xytext=(0.1, -0.32),   # Text to the right
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.0, headwidth=6),
                 fontsize=11, color='black', ha='left', va='center')

    ax1.set_xlabel('Normalized Amplitude')
    ax1.set_ylabel(r'Normalized Depth, $x_2/H$')
    ax1.set_title(r'(a) Mode Amplitude $U(x_2)$', fontsize=13, loc='left')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.0, 0.0)
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, fontsize=10)
    
    # ==========================================
    # Panel (b): Full Field Cloud Map
    # ==========================================
    ax2 = fig.add_subplot(gs[1])
    
    # Generate 2D Grid
    x_grid = np.linspace(0, 4*np.pi/k_res, 200) # 2 wavelengths
    X, Z = np.meshgrid(x_grid, z)
    
    # Reconstruct Field: u1 ~ sin(kx), u2 ~ cos(kx)
    # Note: Phase shift represents standing wave structure
    U1_field = u1[:, np.newaxis] * np.sin(k_res * X)
    U2_field = u2[:, np.newaxis] * np.cos(k_res * X)
    
    # Magnitude
    Mag = np.sqrt(U1_field**2 + U2_field**2)
    
    # Contour Plot
    levels = np.linspace(0, 1.0, 20)
    cf = ax2.contourf(X, Z, Mag, levels=levels, cmap='viridis')
    
    # Quiver Flow (Downsampled)
    skip_x, skip_z = 8, 8
    ax2.quiver(X[::skip_z, ::skip_x], Z[::skip_z, ::skip_x], 
               U1_field[::skip_z, ::skip_x], U2_field[::skip_z, ::skip_x],
               color='white', alpha=0.6, scale=20, width=0.002)
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax2, pad=0.02)
    cbar.set_label(r'Displacement Magnitude $|\mathbf{u}|$', rotation=270, labelpad=15)
    
    # Annotations for Panel (b) - CORRECTED POSITIONS
    
    # 1. Surface Label
    ax2.text(0.64, -0.05, 'Soft Surface (Driven)', color='white', 
             ha='center', va='top', fontsize=10, fontweight='bold',
             path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # 2. Mid-Depth Resonance Label (The Critical Fix)
    # Pointing UP to the energy band at -0.3, text sitting in the quiet zone at -0.55
    ax2.annotate('Energy Concentration\n(Mid-Depth Trap)', 
                 xy=(0.64, -0.30),      # Point to the yellow/green band
                 xytext=(0.64, -0.55),  # Text in the dark blue zone
                 arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, color='white', ha='center', fontweight='bold',
                 path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # 3. Base Label
    ax2.text(0.64, -0.90, 'Clamped Base (Silent)', color='white', 
             ha='center', va='bottom', fontsize=10, fontweight='bold', alpha=0.8)

    ax2.set_xlabel(r'Horizontal Position $x_1$')
    ax2.set_ylabel(r'Depth $x_2/H$')
    ax2.set_title(rf'(b) Eigenmode Cloud Map ($kH={int(k_res)}$)', fontsize=13, loc='left')
    ax2.set_ylim(-1.0, 0.0)
    
    # Save
    outfile = 'Figure8_Resonance.pdf'
    plt.savefig(outfile, format='pdf')
    print(f"Figure saved to {outfile}")

if __name__ == "__main__":
    main()
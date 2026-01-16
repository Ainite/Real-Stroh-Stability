"""
Figure 5 Reproduction Script: Eigenmode Duality (Fixed Logic)
=============================================================

This script visualizes the two fundamental instability types in graded solids:
1. Surface Wrinkling (Short-wave limit): Energy localized at the surface.
2. Global/Resonance Buckling (Long/Intermediate wave): Energy penetrates deep.

Fixes in this version:
- Case I Search Range: Adjusted to (0.6, 0.95) to catch the true Surface Wrinkling mode.
  (Previous range (0.4, 0.6) caught a spurious high-order internal mode).
- Normalization: Ensures consistent visualization of relative amplitudes.

Usage:
    python examples/fig5_eigenmode_duality.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import warnings

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stroh_solver import RealStrohSolver

warnings.filterwarnings("ignore")

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
# 2. Material Definitions
# ==============================================================================

def mu_stiff_skin(x2):
    """ Case I: Stiff Skin (L/H = 0.15, beta = 10) """
    mu_sub = 1.0
    mu_surf = 10.0
    L = 0.15
    return mu_sub + (mu_surf - mu_sub) * np.exp(x2 / L)

def mu_soft_skin(x2):
    """ Case II: Soft Skin (L/H = 0.5, beta = 0.1) """
    mu_sub = 1.0
    mu_surf = 0.1
    L = 0.5
    return mu_sub + (mu_surf - mu_sub) * np.exp(x2 / L)

# ==============================================================================
# 3. Eigenmode Reconstruction Logic
# ==============================================================================

class EigenmodeReconstructor:
    def __init__(self, mu_func, H=1.0):
        self.solver = RealStrohSolver(mu_func, H)
        self.H = H

    def _system_derivative(self, x, y_flat, k, lam):
        y = y_flat.reshape(2, 4).T 
        N = self.solver.get_real_stroh_matrix(x, k, lam)
        dY = N @ y
        return dY.T.flatten()

    def get_residual(self, k, lam):
        # Shooting from clamped base
        y0 = np.zeros((2, 4))
        y0[0, 2] = 1.0; y0[1, 3] = 1.0 
        
        sol = solve_ivp(
            lambda x, y: self._system_derivative(x, y, k, lam),
            [-self.H, 0], y0.flatten(),
            rtol=1e-5, atol=1e-8
        )
        
        Y_surf = sol.y[:, -1].reshape(2, 4).T
        T_surf = Y_surf[2:4, :] 
        return np.linalg.det(T_surf)

    def find_critical_lambda(self, k, search_range):
        def objective(lam):
            return abs(self.get_residual(k, lam))
        
        # 使用 bounded 方法寻找 det(T)=0 的根
        res = minimize_scalar(objective, bounds=search_range, method='bounded')
        return res.x

    def reconstruct_mode(self, k, lam):
        y0 = np.zeros((2, 4))
        y0[0, 2] = 1.0; y0[1, 3] = 1.0
        
        sol = solve_ivp(
            lambda x, y: self._system_derivative(x, y, k, lam),
            [-self.H, 0], y0.flatten(),
            dense_output=True, rtol=1e-8, atol=1e-10
        )
        
        Y_surf = sol.y[:, -1].reshape(2, 4).T
        T_surf = Y_surf[2:4, :]
        u, s, vh = np.linalg.svd(T_surf)
        c = vh[-1, :] 
        
        x_eval = np.linspace(-self.H, 0, 300)
        Y_eval_flat = sol.sol(x_eval)
        Y_eval = Y_eval_flat.reshape(2, 4, 300)
        
        Mode = c[0] * Y_eval[0] + c[1] * Y_eval[1]
        U1 = Mode[0]
        U2 = Mode[1]
        
        # 统一归一化，使得最大分量的幅值为 1.0
        # 对于 Wrinkling，通常 U2 最大；对于 Resonance，可能 U1 最大。
        amp_mag = np.sqrt(U1**2 + U2**2) # 总位移幅值
        max_val = np.max(amp_mag)
        
        return x_eval, U1/max_val, U2/max_val

# ==============================================================================
# 4. Main Execution
# ==============================================================================

def main():
    print("--- Generating Figure 5: Mode Duality (Corrected) ---")
    
    # --- Panel (a): Surface Wrinkling (Stiff Skin) ---
    print("\nProcessing Case I: Stiff-on-Soft (Wrinkling)...")
    rec1 = EigenmodeReconstructor(mu_stiff_skin)
    k1 = 12.0
    # 【关键修正】: 搜索范围提高到 (0.6, 0.95)。
    # 硬皮非常不稳定，临界拉伸比通常较高（接近1）。
    lam1 = rec1.find_critical_lambda(k1, search_range=(0.6, 0.95))
    print(f"  Found eigenvalue: k={k1}, lambda={lam1:.4f}")
    z1, u1_1, u2_1 = rec1.reconstruct_mode(k1, lam1)
    
    # --- Panel (b): Global/Resonance (Soft Skin) ---
    print("\nProcessing Case II: Soft-on-Stiff (Resonance)...")
    rec2 = EigenmodeReconstructor(mu_soft_skin)
    k2 = 1.5
    # 软皮比较稳定，临界值较低，保持原范围
    lam2 = rec2.find_critical_lambda(k2, search_range=(0.2, 0.5))
    print(f"  Found eigenvalue: k={k2}, lambda={lam2:.4f}")
    z2, u1_2, u2_2 = rec2.reconstruct_mode(k2, lam2)

    # --- Plotting ---
    print("\nPlotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    plt.subplots_adjust(wspace=0.15, bottom=0.15, top=0.88, left=0.08, right=0.98)

    c_u1 = '#00429d' # Deep Blue
    c_u2 = '#d62728' # Red
    
    # === Visual Settings ===
    # Case I Visuals
    L_stiff = 0.15
    ax1.axhspan(-L_stiff, 0, color='gold', alpha=0.2, lw=0)
    ax1.text(-1.0, -L_stiff/2, r'Stiff Skin ($L/H=0.15$)', fontsize=9, ha='left', va='center', color='darkgoldenrod')
    
    ax1.plot(u1_1, z1, color=c_u1, lw=2.5, label=r'Tangential, $U_1$')
    ax1.plot(u2_1, z1, color=c_u2, lw=2.5, ls='--', label=r'Normal, $U_2$')
    
    ax1.text(0.7, -0.85, 'Surface Mode\n(Wrinkling)', fontsize=12, fontweight='bold', ha='center',
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax1.text(-0.05, 1.03, '(a) Stiff-on-Soft (' + r'$\beta=10$' + ')', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    
    ax1.set_xlabel('Normalized Amplitude')
    ax1.set_ylabel(r'Normalized Depth, $x_2/H$')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.0, 0.0)
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.legend(loc='lower left', frameon=True, fontsize=10)

    # Case II Visuals
    L_soft = 0.5
    ax2.axhspan(-L_soft, 0, color='blue', alpha=0.08, lw=0)
    ax2.text(-0.3, -L_soft/2, r'Soft Skin ($L/H=0.5$)', fontsize=9, ha='right', va='center', color='darkblue')
    
    ax2.plot(u1_2, z2, color=c_u1, lw=2.5)
    ax2.plot(u2_2, z2, color=c_u2, lw=2.5, ls='--')
    
    ax2.text(-0.60, -0.85, 'Internal Resonance\n(Energy Trap)', fontsize=12, fontweight='bold', ha='center',
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax2.text(-0.05, 1.03, '(b) Soft-on-Stiff (' + r'$\beta=0.1$' + ')', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Normalized Amplitude')
    ax2.set_xlim(-1.1, 1.1)
    ax2.grid(True, ls=':', alpha=0.5)

    ax1.axvline(0, color='k', lw=0.8, alpha=0.3)
    ax2.axvline(0, color='k', lw=0.8, alpha=0.3)

    outfile = 'Figure5_Eigenmodes.pdf'
    plt.savefig(outfile, format='pdf')
    print(f"Figure saved to {outfile}")

if __name__ == "__main__":
    main()
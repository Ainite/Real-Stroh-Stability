"""
Figure 3 Reproduction Script
============================

This script reproduces Figure 3 from the manuscript:
"Benchmarking of the proposed Real-Variable CMM."

It compares the accuracy and numerical stability of:
1. The proposed Real-CMM (Exact).
2. Standard FEM (N=8 elements, showing discretization error).
3. Standard Shooting Method (showing numerical explosion).

Usage:
    python examples/fig3_benchmark.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
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
# 2. Auxiliary Solvers (FEM & Shooting) for Comparison
# ==============================================================================

class StandardShootingSolver:
    """Helper class to demonstrate the failure of standard shooting."""
    def __init__(self, solver):
        self.solver = solver

    def ode_func(self, x, U_flat, k, lam):
        U = U_flat.reshape((4, 4))
        N = self.solver.get_real_stroh_matrix(x, k, lam)
        dU = np.dot(N, U)
        return dU.flatten()

    def get_growth(self, k, lam, n_steps=200):
        t_eval = np.linspace(-self.solver.H, 0, n_steps)
        U0 = np.eye(4).flatten()
        sol = solve_ivp(lambda x, u: self.ode_func(x, u, k, lam), 
                        (-self.solver.H, 0), U0, t_eval=t_eval, method='LSODA')
        mags = [np.linalg.norm(sol.y[:, i].reshape((4, 4))) for i in range(len(sol.t))]
        return sol.t, np.array(mags)

class FEMSolver_Benchmark:
    """Simplified FEM solver to demonstrate shear locking/discretization error."""
    def __init__(self, solver, N_elem=8):
        self.solver = solver
        self.N = N_elem
        self.H = solver.H
        self.dy = self.H / N_elem
        self.y_nodes = np.linspace(-self.H, 0, N_elem + 1)

    def get_det_sign(self, k, lam):
        # Construct global stiffness matrix for Real-Stroh FEM
        dim = 4
        dof = dim * (self.N + 1)
        K_global = np.zeros((dof, dof))
        
        for i in range(self.N):
            y_mid = (self.y_nodes[i] + self.y_nodes[i+1])/2
            # Use the shared physics engine
            N_mat = self.solver.get_real_stroh_matrix(y_mid, k, lam)
            I = np.eye(4)
            # Simple midpoint rule element
            M_right = -(I + (self.dy/2)*N_mat)
            M_left  =  (I - (self.dy/2)*N_mat)
            row, col = i * dim, i * dim
            K_global[row:row+4, col:col+4] = M_right
            K_global[row:row+4, col+4:col+8] = M_left
            
        # BCs: Identity at base (clamped-like in Stroh formalism context)
        # and Surface Traction free check
        row_bc = 4 * self.N
        K_global[row_bc, 0] = 1.0; K_global[row_bc+1, 1] = 1.0
        K_global[row_bc+2, 4*self.N + 2] = 1.0; K_global[row_bc+3, 4*self.N + 3] = 1.0
        
        sign, _ = np.linalg.slogdet(K_global)
        return sign

# ==============================================================================
# 3. Main Execution
# ==============================================================================

def main():
    print("Running Figure 3 Benchmark (Comparing Real-CMM vs FEM vs Shooting)...")

    # --- Define Material Profile ---
    # Exponential Gradient: mu(x) = 1.0 + (5.0 - 1.0) * exp(x/0.2)
    def mu_func(x2):
        mu_sub = 1.0
        mu_surf = 5.0
        L = 0.2
        return mu_sub + (mu_surf - mu_sub) * np.exp(x2 / L)

    # Initialize Physics Engines
    real_solver = RealStrohSolver(mu_func, H=1.0)
    cmm_integrator = CMMIntegrator(real_solver, n_steps=200)
    
    shoot_solver = StandardShootingSolver(real_solver)
    fem_solver = FEMSolver_Benchmark(real_solver, N_elem=8) # Coarse mesh to show error

    # === Panel (a): Accuracy (Dispersion Curves) ===
    print("Computing Panel (a): Dispersion Curves [kH=2.0-7.0]...")
    
    k_vals = np.linspace(2.0, 7.0, 40)
    lam_vals = np.linspace(0.6, 0.95, 60)
    
    # 1. Calculate Real-CMM (Exact)
    cmm_curve_k = []
    cmm_curve_lam = []
    
    for k in k_vals:
        # Scan for root
        dets = [cmm_integrator.check_stability(k, l) for l in lam_vals]
        # Simple root finding
        for i in range(len(dets)-1, 0, -1):
            if dets[i] * dets[i-1] < 0:
                try:
                    root = brentq(lambda l: cmm_integrator.check_stability(k, l), 
                                  lam_vals[i-1], lam_vals[i])
                    cmm_curve_k.append(k)
                    cmm_curve_lam.append(root)
                    break
                except: continue
    
    # 2. Calculate FEM (Approximate)
    print("Computing Panel (a): FEM Points...")
    k_fem = np.linspace(2.5, 6.5, 5)
    fem_points_lam = []
    
    for k in k_fem:
        det_signs = [fem_solver.get_det_sign(k, l) for l in lam_vals]
        root = np.nan
        for i in range(len(det_signs)-1, 0, -1):
            if det_signs[i] * det_signs[i-1] < 0:
                try:
                    root = brentq(lambda l: fem_solver.get_det_sign(k, l), 
                                  lam_vals[i-1], lam_vals[i])
                    break
                except: continue
        fem_points_lam.append(root)

    # === Panel (b): Numerical Stability ===
    print("Computing Panel (b): Numerical Growth at kH=45...")
    k_stiff = 45.0
    lam_ref = 0.85
    
    # Shooting Method Growth
    t_shoot, mag_shoot = shoot_solver.get_growth(k_stiff, lam_ref)
    
    # CMM Growth (Simulated by accessing internal state history if needed, 
    # but here we just re-run a profile logic similar to the manual script)
    # Note: The CMMIntegrator class in src usually returns a scalar.
    # For this plot, we manually run the normalized integration loop to get the profile.
    
    # Manual CMM Profile Extraction for Plotting
    n_steps = 200
    h = 1.0 / n_steps
    y = np.zeros(6); y[5] = 1.0
    x = -1.0
    t_cmm = []
    mag_cmm = []
    
    for _ in range(n_steps):
        t_cmm.append(x)
        mag_cmm.append(abs(y[5])) # Track the relevant minor
        
        N = real_solver.get_real_stroh_matrix(x, k_stiff, lam_ref)
        Q = real_solver.get_compound_matrix(N)
        k1 = Q @ y
        
        # ... (Simplified RK1 for visualization, or full RK4) ...
        # Using RK4 for consistency
        N2 = real_solver.get_real_stroh_matrix(x+0.5*h, k_stiff, lam_ref)
        Q2 = real_solver.get_compound_matrix(N2)
        k2 = Q2 @ (y + 0.5*h*k1)
        k3 = Q2 @ (y + 0.5*h*k2)
        N4 = real_solver.get_real_stroh_matrix(x+h, k_stiff, lam_ref)
        Q4 = real_solver.get_compound_matrix(N4)
        k4 = Q4 @ (y + h*k3)
        
        y = y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        # Normalize
        nm = np.max(np.abs(y))
        if nm > 1e-20: y /= nm
        x += h

    # === Plotting ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    plt.subplots_adjust(wspace=0.3, bottom=0.15, top=0.9, left=0.08, right=0.95)

    # Subplot 1: Accuracy
    ax1.plot(cmm_curve_k, cmm_curve_lam, 'k-', lw=2.5, label='Real-CMM (Exact)')
    ax1.plot(k_fem, fem_points_lam, 'bo', ms=7, fillstyle='none', markeredgewidth=1.5, label='FEM (N=8)')
    
    if len(fem_points_lam) > 1 and not np.isnan(fem_points_lam[1]):
        ax1.annotate('Discretization Error\n(Shear Locking)', 
                     xy=(k_fem[1], fem_points_lam[1]-0.02), 
                     xytext=(k_fem[1], fem_points_lam[1]-0.1),
                     arrowprops=dict(arrowstyle='->', color='b'), fontsize=10, color='b', ha='center')

    ax1.set_xlabel(r'Wavenumber, $kH$')
    ax1.set_ylabel(r'Critical Stretch, $\lambda_{cr}$')
    ax1.set_xlim(2.0, 7.0)
    ax1.set_ylim(0.6, 0.9)
    ax1.legend(loc='lower right', frameon=False)
    ax1.text(-0.15, 1.05, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax1.grid(True, ls=':', alpha=0.5)

    # Subplot 2: Stability
    x_axis_shoot = t_shoot + 1.0
    x_axis_cmm = np.array(t_cmm) + 1.0
    
    ax2.semilogy(x_axis_shoot, mag_shoot, 'r--', lw=2.5, label='Standard Shooting')
    ax2.semilogy(x_axis_cmm, np.array(mag_cmm) + 1e-10, 'k-', lw=2.5, label='Real-CMM')
    
    max_val = np.max(mag_shoot)
    exp_val = int(np.log10(max_val)) if max_val > 0 else 0
    ax2.text(0.9, max_val*0.1, f'$\\sim 10^{{{exp_val}}}$', color='red', ha='right', fontsize=12)
    ax2.text(0.9, 5e-2, r'$O(1)$', color='black', ha='right', fontsize=12)
    
    ax2.set_xlabel(r'Normalized Depth, $(x_2+H)/H$')
    ax2.set_ylabel(r'Solution Magnitude')
    ax2.set_ylim(1e-5, 1e16)
    ax2.legend(loc='upper left', frameon=False)
    ax2.text(-0.15, 1.05, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    ax2.grid(True, ls=':', alpha=0.5)

    outfile = 'Figure3_Benchmark.pdf'
    plt.savefig(outfile, format='pdf')
    print(f"Figure saved to {outfile}")

if __name__ == "__main__":
    main()

# Real-Stroh-Stability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX) This repository contains the reference implementation for the paper:

**"Engineering Stability Band-Gaps in Graded Soft Matter: Mode Duality and Inverse Spectral Design"**
*Feng Yang, et al.*
*Submitted to International Journal of Engineering Science (IJES)*

## 📖 Description

**Real-Stroh-Stability** is a robust computational framework designed to solve the surface instability problem in deep, functionally graded elastomers. 

Unlike standard shooting methods that suffer from **numerical stiffness** (parasitic error growth) in deep inhomogeneous substrates, this project employs a **Real-Variable Stroh Formalism** coupled with the **Compound Matrix Method (CMM)**. This approach guarantees numerical stability and linear independence of the solution space for arbitrary gradient profiles and wavenumbers.

### Key Features
* [cite_start]**Real-Variable Formulation:** Eliminates complex arithmetic and branch-cut ambiguities[cite: 7, 26].
* [cite_start]**Numerical Robustness:** Solves stiff differential equations in deep gradients using CMM (Compound Matrix Method)[cite: 6, 28].
* [cite_start]**Mode Duality Analysis:** Tools to identify and visualize both Surface Wrinkling (Biot mode) and Macroscopic Shear Buckling[cite: 8].
* [cite_start]**Inverse Design:** Includes the Differential Evolution (DE) engine to discover "Stiff-Soft-Stiff" stability band-gap topologies[cite: 10, 78].

---
*(Rest of the README as provided before...)*
## 🚀 Getting Started

### Prerequisites
* Python 3.8+ (Example)
* NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt

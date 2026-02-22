# Curvature Feedback Model (CFM)

**A Four-Paper Program for Geometric Cosmology Without the Dark Sector**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

The Curvature Feedback Model (CFM) replaces dark energy with a geometric curvature return potential and embeds modified gravity within the Horndeski/f(R) framework. The model is validated against Planck 2018 CMB data (TT+TE+EE, 6,405 data points) using [hi_class](https://github.com/miguelzuma/hi_class_public) with a custom `cfm_fR` gravity model.

**Key result:** The native `cfm_fR` model achieves **Delta chi2 = -3.6** vs. LCDM on Planck CMB data (MCMC best-fit), with alpha_M_0 = 0.0013 +/- 0.0007 (1.78 sigma detection, P(alpha_M_0 > 0) = 100%) and 100*theta_s = 1.04173 (identical to LCDM).

## Papers

| Paper | EN | DE | Topic |
|-------|----|----|-------|
| I | `papers/Paper1_EN.tex` | `papers/Paper1_DE.tex` | Game-theoretic foundation, CFM, Pantheon+ validation |
| II | `papers/Paper2_EN.tex` | `papers/Paper2_DE.tex` | MOND unification, baryon-only universe, running coupling |
| III | `papers/Paper3_EN.tex` | `papers/Paper3_DE.tex` | Lagrangian (R + gamma R^2), scalaron dynamics, predictions |
| IV | `papers/Paper4_EN.tex` | `papers/Paper4_DE.tex` | Galactic MOND from curvature saturation (DRAFT) |

## Key Results

| Model | chi2 (TT+TE+EE) | Delta chi2 | sigma8 | 100*theta_s |
|-------|----------------:|----------:|-------:|------------:|
| LCDM | 6628.8 | --- | 0.811 | 1.04173 |
| propto_omega cM=0.0002 | 6628.6 | -0.2 | 0.826 | 1.04173 |
| cfm_fR n=0.5, aM0=0.001 | 6626.1 | -2.7 | 0.899 | 1.04173 |
| cfm_fR n=1.0, aM0=0.0005 | 6627.1 | -1.6 | 0.879 | 1.04173 |
| **cfm_fR MCMC best-fit** | **6625.2** | **-3.6** | --- | 1.04173 |

The cfm_fR model implements:
```
alpha_M(a) = alpha_M_0 * n * a^n / (1 + alpha_M_0 * a^n)
alpha_B(a) = -alpha_M(a) / 2     [f(R) relation]
alpha_T    = 0                     [c_gw = c, consistent with GW170817]
alpha_K    = 0
```

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. hi_class (Horndeski in CLASS Boltzmann code)

hi_class is required for CMB power spectrum computations and the cfm_fR model.

```bash
# Clone hi_class
git clone https://github.com/miguelzuma/hi_class_public.git
cd hi_class_public

# Apply cfm_fR patch (adds the native CFM gravity model)
python /path/to/cfm-cosmology/scripts/patch_cfm.py

# Build hi_class with Python wrapper
cd python
python setup.py build
```

The patch modifies `gravity_models_smg.c` to add the `cfm_fR` gravity model. See [Patch Documentation](#cfm_fr-patch-documentation) below for details.

**Tested with:** hi_class v2.9.4+, Python 3.12, Cython 0.29.37, NumPy 1.26.4 on Ubuntu 24.04 (WSL).

### 3. Pantheon+ Data

The Pantheon+ supernova data (Scolnic et al. 2022) and Planck 2018 CMB spectra are downloaded automatically by the analysis scripts. No manual download required.

## Reproducing the Results

### Paper I: Pantheon+ Analysis
```bash
python scripts/cfm_pantheonplus_test.py    # Main SN analysis
python scripts/cfm_enhanced_analysis.py    # Extended functional forms
```

### Paper II: MOND + Baryon-Only
```bash
python scripts/cfm_baryon_only_test.py     # Baryon-only universe test
python scripts/cfm_mond_mcmc.py            # MOND MCMC analysis
```

### Paper III: hi_class CMB Validation
```bash
# Requires hi_class with cfm_fR patch (see Installation above)
python scripts/test_cfm_fR_native.py       # Systematic cfm_fR parameter scan
python scripts/compute_TT_TE_EE.py         # Planck TT+TE+EE chi2 computation
python scripts/scalaron_alphaM_theta_s.py  # theta_s resolution analysis
python scripts/compute_fsigma8.py          # Growth rate f*sigma8
python scripts/run_full_mcmc.py            # Full MCMC (5 params, ~8h runtime)
python scripts/analyze_mcmc_results.py     # MCMC posterior analysis
```

### Paper IV: Galactic MOND from Vector Sector
```bash
python scripts/paper4/sparc_full_analysis.py         # Full SPARC (171 galaxies) RAR test
python scripts/paper4/multi_galaxy_bvp.py            # Multi-mass BVP MOND attractor scan
python scripts/paper4/rotation_curves_bessel.py      # Bessel rotation curves
python scripts/paper4/a0_discrepancy.py              # a0 = cH0/(2pi) discrepancy analysis
python scripts/paper4/cfm_deep_mond_derivation.py    # Deep-MOND fixed point + Tully-Fisher
```

### Supplementary
```bash
python scripts/poeschl_teller_path_integral.py  # sqrt(pi) path integral
```

## Repository Structure

```
cfm-cosmology/
  README.md                  # This file
  LICENSE                    # CC BY 4.0
  requirements.txt           # Python dependencies
  CHANGES_cfm_fR.md          # Patch documentation
  papers/                    # LaTeX sources (4 papers x 2 languages)
  scripts/                   # All analysis scripts
  scripts/paper4/            # Paper IV galactic analysis scripts
  results/                   # Key numerical results
  results/paper4/            # Paper IV SPARC, BVP, rotation curve results
  figures/                   # Plots referenced in papers
  data/                      # Pantheon+ analysis outputs
```

## cfm_fR Patch Documentation

The file `scripts/patch_cfm.py` applies 4 modifications to `hi_class/gravity_models_smg.c`:

| # | Location | Change |
|---|----------|--------|
| 1 | `gravity_models_init()` | Registers `cfm_fR` as a new gravity model |
| 2 | `gravity_functions_smg()` | Computes alpha_M, alpha_B from parameters `(alpha_M_0, n_exp, M*2_init)` |
| 3 | `gravity_print_smg()` | Adds print output for cfm_fR parameters |
| 4 | Error message | Adds cfm_fR to the list of recognized models |

**Parameters passed via hi_class:**
```python
cosmo.set({
    'gravity_model': 'cfm_fR',
    'parameters_smg': f'{alpha_M_0}, {n_exp}, 1.0',
    'expansion_model': 'lcdm',
    'Omega_smg': -1,
})
```

**Physical interpretation:**
- `alpha_M_0`: Amplitude of the Planck mass running rate
- `n_exp`: Power-law index controlling time evolution (n=0.5 best fit, n=1 reproduces propto_scale)
- At early times (a << 1): alpha_M ~ alpha_M_0 * n * a^n (perturbative)
- At late times (a ~ 1): alpha_M -> n_exp / (1 + alpha_M_0) (saturates)

## Software Citations

This work uses the following open-source software:

- **CLASS** (Cosmic Linear Anisotropy Solving System): Blas, Lesgourgues & Tram (2011), JCAP 07, 034. [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)
- **hi_class** (Horndeski in CLASS): Zumalacarregui, Bellini, Sawicki, Lesgourgues & Ferreira (2017), JCAP 01, 019. [arXiv:1605.06102](https://arxiv.org/abs/1605.06102)
- **emcee** (MCMC sampler): Foreman-Mackey, Hogg, Lang & Goodman (2013), PASP 125, 306. [arXiv:1202.3665](https://arxiv.org/abs/1202.3665)
- **NumPy**: Harris et al. (2020), Nature 585, 357.
- **SciPy**: Virtanen et al. (2020), Nature Methods 17, 261.
- **Matplotlib**: Hunter (2007), Computing in Science & Engineering 9, 90.

**Observational data:**
- **Pantheon+**: Scolnic et al. (2022), ApJ 938, 113. [arXiv:2112.03863](https://arxiv.org/abs/2112.03863)
- **Planck 2018**: Aghanim et al. (2020), A&A 641, A6. [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
- **SPARC**: Lelli, McGaugh & Schombert (2016), AJ 152, 157. [arXiv:1606.09251](https://arxiv.org/abs/1606.09251)

## Citation

```bibtex
@article{Geiger2026CFM,
  author  = {Geiger, Lukas},
  title   = {Game-Theoretic Cosmology and the Curvature Feedback Model},
  year    = {2026},
  note    = {Paper I of the CFM program}
}

@article{Geiger2026CFMMOND,
  author  = {Geiger, Lukas},
  title   = {CFM-MOND Unification: A Baryonic Universe Without Dark Matter},
  year    = {2026},
  note    = {Paper II of the CFM program}
}

@article{Geiger2026CFMQG,
  author  = {Geiger, Lukas},
  title   = {From Curvature Feedback to Quantum Gravity: Lagrangian Foundations
             and Testable Predictions},
  year    = {2026},
  note    = {Paper III of the CFM program}
}

@article{Geiger2026CFMMOND2,
  author  = {Geiger, Lukas},
  title   = {The Galactic-Cosmological Nexus: Deriving MOND Dynamics
             from Curvature Saturation},
  year    = {2026},
  note    = {Paper IV of the CFM program (draft)}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

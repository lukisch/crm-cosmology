# Curvature Relaxation Model (CRM)

**A Four-Paper Program for Geometric Cosmology Without the Dark Sector**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18728935.svg)](https://doi.org/10.5281/zenodo.18728935)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

The Curvature Relaxation Model (CRM) replaces dark energy with a geometric curvature return potential and embeds modified gravity within the Horndeski/f(R) framework. The model is validated against Planck 2018 CMB data (TT+TE+EE, 6,405 data points) using [hi_class](https://github.com/miguelzuma/hi_class_public) with a custom `crm_fR` gravity model.

**Key result:** The native `crm_fR` model achieves **Delta chi2 = -3.7** vs. LCDM on Planck CMB data (MCMC best-fit), with alpha_M_0 = 0.0011 +/- 0.0007 (1.76 sigma detection, P(alpha_M_0 > 0) = 100%) and 100*theta_s = 1.04173 (identical to LCDM).

## Core Papers (I--IV)

| Paper | EN | DE | Topic |
|-------|----|----|-------|
| I | `papers/Paper1_EN.tex` | `papers/Paper1_DE.tex` | Game-theoretic foundation, CRM, Pantheon+ validation |
| II | `papers/Paper2_EN.tex` | `papers/Paper2_DE.tex` | MOND unification, baryon-only universe, running coupling |
| III | `papers/Paper3_EN.tex` | `papers/Paper3_DE.tex` | Lagrangian (R + gamma R^2), scalaron dynamics, predictions |
| IV | `papers/Paper4_EN.tex` | `papers/Paper4_DE.tex` | Galactic MOND from curvature saturation (DRAFT) |

## Extensions

| Paper | EN | DE | Topic | DOI |
|-------|----|----|-------|-----|
| V | `papers/extensions/Paper5_EN.tex` | `papers/extensions/Paper5_DE.tex` | The Saturation Theorem: tanh profile as mathematical necessity from 4 QG axioms | [10.5281/zenodo.19036188](https://doi.org/10.5281/zenodo.19036188) |
| VI | `papers/extensions/Paper6_EN.tex` | `papers/extensions/Paper6_DE.tex` | QG-CRM: Ultraviolet Completion via Quantum Quadratic Gravity (DRAFT) | [10.5281/zenodo.19352448](https://doi.org/10.5281/zenodo.19352448) |

**Paper V -- The Saturation Theorem** proves that the tanh saturation profile of Papers I--IV is not a model choice but a mathematical necessity: any quantum gravity theory satisfying four minimal axioms must produce the tanh form. All major QG programs (LQG, asymptotic safety, strings, causal sets, noncommutative geometry) are shown to satisfy the axioms.

**Paper VI -- QG-CRM: Ultraviolet Completion** answers the open question from Paper V: "which UV completion selects k and Phi_0?" By identifying the gamma*R^2 sector of the CRM Lagrangian with asymptotically free quantum quadratic gravity (QQG), inflation is generated dynamically via RG running without an inflaton field. The Saturation Theorem provides the unique UV-IR interface. Predictions: n_s ~ 1 - 4/(3N) ~ 0.976, r >= 0.01, testable with Stage IV CMB experiments.

## Key Results

| Model | chi2 (TT+TE+EE) | Delta chi2 | sigma8 | 100*theta_s |
|-------|----------------:|----------:|-------:|------------:|
| LCDM | 6628.8 | --- | 0.811 | 1.04173 |
| propto_omega cM=0.0002 | 6628.6 | -0.2 | 0.826 | 1.04173 |
| crm_fR n=0.5, aM0=0.001 | 6626.1 | -2.7 | 0.899 | 1.04173 |
| crm_fR n=1.0, aM0=0.0005 | 6627.1 | -1.6 | 0.879 | 1.04173 |
| **crm_fR MCMC best-fit** | **6625.1** | **-3.7** | --- | 1.04173 |

The crm_fR model implements:
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

hi_class is required for CMB power spectrum computations and the crm_fR model.

```bash
# Clone hi_class
git clone https://github.com/miguelzuma/hi_class_public.git
cd hi_class_public

# Apply crm_fR patch (adds the native CRM gravity model)
python /path/to/crm-cosmology/scripts/patch_cfm.py

# Build hi_class with Python wrapper
cd python
python setup.py build
```

The patch modifies `gravity_models_smg.c` to add the `crm_fR` gravity model. See [Patch Documentation](#crm_fr-patch-documentation) below for details.

**Tested with:** hi_class v2.9.4+, Python 3.12, Cython 0.29.37, NumPy 1.26.4 on Ubuntu 24.04 (WSL).

### 3. Pantheon+ Data

The Pantheon+ supernova data (Scolnic et al. 2022) and Planck 2018 CMB spectra are downloaded automatically by the analysis scripts. No manual download required.

## Reproducing the Results

### Paper I: CMB and MCMC
```bash
python scripts/paper1/run_full_mcmc.py            # Full MCMC (5 params, ~8h runtime)
python scripts/paper1/analyze_mcmc_results.py     # MCMC posterior analysis
python scripts/paper1/compute_TT_TE_EE.py         # Planck TT+TE+EE chi2 computation
python scripts/paper1/compute_fsigma8.py          # Growth rate f*sigma8
python scripts/paper1/full_cl_comparison.py        # Full Cl comparison cfm_fR vs LCDM
```

### Paper II: Model Comparison
```bash
python scripts/paper2/compare_models.py            # LCDM vs constant_alphas vs cfm_fR
python scripts/paper2/plot_contour.py              # 2D chi2 contour from grid scan
python scripts/paper2/plot_tradeoff.py             # chi2-sigma8 tradeoff + convergence
```

### Paper III: Pantheon+ and MOND
```bash
python scripts/paper3/cfm_pantheonplus_test.py     # CFM vs LCDM against Pantheon+ data
python scripts/paper3/cfm_baryon_only_test.py      # Baryon-only universe test
python scripts/paper3/cfm_mond_mcmc.py             # MCMC for CFM+MOND extended model
python scripts/paper3/scalaron_alphaM_theta_s.py   # theta_s resolution analysis
python scripts/paper3/poeschl_teller_path_integral.py  # sqrt(pi) path integral
```

### Paper IV: Galactic MOND from Vector Sector
```bash
python scripts/paper4/sparc_full_analysis.py       # Full SPARC (171 galaxies) RAR test
python scripts/paper4/multi_galaxy_bvp.py          # Multi-mass BVP MOND attractor scan
python scripts/paper4/rotation_curves_bessel.py    # Bessel rotation curves
python scripts/paper4/a0_discrepancy.py            # a0 = cH0/(2pi) discrepancy analysis
python scripts/paper4/cfm_deep_mond_derivation.py  # Deep-MOND fixed point + Tully-Fisher
```

### Infrastructure (cross-paper)
```bash
python scripts/patch_cfm.py                        # hi_class crm_fR gravity model patch
python scripts/test_cfm_fR_native.py               # Native crm_fR model test
```

## Repository Structure

```
crm-cosmology/
  README.md                    # This file
  LICENSE                      # CC BY 4.0
  requirements.txt             # Python dependencies
  papers/                      # Core Papers I-IV (LaTeX + PDF, EN + DE)
    extensions/                # Extension Papers V-VI (and future)
  scripts/                     # Cross-paper infrastructure (patch, tests)
    paper1/                    # Paper I: CMB, MCMC analysis
    paper2/                    # Paper II: model comparison, plots
    paper3/                    # Paper III: Pantheon+, MOND, scalaron
    paper4/                    # Paper IV: galactic MOND, SPARC
  results/                     # Cross-paper results
    paper1/                    # Paper I: MCMC summaries, chi2 results
    paper3/                    # Paper III: baryon-only, MOND posteriors
    paper4/                    # Paper IV: SPARC, BVP, rotation curves
  figures/                     # Plots referenced in papers
    paper1/                    # Paper I: Cl spectra, fsigma8
    paper2/                    # Paper II: contours, tradeoffs
    paper3/                    # Paper III: MOND posteriors
  data/                        # Analysis outputs
    paper3/                    # Paper III: Pantheon+ fits
```

## crm_fR Patch Documentation

The file `scripts/patch_cfm.py` applies 5 modifications to hi_class:

| # | File | Location | Change |
|---|------|----------|--------|
| 0 | `include/background.h` | `gravity_model` enum | Adds `cfm_fR` to the gravity model enum |
| 1 | `gravity_models_smg.c` | `gravity_models_init()` | Registers `cfm_fR` as a new gravity model (3 parameters, M2 evolution) |
| 2 | `gravity_models_smg.c` | `gravity_functions_smg()` | Computes alpha_M, alpha_B from parameters `(alpha_M_0, n_exp, M*2_init)` |
| 3 | `gravity_models_smg.c` | `gravity_print_stdout_smg()` | Adds print output for `cfm_fR` parameters |
| 4 | `gravity_models_smg.c` | Error message | Adds `cfm_fR` to the list of recognized models |

**Parameters passed via hi_class:**
```python
cosmo.set({
    'gravity_model': 'crm_fR',
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

If you use this work, please cite the Zenodo deposit:

```bibtex
@misc{Geiger2026CRM,
  author    = {Geiger, Lukas},
  title     = {The Curvature Relaxation Model: A Four-Paper Program
               for Geometric Cosmology Without the Dark Sector},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18728935},
  url       = {https://doi.org/10.5281/zenodo.18728935}
}
```

Individual papers:

```bibtex
@article{Geiger2026CRM_I,
  author  = {Geiger, Lukas},
  title   = {Game-Theoretic Cosmology and the Curvature Relaxation Model},
  year    = {2026},
  doi     = {10.5281/zenodo.18728935},
  note    = {Paper I of the CRM program}
}

@article{Geiger2026CRM_II,
  author  = {Geiger, Lukas},
  title   = {CRM-MOND Unification: A Baryonic Universe Without Dark Matter},
  year    = {2026},
  doi     = {10.5281/zenodo.18728935},
  note    = {Paper II of the CRM program}
}

@article{Geiger2026CRM_III,
  author  = {Geiger, Lukas},
  title   = {From Curvature Relaxation to Quantum Gravity: Lagrangian Foundations
             and Testable Predictions},
  year    = {2026},
  doi     = {10.5281/zenodo.18728935},
  note    = {Paper III of the CRM program}
}

@article{Geiger2026CRM_IV,
  author  = {Geiger, Lukas},
  title   = {The Galactic-Cosmological Nexus: Deriving MOND Dynamics
             from Curvature Saturation},
  year    = {2026},
  doi     = {10.5281/zenodo.18728935},
  note    = {Paper IV of the CRM program (draft)}
}
```

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## Haftung / Liability

Dieses Projekt ist eine **unentgeltliche Open-Source-Schenkung** im Sinne der §§ 516 ff. BGB. Die Haftung des Urhebers ist gemäß **§ 521 BGB** auf **Vorsatz und grobe Fahrlässigkeit** beschränkt. Ergänzend gelten die Haftungsausschlüsse aus GPL-3.0 / MIT / Apache-2.0 §§ 15–16 (je nach gewählter Lizenz).

Nutzung auf eigenes Risiko. Keine Wartungszusage, keine Verfügbarkeitsgarantie, keine Gewähr für Fehlerfreiheit oder Eignung für einen bestimmten Zweck.

This project is an unpaid open-source donation. Liability is limited to intent and gross negligence (§ 521 German Civil Code). Use at your own risk. No warranty, no maintenance guarantee, no fitness-for-purpose assumed.


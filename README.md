# Curvature Feedback Model (CFM)

**Game-Theoretic Cosmology: An Alternative to Dark Energy**

## Overview

The Curvature Feedback Model (CFM) explains the accelerated expansion of the universe without dark energy. Instead of introducing a cosmological constant $\Lambda$ or a new scalar field, the CFM postulates a time-dependent curvature return potential $\Phi(a)$ -- a geometric "memory" of the initial energy concentration at the Big Bang.

The modified Friedmann equation reads:

$$H^2(a) = H_0^2 \left[\Omega_m \, a^{-3} + \Omega_\Phi(a)\right]$$

with

$$\Omega_\Phi(a) = \Phi_0 \cdot \frac{\tanh(k \cdot (a - a_\mathrm{trans})) + s}{1 + s}$$

where $s = \tanh(k \cdot a_\mathrm{trans})$ ensures $\Omega_\Phi(0) = 0$.

## Key Results (Pantheon+ Test)

The CFM was tested against **1,590 real Type Ia supernovae** from the Pantheon+ catalog (Scolnic et al. 2022, ApJ 938, 113):

| Criterion | LCDM | CFM (flat) | Winner |
|-----------|------|------------|--------|
| chi2 | 729.0 | 716.8 (**-12.2**) | CFM |
| AIC | 733.0 | 724.8 (**-8.2**) | CFM |
| BIC | 743.7 | 746.3 (+2.6) | LCDM (marginal) |
| 5-Fold CV | 0.4519 | 0.4499 | CFM |

**3 of 4 model selection criteria favor CFM over LCDM.**

### Fitted Parameters (flat CFM)
- $\Omega_m = 0.364$ (Planck: 0.315)
- $k = 1.30$ (transition sharpness)
- $a_\mathrm{trans} = 0.75$ ($z_\mathrm{trans} = 0.33$)
- $\Phi_0 = 1.047$ (derived from flatness condition)

## Repository Contents

| File | Description |
|------|-------------|
| `Spieltheorie_Urknall_Artikel.tex` | Full paper (German + English) |
| `cfm_pantheonplus_test.py` | Analysis script (documented) |
| `CFM_Pantheon_Plus_Ergebnis.txt` | Detailed results report |
| `CFM_Pantheon_Plus_Ergebnis.png` | 6-panel visualization |

## Running the Analysis

```bash
pip install numpy pandas scipy matplotlib requests
python cfm_pantheonplus_test.py
```

The script automatically downloads the Pantheon+ dataset from GitHub and produces the results report and visualization.

## Paper

**"Spieltheoretische Kosmologie und das Kruemmungs-Rueckgabepotential-Modell"**
(Game-Theoretic Cosmology and the Curvature Feedback Model)

Lukas Geiger, February 2026

The paper develops a game-theoretic framework where the emergence of spacetime is modeled as a Nash equilibrium between a metastable quantum vacuum ("null space") and a spacetime bubble. The key insight: accelerated expansion is not a new "drive" but a "releasing brake."

## Citation

```bibtex
@article{Geiger2026CFM,
  author  = {Geiger, Lukas},
  title   = {Game-Theoretic Cosmology and the Curvature Feedback Model},
  year    = {2026},
  note    = {Available at \url{https://github.com/lukisch/cfm-cosmology}}
}
```

## License

CC BY 4.0

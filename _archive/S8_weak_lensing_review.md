# Weak Lensing S8 Constraints: Summary for CFM Comparison

**Date compiled:** 2026-02-14
**Context:** CFM model prediction S8 ~ 0.864 vs. weak lensing surveys

---

## 1. Summary Table of S8 Measurements

| Survey / Probe              | S8 Value              | Year | Tension with Planck | Tension with CFM (0.864) |
|-----------------------------|-----------------------|------|---------------------|--------------------------|
| **Planck 2018 (CMB)**       | 0.832 +/- 0.013      | 2018 | --                  | ~2.5 sigma               |
| **Combined CMB (2026)**     | 0.836 +0.012/-0.013  | 2026 | --                  | ~2.2 sigma               |
| **KiDS-1000 (3x2pt)**       | 0.766 +0.020/-0.014  | 2021 | 2.7-3.0 sigma       | ~5.0 sigma               |
| **KiDS-Legacy (cosmic shear)** | 0.815 +0.016/-0.021 | 2025 | 0.7 sigma           | ~2.5 sigma               |
| **DES Y3 (3x2pt)**          | 0.776 +/- 0.017      | 2022 | 2.3 sigma           | ~5.2 sigma               |
| **DES Y6 (3x2pt)**          | 0.789 +/- 0.012      | 2026 | 2.4-2.7 sigma       | ~6.3 sigma               |
| **DES Y3+KiDS-1000 joint**  | 0.790 +0.018/-0.014  | 2023 | 1.8-2.0 sigma       | ~4.1 sigma               |
| **HSC Y3 (cosmic shear)**   | 0.776 +0.032/-0.033  | 2023 | ~1.7 sigma          | ~2.7 sigma               |
| **HSC Y3 + DESI calib.**    | 0.805 +/- 0.018      | 2025 | ~1.5 sigma          | ~3.3 sigma               |
| **Euclid**                  | Not yet available     | --   | --                  | --                       |

### Notes on CFM tension calculation:
- CFM: sigma8 = 0.849, Omega_m = 0.3111 (Planck LCDM)
- S8(CFM) = 0.849 * sqrt(0.3111/0.3) = 0.849 * 1.0184 = 0.864
- Tension = |S8(CFM) - S8(survey)| / sigma(survey)

---

## 2. Key Findings from the Literature

### 2.1 The 2026 Status of S8 Tension (arXiv: 2602.12238)

The most recent comprehensive review establishes a new "Combined CMB" baseline
using Planck + ACT DR6 + SPT-3G:
- **Combined CMB: S8 = 0.836 (+0.012 / -0.013)**

The review reveals a **striking bifurcation** among weak lensing surveys:
- **DES Y6**: 2.4-2.7 sigma tension with CMB (S8 = 0.789)
- **KiDS-Legacy**: consistent at < 1 sigma (S8 = 0.815)
- The cause of this dichotomy is under investigation (photo-z calibration,
  intrinsic alignment modeling, shear pipeline differences)

### 2.2 KiDS-Legacy (March 2025)

The final KiDS release (1347 deg^2, 9-band imaging) finds:
- **S8 = 0.815 (+0.016 / -0.021)**
- Agreement with Planck at the **0.73 sigma** level
- This essentially dissolves the S8 tension from the KiDS side

### 2.3 DES Year 6 (January 2026)

The final DES release (4422 deg^2, 151.9 million galaxies) finds:
- **S8 = 0.789 +/- 0.012**
- Maintains tension at **2.4-2.7 sigma** with the combined CMB baseline
- This is the most precise single-survey weak lensing constraint to date

### 2.4 HSC Year 3 + DESI (November 2025)

Reanalysis of HSC Y3 shear with DESI spectroscopic redshift calibration:
- **S8 = 0.805 +/- 0.018**
- Shifted upward from the original HSC Y3 value (0.776) toward Planck
- 1.8x reduction in error bars from improved photo-z calibration
- Suggests photo-z systematics were driving some of the earlier tension

### 2.5 Euclid

- Q1 data release (March 2025): 2000 deg^2 observed, no S8 constraints yet
- First cosmologically relevant cosmic shear results expected in 2026
- Will cover 15,000 deg^2 with space-based image quality

---

## 3. Modified Gravity: f(R) and the S8 Tension

### 3.1 Key result: f(R) gravity WORSENS the S8 tension

f(R) gravity models (Hu-Sawicki type) predict **enhanced structure growth**
at cluster mass scales compared to GR/LCDM. This means:

- f(R) gravity predicts **higher** sigma8 (more structure) than LCDM
- Since weak lensing already measures S8 LOWER than CMB predictions,
  f(R) gravity moves the theoretical prediction in the WRONG direction
- The S8 tension is **exacerbated** in Hu-Sawicki f(R) gravity

### 3.2 Current observational constraints on f(R)

- Weak lensing peaks: log10(|fR0|) < -4.82
- Combined with CMB + clusters: log10(|fR0|) < -5.32
- DESI 2024 full-shape: Sigma_0 = 0.006 +/- 0.043 (consistent with GR)
- At these constrained levels, f(R) contributes no significant phenomenology
  beyond an effective cosmological constant

---

## 4. Implications for the CFM Model

### 4.1 CFM prediction

- sigma8 = 0.849 (at best-fit cM = 0.0005)
- Omega_m = 0.3111 (Planck LCDM value)
- **S8(CFM) = 0.864**

### 4.2 Comparison with CMB

- CFM S8 = 0.864 vs. Planck S8 = 0.832 +/- 0.013
- Deviation: 0.032, corresponding to **~2.5 sigma**
- vs. Combined CMB (0.836): deviation 0.028, corresponding to **~2.2 sigma**
- CFM predicts MORE structure than Planck LCDM, not less

### 4.3 Comparison with weak lensing surveys

CFM's S8 = 0.864 is **higher** than ALL weak lensing measurements:

| vs. Survey          | Delta S8 | Significance |
|---------------------|----------|-------------|
| vs. KiDS-Legacy     | +0.049   | ~2.5 sigma  |
| vs. DES Y6          | +0.075   | ~6.3 sigma  |
| vs. HSC Y3+DESI     | +0.059   | ~3.3 sigma  |

### 4.4 Assessment

**CFM worsens the S8 tension with weak lensing surveys, not improves it.**

The "S8 problem" in standard cosmology is that weak lensing surveys measure
LESS structure (lower S8) than the CMB predicts. CFM predicts EVEN MORE
structure (higher S8 = 0.864) than the CMB (S8 = 0.832). This moves in
exactly the wrong direction relative to weak lensing data.

This is the same problem faced by f(R) modified gravity: any model that
enhances structure growth will worsen the discrepancy with weak lensing.

### 4.5 Caveats and possible mitigations

1. **KiDS-Legacy partially closes the gap**: With S8 = 0.815 (+0.016/-0.021),
   the KiDS-Legacy result is much closer to Planck than earlier KiDS results.
   The tension between CFM and KiDS-Legacy (~2.5 sigma) is uncomfortable but
   not yet a decisive exclusion.

2. **The bifurcation problem**: The disagreement between DES Y6 (0.789) and
   KiDS-Legacy (0.815) suggests unresolved systematics. If DES Y6 has
   a systematic bias pushing S8 low, the true value might be closer to
   ~0.81-0.82, reducing the tension with CFM to ~2-3 sigma.

3. **Omega_m degeneracy**: If CFM allows Omega_m < 0.3111, the S8 prediction
   would decrease. For S8 = 0.832 (Planck), one would need
   Omega_m = 0.3111 * (0.832/0.849)^2 = 0.299, which is within observational
   uncertainty.

4. **Scale-dependent effects**: S8 integrates structure over all scales
   (sigma8 is at 8 Mpc/h). If CFM enhances structure only at specific scales,
   the effective sigma8 measured by weak lensing could differ from the
   linear-theory prediction.

5. **Nonlinear corrections**: The relationship between the linear power
   spectrum sigma8 and weak lensing observables involves nonlinear corrections.
   CFM's modified growth could alter these corrections in ways not captured
   by the simple S8 comparison.

---

## 5. Summary

The current observational landscape (as of February 2026) shows:

- **Weak lensing surveys consistently measure S8 ~ 0.77-0.82**, lower than CMB
- **The CMB predicts S8 ~ 0.832-0.836**
- **CFM predicts S8 ~ 0.864**, which is higher than both CMB and all WL surveys
- **f(R) modified gravity** similarly worsens the tension by enhancing growth
- **The S8 tension itself is in flux**: KiDS-Legacy shows no tension with CMB,
  while DES Y6 maintains 2.4-2.7 sigma tension
- **Euclid** will be the decisive arbiter, with first results expected in 2026

For the CFM model to remain viable, one would need either:
(a) A mechanism to suppress the effective sigma8 measured by weak lensing
    while keeping the CMB-inferred value high, or
(b) An Omega_m value closer to ~0.30, which would bring S8(CFM) down to ~0.85,
    still above all WL measurements but less dramatically so, or
(c) Evidence that current WL systematics are biasing S8 low by ~0.04-0.07

---

## Sources

- KiDS-1000: Heymans et al. (2021), arXiv:2007.15632
- KiDS-Legacy: Li et al. (2025), arXiv:2503.19441
- DES Y3: Abbott et al. (2022), arXiv:2105.13549
- DES Y6: Abbott et al. (2026), arXiv:2601.14559
- HSC Y3: Li et al. (2023), arXiv:2304.00701
- HSC Y3 + DESI: Choppin de Janvry et al. (2025), arXiv:2511.18134
- S8 Tension 2026 Review: arXiv:2602.12238
- f(R) gravity constraints with WL peaks: Boiza et al. (2024), arXiv:2406.11958
- f(R) gravity and structure growth: arXiv:2510.19569

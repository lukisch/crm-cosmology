# FINAL REVIEW -- CFM Papers I-IV (English)

**Reviewer:** Claude Opus 4.6 (Final Peer Review Pass)
**Date:** 2026-02-22
**Scope:** Read-only review. No edits applied. All 4 papers read in full.

---

## 1. Gesamtbewertung (Overall Assessment)

The four-paper CFM series presents a remarkably ambitious and internally coherent theoretical program. Paper I establishes the game-theoretic foundation and validates the standard CFM against Pantheon+. Paper II extends the framework to a baryon-only universe, achieving competitive joint SN+CMB+BAO fits. Paper III provides the Lagrangian underpinning (R + gamma R^2 + Poeschl-Teller scalar) with full hi_class CMB validation. Paper IV introduces the vector sector for galactic MOND dynamics and tests against the SPARC database.

**Strengths across all papers:**
- Consistent conceptual architecture from axioms through Lagrangian to predictions
- Honest self-assessment sections (especially Paper II Sec. 4.4 and Paper IV Sec. 9)
- Open-source code and reproducibility emphasis
- Quantitative falsifiable predictions (w(z), gravitational slip, S_8, a_0)

**Systemic weaknesses:**
- Multiple layers of phenomenological parametrizations (running beta, running mu, screening functions) whose mutual consistency is claimed but not rigorously demonstrated
- The diagonal chi^2 approximation for Planck data (Paper III) limits the statistical significance of Delta-chi^2 claims
- The SPARC test uses the McGaugh interpolation, not the CFM-native interpolation -- this is honestly stated but weakens the "zero-parameter test" claim

**Zenodo-readiness:** Papers I-III are at a publishable-preprint standard suitable for Zenodo. Paper IV is more clearly in draft territory (acknowledged by the authors' use of BVP results with empirical rather than native interpolation, and the elevated chi^2/dof values).

---

## 2. Kritische Befunde (Critical Findings)

### C1. Internal z_trans inconsistency in Paper I

Paper I, Table 1 (line 335): a_trans = 0.75, corresponding to z_trans = 0.33.
Paper I, Section 4.6 MCMC (line 379): "derived quantities ... z_trans = 0.35".

These two values (z_trans = 0.33 from the point estimate vs. z_trans = 0.35 from MCMC) are presented without reconciliation. While the difference is within MCMC uncertainties, it is confusing that the text uses both values without noting the distinction. The table should either show the MCMC posterior median or the text should state "z_trans = 0.33 (point estimate) / 0.35 (MCMC posterior median)".

### C2. Power law Delta-chi^2 sign error in Table 3 (Paper I)

Paper I, Table 3 (line 400): The power law entry shows Delta-chi^2 = "8.9" (positive, no minus sign), while the text (line 230) states "all yield Delta-chi^2 approx -9 to -12". This appears to be a typesetting error: the value should read -8.9. As printed, it implies the power law is *worse* than LCDM, contradicting the text.

### C3. Paper III alpha_M0 value inconsistency

Paper III, Section 8 MCMC results (line 591): alpha_M0 = 0.0011 +0.0010/-0.0006.
Paper III, Abstract (line 44): references alpha_M0 from Paper II as 0.0013 +/- 0.0007.
Paper II, Section 2.2 (line 183): cites Paper III for "alpha_M,0 = 0.0013 +/- 0.0007".

The MCMC posterior in Paper III itself gives 0.0011, but Paper II and the Paper III abstract reference 0.0013. This looks like a version mismatch -- Paper II was likely written before the final Paper III MCMC converged. The values are consistent within errors but should be harmonized to avoid confusion about which is the definitive result.

### C4. Omega_m in Paper I tanh table vs. MCMC

Paper I, Table 3 (line 397): tanh form Omega_m = 0.364.
Paper I, Table 1 (line 331): CFM (flat) Omega_m = 0.368 +/- 0.024 (MCMC).
Paper I, Section 4.6 (line 379): Omega_m = 0.368.

The 0.364 in Table 3 vs. 0.368 in Table 1 reflects the difference between point estimate (optimizer) and MCMC posterior. This is not an error per se, but the reader encounters two slightly different Omega_m values for what appears to be the same model without explanation.

---

## 3. Wichtige Befunde (Important Findings)

### W1. Paper III skip_stability_tests_smg disclosure

Paper III (line 538) transparently discloses using `skip_stability_tests_smg = yes` in hi_class, bypassing automated stability checks. The justification (analytical stability proof in Sec. 3.4) is adequate, but a referee will likely flag this. Recommendation: add a sentence noting that the stability tests produce false positives specifically because alpha_M approaches zero at early times faster than the numerical tolerance.

### W2. Diagonal chi^2 limitation (Paper III)

Paper III (line 542) explicitly states that the chi^2 is computed diagonally without the Planck covariance matrix. This is an important caveat that limits the absolute Delta-chi^2 values. The paper honestly flags this, but it should be more prominent -- currently buried in a mid-paragraph note. A referee for PRD/JCAP will require either full-likelihood analysis or a very prominent disclaimer.

### W3. S_8 tension as a potential falsifier

Papers II and III both note the S_8 tension: CFM predicts S_8 = 0.845-0.855, while cosmic shear surveys give S_8 ~ 0.76-0.78. This is honestly presented as a challenge. The argument that eROSITA (S_8 = 0.86) supports the CFM and that Euclid will be decisive is reasonable. However, the reader should be made aware that this is the single most problematic observational constraint for the model.

### W4. Paper IV SPARC chi^2/dof values

Paper IV, Section 8.6 (line 916-920):
- Run A (free a_0): chi^2/dof = 4.88
- Run B (CFM fixed): chi^2/dof = 9.67
- Run C (best global): chi^2/dof = 9.30

All chi^2/dof values are significantly above 1, indicating the McGaugh interpolation is a poor description of the SPARC data at this level. The paper honestly attributes this to using the empirical rather than CFM-native interpolation. However, chi^2/dof ~ 5-10 would normally indicate a rejected model. The relative comparison between runs is informative, but the absolute values weaken the "zero-parameter test" narrative. A referee will note that Run B (CFM fixed) is nearly 2x worse than Run A (free a_0), which is a significant penalty.

### W5. Mu notation overloaded

Three distinct quantities share the symbol mu across papers:
- mu_eff = sqrt(pi) ~ 1.77: background-level MOND enhancement (Paper II)
- mu(k,a) -> 4/3: perturbation-level modified Poisson equation (Paper III)
- mu(x) = MOND interpolation function (Paper IV, standard MOND)

While footnotes distinguish these (Paper II line 126, Paper III line 403, Paper III line 425), a reader studying all four papers encounters significant notational confusion. A uniform notation table in each paper's conventions section would help.

### W6. Missing comparison with w0-wa CDM

Paper I (line 634) notes that "a fairer benchmark for the phantom crossing behavior would include the w0waCDM parametrization" and defers to Paper II. Paper II does not explicitly perform this comparison either -- it compares against LCDM but not w0waCDM. Given that DESI DR2 results (cited in Papers II-III) favor w0waCDM, this comparison is overdue.

### W7. Geiger2026c referenced before published

Paper I (line 476, 663) references Geiger2026c (Paper III) in the deductive structure section. Since Paper I is supposed to be the first published, forward-referencing unpublished companion papers is acceptable for a series but weakens Paper I as a standalone work. A referee may ask for Paper I to be self-contained.

---

## 4. Kleinere Befunde (Minor Findings)

### M1. Paper I abstract is extremely long

The abstract (lines 43-45) is approximately 450 words, far exceeding the typical PRD limit of ~250 words. It contains multiple sentences that belong in the introduction or conclusions. Recommendation: trim to ~250 words focusing on core results.

### M2. Inconsistent DESI citation years

Paper I (line 70): "DESI2024" -- DESI 2024 VI (arXiv:2404.03002).
Papers II, III, IV: "DESI2025" -- DESI DR2 (arXiv:2503.14738).
Both references are correct (different data releases), but the progression is not explained. Paper I should note that DESI Year 1 (2024) was the available data at time of writing, while later papers incorporate DR2 (2025).

### M3. Euclid citation inconsistency

Paper I (line 793-796): Euclid Collaboration (2025), "Euclid Quick Data Release 1" -- no DOI.
This appears to be a placeholder reference. It should be updated to a specific Euclid publication with DOI if available, or explicitly noted as "in preparation."

### M4. BekensteinMilgrom1984 cited but not in bibliography (Paper III)

Paper III, line 429: references "AQUAL [BekensteinMilgrom1984]" but this citation key does not appear in Paper III's bibliography. The Bekenstein2004 reference is present, but the 1984 AQUAL paper is missing.

### M5. Minor LaTeX: \Geissbuhlweg in email

All papers use `Gei\ss{}b\"uhlweg` in the email field. This is correctly typeset for LaTeX but may not render in all REVTeX compilation environments. Consider testing with different LaTeX distributions.

### M6. "Granddaughter" terminology (Paper II)

Paper II, Section 4.5 (line 798): The ontological hierarchy uses "Mother / Daughter / Granddaughter" terminology. While creative, this anthropomorphic language may not be received well by all referees. The "Null Space / Geometry / Matter" labeling (same section) is more appropriate for a physics journal.

### M7. Paper IV Figure references but no figures directory confirmed

Paper III references figures (lines 571, 577, 601): cfm_cl_comparison.png, cfm_cl_peaks.png, cfm_contour.png. Paper IV does not reference figures. The graphicspath is set to ../figures/ in all papers. These figures should exist in the figures directory -- cannot verify from the tex source alone.

### M8. P3/P1 ratio: 0.4295 vs. 0.4433

Paper II (line 709): P3/P1 = 0.4295 described as "exact Planck match."
Paper III (line 585): P3/P1 = 0.4433 for all conservative cfm_fR models (described as matching LCDM).
These are different quantities: Paper II uses the "effective CDM mapping" approach, while Paper III uses full hi_class Boltzmann integration. However, both are described as "exact Planck match" even though they give different values. Clarification needed on which P3/P1 value Planck actually measures and which method is more reliable.

### M9. beta_early values vary slightly across papers

Paper II line 527: beta_early = 2.82
Paper II line 707: beta_early = 2.829 (after adjustment)
Paper II line 716: beta_early = 2.834 (combined optimal)
Paper III line 322: beta_early = 2.78

The range 2.78-2.834 is narrow but the exact value used varies. Each instance has its own optimization context, but a reader comparing across papers may be confused about the "canonical" value.

---

## 5. Cross-Paper Zahlenwert-Check (Numerical Consistency Table)

| Quantity | Paper I | Paper II | Paper III | Paper IV | Consistent? |
|----------|---------|----------|-----------|----------|-------------|
| Delta-chi^2 (SN, standard CFM) | -12.2 | -12.2 (ref) | -12.2 (ref) | -- | YES |
| Delta-chi^2 (SN, extended) | -- | -26.3 | -- | -- | N/A |
| Delta-chi^2 (joint, preferred) | -- | -5.5 | -5.5 (ref) | -- | YES |
| Delta-chi^2 (CMB MCMC) | -- | -- | -3.7 | -- | N/A |
| Omega_m (CFM flat, MCMC) | 0.368 +/- 0.024 | 0.368 (ref) | -- | -- | YES |
| Omega_m (CFM flat, optimizer) | 0.364 | -- | -- | -- | Note: differs from MCMC |
| k (transition sharpness) | 1.44 | -- | -- | -- | N/A |
| a_trans (z_trans) | 0.75 (0.33) / 0.35 MCMC | -- | -- | -- | **MINOR ISSUE** (C1) |
| H0 (SH0ES calib.) | 76.1 | -- | -- | -- | N/A |
| H0 (joint fit) | -- | 67.3 | -- | -- | N/A |
| alpha (geometric DM) | -- | 0.68 +0.02/-0.07 | -- | -- | N/A |
| beta (SN-only MCMC) | -- | 2.02 +0.26/-0.14 | -- | -- | N/A |
| beta_early (running) | -- | 2.82 | 2.78 | -- | **MINOR** (M9) |
| a_t (z_t) | -- | 0.098 (9.2) | 0.124 (7.1) | -- | **ISSUE**: 9.2 vs 7.1 |
| alpha_M0 (MCMC) | -- | 0.0013 (ref P3) | 0.0011 +0.0010/-0.0006 | -- | **ISSUE** (C3) |
| n (MCMC) | -- | -- | 0.55 +0.58/-0.29 | -- | N/A |
| mu_eff (late) | -- | sqrt(pi) = 1.7725 | -- | -- | N/A |
| a_0 (CFM prediction) | -- | -- | -- | cH0/(2pi) = 1.042e-10 | N/A |
| a_0 (observed) | -- | -- | -- | 1.2e-10 | -- |
| ell_A | -- | 301.471 | -- | -- | N/A |
| R (shift param) | -- | 1.7502 | -- | -- | N/A |
| r_d (Mpc) | -- | 146.9 | 147.10 | -- | YES (within rounding) |
| sigma_8 (conservative) | -- | -- | 0.826 | -- | N/A |
| S_8 | -- | 0.847 (ref) | 0.847 | -- | YES |
| SPARC chi^2/dof (free a_0) | -- | -- | -- | 4.88 | N/A |
| SPARC chi^2/dof (CFM fixed) | -- | -- | -- | 9.67 | N/A |
| chi^2 full cov. (Paper I) | -11.2 | -- | -- | -- | N/A |

**New critical finding from table:**

### C5. Transition redshift z_t discrepancy between Papers II and III

Paper II (line 527): a_t = 0.098 (z_t = 9.2)
Paper III (line 322): a_t = 0.124 (z_t = 7.1)

These represent the same physical quantity (the running-beta transition scale) but differ by ~30%. Paper III (line 322) attributes its value to "best-fit values" while Paper II attributes its to the "preferred mu(a) variant". If these are from different optimization runs with different fixed parameters, this should be explicitly noted. A 30% discrepancy in a key transition parameter between papers claiming to describe the same model requires explanation.

---

## 6. Paper-spezifische Anmerkungen

### Paper I: Game-Theoretic Cosmology and the CFM

**Quality:** Strong phenomenological paper. The game-theoretic framework is creative and the Pantheon+ validation is methodically sound.

**Specific notes:**
- The deductive structure section (Sec. 8.3) references Paper III for "rigorous derivation" but contains strong claims ("uniquely selects R^2") that the reader cannot verify without Paper III.
- The phantom stability analysis (Sec. 4.8) is well-argued but could cite more recent phantom dark energy literature.
- The thermodynamic equivalence (Sec. 2.5) is the most novel section and would benefit from a more formal mathematical treatment.
- Cross-validation results (Table 2) are convincing for overfitting concerns.

### Paper II: Eliminating the Dark Sector

**Quality:** The most technically ambitious paper. The progressive build-up from constant-beta to running-beta to scale-dependent-mu is well-structured. The self-assessment section is exemplary.

**Specific notes:**
- The sqrt(pi) conjecture (Sec. 2.6) is presented with appropriate epistemic caution ("conjecture requiring independent derivation") but the numerical evidence (1.3% match to Omega_CDM) is striking.
- The "CMB catastrophe" framing (Sec. 3.4.1) effectively communicates the severity of the constant-beta problem.
- The Bullet Cluster argument (Sec. 4.3.2) is qualitatively reasonable but the background-level estimate is acknowledged as insufficient. The Sigma = 1 result is more convincing.
- The honest assessment paragraph (line 831) is unusually thorough for a physics paper and should be preserved.

### Paper III: Microscopic Foundations

**Quality:** The mathematical core of the series. The ghost-freedom proof and chameleon screening analysis are the strongest technical results.

**Specific notes:**
- The UV completion candidates (Sec. 3.2, Appendix A) are appropriately flagged as "motivational, not essential." The experimental scorecard is a creative addition.
- The native cfm_fR model in hi_class is the most important technical contribution -- this enables reproducible testing by the community.
- The MCMC analysis (48 walkers, 5000 steps) is adequate for a first exploration but small by Planck-level standards (typical: 10000+ steps, convergence via Gelman-Rubin R-hat). The paper reports R-hat < 1.02, which is acceptable.
- The cosmic birefringence connection (Sec. 5.1.1) is speculative but provides a specific falsifiable prediction.
- The theta_s resolution (Sec. 7.6) is an important self-correction of Paper II's offset.

### Paper IV: The Galactic-Cosmological Nexus

**Quality:** Clearly the most preliminary of the four papers. The theoretical framework is well-motivated but the numerical validation is incomplete.

**Specific notes:**
- The thermodynamic derivation of Daughter 2 (Sec. 2) is creative but relies heavily on the MEPP principle, which itself is debated in the thermodynamics literature.
- The $2\pi$ factor in a_0 = cH_0/(2pi) is described as arising from a "Fourier relationship" (line 369-379). The argument is heuristic; a rigorous derivation is flagged as outstanding.
- The parasitic screening proof (Sec. 6.2) is elegant: the exponential Chameleon suppression (10^{-3e9}) completely dominates the density enhancement (10^{30}).
- The BVP results (Sec. 8.3) are promising: the MOND attractor at slope 0.500 +/- 0.001 is numerically compelling for the Plummer sphere, but the multi-galaxy scan shows mass-dependent slopes (0.55-0.72) that deviate from the ideal 0.5.
- The SPARC test uses the McGaugh interpolation, not the CFM-native form. This is the single biggest limitation of Paper IV -- honestly acknowledged but leaving the key prediction untested.
- The chi^2/dof = 9.67 for the CFM-fixed a_0 vs. 4.88 for free a_0 represents a factor-of-2 degradation. While the paper attributes this to the McGaugh approximation, a referee will note that this is the definitive test of the theory and the result is ambiguous.
- The "assessment" paragraph (line 929-930) about the 13% a_0 discrepancy being 0.66 sigma is correct given the McGaugh et al. error bars, but the systematic uncertainty in a_0 is dominated by the choice of interpolation function, not the statistical error.
- Open theoretical question list (Sec. 9.2) is comprehensive and honest. The perturbation stability analysis for the vector sector (item 5) is a significant missing piece.

---

## 7. Empfehlung (Recommendation)

### Papers I-III: Zenodo-ready with minor corrections

The following should be addressed before upload:
1. Fix the power-law Delta-chi^2 sign in Paper I Table 3 (C2)
2. Harmonize alpha_M0 values between Papers II and III (C3)
3. Clarify the z_trans = 0.33 vs 0.35 distinction in Paper I (C1)
4. Note the z_t = 9.2 (Paper II) vs z_t = 7.1 (Paper III) discrepancy, or explain the different contexts (C5)
5. Ensure BekensteinMilgrom1984 is in Paper III's bibliography (M4)

### Paper IV: Zenodo-ready as explicit draft/preprint

The paper should clearly indicate its preliminary status regarding:
- The SPARC test uses the McGaugh interpolation as a placeholder
- The chi^2/dof >> 1 reflects this approximation, not the theory's viability
- The emergent mu(x) from the BVP solver is the key missing piece
- Vector sector perturbation stability analysis is outstanding

### For journal submission (PRD/JCAP)

Essential prerequisites beyond Zenodo corrections:
1. Full Planck likelihood analysis (not diagonal chi^2) for Paper III
2. Explicit w0-wa comparison (missing from the series)
3. Native CFM interpolation function for SPARC test (Paper IV)
4. Vector perturbation stability analysis (Paper IV)
5. Consistent parameter table across all four papers

### SPARC-Bewertung (per user request)

**Are the SPARC limitations honestly named?** Yes. Paper IV explicitly states:
- "The elevated chi^2/dof values reflect the fact that we use the empirical McGaugh interpolation rather than the CFM-native interpolation function" (line 922)
- "Clarification: 'Zero-parameter' refers to the absence of free dark matter or MOND parameters. The per-galaxy Upsilon_* values are standard astrophysical nuisance parameters" (line 908)
- The 13% a_0 discrepancy and its possible explanations (B_0, Hubble tension, interpolation form) are discussed (line 929-930)

**Could the SPARC results be stronger?** Yes, if:
- The CFM-native interpolation from the BVP solver were used instead of McGaugh
- The per-galaxy free a_0 (Run A) values were compared to the CFM prediction as a distribution
- A Bayesian evidence comparison (not just chi^2) were performed

---

*End of review. No files were modified.*

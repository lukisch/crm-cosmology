# Strict Review: CFM Cosmology Papers II & III (Gen 2)

**Reviewer:** Claude Opus 4.6 (Simulated PRD/JCAP Referee)
**Papers under review:** CFM Paper III ("Eliminating the Dark Sector: Unifying the Curvature Feedback Model with MOND"), Paper I ("Microscopic Foundations of the Curvature Feedback Model")
**Review type:** Post-Revision Assessment (following Gen 1 Major Revision)
**Date:** 15 February 2026

---

## Overall Verdict: MINOR REVISION

---

## Executive Summary

The revision has addressed the most critical concerns of the Gen 1 review with substantial effort and, in several cases, genuine scientific progress. The full Boltzmann integration using a native `cfm_fR` gravity model in hi_class is a major step forward: the CMB TT+TE+EE analysis against 6,405 Planck data points yields Delta_chi2 = -3.6 (MCMC best-fit), the corner plots are now provided, the trace coupling is rigorously derived from the f(R) field equations rather than postulated, and the theta_s offset is resolved. The S8 tension is honestly acknowledged. However, several issues remain that prevent an unconditional accept: (1) the MCMC chain is marginal (9,600 samples, 24 walkers, only 400 production steps) and convergence diagnostics are absent; (2) the diagonal chi2 (no covariance matrix) inflates the claimed improvement; (3) the running beta and running mu transitions remain phenomenological fitting functions without rigorous Lagrangian derivation despite claims to the contrary; (4) the relationship between the SN-only framework (Paper III) and the Boltzmann framework (Paper I) contains an unresolved conceptual tension regarding what omega_cdm represents; and (5) the sqrt(pi) conjecture remains numerologically motivated rather than derived. Overall, the papers present a coherent and testable modified gravity hypothesis that merits publication after the issues below are addressed.

---

## Assessment of Previous Review Points (Gen 1)

### Point 1: CMB Power Spectrum
**RESOLVED.**

The Gen 1 reviewer demanded a full C_l computation. This has been provided: hi_class v2.9.4 with a native `cfm_fR` gravity model, full Boltzmann integration from a ~ 10^{-14} to a = 1, TT+TE+EE spectra against 6,405 Planck data points, and two publication-quality figures (cfm_cl_comparison.png, cfm_cl_peaks.png). The grid scan yields Delta_chi2 = -2.7, the MCMC yields Delta_chi2 = -3.6. The polarization spectra are essentially unchanged (Delta_chi2 < 0.1), which is physically consistent with the perturbative nature of the f(R) modification. This is a substantial and credible piece of work.

**Remaining concern:** The chi2 is computed as a *diagonal* chi2 (no Planck likelihood covariance matrix). The authors state this is "conservative," but this is incorrect. Without the covariance matrix, the chi2 values are not directly comparable to results from full Planck likelihood analyses (e.g., Planck TTTEEE+lowl+lowE). The claimed Delta_chi2 = -3.6 is therefore an *approximation* whose reliability is unknown. The authors should either use the Planck likelihood (via MontePython or CosmoMC) or clearly state the limitation of the diagonal approximation and provide an estimate of the systematic bias.

### Point 2: Sound Speed & Anisotropic Stress
**RESOLVED.**

Paper I Section 6.1 now provides a complete perturbation analysis: c_s^2 = 1 (exact for f(R)), the modified Poisson equation mu(k,a) = 1 + (1/3) k^2/(k^2 + a^2 m_eff^2), the lensing parameter Sigma = 1 (since alpha_T = 0), and the gravitational slip eta(k,a). The clarification that the scalaron does NOT cluster like a fluid but modifies the Poisson equation is correct and important. The structure formation mechanism through enhanced gravitational coupling (mu = 4/3 at sub-Compton scales) is well established in the f(R) literature and is correctly applied here.

### Point 3: Bullet Cluster Lensing
**RESOLVED.**

The derivation Sigma = 1 + alpha_T/2 = 1 (since alpha_T = 0 identically) is rigorous and correct. This is a standard result for f(R) theories in the Horndeski framework. The consequence -- lensing identical to GR at all scales -- is properly presented. The background mass ratio M_lens/M_baryon ~ 10.6 at z = 0.296 is a useful order-of-magnitude check. The prediction that Euclid/Rubin can test Sigma to percent level is well-motivated.

**One caveat:** The Sigma = 1 result ensures that the *total* lensing potential is GR-like given the total matter content. However, in the CFM framework, the "total matter" at the Bullet Cluster is baryons only (Omega_m = Omega_b = 0.05), and the extra lensing comes from the geometric scalaron field. The authors need to demonstrate that the scalaron field perturbation delta_chi at the cluster scale produces a convergence map that is spatially offset from the gas, tracking the galaxies rather than the X-ray gas. The current argument ("geometry is automatically collisionless") is physically reasonable but qualitative. A simulation or at least a semi-analytic calculation of the scalaron field around a cluster merger would strengthen this considerably.

### Point 4: Corner Plots
**RESOLVED.**

The corner plot (cfm_contour.png) is now provided with filled 68%/95% credible regions and KDE-smoothed 1D marginals. The anti-correlation alpha_M_0 vs n (rho = -0.633) is discussed and physically motivated. All cross-correlations between modified gravity and standard parameters are < 0.11, confirming the perturbative nature of the extension.

**Remaining concern:** See Section "New Issues" regarding MCMC chain quality.

### Point 5: Kitchen Sink Problem
**PARTIALLY RESOLVED.**

The QG candidates are now compressed to Section 3.2 (~5 lines each) with the explicit caveat that they are "UV-completion candidates, not derivations" and that "all testable predictions derive exclusively from the effective action." This is a significant improvement. The detailed material is moved to Appendix A. However, Paper I still attempts to cover too much ground: Lagrangian derivation, ghost analysis, perturbation equations, full Boltzmann numerics, MCMC, S8 analysis, DESI comparison, QG candidates, fractal game theory, technological horizons, and metric engineering. For a journal submission, I would strongly recommend splitting: the hard physics (Lagrangian, perturbations, numerics) should be one paper, and the speculative material (fractal game theory, QG connections, technological horizons) should be a separate paper or review article.

The "Fractal Game Theory" section (Section 5 of Paper I) is particularly problematic for a PRD submission. The conjectures on quantum-game duality, the Standard Model as Nash-optimal toolkit, and game-theoretic fine-tuning are intellectually stimulating but entirely speculative -- there is no calculation, no derivation, and no testable prediction. A PRD referee would almost certainly request removal of this section. The "Technological Horizons" section (metric engineering, vacuum energy access) is completely inappropriate for a physics journal and should be removed.

### Point 6: Trace Coupling
**RESOLVED.**

The derivation from the trace of the f(R) = R + 2*gamma*R^2 field equations is now presented rigorously in both Papers II and III:

R + 12*gamma*Box(R) = -8*pi*G*T

During the radiation era, T_rad = 0 (conformal symmetry), implying R + 12*gamma*Box(R) = 0, whose decaying FLRW solution gives R -> 0. Since the scalaron is sourced by R^2, it vanishes when R = 0, automatically suppressing the geometric DM during the radiation era. This is correct and well-known in the f(R) literature (Starobinsky 1980, Hu & Sawicki 2007). The authors correctly identify that the phenomenological suppression factor S(a) = 1/(1 + a_eq/a) parametrizes this rigorous result. The only genuine postulate is f(R) = R + 2*gamma*R^2. This is a satisfactory resolution.

---

## New Issues Identified

### Issue N1: MCMC Chain Quality and Convergence (CRITICAL)

The MCMC analysis uses emcee with 24 walkers, 80 burn-in steps, and 400 production steps, yielding 9,600 samples total. This is *far below* the standard for a publication-quality MCMC in cosmology. Typical analyses use:
- 48-128 walkers
- 1,000-5,000 burn-in steps
- 5,000-50,000 production steps
- Gelman-Rubin R-hat convergence diagnostic (R-hat < 1.01)
- Effective sample size (ESS) reporting
- Autocorrelation length estimation

None of these convergence diagnostics are reported. The acceptance fraction of 0.478 is healthy, but acceptance fraction alone does not guarantee convergence. With only 400 production steps and 24 walkers, the chain may not have explored the full posterior, particularly in the tails. The strong anti-correlation between alpha_M_0 and n (rho = -0.633) suggests a banana-shaped degeneracy that requires many more samples to characterize reliably.

**Requirement:** Either (a) run a longer chain (minimum 5,000 production steps, 48+ walkers) with Gelman-Rubin convergence diagnostic, or (b) explicitly state the limitations and present the current results as "preliminary MCMC constraints."

### Issue N2: Diagonal Chi-Squared vs. Full Planck Likelihood (IMPORTANT)

The chi2 comparison against 6,405 Planck data points uses a *diagonal* chi2 (sum of (data - theory)^2 / sigma^2) without the Planck covariance matrix. This is stated in the text ("The diagonal chi2 (without covariance matrix) provides a conservative estimate"), but the claim that it is "conservative" is not demonstrated and is likely incorrect. The Planck covariance matrix encodes correlations between multipoles that can either increase or decrease the effective chi2. Without it:
1. The absolute chi2 values are unreliable.
2. The Delta_chi2 between models may be biased (since correlations can affect the two models differently).
3. The results cannot be directly compared to any published Planck analysis.

For a PRD/JCAP submission, a proper likelihood analysis (using MontePython or CosmoMC with the Planck 2018 likelihood) is expected. Alternatively, at minimum the authors should use the Planck compressed likelihood (CMB-only: l_A, R, z_*, omega_b, n_s).

### Issue N3: Conceptual Tension Between Paper III and Paper I Frameworks (IMPORTANT)

Paper III presents a *baryon-only* universe (Omega_m = Omega_b = 0.05) where the geometric term alpha*a^{-beta} replaces dark matter cosmologically. Paper I implements the model in hi_class using the *standard* omega_cdm parameter (best-fit omega_cdm = 0.11994), interpreting it as the scalaron background energy density. These two pictures are presented as consistent, but there is a fundamental tension:

- **Paper III claim:** "The dark sector is eliminated. Omega_m = Omega_b = 0.05."
- **Paper I practice:** omega_cdm = 0.120 is used as a fitting parameter in the MCMC. The scalaron's background energy density is identified with omega_cdm.

If omega_cdm = 0.120 represents the scalaron field, then the universe is *not* baryon-only in any meaningful sense -- it contains baryons plus a massive scalar field whose energy density behaves like CDM. The ontological claim of Paper III ("eliminating the dark sector") is undermined if the dark matter is simply relabeled from "CDM particle" to "scalaron field." This is not a mere semantic issue: the scalaron has a definite mass, definite perturbation dynamics, and contributes to the energy budget exactly like CDM at the background level. The physical distinction is in the perturbation sector (mu = 4/3 at sub-Compton scales, gravitational slip), not in the background.

**Requirement:** The authors must clearly state that the CFM replaces CDM *particles* with a *scalaron field* whose background energy density is CDM-like. The claim of a "baryon-only universe" should be qualified: it is baryon-only in the sense that no new *particle species* is introduced, but the total energy budget still includes a dark component (the scalaron), which is geometric in origin. The more accurate description is "a universe without dark matter particles," not "a universe without a dark sector."

### Issue N4: The Running mu(a) -- Empirical Fitting Without Lagrangian Derivation (IMPORTANT)

The scale-dependent MOND background coupling mu(a) is given by:

mu(a) = mu_late + (mu_early - mu_late) / (1 + (a/a_mu)^4)

with mu_late = sqrt(pi), mu_early = 1.00, a_mu = 2.55 x 10^{-4}. This is a phenomenological transition function with three parameters. Despite repeated claims that Paper I "provides the Lagrangian derivation," no such derivation of mu(a) is presented. Paper I derives beta_eff(a) from the scalaron equation of motion (Eq. 36), which is a genuine result, but the running mu(a) has no Lagrangian origin whatsoever. It is an ad-hoc interpolation function introduced to fix the sound horizon problem.

The physical motivation ("at z > 4000, the cosmological acceleration exceeds a_0, so standard gravity applies") is qualitatively reasonable but does not constitute a derivation. In particular:
1. The transition scale a_mu = 2.55 x 10^{-4} is fitted, not derived.
2. The transition sharpness n = 4 is assumed, not derived.
3. The late-time value mu = sqrt(pi) is conjectured, not derived.

**Requirement:** The authors should honestly label mu(a) as a phenomenological parametrization throughout both papers. Claims of "Lagrangian derivation" should be restricted to the running beta, which is genuinely derived.

### Issue N5: The sqrt(pi) Conjecture -- Numerology vs. Physics (MINOR)

The "sqrt(pi) Conjecture" (Section 2.7 of Paper III) proposes that the MOND enhancement factor at cosmological scales is mu_eff = sqrt(pi) based on:
1. The "projection amplitude" of the 2-sphere onto observational space.
2. The Gaussian integral Gamma(1/2) = sqrt(pi).
3. "Thermodynamic normalization of gravitational modes on the cosmological 2-sphere."

None of these arguments is a derivation. The first is a vague geometric analogy. The second is a mathematical identity with no demonstrated physical relevance. The third is a hand-waving statement. The fact that the fitted value 1.77 is numerically close to sqrt(pi) = 1.7725 (deviation 0.2%) is interesting but could be coincidental. Many irrational numbers are close to sqrt(pi). The claim "3*sqrt(pi)*Omega_b = 0.2606 matches Omega_CDM = 0.2660 to 1.3%" is also suggestive but not a prediction -- it is a post-hoc observation.

I note that the factor 3 in "3*sqrt(pi)" appears without derivation. Why 3? This needs justification.

The conjecture is not wrong to include, but it should be clearly labeled as speculative numerology, not as a "remarkable quantitative prediction."

### Issue N6: S8 Tension -- Potentially Fatal

The authors honestly acknowledge that the CFM predicts S8 = 0.845-0.920, in >= 3 sigma tension with DES Y3 (S8 = 0.776 +/- 0.017). This is presented as a "falsifiable prediction" contingent on Euclid results. However, the tension is more serious than presented:

1. DES Y6 (mentioned as S8 = 0.789 +/- 0.012) shows 4.8 sigma tension with the CFM prediction. This is not a marginal discrepancy.
2. The authors cite eROSITA (S8 = 0.86 +/- 0.01) as supporting evidence, but the eROSITA value is in tension with all weak lensing surveys. The authors should not cherry-pick the most favorable measurement.
3. The generic prediction of f(R) gravity is *enhanced* structure growth (mu > 1), which deepens the S8 tension. This is a structural problem, not a tunable parameter.

If Euclid confirms S8 < 0.80, the cfm_fR model would need qualitative modification (e.g., alpha_K != 0, scale-dependent screening at k ~ 0.1-1 h/Mpc). The authors should explicitly discuss what modifications would be needed and whether they preserve the model's other successes.

### Issue N7: skip_stability_tests_smg = yes (MINOR)

The justification for bypassing hi_class stability tests is reasonable (analytic proof of stability, false positives at early times when alpha_M -> 0). However, this is a red flag for any referee. The authors should:
1. Report what specifically the automated tests flag (which stability condition, at which epoch).
2. Demonstrate explicitly that the flagged instability is numerical (not physical) by showing that the perturbation solutions remain bounded.
3. Ideally, fix the numerical issue rather than bypassing the test.

### Issue N8: DESI DR2 Citation and Interpretation (MINOR)

The DESI DR2 results are cited as "arXiv:2503.14738" with w_0 = -0.42 +/- 0.21 and w_a = -1.75 +/- 0.58. The authors claim "The CFM effective equation of state w_eff(z=0) ~ -0.33 lies within 0.4 sigma of the DESI w_0 value." This comparison is misleading:
1. w_0 in the CPL parametrization is the equation of state at z = 0 of a *dark energy fluid*. The CFM has no dark energy fluid; the geometric term has w_eff = beta/3 - 1. These are different physical quantities.
2. The DESI result disfavors Lambda (w = -1) but does not specifically favor w = -1/3. Many dynamical dark energy models are consistent with DESI.
3. The correct comparison would be to compute the CFM prediction in the w_0-w_a space and show a contour overlap.

---

## Critical Questions ("Killer Questions")

These are the questions a hostile but competent PRD referee would ask. Failure to answer any one of them convincingly could result in rejection.

### Q1: If omega_cdm = 0.120 in your hi_class fits, how is this a "baryon-only universe"?

The scalaron field has energy density rho_scalaron ~ omega_cdm * rho_crit, with CDM-like background evolution (w ~ 0). It clusters differently from CDM (mu = 4/3 at sub-Compton scales), but its gravitational effect on the expansion history is identical. In what precise, quantifiable sense does this differ from CDM with modified clustering properties? Are you not simply replacing one dark component (particles) with another (a scalar field)? Please provide a sharp, testable distinction between "scalaron dark matter" and "CDM with modified perturbations."

### Q2: What is the Compton wavelength of the scalaron, and does it conflict with local gravity tests?

From the MCMC best-fit alpha_M_0 = 0.0013, what is the scalaron mass m_s today? What is the corresponding Compton wavelength lambda_C = 2*pi/m_s? Does this conflict with Lunar Laser Ranging, Cassini, or fifth-force experiments? The authors claim chameleon screening with lambda_C^solar ~ 20 m, but this depends on gamma, which is never numerically specified. Please provide a concrete value of gamma (or equivalently m_s) consistent with the MCMC and demonstrate that local tests are satisfied.

### Q3: The SN-only fit (Paper III, Delta_chi2 = -26.3) uses a completely different framework from the Boltzmann fit (Paper I, Delta_chi2 = -3.6). Which is the CFM?

Paper III fits Pantheon+ with a phenomenological extended Friedmann equation using 5 parameters (Phi_0, k, a_trans, alpha, beta). Paper I fits Planck CMB with hi_class using standard Lambda_CDM parameters plus alpha_M_0 and n. These are two different models with different parameter spaces, different physics, and different datasets. The authors present both as "the CFM," but they are not the same model. Can the authors demonstrate, quantitatively, that the Paper III SN fit and the Paper I CMB fit are simultaneously satisfied by a *single* set of Lagrangian parameters (gamma, V_0, phi_0)?

### Q4: The running beta transition at z_t ~ 7-10 is coincident with first galaxy formation. Is this a prediction or a retrodiction?

The transition redshift z_t is a *fitted* parameter (best-fit z_t = 9.2 from the joint SN+CMB+BAO fit). The authors interpret this as physically meaningful ("the geometric transition triggers MOND on galactic scales"). But z_t was optimized against the data, not predicted from first principles. If z_t had come out at z = 3 or z = 50, the authors would presumably have found a different physical interpretation. This is a retrodiction, not a prediction. Can the authors derive z_t from the Lagrangian parameters?

### Q5: How does the model handle the Lyman-alpha forest power spectrum?

The Lyman-alpha forest at z ~ 2-4 constrains the matter power spectrum on small scales (k ~ 0.1-10 h/Mpc). With mu = 4/3 at sub-Compton scales, the CFM predicts enhanced small-scale power. The Lyman-alpha constraints are among the most stringent tests of modified gravity at these redshifts. Have the authors checked consistency with BOSS/eBOSS Lyman-alpha data?

### Q6: What is the scalaron's contribution to N_eff during BBN?

The trace coupling suppresses the scalaron during the radiation era, but "suppression" is not the same as "absence." At T ~ 1 MeV (BBN), is the scalaron energy density exactly zero, or is there a residual contribution? If the latter, what is the effective Delta_N_eff? The authors claim Delta_N_eff ~ 0.000 but do not show the calculation. A PRD referee would require this.

### Q7: The claimed "exact Planck match" (l_1 = 220, P_3/P_1 = 0.4295) uses the effective CDM mapping, not the native cfm_fR model. Can you reproduce these numbers with the native model?

Paper III claims l_1 = 220 and P_3/P_1 = 0.4295 using the "effective CDM" mapping (replacing the geometric term with an effective omega_cdm in CAMB/hi_class). Paper I uses the native cfm_fR model but reports only total chi2 values, not individual peak positions and ratios. Do the native cfm_fR results reproduce l_1 = 220 and P_3/P_1 = 0.4295? If so, show the numbers. If not, which result should the reader trust?

---

## Minor Comments

1. **Notation inconsistency:** Paper III uses mu_eff for the MOND background coupling, while the perturbation analysis in Paper I uses mu(k,a) for the effective gravitational coupling. These are different quantities sharing the same symbol. This will confuse readers.

2. **Equation numbering:** The "boxed" equations are distracting and non-standard for PRD/JCAP. Remove the boxes.

3. **AI Disclosure section:** While transparency is commendable, the current AI disclosure is excessively detailed for a physics journal. "Claude Opus 4.6" and "Gemini" are listed as co-writers and reviewers. For a journal submission, a brief footnote stating "AI tools were used for mathematical formalization and text generation" would suffice. The current format may prejudice some referees.

4. **"Working Paper" label:** If submitting to PRD or JCAP, remove all "Working Paper" designations. These signal that the authors do not consider the work complete, which is incompatible with journal submission.

5. **Game-theoretic language:** Terms like "Nash equilibrium," "gradient reduction game," "null space player," and "Mother-Daughter-Granddaughter ontology" are metaphorical. While the game-theoretic foundation is the subject of Paper II, in Papers II and III these terms should be used sparingly or replaced with standard physics terminology. A PRD referee unfamiliar with Paper II will find this language obscure.

6. **The Efficiency Hypothesis (Paper III, Section 2.5):** The argument that a Nash-optimal universe would not "waste" energy on dark matter because baryons are more "efficient" entropy producers is philosophical, not physical. It assumes an objective function (entropy maximization) that is not derived from the action. This section can be shortened to a paragraph.

7. **"Decaying Dark Geometry" as a brand name:** Introducing named hypotheses ("Decaying Dark Geometry hypothesis," "Geometric Crystallization," "sqrt(pi) Conjecture") is unusual in physics papers and gives the impression of marketing rather than science. Let the physics speak for itself.

8. **Reference [Geiger2026c] is self-referential.** Paper I cites itself as "in preparation." If submitting simultaneously, use "companion paper" with a shared submission note.

9. **Table formatting:** Tables in Paper III (e.g., Table 5 with 5 model variants) are difficult to parse. Consolidate to the most relevant comparison (preferred CFM vs. Lambda_CDM).

10. **The 1,590 SN count:** The Pantheon+ catalog contains 1,701 SN (1,550 unique after cuts). The number 1,590 should be clarified -- is this after a specific redshift cut (z > 0.01)? State the cut explicitly.

---

## Strengths

1. **The beta ~ 2.0 result (Paper III)** remains the strongest single finding of the entire program. That a free MCMC parameter independently recovers the curvature scaling exponent is nontrivial and demands explanation, regardless of whether the full CFM framework is correct. This result alone merits publication.

2. **The native hi_class implementation (Paper I)** is a substantial technical achievement. Patching the C source code of hi_class with a custom gravity model and running full Boltzmann integration is the gold standard for modified gravity cosmology. The Delta_chi2 = -3.6 (MCMC) against Planck is a legitimate, if marginal, result.

3. **Rigorous trace coupling derivation.** The demonstration that the trace coupling follows from the f(R) field equations rather than being ad-hoc is the single most important theoretical improvement in this revision. It elevates the model from phenomenology to a well-defined f(R) theory.

4. **Honest self-assessment.** The authors' treatment of the S8 tension, the theta_s offset (now resolved), the skip_stability_tests issue, and the phenomenological nature of several fitting functions is commendably transparent. This is rare in working papers.

5. **Testable predictions.** The model makes clear, falsifiable predictions: Sigma = 1 (testable by Euclid), S8 = 0.845-0.920 (testable by Euclid), alpha_M_0 = 0.0013 +/- 0.0007 (testable by next-generation CMB), and c_GW = c (already consistent with GW170817). This is a strength relative to many modified gravity proposals that avoid making sharp predictions.

6. **The theta_s resolution (Paper I, Section 7.4):** The demonstration that the scalaron's background energy density is CDM-like (w ~ 0), yielding 100*theta_s = 1.04173 for all alpha_M values, is a clean and convincing resolution of what was a serious problem in Paper III.

7. **Cross-validation (Paper III, Section 3.3):** The 5-fold cross-validation on the SN-only fit is a rigorous and underused technique in cosmology. The result that the extended CFM generalizes better than Lambda_CDM (lower mean predictive chi2/n on held-out data) is compelling evidence against overfitting.

---

## Final Recommendation

**Verdict: MINOR REVISION**

The revision has resolved the three most critical concerns of Gen 1 (CMB C_l, trace coupling derivation, corner plots) and has added substantial new content (full Boltzmann numerics, MCMC, ghost analysis, theta_s resolution). The papers now present a well-defined f(R) modified gravity model with concrete numerical predictions tested against Planck data. The beta ~ 2.0 result remains intriguing, and the native hi_class implementation is technically solid.

However, the following must be addressed before acceptance:

**Required changes (blocking):**

1. **MCMC convergence:** Either run a longer chain with convergence diagnostics (Gelman-Rubin, ESS, autocorrelation length) or explicitly label the current results as preliminary. (Issue N1)

2. **Diagonal chi2 limitation:** Add a clear statement that the chi2 comparison uses a diagonal approximation without the Planck covariance matrix, and discuss the potential systematic bias. (Issue N2)

3. **Baryon-only claim:** Revise the language throughout both papers to accurately reflect that the model replaces CDM particles with a scalaron field, not that the dark sector is "eliminated." The scalaron is a dark component -- it is simply geometric rather than particulate. (Issue N3)

4. **Running mu(a) status:** Remove claims that mu(a) is "derived from the Lagrangian." It is a phenomenological fitting function. State this clearly. (Issue N4)

**Recommended changes (non-blocking but strongly advised):**

5. Split Paper I into (a) Lagrangian + numerics and (b) QG connections + speculative material.
6. Remove the "Fractal Game Theory" section and "Technological Horizons" section from a journal submission.
7. Provide a concrete value of gamma (scalaron mass) and demonstrate consistency with local gravity tests.
8. Address the Lyman-alpha forest constraint.
9. Harmonize the SN-only framework (Paper III) and the Boltzmann framework (Paper I) into a single consistent parameter space.

**Assessment of publication venue:**

In its current form, after the required changes, the combined material would be appropriate for Physical Review D (as a regular article, not a letter) or JCAP. The speculative sections (fractal game theory, QG connections) could form a separate submission to Foundations of Physics, Classical and Quantum Gravity, or a review journal. Physical Review Letters would require a much shorter paper focused exclusively on the beta ~ 2.0 result and the Delta_chi2 = -3.6 MCMC improvement, with all derivations in supplemental material.

---

*End of Review. Prepared by Claude Opus 4.6, simulating a strict but fair PRD/JCAP referee. February 15, 2026.*

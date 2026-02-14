#!/usr/bin/env python3
"""
=============================================================================
Ergaenzende Berechnungen fuer Paper II:
"Eliminating the Dark Sector: Unifying the CFM with MOND"

Adressiert 5 Review-Punkte:
  1. 5-Fold-Kreuzvalidierung fuer das 5-Parameter-Modell
  2. Semi-analytische CMB-Peak-Ratio-Abschaetzung
  3. Bullet-Cluster-Lensing: Groessenordnungsabschaetzung
  4. Effizienz-Hypothese: Formalisierung (kein Zirkelschluss)
  5. Zusammenfassung und Empfehlungen

Baut auf bestehendem Code auf: cfm_baryon_only_test.py, cfm_mond_mcmc.py
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "_data", "Pantheon+SH0ES.dat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_MIN = 0.01
N_GRID = 2000
C_LIGHT = 299792.458  # km/s
OMEGA_B = 0.05        # baryonic density (fixed)

# Physical constants for CMB calculation
Omega_gamma = 5.38e-5     # photon density
Omega_nu = 3.65e-5        # neutrino density (3 massless, N_eff=3.046)
Omega_r = Omega_gamma + Omega_nu  # total radiation ~9.03e-5
z_star = 1089.92           # recombination redshift
H0_planck = 67.36          # Planck H0 [km/s/Mpc]

# Best-fit MCMC values from Paper II
MCMC_k = 9.81
MCMC_a_trans = 0.971
MCMC_alpha = 0.68
MCMC_beta = 2.02
MCMC_chi2 = 702.69
LCDM_chi2 = 729.0

# ================================================================
# DATA LOADING
# ================================================================

def load_data():
    df = pd.read_csv(DATA_FILE, sep=r'\s+', comment='#')
    mask = (
        (df['zHD'] > Z_MIN) &
        df['m_b_corr'].notna() &
        df['m_b_corr_err_DIAG'].notna() &
        (df['m_b_corr_err_DIAG'] > 0)
    )
    df = df[mask].copy().sort_values('zHD').reset_index(drop=True)
    z = df['zHD'].values.astype(np.float64)
    m_obs = df['m_b_corr'].values.astype(np.float64)
    m_err = df['m_b_corr_err_DIAG'].values.astype(np.float64)
    return z, m_obs, m_err


# ================================================================
# MODEL FUNCTIONS (from existing code)
# ================================================================

def _z_grid(z_max):
    return np.linspace(0, z_max * 1.05, N_GRID)

def _cumulative_integral(z_grid, E_inverse):
    dz = z_grid[1] - z_grid[0]
    cum = np.cumsum(E_inverse) * dz
    cum[0] = 0.0
    return cum

def distance_modulus_lcdm(z_data, Omega_m):
    zg = _z_grid(z_data.max())
    E = np.sqrt(Omega_m * (1 + zg)**3 + (1.0 - Omega_m))
    cum = _cumulative_integral(zg, 1.0 / E)
    chi_r = np.interp(z_data, zg, cum)
    d_L = np.maximum((1 + z_data) * chi_r, 1e-30)
    return 5.0 * np.log10(d_L)

def omega_phi_extended(a, Phi0, k, a_trans, alpha, beta):
    s = np.tanh(k * a_trans)
    phi_de = Phi0 * (np.tanh(k * (a - a_trans)) + s) / (1.0 + s)
    phi_dm = alpha * a**(-beta)
    return phi_de + phi_dm

def phi0_from_flatness_ext(k, a_trans, alpha):
    s = np.tanh(k * a_trans)
    f_at_1 = (np.tanh(k * (1.0 - a_trans)) + s) / (1.0 + s)
    if abs(f_at_1) < 1e-15:
        return 1e10
    return (1.0 - OMEGA_B - alpha) / f_at_1

def distance_modulus_ext_cfm(z_data, k, a_trans, alpha, beta):
    zg = _z_grid(z_data.max())
    ag = 1.0 / (1.0 + zg)
    Phi0 = phi0_from_flatness_ext(k, a_trans, alpha)
    Omega_Phi = omega_phi_extended(ag, Phi0, k, a_trans, alpha, beta)
    E2 = OMEGA_B * (1 + zg)**3 + Omega_Phi
    E2 = np.maximum(E2, 1e-30)
    E = np.sqrt(E2)
    cum = _cumulative_integral(zg, 1.0 / E)
    chi_r = np.interp(z_data, zg, cum)
    d_L = np.maximum((1 + z_data) * chi_r, 1e-30)
    return 5.0 * np.log10(d_L)

def chi2_marginalized(mu_theory, m_obs, m_err):
    w = 1.0 / m_err**2
    delta = m_obs - mu_theory
    M_best = np.sum(w * delta) / np.sum(w)
    chi2 = np.sum(((delta - M_best) / m_err)**2)
    return chi2, M_best


# ================================================================
# SECTION 1: 5-FOLD CROSS-VALIDATION
# ================================================================

def section_1_crossvalidation():
    print("=" * 74)
    print("  SECTION 1: 5-FOLD-KREUZVALIDIERUNG")
    print("  Extended CFM+MOND (5 Parameter) vs. LCDM (2 Parameter)")
    print("=" * 74)
    print()

    z, m_obs, m_err = load_data()
    n = len(z)
    print(f"  {n} Supernovae geladen")
    print()

    # Create 5 folds (random, reproducible)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, 5)

    results_lcdm = []
    results_cfm = []

    for fold_idx in range(5):
        print(f"  --- Fold {fold_idx+1}/5 ---")

        # Split data
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(5) if j != fold_idx])

        z_train, m_train, e_train = z[train_idx], m_obs[train_idx], m_err[train_idx]
        z_test, m_test, e_test = z[test_idx], m_obs[test_idx], m_err[test_idx]

        # --- LCDM ---
        def obj_lcdm(p):
            mu = distance_modulus_lcdm(z_train, p[0])
            return chi2_marginalized(mu, m_train, e_train)[0]

        res_l = differential_evolution(obj_lcdm, [(0.05, 0.60)], seed=42+fold_idx,
                                       maxiter=100, tol=1e-8, polish=True)
        # Evaluate on test set
        mu_test_l = distance_modulus_lcdm(z_test, res_l.x[0])
        chi2_test_l, _ = chi2_marginalized(mu_test_l, m_test, e_test)
        chi2_per_n_l = chi2_test_l / len(z_test)
        results_lcdm.append(chi2_per_n_l)

        # --- Extended CFM ---
        def obj_cfm(p):
            kk, at, alpha, beta = p
            P0 = phi0_from_flatness_ext(kk, at, alpha)
            if P0 < -5 or P0 > 10:
                return 1e10
            try:
                mu = distance_modulus_ext_cfm(z_train, kk, at, alpha, beta)
                if np.any(np.isnan(mu)):
                    return 1e10
                return chi2_marginalized(mu, m_train, e_train)[0]
            except:
                return 1e10

        bounds_cfm = [(0.5, 50.0), (0.05, 0.99), (0.05, 0.70), (1.0, 3.5)]
        res_c = differential_evolution(obj_cfm, bounds_cfm, seed=42+fold_idx,
                                       maxiter=500, tol=1e-8, popsize=30,
                                       mutation=(0.5, 1.5), recombination=0.9,
                                       polish=True)
        # Evaluate on test set
        mu_test_c = distance_modulus_ext_cfm(z_test, *res_c.x)
        chi2_test_c, _ = chi2_marginalized(mu_test_c, m_test, e_test)
        chi2_per_n_c = chi2_test_c / len(z_test)
        results_cfm.append(chi2_per_n_c)

        print(f"    LCDM:    chi2/n_test = {chi2_per_n_l:.4f}  (Om={res_l.x[0]:.3f})")
        print(f"    Ext.CFM: chi2/n_test = {chi2_per_n_c:.4f}  (k={res_c.x[0]:.1f}, at={res_c.x[1]:.3f}, "
              f"a={res_c.x[2]:.3f}, b={res_c.x[3]:.2f})")
        print()

    # Summary
    mean_lcdm = np.mean(results_lcdm)
    std_lcdm = np.std(results_lcdm)
    mean_cfm = np.mean(results_cfm)
    std_cfm = np.std(results_cfm)

    print("  ERGEBNIS: 5-Fold-Kreuzvalidierung")
    print("  " + "=" * 50)
    print(f"  {'Modell':<20} {'<chi2/n>':<12} {'std':<12} {'Folds'}")
    print("  " + "-" * 50)
    fold_str_l = ", ".join([f"{x:.4f}" for x in results_lcdm])
    fold_str_c = ", ".join([f"{x:.4f}" for x in results_cfm])
    print(f"  {'LCDM (2 Param.)':<20} {mean_lcdm:<12.4f} {std_lcdm:<12.4f} [{fold_str_l}]")
    print(f"  {'Ext.CFM (5 Param.)':<20} {mean_cfm:<12.4f} {std_cfm:<12.4f} [{fold_str_c}]")
    print()

    if mean_cfm < mean_lcdm:
        print(f"  => Extended CFM generalisiert BESSER auf ungesehene Daten!")
        print(f"     Delta <chi2/n> = {mean_cfm - mean_lcdm:+.4f}")
        print(f"     KEIN Overfitting trotz 5 vs. 2 Parameter.")
    else:
        print(f"  => LCDM generalisiert besser auf ungesehene Daten.")
        print(f"     Delta <chi2/n> = {mean_cfm - mean_lcdm:+.4f}")
        print(f"     Moegliches Overfitting des Extended CFM.")
    print()

    return mean_lcdm, mean_cfm, results_lcdm, results_cfm


# ================================================================
# SECTION 2: CMB PEAK RATIO SEMI-ANALYTISCHE ABSCHAETZUNG
# ================================================================

def section_2_cmb_peaks():
    print()
    print("=" * 74)
    print("  SECTION 2: CMB-PEAK-VERHAELTNISSE (SEMI-ANALYTISCH)")
    print("=" * 74)
    print()

    # Key question: What is the effective "DM-like" contribution
    # at recombination in the extended CFM?

    a_rec = 1.0 / (1 + z_star)  # ~9.17e-4

    print(f"  Recombination: z* = {z_star}, a* = {a_rec:.6e}")
    print()

    # --- Trace-coupling suppression factor ---
    # S(a) = 1 / (1 + (a_eq/a))  where a_eq = Omega_r / Omega_b
    a_eq_cfm = Omega_r / OMEGA_B  # baryon-only: ~1.8e-3
    a_eq_lcdm = Omega_r / 0.315   # LCDM: ~2.9e-4

    S_rec = 1.0 / (1.0 + (a_eq_cfm / a_rec))

    print(f"  Matter-Radiation Equality:")
    print(f"    LCDM (Om=0.315):   a_eq = {a_eq_lcdm:.4e}  (z_eq = {1/a_eq_lcdm - 1:.0f})")
    print(f"    CFM  (Ob=0.05):    a_eq = {a_eq_cfm:.4e}  (z_eq = {1/a_eq_cfm - 1:.0f})")
    print()
    print(f"  Trace-coupling suppression at recombination:")
    print(f"    S(a*) = 1/(1 + a_eq/a*) = {S_rec:.4f}")
    print()

    # --- Geometric DM term at recombination ---
    # alpha * a^{-beta} * S(a)
    geom_dm_rec = MCMC_alpha * a_rec**(-MCMC_beta) * S_rec

    # Baryonic matter term at recombination
    baryon_rec = OMEGA_B * a_rec**(-3)

    # In LCDM, CDM term at recombination
    Omega_cdm = 0.315 - 0.0493  # 0.2657
    cdm_rec = Omega_cdm * a_rec**(-3)
    total_matter_lcdm = 0.315 * a_rec**(-3)

    ratio_geom_to_baryon = geom_dm_rec / baryon_rec
    ratio_cdm_to_baryon_lcdm = cdm_rec / (0.0493 * a_rec**(-3))

    print(f"  Contributions to H^2(a*) / H0^2:")
    print(f"    " + "-" * 60)
    print(f"    {'Component':<30} {'LCDM':>14} {'Ext. CFM':>14}")
    print(f"    " + "-" * 60)
    print(f"    {'Baryonic (Ob*a^-3)':<30} {0.0493*a_rec**(-3):>14.2e} {OMEGA_B*a_rec**(-3):>14.2e}")
    print(f"    {'CDM (Ocdm*a^-3)':<30} {cdm_rec:>14.2e} {'---':>14}")
    print(f"    {'Geom. DM (alpha*a^-b*S)':<30} {'---':>14} {geom_dm_rec:>14.2e}")
    print(f"    {'Radiation (Or*a^-4)':<30} {Omega_r*a_rec**(-4):>14.2e} {Omega_r*a_rec**(-4):>14.2e}")
    print(f"    " + "-" * 60)
    print(f"    {'Total \"matter-like\"':<30} {total_matter_lcdm:>14.2e} {OMEGA_B*a_rec**(-3)+geom_dm_rec:>14.2e}")
    print()
    print(f"  Ratio geom.DM / baryonic at z*: {ratio_geom_to_baryon:.4f} ({ratio_geom_to_baryon*100:.2f}%)")
    print(f"  Ratio CDM / baryonic at z* (LCDM): {ratio_cdm_to_baryon_lcdm:.2f} ({ratio_cdm_to_baryon_lcdm*100:.0f}%)")
    print()

    # --- Peak ratio analysis ---
    print("  CMB PEAK-VERHAELTNISSE:")
    print("  " + "-" * 60)
    print()

    # In LCDM, the ratio of the 1st to 3rd peak depends on R = rho_b / rho_m
    # R_LCDM = Omega_b / Omega_m = 0.0493 / 0.315 = 0.1565
    R_lcdm = 0.0493 / 0.315
    # In pure baryon-only (no geom DM): R_baryon = 1.0
    R_baryon_only = 1.0

    # In extended CFM: effective R at recombination
    # R_eff = baryon / (baryon + geom_DM) at z*
    rho_b_star = OMEGA_B * (1 + z_star)**3
    rho_geom_star = MCMC_alpha * (1 + z_star)**MCMC_beta * S_rec
    R_cfm_eff = rho_b_star / (rho_b_star + rho_geom_star)

    print(f"  Baryon fraction R = rho_b / rho_total_matter:")
    print(f"    LCDM:         R = {R_lcdm:.4f}  (baryons = {R_lcdm*100:.1f}% of matter)")
    print(f"    Baryon-only:  R = {R_baryon_only:.4f}  (baryons = 100% of matter)")
    print(f"    Ext. CFM:     R = {R_cfm_eff:.4f}  (baryons = {R_cfm_eff*100:.1f}% of matter)")
    print()

    # Qualitative peak ratio predictions
    # The 3rd peak is enhanced by CDM (which doesn't oscillate)
    # while the 1st peak is enhanced by baryons (which compress)
    # Standard result: peak3/peak1 ~ (Omega_m / Omega_b)^{0.5-1}
    # Simplified scaling: peak3/peak1 ~ (1/R)^{0.7} (approximate)
    peak_ratio_lcdm = (1/R_lcdm)**0.7
    peak_ratio_baryon = (1/R_baryon_only)**0.7
    peak_ratio_cfm = (1/R_cfm_eff)**0.7

    print(f"  Approximate peak3/peak1 scaling ~ (1/R)^0.7:")
    print(f"    LCDM:         {peak_ratio_lcdm:.2f}")
    print(f"    Baryon-only:  {peak_ratio_baryon:.2f}  (catastrophically low 3rd peak!)")
    print(f"    Ext. CFM:     {peak_ratio_cfm:.2f}")
    print()

    # The critical question: the geometric DM term is negligible at z*
    print("  KRITISCHES ERGEBNIS:")
    print("  " + "=" * 60)
    print()

    if ratio_geom_to_baryon < 0.1:
        print(f"  Der geometrische DM-Term traegt nur {ratio_geom_to_baryon*100:.1f}%")
        print(f"  der baryonischen Materiedichte bei Rekombination bei.")
        print(f"  In LCDM betraegt der CDM-Beitrag {ratio_cdm_to_baryon_lcdm*100:.0f}%.")
        print()
        print(f"  PROBLEM: Der alpha*a^{{-beta}}-Term mit beta~2 skaliert")
        print(f"  LANGSAMER als Materie (a^{{-3}}). Bei hohen z dominiert")
        print(f"  die baryonische Materie, und der geometrische DM-Term")
        print(f"  wird vernachlaessigbar.")
        print()
        print(f"  Dies bedeutet: Die CMB-Physik (z~1000) unterscheidet sich")
        print(f"  fundamental von LCDM. Das erweiterte CFM muss entweder:")
        print(f"    (a) einen anderen Mechanismus fuer CMB-Peaks liefern")
        print(f"        (z.B. ueber die Perturbationstheorie des R^2-Terms")
        print(f"         aus Paper III, der anisotropen Stress erzeugt), oder")
        print(f"    (b) akzeptieren, dass die SN-Validierung allein nicht")
        print(f"        ausreicht und die CMB-Kompatibilitaet ungeklaert ist.")
    print()

    # --- What beta would be needed? ---
    print("  GEDANKENEXPERIMENT: Welches beta waere noetig?")
    print("  " + "-" * 50)
    # For the geometric DM to contribute significantly at z*:
    # alpha * a*^(-beta) ~ CDM contribution = 0.265 * a*^(-3)
    # => beta must be close to 3 for significant contribution at z*
    # With alpha=0.68: 0.68 * a*^(-beta) = 0.265 * a*^(-3)
    # => a*^(3-beta) = 0.265/0.68 = 0.39
    # => (3-beta) * ln(a*) = ln(0.39)
    # => 3-beta = ln(0.39)/ln(9.17e-4) = -0.94 / (-7.0) = 0.134
    # => beta_needed = 3 - 0.134 = 2.87
    beta_needed = 3 - np.log(Omega_cdm / MCMC_alpha) / np.log(a_rec)
    print(f"  Fuer CDM-aequivalenten Beitrag bei z*:")
    print(f"    beta_needed = {beta_needed:.2f}")
    print(f"    MCMC-Ergebnis: beta = {MCMC_beta:.2f} +/- 0.20")
    print(f"    Spannung: {abs(beta_needed - MCMC_beta)/0.20:.1f} sigma")
    print()
    print(f"  ABER: beta~2 wird durch die SN-Daten erzwungen (Kruemmungs-")
    print(f"  skalierung). Ein beta~3 wuerde den SN-Fit verschlechtern.")
    print(f"  Dies ist die fundamentale Spannung des Modells:")
    print(f"    - SN erfordern beta~2 (Kruemmung)")
    print(f"    - CMB-Peaks erfordern beta~3 (Materie)")
    print()

    return ratio_geom_to_baryon, R_cfm_eff


# ================================================================
# SECTION 3: BULLET CLUSTER LENSING ESTIMATE
# ================================================================

def section_3_bullet_cluster():
    print()
    print("=" * 74)
    print("  SECTION 3: BULLET-CLUSTER LENSING-ABSCHAETZUNG")
    print("=" * 74)
    print()

    # The Bullet Cluster (1E 0657-56) at z = 0.296
    z_bullet = 0.296
    a_bullet = 1.0 / (1 + z_bullet)

    # Key observation: gravitational lensing shows mass concentration
    # offset from the X-ray gas by ~250 kpc
    # The "dark matter" mass inferred from lensing: ~2.4 × 10^14 M_sun
    # (Clowe et al. 2006)

    M_lensing = 2.4e14  # M_sun, inferred "DM" mass from lensing offset
    offset = 250        # kpc, offset between lensing center and gas

    print(f"  Bullet Cluster: z = {z_bullet}, a = {a_bullet:.4f}")
    print(f"  Beobachteter Lensing-Masseversatz: ~{offset} kpc")
    print(f"  Inferierte 'DM-Masse': ~{M_lensing:.1e} M_sun")
    print()

    print("  FRAGE: Kann der geometrische DM-Term des CFM den")
    print("  Lensing-Versatz des Bullet Clusters erklaeren?")
    print()

    # In the extended CFM, the "dark matter" is spacetime geometry.
    # The geometric DM term alpha*a^{-beta} is a BACKGROUND term.
    # Perturbations in this term require the Lagrangian from Paper III.

    # The key argument from Paper II (Section 4.4.2):
    # The geometric potential is sourced by the total energy distribution
    # INCLUDING ITS OWN HISTORY (curvature memory).
    # During a cluster collision:
    # - Gas is decelerated (ram pressure)
    # - Geometric potential traces the PRE-COLLISION mass distribution
    # - Galaxies (collisionless) pass through with the geometric potential

    # Order-of-magnitude: Does the geometric potential provide enough
    # lensing convergence?

    # In LCDM, the convergence kappa from a NFW halo:
    # kappa ~ Sigma_DM / Sigma_crit
    # Sigma_crit = c^2 / (4*pi*G) * D_s / (D_l * D_ls)

    # For the CFM, we need to estimate the perturbation in the geometric
    # potential that would mimic the DM lensing signal.

    # The Poisson equation in the CFM (from Paper III, eq. 10):
    # nabla^2 Phi_N = 4*pi*G*rho_b + (perturbation from R^2 term)
    # The R^2 term contributes an additional gravitational potential
    # that depends on the curvature distribution.

    # Effective surface density from geometric DM at z=0.296:
    geom_dm_bg = MCMC_alpha * a_bullet**(-MCMC_beta)
    baryon_bg = OMEGA_B * (1+z_bullet)**3

    # The ratio at the cluster epoch
    ratio_at_cluster = geom_dm_bg / baryon_bg
    print(f"  Hintergrund-Verhaeltnisse bei z = {z_bullet}:")
    print(f"    Geom. DM / Baryonen = {ratio_at_cluster:.2f}")
    print(f"    (In LCDM: CDM / Baryonen = {0.265/0.0493:.2f})")
    print()

    # Critical assessment
    print("  BEWERTUNG:")
    print("  " + "-" * 50)
    print()
    print("  Das Paper-II-Argument beruht auf zwei Saeulen:")
    print()
    print("  (A) Das geometrische Potential hat GEDAECHTNIS:")
    print("      Es koppelt an die GESCHICHTE der Masseverteilung,")
    print("      nicht an die instantane Verteilung. Daher kann es")
    print("      nach einer Kollision die urspruengliche Position")
    print("      'erinnern' -- aehnlich wie CDM.")
    print()
    print("  (B) AeST-Analogie: In der relativistischen MOND-Theorie")
    print("      AeST erzeugen Skalar- und Vektorfelder Lensing-")
    print("      Signale, die vom Gas versetzt sind.")
    print()
    print("  KRITIK: Beide Argumente sind qualitativ, nicht quantitativ.")
    print("  Fuer eine quantitative Vorhersage wird benoetigt:")
    print("    1. Perturbationsgleichungen des erweiterten CFM")
    print("       (aus der Lagrangian-Formulierung, Paper III)")
    print("    2. Numerische Simulation einer Cluster-Kollision")
    print("       mit dem geometrischen Potential")
    print("    3. Vergleich der vorhergesagten Lensing-Karte")
    print("       mit den beobachteten Daten")
    print()
    print("  STATUS: Nicht-quantifizierbar ohne Perturbationstheorie.")
    print("  EMPFEHLUNG: Als 'offenes Problem' kennzeichnen, nicht")
    print("  als 'geloest'. Das Argument ist plausibel, aber nicht")
    print("  bewiesen.")
    print()


# ================================================================
# SECTION 4: EFFICIENCY HYPOTHESIS -- FORMALISIERUNG
# ================================================================

def section_4_efficiency():
    print()
    print("=" * 74)
    print("  SECTION 4: EFFIZIENZ-HYPOTHESE -- ZIRKELSCHLUSS-CHECK")
    print("=" * 74)
    print()

    print("  Paper II, Proposition 1 (Efficiency Principle):")
    print("  'In einem Nash-Gleichgewichts-Universum besteht der")
    print("   Materieinhalt ausschliesslich aus baryonischer Materie,")
    print("   weil DM eine ineffiziente Allokation des Energiebudgets")
    print("   darstellt.'")
    print()
    print("  REVIEW-KRITIK: Zirkelschluss-Risiko.")
    print("  'Man setzt voraus, dass das Universum Entropieproduktion")
    print("   optimiert, und schliesst dann, dass DM ineffizient waere.'")
    print()

    print("  ANALYSE DES ZIRKELSCHLUSS-VORWURFS:")
    print("  " + "=" * 50)
    print()
    print("  Die Argumentstruktur ist:")
    print("    Praemisse P1: Das Universum befindet sich in einem")
    print("      Nash-Gleichgewicht, das Entropieproduktion optimiert.")
    print("    Praemisse P2: Baryonen produzieren pro Masseneinheit")
    print("      mehr Entropie als DM.")
    print("    Schluss S:    DM ist nicht Teil der optimalen Loesung.")
    print()
    print("  IST DAS EIN ZIRKELSCHLUSS?")
    print("  " + "-" * 50)
    print()
    print("  Nein, im strengen Sinne nicht. Ein Zirkelschluss waere:")
    print("    'Es gibt keine DM, WEIL es keine DM gibt.'")
    print()
    print("  Hier ist die Struktur:")
    print("    P1 (Optimierung) + P2 (Effizienzranking) => S (keine DM)")
    print()
    print("  P2 ist empirisch verifizierbar und unstrittig:")
    print("    Baryonen: Sternbildung, Nukleosynthese, Schwarze Loecher")
    print("    DM (hypothetisch): nur gravitativ, keine Entropieproduktion")
    print()
    print("  ABER: P1 ist die UNBEWIESENE Praemisse.")
    print("  Das Argument ist formal gueltig, aber nur so stark wie P1.")
    print()
    print("  EMPFEHLUNG fuer Paper II:")
    print("  " + "-" * 50)
    print("  1. Proposition 1 umformulieren als KONDITIONAL:")
    print("     'WENN das Nash-Gleichgewicht Entropieproduktion")
    print("      optimiert (P1), DANN ist DM disfavorisiert (S).'")
    print()
    print("  2. P1 nicht als Axiom, sondern als Hypothese kennzeichnen,")
    print("     die durch den empirischen Erfolg des Modells gestuetzt")
    print("     (aber nicht bewiesen) wird.")
    print()
    print("  3. Den praeditiven Wert betonen: P1 + P2 => S ist eine")
    print("     testbare Vorhersage, nicht eine zirkulaere Definition.")
    print("     Das Modell waere falsifiziert, wenn DM experimentell")
    print("     nachgewiesen wuerde.")
    print()


# ================================================================
# SECTION 5: ZUSAMMENFASSUNG
# ================================================================

def section_5_summary(cv_mean_lcdm, cv_mean_cfm, geom_ratio, R_eff):
    print()
    print("=" * 74)
    print("  ZUSAMMENFASSUNG: PAPER II -- OFFENE PUNKTE UND LOESUNGEN")
    print("=" * 74)
    print()

    print("  1. KREUZVALIDIERUNG (neu berechnet):")
    if cv_mean_cfm < cv_mean_lcdm:
        print(f"     [ERLEDIGT] Extended CFM generalisiert besser als LCDM")
        print(f"     <chi2/n>: LCDM={cv_mean_lcdm:.4f}, Ext.CFM={cv_mean_cfm:.4f}")
        print(f"     => Kein Overfitting trotz 5 vs. 2 Parametern")
    else:
        print(f"     [WARNUNG] LCDM generalisiert besser!")
        print(f"     => Overfitting moeglich, muss im Paper diskutiert werden")
    print()

    print(f"  2. CMB-PEAKS (semi-analytisch berechnet):")
    print(f"     [OFFENES PROBLEM] Der geometrische DM-Term (beta~2)")
    print(f"     traegt nur {geom_ratio*100:.1f}% der Baryonendichte bei z* bei.")
    print(f"     Dies ist unzureichend fuer die CMB-Peak-Struktur.")
    print(f"     Moegliche Loesung: Perturbationen des R^2-Terms")
    print(f"     (Paper III Lagrangian) koennten zusaetzliche")
    print(f"     Gravitationspotentiale liefern.")
    print(f"     STATUS: Erfordert numerische Boltzmann-Rechnung (CLASS/CAMB)")
    print()

    print(f"  3. BULLET CLUSTER:")
    print(f"     [QUALITATIV PLAUSIBEL] Das Gedaechtnis-Argument und")
    print(f"     die AeST-Analogie sind konzeptuell ueberzeugend,")
    print(f"     aber nicht quantifiziert.")
    print(f"     STATUS: Erfordert Perturbationstheorie")
    print()

    print(f"  4. MUTTER-TOCHTER-ONTOLOGIE:")
    print(f"     [EMPFEHLUNG] In Diskussion verschieben, nicht als")
    print(f"     Definition 1 positionieren. Physikalisches Paper")
    print(f"     sollte Physik betonen, nicht Ontologie.")
    print()

    print(f"  5. EFFIZIENZ-HYPOTHESE:")
    print(f"     [KEIN ZIRKELSCHLUSS, aber schwache Praemisse]")
    print(f"     Umformulieren als Konditional: 'Falls P1, dann S.'")
    print(f"     P1 (Entropie-Optimierung) ist die zu beweisende Hypothese.")
    print()

    print(f"  KRITISCHSTER OFFENER PUNKT:")
    print(f"  Die CMB-Peak-Kompatibilitaet ist das groesste Risiko.")
    print(f"  Der Titel 'Eliminating the Dark Sector' sollte zu")
    print(f"  'Toward Eliminating the Dark Sector' abgemildert werden,")
    print(f"  bis eine quantitative C_l-Berechnung vorliegt.")
    print()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print()
    print("######################################################################")
    print("#  ERGAENZENDE BERECHNUNGEN FUER PAPER II (CFM+MOND)                #")
    print("#  Adressiert alle 5 Review-Punkte                                  #")
    print("######################################################################")
    print()

    # Section 1: Cross-validation (computationally intensive)
    cv_l, cv_c, folds_l, folds_c = section_1_crossvalidation()

    # Section 2: CMB peak ratios
    geom_ratio, R_eff = section_2_cmb_peaks()

    # Section 3: Bullet Cluster
    section_3_bullet_cluster()

    # Section 4: Efficiency Hypothesis
    section_4_efficiency()

    # Section 5: Summary
    section_5_summary(cv_l, cv_c, geom_ratio, R_eff)

    # Write results to file
    print("  Ergebnisse werden in _results/ gespeichert...")
    outpath = os.path.join(OUTPUT_DIR, 'Paper2_Supplement_Results.txt')
    # Redirect stdout to capture all output
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    section_1_crossvalidation()
    section_2_cmb_peaks()
    section_3_bullet_cluster()
    section_4_efficiency()
    section_5_summary(cv_l, cv_c, geom_ratio, R_eff)

    sys.stdout = old_stdout
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())
    print(f"  Gespeichert: {outpath}")

    print()
    print("  FERTIG.")

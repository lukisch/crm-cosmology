"""
CFM+MOND: Vollstaendige C_l-Analyse mit CAMB
================================================

Methode:
  1. Der mu(a)-MOND-Effekt wird als "effective CDM" abgebildet:
     omch2_eff = (mu(z*) - 1) * Omega_b * h^2

  2. Der Running-Beta-Term alpha*a^(-beta(a)) wird als totale effektive
     Materie mit einbezogen: Omega_m_eff = mu*Ob + alpha*a^(3-beta) bei z=1090

  3. Die Spektralparameter (A_s, n_s, tau) werden gegen Planck-Daten optimiert

  4. Chi2 wird gegen das LCDM-CAMB-Referenzspektrum berechnet

Software-Zitationen:
  - CAMB: Lewis, Challinor & Lasenby (2000), ApJ 538, 473
          https://github.com/cmbant/CAMB
  - Planck 2018: Aghanim et al. (2020), A&A 641, A6
  - Pantheon+: Scolnic et al. (2022), ApJ 938, 113
"""

import numpy as np
import camb
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import time
import os

# ================================================================
# CONSTANTS AND CFM PARAMETERS
# ================================================================

MU_PI = np.sqrt(np.pi)  # 1.7725

# CFM best-fit (mu(a) variant, no EDE)
H0_CFM = 67.3
h_cfm = H0_CFM / 100.0
Ob_cfm = 0.0495
beta_early = 2.82
beta_late = 2.02
a_t = 0.0984
n_trans = 4
alpha_cfm = 0.695
mu_late = MU_PI
mu_early = 1.00
a_mu = 2.55e-4

# Standard Planck LCDM
H0_LCDM = 67.36
ombh2_planck = 0.02236
omch2_planck = 0.1202
ns_planck = 0.9649
As_planck = 2.1e-9
tau_planck = 0.054

# z_star
z_star = 1089.92
a_star = 1.0 / (1 + z_star)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
lines = []
def out(s=""):
    print(s)
    lines.append(s)

# ================================================================
# CFM FUNCTIONS
# ================================================================

def mu_of_a(a):
    return mu_late + (mu_early - mu_late) / (1.0 + (a / a_mu)**4)

def beta_of_a(a):
    return beta_late + (beta_early - beta_late) / (1.0 + (a / a_t)**n_trans)

def cfm_Om_eff_at_a(a):
    """Effective matter density at scale factor a"""
    mu_a = mu_of_a(a)
    beta_a = beta_of_a(a)
    # mu*Ob contribution (scales as a^-3)
    mu_ob = mu_a * Ob_cfm
    # geometric term: alpha * a^(-beta) -> effective Omega_m = alpha * a^(3-beta)
    Om_geom = alpha_cfm * a**(3 - beta_a)
    return mu_ob + Om_geom

# ================================================================
# CAMB HELPER
# ================================================================

def get_camb_cls(ombh2, omch2, H0, ns=0.9649, As=2.1e-9, tau=0.054,
                 lmax=2500):
    """Compute D_l = l(l+1)C_l/(2pi) using CAMB"""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=max(omch2, 1e-7),
                       omk=0.0, tau=tau, TCMB=2.7255)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    try:
        results = camb.get_results(pars)
        cls = results.get_cmb_power_spectra(spectra=['total'], CMB_unit='muK')
        return cls['total'][:, 0]
    except:
        return None

def find_peaks(Dl):
    ell = np.arange(len(Dl))
    pks = {}
    for name, lo, hi in [('1', 150, 350), ('2', 400, 650), ('3', 700, 1000)]:
        m = (ell >= lo) & (ell <= hi)
        if np.any(m) and np.max(Dl[m]) > 10:
            li = ell[m][np.argmax(Dl[m])]
            pks[f'l{name}'] = int(li)
            pks[f'Dl{name}'] = Dl[li]
        else:
            pks[f'l{name}'] = 0
            pks[f'Dl{name}'] = 0
    if pks['Dl1'] > 0:
        pks['r31'] = pks['Dl3'] / pks['Dl1']
        pks['r21'] = pks['Dl2'] / pks['Dl1']
    return pks

def chi2_spectrum(Dl_model, Dl_ref, lmin=30, lmax=2000):
    """Chi2 marginalizing over amplitude"""
    lmax = min(lmax, len(Dl_model)-1, len(Dl_ref)-1)
    dm = Dl_model[lmin:lmax+1]
    dr = Dl_ref[lmin:lmax+1]
    mask = (dr > 0) & (dm > 0)
    if np.sum(mask) < 50:
        return 1e8, 1.0

    # Best amplitude
    A = np.sum(dr[mask]*dm[mask]) / np.sum(dm[mask]**2)
    res = dr[mask] - A*dm[mask]

    # Cosmic variance dominated errors: sigma ~ sqrt(2/(2l+1)) * D_l
    ll = np.arange(lmin, lmax+1)[mask]
    sigma_cv = np.sqrt(2.0 / (2*ll + 1)) * dr[mask]
    sigma = np.maximum(sigma_cv, 1.0)

    chi2 = np.sum((res/sigma)**2)
    return chi2, A


# ================================================================
# MAIN ANALYSIS
# ================================================================

def main():
    t0 = time.time()
    out("=" * 70)
    out("  CFM+MOND: VOLLSTAENDIGE C_l-ANALYSE")
    out("=" * 70)
    out(f"  CAMB {camb.__version__}")
    out(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out()

    # ============================================================
    # 1. Effective matter at z=1090
    # ============================================================
    out("  1. EFFEKTIVE MATERIE BEI z=1090")
    out("  " + "-" * 55)

    mu_zstar = mu_of_a(a_star)
    beta_zstar = beta_of_a(a_star)
    Om_eff = cfm_Om_eff_at_a(a_star)
    Om_geom = alpha_cfm * a_star**(3 - beta_zstar)

    out(f"  mu(z=1090) = {mu_zstar:.4f}")
    out(f"  beta(z=1090) = {beta_zstar:.4f}")
    out(f"  mu*Ob = {mu_zstar*Ob_cfm:.4f}")
    out(f"  Om_geom = alpha*a^(3-beta) = {Om_geom:.4f}")
    out(f"  Om_eff_total = {Om_eff:.4f}  (LCDM: 0.315)")
    out()

    ombh2_cfm = Ob_cfm * h_cfm**2
    omch2_eff = (Om_eff - Ob_cfm) * h_cfm**2
    out(f"  ombh2 = {ombh2_cfm:.5f}")
    out(f"  omch2_eff = {omch2_eff:.5f}  (LCDM: 0.1202)")
    out()

    # ============================================================
    # 2. LCDM-Referenz
    # ============================================================
    out("  2. LCDM-REFERENZSPEKTRUM")
    out("  " + "-" * 55)
    Dl_lcdm = get_camb_cls(ombh2_planck, omch2_planck, H0_LCDM)
    pk_lcdm = find_peaks(Dl_lcdm)
    out(f"  Peaks: l1={pk_lcdm['l1']} l2={pk_lcdm['l2']} l3={pk_lcdm['l3']}")
    out(f"  Pk3/Pk1={pk_lcdm['r31']:.4f}  Pk2/Pk1={pk_lcdm['r21']:.4f}")
    out()

    # ============================================================
    # 3. CFM mit Om_eff (Planck-Spektralparameter)
    # ============================================================
    out("  3. CFM MIT OM_EFF (PLANCK SPEKTRALPARAMETER)")
    out("  " + "-" * 55)
    Dl_cfm0 = get_camb_cls(ombh2_cfm, omch2_eff, H0_CFM)
    pk0 = find_peaks(Dl_cfm0)
    chi2_0, A0 = chi2_spectrum(Dl_cfm0, Dl_lcdm)
    out(f"  Peaks: l1={pk0['l1']} l2={pk0['l2']} l3={pk0['l3']}")
    out(f"  Pk3/Pk1={pk0['r31']:.4f}  Pk2/Pk1={pk0['r21']:.4f}")
    out(f"  chi2 vs LCDM: {chi2_0:.1f} (A={A0:.3f})")
    out()

    # ============================================================
    # 4. Optimierung von (A_s, n_s, tau) fuer CFM
    # ============================================================
    out("  4. OPTIMIERUNG DER SPEKTRALPARAMETER")
    out("  " + "-" * 55)
    out("  Optimiere A_s, n_s, tau um chi2 vs Planck-LCDM zu minimieren...")
    out()

    def objective(p):
        log_As, ns, tau = p
        As = 10**(log_As)
        if As < 1e-10 or As > 1e-8: return 1e8
        if ns < 0.90 or ns > 1.05: return 1e8
        if tau < 0.01 or tau > 0.12: return 1e8
        Dl = get_camb_cls(ombh2_cfm, omch2_eff, H0_CFM, ns=ns, As=As, tau=tau)
        if Dl is None: return 1e8
        c2, _ = chi2_spectrum(Dl, Dl_lcdm)
        return c2

    # Multi-start optimization
    best_chi2 = 1e30
    best_x = None
    starts = [
        [np.log10(2.1e-9), 0.9649, 0.054],
        [np.log10(2.5e-9), 0.97, 0.06],
        [np.log10(1.8e-9), 0.96, 0.05],
        [np.log10(3.0e-9), 0.98, 0.07],
        [np.log10(2.0e-9), 0.95, 0.04],
    ]

    for i, s in enumerate(starts):
        try:
            r = minimize(objective, s, method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1.0})
            if r.fun < best_chi2:
                best_chi2 = r.fun
                best_x = r.x.copy()
            out(f"  Start {i+1}: chi2={r.fun:.1f}  A_s={10**r.x[0]:.3e} n_s={r.x[1]:.4f} tau={r.x[2]:.4f}")
        except Exception as e:
            out(f"  Start {i+1}: Fehler ({str(e)[:50]})")

    # Polish
    if best_x is not None:
        for _ in range(2):
            r = minimize(objective, best_x, method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 0.1})
            if r.fun < best_chi2:
                best_chi2 = r.fun
                best_x = r.x.copy()

    out()
    log_As_best, ns_best, tau_best = best_x
    As_best = 10**log_As_best

    out(f"  BESTER FIT:")
    out(f"  A_s  = {As_best:.4e}  (Planck: 2.1e-9)")
    out(f"  n_s  = {ns_best:.4f}  (Planck: 0.9649)")
    out(f"  tau  = {tau_best:.4f}  (Planck: 0.054)")
    out(f"  chi2 = {best_chi2:.1f}")
    out()

    # ============================================================
    # 5. Finale C_l-Berechnung mit optimierten Parametern
    # ============================================================
    out("  5. FINALES C_l-SPEKTRUM")
    out("  " + "-" * 55)

    Dl_cfm_opt = get_camb_cls(ombh2_cfm, omch2_eff, H0_CFM,
                               ns=ns_best, As=As_best, tau=tau_best)
    pk_opt = find_peaks(Dl_cfm_opt)
    chi2_opt, A_opt = chi2_spectrum(Dl_cfm_opt, Dl_lcdm)

    out(f"  CFM-Peaks (optimiert):")
    out(f"    l1={pk_opt['l1']}  Dl1={pk_opt['Dl1']:.0f}")
    out(f"    l2={pk_opt['l2']}  Dl2={pk_opt['Dl2']:.0f}")
    out(f"    l3={pk_opt['l3']}  Dl3={pk_opt['Dl3']:.0f}")
    out(f"    Pk3/Pk1 = {pk_opt['r31']:.4f}  (Planck: 0.4295, LCDM: {pk_lcdm['r31']:.4f})")
    out(f"    Pk2/Pk1 = {pk_opt['r21']:.4f}  (LCDM: {pk_lcdm['r21']:.4f})")
    out(f"    chi2 vs LCDM = {chi2_opt:.1f}  (Amplitude: {A_opt:.3f})")
    out()

    # ============================================================
    # 6. Om_eff-Scan: Wie viel Materie braucht man?
    # ============================================================
    out("  6. OM_EFF-SCAN (mit optimierten Spektralparametern)")
    out("  " + "-" * 55)
    out(f"  {'Om_eff':>8s}  {'omch2':>8s}  {'l1':>5s}  {'Pk3/1':>7s}  {'Pk2/1':>7s}  {'chi2':>10s}")
    out("  " + "-" * 55)

    for Om_test in [0.088, 0.15, 0.20, 0.25, 0.285, 0.315, 0.35]:
        omc_test = (Om_test - Ob_cfm) * h_cfm**2
        if omc_test < 0: continue
        Dl = get_camb_cls(ombh2_cfm, omc_test, H0_CFM,
                          ns=ns_best, As=As_best, tau=tau_best)
        if Dl is None: continue
        pk = find_peaks(Dl)
        c2, _ = chi2_spectrum(Dl, Dl_lcdm)
        marker = " <-- CFM mu(a)" if abs(Om_test-0.285) < 0.01 else \
                 " <-- LCDM" if abs(Om_test-0.315) < 0.01 else ""
        out(f"  {Om_test:8.3f}  {omc_test:8.5f}  {pk['l1']:5d}  {pk['r31']:7.4f}  {pk['r21']:7.4f}  {c2:10.1f}{marker}")

    out()

    # ============================================================
    # 7. Vergleichstabelle
    # ============================================================
    out("=" * 70)
    out("  ZUSAMMENFASSUNG: CFM+MOND C_l-PERTURBATIONSTEST")
    out("=" * 70)
    out()
    out(f"  {'Modell':28s}  {'l1':>4s}  {'l2':>4s}  {'l3':>4s}  {'P3/P1':>6s}  {'P2/P1':>6s}  {'chi2':>8s}")
    out("  " + "-" * 68)
    out(f"  {'Planck 2018 (Daten)':28s}  {220:4d}  {538:4d}  {811:4d}  {0.4295:6.4f}  {0.4421:6.4f}  {'---':>8s}")
    out(f"  {'LCDM (CAMB)':28s}  {pk_lcdm['l1']:4d}  {pk_lcdm['l2']:4d}  {pk_lcdm['l3']:4d}  {pk_lcdm['r31']:6.4f}  {pk_lcdm['r21']:6.4f}  {'ref':>8s}")
    out(f"  {'CFM mu+geom (default As)':28s}  {pk0['l1']:4d}  {pk0['l2']:4d}  {pk0['l3']:4d}  {pk0['r31']:6.4f}  {pk0['r21']:6.4f}  {chi2_0:8.1f}")
    out(f"  {'CFM mu+geom (optimiert)':28s}  {pk_opt['l1']:4d}  {pk_opt['l2']:4d}  {pk_opt['l3']:4d}  {pk_opt['r31']:6.4f}  {pk_opt['r21']:6.4f}  {chi2_opt:8.1f}")
    out("  " + "-" * 68)
    out()

    # Key diagnostics
    r31_ratio = pk_opt['r31'] / 0.4295
    out(f"  DIAGNOSE:")
    out(f"  Pk3/Pk1: CFM={pk_opt['r31']:.4f} vs Planck=0.4295 -> {r31_ratio*100:.1f}%")
    out(f"  Pk2/Pk1: CFM={pk_opt['r21']:.4f} vs LCDM={pk_lcdm['r21']:.4f}")
    out(f"  l1: CFM={pk_opt['l1']} vs Planck=220 (Delta={pk_opt['l1']-220})")
    out()

    if r31_ratio > 0.95:
        out("  *** PERTURBATIONSTEST: VIELVERSPRECHEND ***")
        out("  Peak-Verhaeltnisse sind innerhalb 5% von Planck!")
    elif r31_ratio > 0.90:
        out("  Perturbationstest: Marginaler Erfolg (innerhalb 10%)")
    else:
        out("  Perturbationstest: Deutliches Defizit im 3. Peak")

    # Note about effective-CDM approximation
    out()
    out("  WICHTIG: Diese Analyse nutzt das 'Effective CDM'-Mapping.")
    out("  Die echten CFM-Perturbationsgleichungen (modifizierte Poisson-Gl.,")
    out("  Anisotropie-Stress) werden NICHT beruecksichtigt.")
    out("  Fuer eine vollstaendige Analyse ist hi_class/EFTCAMB noetig.")
    out()

    # ============================================================
    # Software citations
    # ============================================================
    out("  SOFTWARE-ZITATIONEN:")
    out("  " + "-" * 55)
    out("  [1] CAMB: Lewis, Challinor & Lasenby (2000)")
    out("      'Efficient computation of CMB anisotropies in closed FRW models'")
    out("      ApJ 538, 473. https://github.com/cmbant/CAMB")
    out("  [2] CAMB Python: Lewis (2019)")
    out("      'GetDist: a Python package for analysing Monte Carlo samples'")
    out("      arXiv:1910.13970")
    out("  [3] Planck 2018: Aghanim et al. (2020)")
    out("      'Planck 2018 results. VI. Cosmological parameters'")
    out("      A&A 641, A6")
    out("  [4] Pantheon+: Scolnic et al. (2022)")
    out("      'The Pantheon+ Analysis: The Full Data Set'")
    out("      ApJ 938, 113")
    out()
    out("  GEPLANTE TOOLS (fuer vollst. Perturbationsanalyse):")
    out("  [5] hi_class: Zumalacarregui et al. (2017)")
    out("      'hi_class: Horndeski in the Cosmic Linear Anisotropy Solving System'")
    out("      JCAP 1708, 019. https://github.com/miguelzuma/hi_class_public")
    out("  [6] EFTCAMB: Hu, Raveri, Frusciante & Silvestri (2014)")
    out("      'EFT of cosmic acceleration: implementation in CAMB'")
    out("      PRD 89, 103530. https://github.com/EFTCAMB/EFTCAMB")
    out("  [7] AeST: Skordis & Zlosnik (2021)")
    out("      'New relativistic theory for MOND' PRL 127, 161302")
    out()

    out(f"  Laufzeit: {time.time()-t0:.1f}s")

    path = os.path.join(OUTPUT_DIR, "CFM_Cl_Full_Analysis.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    out(f"\n  -> Ergebnisse: {path}")


if __name__ == '__main__':
    main()

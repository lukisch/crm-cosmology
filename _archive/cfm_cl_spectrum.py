"""
CFM+MOND C_l Spektrums-Analyse
================================
Berechnet das volle CMB-Powerspektrum C_l fuer das CFM+MOND-Modell
mit scale-dependent mu(a) unter Verwendung des "Effective CDM"-Mappings.

Methode:
  - CAMB wird mit effektiver CDM-Dichte gefuettert, die dem CFM-Hintergrund
    bei z~1090 entspricht
  - mu(a) = sqrt(pi) bei z<1000, transitioning zu mu=1 bei z>4000
  - Effektives CDM: omch2_eff = (mu(z*) - 1) * Omega_b * h^2
  - Vergleich mit Planck TT-Spektrum (Peak-Positionen und -Verhaeltnisse)
"""

import numpy as np
import camb
import time
import os

c_light = 299792.458
MU_PI = np.sqrt(np.pi)
T_CMB = 2.7255

# Planck 2018 peak measurements (D_l = l(l+1)C_l/(2pi) in muK^2)
PLANCK_PEAKS = {
    'l1': 220.0, 'Dl1': 5720.0,
    'l2': 537.5, 'Dl2': 2529.0,
    'l3': 810.8, 'Dl3': 2457.0,
    'ratio31': 2457.0 / 5720.0,  # 0.4296
    'ratio21': 2529.0 / 5720.0,  # 0.4421
}

# CFM best-fit parameters (mu(a) variant, no EDE)
CFM_PARAMS = {
    'H0': 67.3,
    'mu_late': MU_PI,       # sqrt(pi) = 1.7725
    'mu_early': 1.00,
    'a_mu': 2.55e-4,        # z_mu = 3918
    'beta_early': 2.82,
    'a_t': 0.0984,          # z_t = 9.2
    'alpha': 0.695,
    'Omega_b': 0.0495,      # omega_b_BBN / h^2
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

lines = []
def out(s=""):
    print(s)
    lines.append(s)


def mu_of_a(a, mu_late=MU_PI, mu_early=1.0, a_mu=2.55e-4):
    """Scale-dependent MOND coupling"""
    return mu_late + (mu_early - mu_late) / (1.0 + (a / a_mu)**4)


def get_camb_cls(ombh2, omch2, H0=67.36, ns=0.9649, As=2.1e-9,
                 tau=0.054, lmax=2500):
    """Compute D_l using CAMB"""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=max(omch2, 1e-7),
                       omk=0.0, tau=tau, TCMB=T_CMB)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    try:
        results = camb.get_results(pars)
        cls = results.get_cmb_power_spectra(spectra=['total'], CMB_unit='muK')
        return cls['total'][:, 0]
    except Exception as e:
        out(f"  [CAMB Fehler: {str(e)[:80]}]")
        return None


def find_peaks(Dl):
    """Find acoustic peak positions and heights"""
    ell = np.arange(len(Dl))
    peaks = {}

    # 1st peak: 150-350
    m = (ell >= 150) & (ell <= 350)
    if np.any(m):
        l1 = ell[m][np.argmax(Dl[m])]
        peaks['l1'] = int(l1)
        peaks['Dl1'] = Dl[l1]

    # 2nd peak: 400-650
    m = (ell >= 400) & (ell <= 650)
    if np.any(m):
        l2 = ell[m][np.argmax(Dl[m])]
        peaks['l2'] = int(l2)
        peaks['Dl2'] = Dl[l2]

    # 3rd peak: 700-1000
    m = (ell >= 700) & (ell <= 1000)
    if np.any(m) and np.max(Dl[m]) > 10:
        l3 = ell[m][np.argmax(Dl[m])]
        peaks['l3'] = int(l3)
        peaks['Dl3'] = Dl[l3]
    else:
        peaks['l3'] = 0
        peaks['Dl3'] = 0.0

    if peaks['Dl1'] > 0:
        peaks['ratio31'] = peaks['Dl3'] / peaks['Dl1']
        peaks['ratio21'] = peaks.get('Dl2', 0) / peaks['Dl1']
    else:
        peaks['ratio31'] = 0
        peaks['ratio21'] = 0

    return peaks


def chi2_vs_reference(Dl_model, Dl_ref, lmin=30, lmax=2000):
    """Chi2 of model vs reference spectrum (marginalized over amplitude)"""
    lmax = min(lmax, len(Dl_model)-1, len(Dl_ref)-1)
    d_m = Dl_model[lmin:lmax+1]
    d_r = Dl_ref[lmin:lmax+1]
    mask = (d_r > 0) & (d_m > 0)
    if np.sum(mask) < 50:
        return 1e6, 1.0

    # Best amplitude rescaling
    A = np.sum(d_r[mask] * d_m[mask]) / np.sum(d_m[mask]**2)
    residuals = d_r[mask] - A * d_m[mask]

    # Approximate cosmic variance + noise: sigma ~ 5% of D_l
    sigma = np.maximum(0.05 * d_r[mask], 1.0)
    chi2 = np.sum((residuals / sigma)**2)
    ndof = np.sum(mask)
    return chi2, A


def main():
    t0 = time.time()
    out("=" * 70)
    out("  CFM+MOND: C_l SPEKTRUMS-ANALYSE")
    out("=" * 70)
    out(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"  CAMB Version: {camb.__version__}")
    out()

    h = CFM_PARAMS['H0'] / 100.0
    Ob = CFM_PARAMS['Omega_b']
    ombh2 = Ob * h**2

    # ============================================================
    # 1. LCDM Referenz
    # ============================================================
    out("  1. LCDM-Referenzspektrum (Planck best-fit)")
    out("  " + "-" * 55)
    Dl_lcdm = get_camb_cls(0.02236, 0.1202, H0=67.36)
    pk_lcdm = find_peaks(Dl_lcdm)
    out(f"  l1={pk_lcdm['l1']}  Dl1={pk_lcdm['Dl1']:.0f}")
    out(f"  l2={pk_lcdm['l2']}  Dl2={pk_lcdm['Dl2']:.0f}")
    out(f"  l3={pk_lcdm['l3']}  Dl3={pk_lcdm['Dl3']:.0f}")
    out(f"  Pk3/Pk1 = {pk_lcdm['ratio31']:.4f}  (Planck: {PLANCK_PEAKS['ratio31']:.4f})")
    out(f"  Pk2/Pk1 = {pk_lcdm['ratio21']:.4f}")
    out()

    # ============================================================
    # 2. Baryon-only (kein CDM, kein MOND)
    # ============================================================
    out("  2. Baryon-only (kein CDM, kein MOND)")
    out("  " + "-" * 55)
    Dl_bare = get_camb_cls(ombh2, 1e-7, H0=CFM_PARAMS['H0'])
    if Dl_bare is not None:
        pk_bare = find_peaks(Dl_bare)
        out(f"  l1={pk_bare['l1']}  Dl1={pk_bare['Dl1']:.0f}")
        out(f"  l3={pk_bare['l3']}  Dl3={pk_bare['Dl3']:.0f}")
        out(f"  Pk3/Pk1 = {pk_bare['ratio31']:.4f}  -> KATASTROPHAL niedrig")
    out()

    # ============================================================
    # 3. Effective-CDM Mapping fuer mu(a) bei z=1090
    # ============================================================
    out("  3. CFM+MOND: Effective-CDM Mapping")
    out("  " + "-" * 55)

    a_star = 1.0 / 1090.0
    mu_at_zstar = mu_of_a(a_star)
    omcdm_eff = (mu_at_zstar - 1.0) * Ob
    omch2_eff = omcdm_eff * h**2
    om_total = Ob + omcdm_eff

    out(f"  mu(z=1090)  = {mu_at_zstar:.4f}")
    out(f"  Omega_b     = {Ob:.5f}")
    out(f"  Omega_cdm_eff = (mu-1)*Ob = {omcdm_eff:.5f}")
    out(f"  Omega_m_eff = {om_total:.5f}")
    out(f"  ombh2       = {ombh2:.5f}")
    out(f"  omch2_eff   = {omch2_eff:.5f}")
    out(f"  H0          = {CFM_PARAMS['H0']}")
    out()

    out("  Vergleich mit LCDM:")
    out(f"    LCDM:     Omega_b=0.049, Omega_cdm=0.265, Omega_m=0.315")
    out(f"    CFM mu(a): Omega_b={Ob:.3f}, Omega_cdm_eff={omcdm_eff:.3f}, Omega_m_eff={om_total:.3f}")
    out()

    # Compute C_l with effective CDM
    out("  Berechne C_l mit effective CDM...")
    Dl_cfm = get_camb_cls(ombh2, omch2_eff, H0=CFM_PARAMS['H0'])
    if Dl_cfm is None:
        out("  FEHLER: CAMB-Berechnung fehlgeschlagen!")
        return

    pk_cfm = find_peaks(Dl_cfm)
    out(f"  l1={pk_cfm['l1']}  Dl1={pk_cfm['Dl1']:.0f}")
    out(f"  l2={pk_cfm['l2']}  Dl2={pk_cfm['Dl2']:.0f}")
    out(f"  l3={pk_cfm['l3']}  Dl3={pk_cfm['Dl3']:.0f}")
    out(f"  Pk3/Pk1 = {pk_cfm['ratio31']:.4f}  (Planck: {PLANCK_PEAKS['ratio31']:.4f})")
    out(f"  Pk2/Pk1 = {pk_cfm['ratio21']:.4f}")
    out()

    # Chi2 vs LCDM reference
    chi2_cfm, A_cfm = chi2_vs_reference(Dl_cfm, Dl_lcdm)
    out(f"  chi2 vs LCDM-Referenz: {chi2_cfm:.1f}  (Amplitude-Faktor: {A_cfm:.3f})")
    out()

    # ============================================================
    # 4. mu_eff-Scan: Optimales effektives mu fuer Peak-Verhaeltnisse
    # ============================================================
    out("  4. mu_eff-Scan: C_l-Spektren fuer verschiedene mu_eff")
    out("  " + "-" * 55)
    out(f"  {'mu_eff':>8s}  {'Om_cdm':>8s}  {'Om_m':>8s}  {'l1':>5s}  {'Pk3/Pk1':>8s}  {'Pk2/Pk1':>8s}  {'chi2':>10s}")
    out("  " + "-" * 65)

    scan_results = []
    for mu_val in [1.0, 1.33, 1.50, 1.77, 2.0, 2.5, 3.0, 4.0, 5.0, 6.3]:
        omc = (mu_val - 1.0) * Ob
        och2 = omc * h**2
        Dl = get_camb_cls(ombh2, och2, H0=CFM_PARAMS['H0'])
        if Dl is None:
            continue

        pk = find_peaks(Dl)
        c2, _ = chi2_vs_reference(Dl, Dl_lcdm)
        scan_results.append({
            'mu': mu_val, 'omc': omc, 'om': Ob + omc,
            'peaks': pk, 'chi2': c2, 'Dl': Dl
        })
        out(f"  {mu_val:8.2f}  {omc:8.4f}  {Ob+omc:8.4f}  {pk['l1']:5d}  {pk['ratio31']:8.4f}  {pk['ratio21']:8.4f}  {c2:10.1f}")

    out()

    # Find optimal mu for peak ratio
    if len(scan_results) > 3:
        mus = np.array([r['mu'] for r in scan_results])
        r31s = np.array([r['peaks']['ratio31'] for r in scan_results])
        target = PLANCK_PEAKS['ratio31']

        # Find where ratio31 crosses the Planck value
        diffs = r31s - target
        for i in range(len(diffs)-1):
            if diffs[i] * diffs[i+1] < 0:
                # Linear interpolation
                mu_opt = mus[i] + (mus[i+1] - mus[i]) * (-diffs[i]) / (diffs[i+1] - diffs[i])
                out(f"  OPTIMALES mu_eff fuer Pk3/Pk1 = {target:.4f}: mu = {mu_opt:.2f}")
                out(f"  => Omega_cdm_eff = {(mu_opt-1)*Ob:.4f}")
                out(f"  => Omega_m_eff = {Ob + (mu_opt-1)*Ob:.4f}")
                break

    out()

    # ============================================================
    # 5. Detaillierter Vergleich: CFM mu(a) vs LCDM
    # ============================================================
    out("  5. Detaillierter C_l-Vergleich: CFM mu(a) vs LCDM")
    out("  " + "-" * 55)
    out(f"  {'l':>6s}  {'D_l(LCDM)':>12s}  {'D_l(CFM)':>12s}  {'Ratio':>8s}")
    out("  " + "-" * 45)

    for l in [2, 10, 50, 100, 150, 220, 300, 400, 538, 700, 811, 1000, 1500, 2000]:
        if l < len(Dl_lcdm) and l < len(Dl_cfm):
            ratio = Dl_cfm[l] / Dl_lcdm[l] if Dl_lcdm[l] > 0 else 0
            out(f"  {l:6d}  {Dl_lcdm[l]:12.2f}  {Dl_cfm[l]:12.2f}  {ratio:8.4f}")

    out()

    # ============================================================
    # 6. Erweiterte Analyse: Running Beta Beitrag
    # ============================================================
    out("  6. Zusaetzlicher Beitrag des Running Beta")
    out("  " + "-" * 55)

    # At z=1090, the running beta term contributes:
    # alpha * a^(-beta_eff) where beta_eff(z=1090) ~ 2.82
    a_star = 1.0 / 1091.0
    beta_z1090 = CFM_PARAMS['beta_early']  # ~2.82 at z>>z_t
    geom_contribution = CFM_PARAMS['alpha'] * a_star**(-beta_z1090)

    # In matter-equivalent terms: alpha*a^(-beta) ~ Omega_m_geom * a^(-3)
    # At a=a_star: alpha * a_star^(-2.82) vs Omega_m * a_star^(-3)
    # Omega_m_geom(z=1090) = alpha * a_star^(-2.82+3) = alpha * a_star^(0.18)
    Om_geom_at_zstar = CFM_PARAMS['alpha'] * a_star**(3 - beta_z1090)

    out(f"  alpha = {CFM_PARAMS['alpha']:.3f}")
    out(f"  beta_eff(z=1090) = {beta_z1090:.2f}")
    out(f"  Geom. Term bei z=1090: alpha*a^(-beta) = {geom_contribution:.2f}")
    out(f"  Matter-Aequivalent: Omega_m_geom ~ {Om_geom_at_zstar:.4f}")
    out()

    # Total effective matter at z=1090
    Om_total_eff = mu_at_zstar * Ob + Om_geom_at_zstar
    out(f"  Totale effektive Materie bei z=1090:")
    out(f"    mu*Ob = {mu_at_zstar:.4f} * {Ob:.4f} = {mu_at_zstar*Ob:.4f}")
    out(f"    + geom = {Om_geom_at_zstar:.4f}")
    out(f"    = Om_eff_total = {Om_total_eff:.4f}")
    out(f"    (LCDM: Omega_m = 0.315)")
    out()

    # Compute C_l with total effective matter
    omch2_total = (Om_total_eff - Ob) * h**2
    out(f"  Berechne C_l mit totalem Om_eff = {Om_total_eff:.4f}...")
    Dl_total = get_camb_cls(ombh2, max(omch2_total, 1e-7), H0=CFM_PARAMS['H0'])
    if Dl_total is not None:
        pk_total = find_peaks(Dl_total)
        chi2_total, A_total = chi2_vs_reference(Dl_total, Dl_lcdm)
        out(f"  l1={pk_total['l1']}  Pk3/Pk1={pk_total['ratio31']:.4f}  chi2={chi2_total:.1f}")
    out()

    # ============================================================
    # 7. Zusammenfassung
    # ============================================================
    out("=" * 70)
    out("  ZUSAMMENFASSUNG: C_l-ANALYSE")
    out("=" * 70)
    out()
    out(f"  {'Modell':25s}  {'l1':>5s}  {'Pk3/Pk1':>8s}  {'Pk2/Pk1':>8s}  {'chi2':>10s}")
    out("  " + "-" * 65)
    out(f"  {'Planck 2018':25s}  {int(PLANCK_PEAKS['l1']):5d}  {PLANCK_PEAKS['ratio31']:8.4f}  {PLANCK_PEAKS['ratio21']:8.4f}  {'(Daten)':>10s}")
    out(f"  {'LCDM (CAMB)':25s}  {pk_lcdm['l1']:5d}  {pk_lcdm['ratio31']:8.4f}  {pk_lcdm['ratio21']:8.4f}  {'ref':>10s}")
    if Dl_bare is not None:
        out(f"  {'Baryon-only':25s}  {pk_bare['l1']:5d}  {pk_bare['ratio31']:8.4f}  {pk_bare.get('ratio21',0):8.4f}  {'---':>10s}")
    out(f"  {'CFM mu(z*)=1.77':25s}  {pk_cfm['l1']:5d}  {pk_cfm['ratio31']:8.4f}  {pk_cfm['ratio21']:8.4f}  {chi2_cfm:10.1f}")
    if Dl_total is not None:
        out(f"  {'CFM mu+geom':25s}  {pk_total['l1']:5d}  {pk_total['ratio31']:8.4f}  {pk_total['ratio21']:8.4f}  {chi2_total:10.1f}")
    out("  " + "-" * 65)
    out()

    # Key diagnostic
    out("  DIAGNOSE:")
    deficit = pk_cfm['ratio31'] / PLANCK_PEAKS['ratio31']
    out(f"  CFM Pk3/Pk1 = {pk_cfm['ratio31']:.4f} vs Planck {PLANCK_PEAKS['ratio31']:.4f}")
    out(f"  Verhaeltnis: {deficit:.3f}  ({'OK' if 0.9 < deficit < 1.1 else 'DEFIZIT' if deficit < 0.9 else 'UEBERSCHUSS'})")
    out()

    if deficit < 0.9:
        out("  Der 3. Peak ist zu niedrig relativ zum 1. Peak.")
        out("  Das bedeutet: Die effektive CDM-Dichte bei z=1090 ist zu gering.")
        out(f"  Benoetigtes mu_eff (geschaetzt): {1 + (0.265/Ob) * (PLANCK_PEAKS['ratio31']/pk_cfm['ratio31']):.1f}")
    elif deficit > 1.1:
        out("  Der 3. Peak ist zu hoch relativ zum 1. Peak.")
        out("  Die effektive Materie bei z=1090 ist zu gross.")

    out()
    out(f"  Laufzeit: {time.time()-t0:.1f}s")

    # Save
    path = os.path.join(OUTPUT_DIR, "CFM_Cl_Spectrum.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    out(f"\n  -> Ergebnisse in {path}")


if __name__ == '__main__':
    main()

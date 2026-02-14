"""
CFM+MOND: Perturbationsanalyse mit hi_class
=============================================

Erste vollstaendige Perturbationsanalyse des CFM+MOND-Modells
unter Verwendung von hi_class (Horndeski in CLASS).

Strategie:
  1. Baseline: Effective CDM in standard CLASS (Reproduktion des CAMB-Ergebnisses)
  2. Modified Gravity: alpha_M Scan (effektive Gravitationsverstaerkung)
  3. Optimierung: Finde alpha-Parameter die Planck am besten fitten
  4. Physikalische Diagnostik: G_eff, slip, Lensing

Software-Zitationen:
  - hi_class: Zumalacarregui, Bellini, Sawicki, Lesgourgues, Ferreira (2017)
              JCAP 1708, 019. arXiv:1605.06102
  - hi_class BG: Bellini, Sawicki, Zumalacarregui (2020) arXiv:1909.01828
  - CLASS: Lesgourgues (2011), arXiv:1104.2932
  - Planck 2018: Aghanim et al. (2020), A&A 641, A6
"""

import numpy as np
from classy import Class
import time
import os

# ================================================================
# CONSTANTS
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

# Planck LCDM
H0_LCDM = 67.36
ombh2_planck = 0.02236
omch2_planck = 0.1202
ns_planck = 0.9649
As_planck = 2.1e-9
tau_planck = 0.054

# CFM optimized spectral parameters
As_cfm = 3.039e-9
ns_cfm = 0.9638
tau_cfm = 0.074

# z_star
z_star = 1089.92
a_star = 1.0 / (1 + z_star)

# Effective matter
def mu_of_a(a):
    return mu_late + (mu_early - mu_late) / (1.0 + (a / a_mu)**4)

def beta_of_a(a):
    return beta_late + (beta_early - beta_late) / (1.0 + (a / a_t)**n_trans)

mu_zs = mu_of_a(a_star)
Om_geom_zs = alpha_cfm * a_star**(3 - beta_of_a(a_star))
Om_eff_zs = mu_zs * Ob_cfm + Om_geom_zs
ombh2_cfm = Ob_cfm * h_cfm**2
omch2_eff = (Om_eff_zs - Ob_cfm) * h_cfm**2

# Output
OUTPUT_DIR = "/mnt/c/Users/User/OneDrive/Desktop/Forschung/Natur&Technik/Spieltheorie Urknall/_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
lines = []
def out(s=""):
    print(s)
    lines.append(s)


def find_peaks(ell, Dl):
    """Find CMB TT peaks"""
    pks = {}
    for name, lo, hi in [('1', 150, 350), ('2', 400, 650), ('3', 700, 1000)]:
        mask = (ell >= lo) & (ell <= hi)
        if np.any(mask) and np.max(Dl[mask]) > 10:
            idx = np.argmax(Dl[mask])
            li = ell[mask][idx]
            pks[f'l{name}'] = int(li)
            pks[f'Dl{name}'] = Dl[mask][idx]
        else:
            pks[f'l{name}'] = 0
            pks[f'Dl{name}'] = 0
    if pks['Dl1'] > 0:
        pks['r31'] = pks['Dl3'] / pks['Dl1']
        pks['r21'] = pks['Dl2'] / pks['Dl1']
    else:
        pks['r31'] = 0
        pks['r21'] = 0
    return pks


def chi2_cls(ell1, Dl1, ell2, Dl2, lmin=30, lmax=2000):
    """Chi2 between two C_l spectra with amplitude marginalization"""
    lmax = min(lmax, max(ell1), max(ell2))
    # Interpolate to common ell range
    ell_common = np.arange(lmin, lmax+1)
    d1 = np.interp(ell_common, ell1, Dl1)
    d2 = np.interp(ell_common, ell2, Dl2)
    mask = (d1 > 0) & (d2 > 0)
    if np.sum(mask) < 50:
        return 1e8, 1.0
    A = np.sum(d2[mask]*d1[mask]) / np.sum(d1[mask]**2)
    res = d2[mask] - A*d1[mask]
    sigma_cv = np.sqrt(2.0 / (2*ell_common[mask] + 1)) * d2[mask]
    sigma = np.maximum(sigma_cv, 1.0)
    return np.sum((res/sigma)**2), A


def run_class_standard(ombh2, omch2, H0, ns, As, tau, lmax=2500):
    """Run standard CLASS (no modified gravity)"""
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'lensing': 'yes',
        'l_max_scalars': lmax,
        'omega_b': ombh2,
        'omega_cdm': omch2,
        'H0': H0,
        'n_s': ns,
        'A_s': As,
        'tau_reio': tau,
        'T_cmb': 2.7255,
        'P_k_max_h/Mpc': 10.,
        'z_pk': '0., 0.5, 1.0, 2.0',
    }
    cosmo.set(params)
    try:
        cosmo.compute()
        cls_tt = cosmo.lensed_cl(lmax)
        ell = np.arange(len(cls_tt['ell']))
        # Convert to D_l = l(l+1)C_l/(2pi) in muK^2
        factor = ell * (ell + 1) / (2 * np.pi) * (2.7255e6)**2
        Dl = cls_tt['tt'] * factor
        return ell, Dl, cosmo
    except Exception as e:
        print(f"  CLASS error: {e}")
        return None, None, None


def run_hiclass_mg(ombh2, omch2, H0, ns, As, tau,
                   alpha_M=0., alpha_B=0., alpha_K=1., alpha_T=0.,
                   lmax=2500):
    """Run hi_class with modified gravity (propto_omega parametrization)"""
    cosmo = Class()

    # Omega_smg = -1 means it's inferred from closure
    # expansion_model = lcdm means LCDM-like expansion
    # gravity_model = propto_omega means alphas proportional to DE density
    # parameters_smg = alpha_K, alpha_B, alpha_M, alpha_T, M*^2_ini

    params = {
        'output': 'tCl,pCl,lCl',
        'lensing': 'yes',
        'l_max_scalars': lmax,
        'omega_b': ombh2,
        'omega_cdm': omch2,
        'H0': H0,
        'n_s': ns,
        'A_s': As,
        'tau_reio': tau,
        'T_cmb': 2.7255,
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'propto_omega',
        'parameters_smg': f'{alpha_K}, {alpha_B}, {alpha_M}, {alpha_T}, 1.',
        'expansion_model': 'lcdm',
        'expansion_smg': 0.5,  # Will be overwritten by closure
        'skip_stability_tests_smg': 'yes',  # For exploration
        'output_background_smg': 2,
    }
    cosmo.set(params)
    try:
        cosmo.compute()
        cls_tt = cosmo.lensed_cl(lmax)
        ell = np.arange(len(cls_tt['ell']))
        factor = ell * (ell + 1) / (2 * np.pi) * (2.7255e6)**2
        Dl = cls_tt['tt'] * factor
        return ell, Dl, cosmo
    except Exception as e:
        print(f"  hi_class error: {e}")
        return None, None, None


# ================================================================
# MAIN
# ================================================================

def main():
    t0 = time.time()
    out("=" * 70)
    out("  CFM+MOND: PERTURBATIONSANALYSE MIT hi_class")
    out("=" * 70)
    out(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out()

    out(f"  CFM-Parameter:")
    out(f"    Om_eff(z*) = {Om_eff_zs:.4f}, mu(z*) = {mu_zs:.4f}")
    out(f"    ombh2 = {ombh2_cfm:.5f}, omch2_eff = {omch2_eff:.5f}")
    out(f"    As = {As_cfm:.3e}, ns = {ns_cfm:.4f}, tau = {tau_cfm:.3f}")
    out()

    # ============================================================
    # 1. LCDM-REFERENZ (Standard CLASS)
    # ============================================================
    out("=" * 70)
    out("  1. LCDM-REFERENZ (Standard CLASS)")
    out("=" * 70)
    out()

    ell_lcdm, Dl_lcdm, cosmo_lcdm = run_class_standard(
        ombh2_planck, omch2_planck, H0_LCDM,
        ns_planck, As_planck, tau_planck)

    if Dl_lcdm is not None:
        pk_lcdm = find_peaks(ell_lcdm, Dl_lcdm)
        out(f"  LCDM Peaks: l1={pk_lcdm['l1']}, l2={pk_lcdm['l2']}, l3={pk_lcdm['l3']}")
        out(f"  Pk3/Pk1={pk_lcdm['r31']:.4f}  Pk2/Pk1={pk_lcdm['r21']:.4f}")
        sigma8_lcdm = cosmo_lcdm.sigma8()
        out(f"  sigma8 = {sigma8_lcdm:.4f}")
        cosmo_lcdm.struct_cleanup()
        cosmo_lcdm.empty()
    out()

    # ============================================================
    # 2. CFM EFFECTIVE CDM (Standard CLASS, Sanity Check)
    # ============================================================
    out("=" * 70)
    out("  2. CFM EFFECTIVE CDM (Standard CLASS)")
    out("=" * 70)
    out()

    ell_cfm0, Dl_cfm0, cosmo_cfm0 = run_class_standard(
        ombh2_cfm, omch2_eff, H0_CFM,
        ns_cfm, As_cfm, tau_cfm)

    if Dl_cfm0 is not None and Dl_lcdm is not None:
        pk_cfm0 = find_peaks(ell_cfm0, Dl_cfm0)
        c2_0, A0 = chi2_cls(ell_cfm0, Dl_cfm0, ell_lcdm, Dl_lcdm)
        out(f"  CFM (eff. CDM) Peaks: l1={pk_cfm0['l1']}, l2={pk_cfm0['l2']}, l3={pk_cfm0['l3']}")
        out(f"  Pk3/Pk1={pk_cfm0['r31']:.4f}  Pk2/Pk1={pk_cfm0['r21']:.4f}")
        out(f"  chi2 vs LCDM: {c2_0:.1f} (A={A0:.3f})")
        sigma8_cfm0 = cosmo_cfm0.sigma8()
        out(f"  sigma8 = {sigma8_cfm0:.4f}")
        cosmo_cfm0.struct_cleanup()
        cosmo_cfm0.empty()
    out()

    # ============================================================
    # 3. MODIFIED GRAVITY: alpha_M SCAN
    # ============================================================
    out("=" * 70)
    out("  3. MODIFIED GRAVITY: alpha_M SCAN (propto_omega)")
    out("=" * 70)
    out()
    out("  alpha_M modifiziert die effektive Gravitationskonstante:")
    out("  G_eff/G_N = 1 + f(alpha_M) -> simuliert mu(a)-Verstaerkung")
    out()

    # Scan over alpha_M with effective CDM parameters
    out(f"  {'aM':>6s}  {'l1':>5s}  {'Pk3/1':>7s}  {'Pk2/1':>7s}  {'chi2':>10s}  {'sig8':>6s}  Status")
    out("  " + "-" * 65)

    best_chi2 = 1e30
    best_aM = 0

    for aM in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        try:
            ell_mg, Dl_mg, cosmo_mg = run_hiclass_mg(
                ombh2_cfm, omch2_eff, H0_CFM,
                ns_cfm, As_cfm, tau_cfm,
                alpha_M=aM, alpha_K=1.0)

            if Dl_mg is not None and Dl_lcdm is not None:
                pk_mg = find_peaks(ell_mg, Dl_mg)
                c2, A = chi2_cls(ell_mg, Dl_mg, ell_lcdm, Dl_lcdm)
                status = ""
                if c2 < best_chi2:
                    best_chi2 = c2
                    best_aM = aM
                    status = " <-- best"
                out(f"  {aM:6.2f}  {pk_mg['l1']:5d}  {pk_mg['r31']:7.4f}  {pk_mg['r21']:7.4f}  {c2:10.1f}  {'---':>6s}{status}")
                cosmo_mg.struct_cleanup()
                cosmo_mg.empty()
            else:
                out(f"  {aM:6.2f}  -- computation failed --")
        except Exception as e:
            out(f"  {aM:6.2f}  ERROR: {str(e)[:50]}")

    out()
    out(f"  Bester alpha_M: {best_aM:.2f} (chi2={best_chi2:.1f})")
    out()

    # ============================================================
    # 4. alpha_B SCAN (Braiding)
    # ============================================================
    out("=" * 70)
    out("  4. BRAIDING alpha_B SCAN")
    out("=" * 70)
    out()
    out("  alpha_B kontrolliert die Kopplung zwischen Skalarfeld und Materie")
    out()

    out(f"  {'aB':>6s}  {'l1':>5s}  {'Pk3/1':>7s}  {'Pk2/1':>7s}  {'chi2':>10s}  Status")
    out("  " + "-" * 55)

    best_chi2_B = 1e30
    best_aB = 0

    for aB in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, -0.1, -0.2, -0.5]:
        try:
            ell_b, Dl_b, cosmo_b = run_hiclass_mg(
                ombh2_cfm, omch2_eff, H0_CFM,
                ns_cfm, As_cfm, tau_cfm,
                alpha_M=0., alpha_B=aB, alpha_K=1.0)

            if Dl_b is not None and Dl_lcdm is not None:
                pk_b = find_peaks(ell_b, Dl_b)
                c2, A = chi2_cls(ell_b, Dl_b, ell_lcdm, Dl_lcdm)
                status = ""
                if c2 < best_chi2_B:
                    best_chi2_B = c2
                    best_aB = aB
                    status = " <-- best"
                out(f"  {aB:6.2f}  {pk_b['l1']:5d}  {pk_b['r31']:7.4f}  {pk_b['r21']:7.4f}  {c2:10.1f}{status}")
                cosmo_b.struct_cleanup()
                cosmo_b.empty()
            else:
                out(f"  {aB:6.2f}  -- computation failed --")
        except Exception as e:
            out(f"  {aB:6.2f}  ERROR: {str(e)[:50]}")

    out()
    out(f"  Bester alpha_B: {best_aB:.2f} (chi2={best_chi2_B:.1f})")
    out()

    # ============================================================
    # 5. KOMBINIERTER alpha_M + alpha_B SCAN
    # ============================================================
    out("=" * 70)
    out("  5. KOMBINIERTER alpha_M + alpha_B SCAN")
    out("=" * 70)
    out()

    out(f"  {'aM':>5s}  {'aB':>5s}  {'l1':>5s}  {'Pk3/1':>7s}  {'chi2':>10s}  Status")
    out("  " + "-" * 50)

    best_chi2_comb = 1e30
    best_params = (0, 0)

    for aM in [0.0, 0.1, 0.2, 0.3, 0.5]:
        for aB in [0.0, 0.1, 0.2, -0.1, -0.2]:
            try:
                ell_c, Dl_c, cosmo_c = run_hiclass_mg(
                    ombh2_cfm, omch2_eff, H0_CFM,
                    ns_cfm, As_cfm, tau_cfm,
                    alpha_M=aM, alpha_B=aB, alpha_K=1.0)

                if Dl_c is not None and Dl_lcdm is not None:
                    pk_c = find_peaks(ell_c, Dl_c)
                    c2, A = chi2_cls(ell_c, Dl_c, ell_lcdm, Dl_lcdm)
                    status = ""
                    if c2 < best_chi2_comb:
                        best_chi2_comb = c2
                        best_params = (aM, aB)
                        status = " <-- best"
                    out(f"  {aM:5.2f}  {aB:5.2f}  {pk_c['l1']:5d}  {pk_c['r31']:7.4f}  {c2:10.1f}{status}")
                    cosmo_c.struct_cleanup()
                    cosmo_c.empty()
            except Exception as e:
                out(f"  {aM:5.2f}  {aB:5.2f}  ERROR: {str(e)[:40]}")

    out()
    out(f"  Bestes Paar: alpha_M={best_params[0]:.2f}, alpha_B={best_params[1]:.2f}")
    out(f"  chi2 = {best_chi2_comb:.1f}")
    out()

    # ============================================================
    # 6. PHYSIKALISCHE DIAGNOSTIK MIT BESTEM MODELL
    # ============================================================
    out("=" * 70)
    out("  6. PHYSIKALISCHE DIAGNOSTIK")
    out("=" * 70)
    out()

    aM_best, aB_best = best_params
    try:
        ell_best, Dl_best, cosmo_best = run_hiclass_mg(
            ombh2_cfm, omch2_eff, H0_CFM,
            ns_cfm, As_cfm, tau_cfm,
            alpha_M=aM_best, alpha_B=aB_best, alpha_K=1.0)

        if cosmo_best is not None:
            pk_best = find_peaks(ell_best, Dl_best)
            c2_best, A_best = chi2_cls(ell_best, Dl_best, ell_lcdm, Dl_lcdm)

            out(f"  Bestes Modell: alpha_M={aM_best:.2f}, alpha_B={aB_best:.2f}")
            out(f"  Peaks: l1={pk_best['l1']}, l2={pk_best['l2']}, l3={pk_best['l3']}")
            out(f"  Pk3/Pk1 = {pk_best['r31']:.4f} (Planck: 0.4295)")
            out(f"  chi2 vs LCDM = {c2_best:.1f}")
            out()

            # G_eff at various redshifts
            out("  G_eff/G_N bei verschiedenen Rotverschiebungen:")
            out(f"  {'z':>8s}  {'G_eff':>10s}  {'G_light':>10s}  {'slip':>10s}")
            out("  " + "-" * 45)
            for z in [0, 0.5, 1, 2, 5, 10, 50, 100, 500, 1090]:
                try:
                    geff = cosmo_best.G_eff_at_z_smg(z)
                    glight = cosmo_best.G_light_at_z_smg(z)
                    slip = cosmo_best.slip_eff_at_z_smg(z)
                    out(f"  {z:8.0f}  {geff:10.6f}  {glight:10.6f}  {slip:10.6f}")
                except Exception as e:
                    out(f"  {z:8.0f}  Error: {str(e)[:30]}")
            out()

            cosmo_best.struct_cleanup()
            cosmo_best.empty()

    except Exception as e:
        out(f"  Diagnostik-Fehler: {e}")

    # ============================================================
    # ZUSAMMENFASSUNG
    # ============================================================
    out("=" * 70)
    out("  ZUSAMMENFASSUNG: hi_class PERTURBATIONSANALYSE")
    out("=" * 70)
    out()

    out("  Vergleichstabelle:")
    out(f"  {'Modell':>30s}  {'l1':>4s}  {'P3/P1':>6s}  {'chi2':>8s}")
    out("  " + "-" * 55)
    if Dl_lcdm is not None:
        out(f"  {'Planck 2018':>30s}  {220:4d}  {0.4295:6.4f}  {'---':>8s}")
        out(f"  {'LCDM (CLASS)':>30s}  {pk_lcdm['l1']:4d}  {pk_lcdm['r31']:6.4f}  {'ref':>8s}")
    if Dl_cfm0 is not None:
        out(f"  {'CFM eff.CDM (CLASS)':>30s}  {pk_cfm0['l1']:4d}  {pk_cfm0['r31']:6.4f}  {c2_0:8.1f}")
    out(f"  {'CFM + MG (hi_class)':>30s}  {pk_best['l1']:4d}  {pk_best['r31']:6.4f}  {c2_best:8.1f}")
    out()

    out("  SOFTWARE-ZITATIONEN:")
    out("  " + "-" * 55)
    out("  [1] hi_class: Zumalacarregui et al. (2017)")
    out("      JCAP 1708, 019. arXiv:1605.06102")
    out("  [2] hi_class BG: Bellini, Sawicki, Zumalacarregui (2020)")
    out("      arXiv:1909.01828")
    out("  [3] CLASS: Lesgourgues (2011), arXiv:1104.2932")
    out("  [4] Planck 2018: Aghanim et al. (2020), A&A 641, A6")
    out()

    out(f"  Laufzeit: {time.time()-t0:.1f}s")

    path = os.path.join(OUTPUT_DIR, "CFM_hiclass_Perturbations.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    out(f"\n  -> Ergebnisse: {path}")


if __name__ == '__main__':
    main()

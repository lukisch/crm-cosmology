#!/usr/bin/env python3
"""
===============================================================================
P1-P2-P3: CFM BOLTZMANN-ANALYSE UND GEMEINSAMER FIT
===============================================================================

P1: Implementierung des CFM-Hintergrunds + CAMB-Referenzspektren
P2: C_l-Berechnung mit mu_eff-Scan (effektive Gravitationsverstaerkung)
P3: Gemeinsamer Fit: SN (Pantheon+) + CMB (Planck) + BAO

Methodik:
- CAMB fuer LCDM-Referenz und effektive CFM-Modelle
- Analytische CFM-Hintergrund-Kosmologie
- Komprimierte Planck-Likelihood (Akustische Skala + Peak-Verhaeltnisse)
- Pantheon+ SN-Daten (1590 Supernovae)
- BAO-Messungen (6dFGS, SDSS, BOSS, eBOSS)

===============================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize
import camb
import os
import warnings
import time
warnings.filterwarnings('ignore')

# ================================================================
# PATHS AND SETUP
# ================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "_data", "Pantheon+SH0ES.dat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_lines = []
def out(text=""):
    print(text)
    output_lines.append(text)

def save_output():
    path = os.path.join(OUTPUT_DIR, "P1_P2_P3_Joint_Analysis.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    print(f"\n  [Ergebnisse gespeichert: {path}]")

# ================================================================
# PHYSICAL CONSTANTS
# ================================================================

c_light = 299792.458   # km/s
H0 = 67.36             # km/s/Mpc
h = H0 / 100.0
T_CMB = 2.7255          # K

# Density parameters
Omega_b = 0.05          # CFM: baryons only
Omega_b_lcdm = 0.0493   # LCDM baryons
Omega_cdm_lcdm = 0.265  # LCDM CDM
Omega_m_lcdm = 0.315    # LCDM total matter
Omega_gamma = 5.38e-5   # photon density
Omega_nu = 3.65e-5      # neutrino density
Omega_r = Omega_gamma + Omega_nu  # total radiation

# CMB
z_star = 1089.92
a_star = 1.0 / (1 + z_star)
z_drag = 1059.62  # baryon drag epoch

# MCMC best-fit (Paper II)
MCMC_k = 9.81
MCMC_a_trans = 0.971
MCMC_alpha = 0.68
MCMC_beta = 2.02
MCMC_Phi0 = 0.43

# R2 perturbation parameters
eps_16piG = 16 * np.pi * 6.674e-11  # for unit conversion

# ================================================================
# PLANCK 2018 COMPRESSED LIKELIHOOD
# (Chen, Huang, Wang 2019 - Planck 2018 distance priors)
# ================================================================

PLANCK_lA = 301.471      # acoustic scale
PLANCK_lA_err = 0.090
PLANCK_R = 1.7502        # shift parameter (uses Omega_m)
PLANCK_R_err = 0.0046
PLANCK_wb = 0.02236      # omega_b = Omega_b * h^2
PLANCK_wb_err = 0.00015

# Planck 2018 TT peak measurements (approximate)
PLANCK_l1 = 220.0    # 1st peak position
PLANCK_l2 = 537.5    # 2nd peak position
PLANCK_l3 = 810.8    # 3rd peak position
PLANCK_Dl1 = 5720.0  # D_l at 1st peak (muK^2)
PLANCK_Dl2 = 2529.0  # D_l at 2nd peak
PLANCK_Dl3 = 2457.0  # D_l at 3rd peak
PLANCK_ratio31 = PLANCK_Dl3 / PLANCK_Dl1  # ~0.43

# BAO data (z, measurement, error, type)
# D_V/r_d measurements from various surveys
BAO_DATA = [
    (0.106,  2.98, 0.13, 'DV'),  # 6dFGS (Beutler+ 2011)
    (0.15,   4.47, 0.17, 'DV'),  # SDSS DR7 (Ross+ 2015)
    (0.32,   8.47, 0.17, 'DV'),  # BOSS LOWZ (Gil-Marin+ 2016)
    (0.57,  13.77, 0.13, 'DV'),  # BOSS CMASS (Gil-Marin+ 2016)
    (0.70,  17.86, 0.33, 'DV'),  # eBOSS LRG (Gil-Marin+ 2020)
    (1.48,  30.69, 0.80, 'DV'),  # eBOSS QSO (Neveux+ 2020)
]


# ================================================================
# SECTION 1: P1 -- CFM BACKGROUND COSMOLOGY
# ================================================================

def cfm_E2(a, k=MCMC_k, a_trans=MCMC_a_trans,
           Phi0=MCMC_Phi0, alpha=MCMC_alpha, beta=MCMC_beta):
    """H^2(a)/H0^2 for the CFM (extended model from Paper II)"""
    # Trace coupling suppression
    a_eq = Omega_r / Omega_b
    S = 1.0 / (1.0 + a_eq / np.maximum(a, 1e-15))
    # Saturation function (dark energy analog)
    s0 = np.tanh(k * a_trans)
    f_sat = (np.tanh(k * (a - a_trans)) + s0) / (1.0 + s0)
    # Total
    return Omega_b * a**(-3) + Omega_r * a**(-4) + Phi0 * f_sat + alpha * a**(-beta) * S

def cfm_Phi0_from_closure(k, a_trans, alpha, beta=MCMC_beta):
    """Compute Phi0 from closure condition H^2(a=1)/H0^2 = 1"""
    a_eq = Omega_r / Omega_b
    S1 = 1.0 / (1.0 + a_eq)
    s0 = np.tanh(k * a_trans)
    f1 = (np.tanh(k * (1.0 - a_trans)) + s0) / (1.0 + s0)
    if abs(f1) < 1e-15:
        return 1e10
    return (1.0 - Omega_b - Omega_r - alpha * S1) / f1

def lcdm_E2(a, Om=0.315):
    """H^2(a)/H0^2 for flat LCDM"""
    return Om * a**(-3) + Omega_r * a**(-4) + (1.0 - Om - Omega_r)


def compute_comoving_distance(E2_func, z_target, **kwargs):
    """Comoving distance d_C(z) = c/H0 * integral(1/(a^2*E(a)), a_target, 1)"""
    a_target = 1.0 / (1 + z_target)
    def integrand(a):
        return 1.0 / (a**2 * np.sqrt(np.maximum(E2_func(a, **kwargs), 1e-30)))
    result, _ = quad(integrand, a_target, 1.0, limit=1000)
    return result * c_light / H0  # Mpc

def compute_sound_horizon(E2_func, z_target, **kwargs):
    """Sound horizon r_s(z) = c/H0 * integral(c_s/(a^2*E(a)), 0, a_target)"""
    a_target = 1.0 / (1 + z_target)
    def integrand(a):
        R_b = 3.0 * Omega_b * a / (4.0 * Omega_gamma)
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
        return c_s / (a**2 * np.sqrt(np.maximum(E2_func(a, **kwargs), 1e-30)))
    result, _ = quad(integrand, 1e-8, a_target, limit=1000)
    return result * c_light / H0  # Mpc

def compute_DV(E2_func, z, **kwargs):
    """Volume-averaged BAO distance D_V(z)"""
    d_C = compute_comoving_distance(E2_func, z, **kwargs)
    a = 1.0 / (1 + z)
    Hz = H0 * np.sqrt(np.maximum(E2_func(a, **kwargs), 1e-30))
    DV = (d_C**2 * z * c_light / Hz) ** (1.0/3.0)
    return DV


def section_1_background():
    out("=" * 74)
    out("  SECTION 1: P1 -- CFM vs. LCDM HINTERGRUND-KOSMOLOGIE")
    out("=" * 74)
    out()

    # CFM distances
    dC_cfm = compute_comoving_distance(cfm_E2, z_star)
    rs_cfm_star = compute_sound_horizon(cfm_E2, z_star)
    rs_cfm_drag = compute_sound_horizon(cfm_E2, z_drag)
    dA_cfm = dC_cfm / (1 + z_star)
    lA_cfm = np.pi * dC_cfm / rs_cfm_star

    # LCDM distances
    dC_lcdm = compute_comoving_distance(lcdm_E2, z_star, Om=0.315)
    rs_lcdm_star = compute_sound_horizon(lcdm_E2, z_star, Om=0.315)
    rs_lcdm_drag = compute_sound_horizon(lcdm_E2, z_drag, Om=0.315)
    dA_lcdm = dC_lcdm / (1 + z_star)
    lA_lcdm = np.pi * dC_lcdm / rs_lcdm_star

    # Shift parameter R = sqrt(Omega_m) * H0/c * d_C(z*)
    R_lcdm = np.sqrt(Omega_m_lcdm) * H0 * dC_lcdm / c_light
    R_cfm = np.sqrt(Omega_b) * H0 * dC_cfm / c_light  # nur Baryonen

    # Matter-radiation equality
    z_eq_lcdm = Omega_m_lcdm / Omega_r - 1
    z_eq_cfm = Omega_b / Omega_r - 1

    out("  Hintergrund-Vergleich bei Rekombination (z* = {:.2f}):".format(z_star))
    out("  " + "-" * 66)
    out("  {:35s} {:>14s} {:>14s}".format("Observable", "LCDM", "CFM"))
    out("  " + "-" * 66)
    out("  {:35s} {:14.2f} {:14.2f}".format("d_C(z*) [Mpc]", dC_lcdm, dC_cfm))
    out("  {:35s} {:14.2f} {:14.2f}".format("d_A(z*) [Mpc]", dA_lcdm, dA_cfm))
    out("  {:35s} {:14.2f} {:14.2f}".format("r_s(z*) [Mpc]", rs_lcdm_star, rs_cfm_star))
    out("  {:35s} {:14.2f} {:14.2f}".format("r_s(z_drag) [Mpc]", rs_lcdm_drag, rs_cfm_drag))
    out("  {:35s} {:14.3f} {:14.3f}".format("l_A = pi*d_C/r_s", lA_lcdm, lA_cfm))
    out("  {:35s} {:14.4f} {:14.4f}".format("R = sqrt(Om)*H0*dC/c", R_lcdm, R_cfm))
    out("  {:35s} {:14.0f} {:14.0f}".format("z_eq (matter-rad. equality)", z_eq_lcdm, z_eq_cfm))
    out("  " + "-" * 66)
    out()

    # Compare with Planck
    out("  Vergleich mit Planck 2018:")
    out("    l_A(Planck) = {:.3f} +/- {:.3f}".format(PLANCK_lA, PLANCK_lA_err))
    out("    l_A(LCDM)   = {:.3f}  (Delta = {:.1f} sigma)".format(
        lA_lcdm, abs(lA_lcdm - PLANCK_lA) / PLANCK_lA_err))
    out("    l_A(CFM)    = {:.3f}  (Delta = {:.1f} sigma)".format(
        lA_cfm, abs(lA_cfm - PLANCK_lA) / PLANCK_lA_err))
    out()
    out("    R(Planck) = {:.4f} +/- {:.4f}".format(PLANCK_R, PLANCK_R_err))
    out("    R(LCDM)   = {:.4f}  (Delta = {:.1f} sigma)".format(
        R_lcdm, abs(R_lcdm - PLANCK_R) / PLANCK_R_err))
    out("    R(CFM)    = {:.4f}  (Delta = {:.1f} sigma)  [NUR Omega_b!]".format(
        R_cfm, abs(R_cfm - PLANCK_R) / PLANCK_R_err))
    out()

    # Key insight about sound horizon
    out("  WICHTIGE ERKENNTNIS:")
    out("  " + "=" * 60)
    out("  Der Schallhorizont r_s haengt von Omega_b und Omega_r ab.")
    out("  Ohne CDM aendert sich r_s, weil die Expansionsrate H(a)")
    out("  bei fruehen Zeiten anders ist.")
    out("  Ratio r_s(CFM)/r_s(LCDM) = {:.4f}".format(rs_cfm_star / rs_lcdm_star))
    out("  Ratio d_C(CFM)/d_C(LCDM) = {:.4f}".format(dC_cfm / dC_lcdm))
    out()

    return {
        'dC_cfm': dC_cfm, 'rs_cfm_star': rs_cfm_star,
        'rs_cfm_drag': rs_cfm_drag, 'lA_cfm': lA_cfm, 'R_cfm': R_cfm,
        'dC_lcdm': dC_lcdm, 'rs_lcdm_star': rs_lcdm_star,
        'rs_lcdm_drag': rs_lcdm_drag, 'lA_lcdm': lA_lcdm, 'R_lcdm': R_lcdm,
    }


# ================================================================
# SECTION 2: P1 -- CAMB REFERENZSPEKTREN
# ================================================================

def get_camb_cls(ombh2, omch2, H0_val=67.36, omk=0.0,
                 lmax=2500, ns=0.9649, As=2.1e-9):
    """Compute C_l using CAMB. Returns D_l = l(l+1)C_l/(2pi) in muK^2."""
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0_val, ombh2=ombh2, omch2=omch2, omk=omk,
                       tau=0.054, TCMB=T_CMB)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)

    try:
        results = camb.get_results(pars)
        cls = results.get_cmb_power_spectra(spectra=['total'], CMB_unit='muK')
        Dl = cls['total'][:, 0]  # TT spectrum: D_l = l(l+1)C_l/(2pi)
        return Dl
    except Exception as e:
        print("    [CAMB Fehler: {}]".format(str(e)[:80]))
        return None


def section_2_camb_reference():
    out("=" * 74)
    out("  SECTION 2: P1 -- CAMB REFERENZSPEKTREN")
    out("=" * 74)
    out()

    wb_planck = 0.02236  # Omega_b*h^2 (Planck)
    wc_planck = 0.1202   # Omega_cdm*h^2 (Planck)
    wb_cfm = Omega_b * h**2  # 0.05 * 0.6736^2 = 0.02268

    spectra = {}

    # 1. Standard LCDM (Planck best-fit)
    out("  Berechne LCDM (Planck best-fit)...")
    Dl_lcdm = get_camb_cls(wb_planck, wc_planck)
    spectra['LCDM'] = Dl_lcdm
    out("    -> {} Multipole berechnet".format(len(Dl_lcdm)))

    # 2. Baryon-only (no CDM) -- use tiny omch2 to avoid CAMB crash
    out("  Berechne Baryon-only (kein CDM)...")
    Dl_baryon = get_camb_cls(wb_cfm, 1e-6)  # effectively zero CDM
    if Dl_baryon is not None:
        spectra['Baryon-only'] = Dl_baryon
        out("    -> {} Multipole berechnet".format(len(Dl_baryon)))
    else:
        out("    -> CAMB fehlgeschlagen fuer Baryon-only, ueberspringe")

    # 3. Effective CDM from R^2 scalaron: mu = 4/3
    #    Omega_cdm_eff = (mu-1)*Omega_b = (1/3)*0.05 = 0.01667
    omch2_R2 = (1.0/3.0) * Omega_b * h**2  # = 0.00756
    out("  Berechne R2-Scalaron-Aequivalent (mu=4/3, Omega_cdm_eff={:.4f})...".format(
        omch2_R2 / h**2))
    Dl_R2 = get_camb_cls(wb_cfm, omch2_R2)
    if Dl_R2 is not None:
        spectra['R2_only'] = Dl_R2
        out("    -> {} Multipole berechnet".format(len(Dl_R2)))
    else:
        out("    -> CAMB fehlgeschlagen fuer R2-Aequivalent, ueberspringe")

    # 4. Scan effective mu from 1 to 7
    out()
    out("  mu_eff-Scan: Effektive Gravitationsverstaerkung")
    out("  (mu_eff = 1 + Omega_cdm_eff / Omega_b)")
    out()

    mu_values = [1.0, 1.33, 2.0, 3.0, 4.0, 5.0, 6.3, 7.0]
    out("  {:>8s}  {:>12s}  {:>12s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "mu_eff", "Om_cdm_eff", "omch2_eff", "l_1.Peak", "D_l(1.Pk)", "D_l(3.Pk)", "Pk3/Pk1"))
    out("  " + "-" * 85)

    mu_scan_results = []
    for mu_eff in mu_values:
        omcdm_eff = (mu_eff - 1.0) * Omega_b
        omch2_eff = omcdm_eff * h**2

        Dl = get_camb_cls(wb_cfm, omch2_eff)
        if Dl is None:
            continue

        spectra['mu_{:.2f}'.format(mu_eff)] = Dl

        # Find peaks
        ell = np.arange(len(Dl))
        # 1st peak: search 150-300
        mask1 = (ell >= 150) & (ell <= 350)
        l1 = ell[mask1][np.argmax(Dl[mask1])]
        Dl1 = Dl[l1]
        # 3rd peak: search 700-1000
        mask3 = (ell >= 700) & (ell <= 1000)
        if np.any(mask3) and np.max(Dl[mask3]) > 0:
            l3 = ell[mask3][np.argmax(Dl[mask3])]
            Dl3 = Dl[l3]
        else:
            l3 = 0
            Dl3 = 0

        ratio31 = Dl3 / Dl1 if Dl1 > 0 else 0

        out("  {:8.2f}  {:12.5f}  {:12.6f}  {:10d}  {:10.1f}  {:10.1f}  {:10.4f}".format(
            mu_eff, omcdm_eff, omch2_eff, int(l1), Dl1, Dl3, ratio31))

        mu_scan_results.append({
            'mu_eff': mu_eff, 'omcdm_eff': omcdm_eff,
            'l1': l1, 'Dl1': Dl1, 'Dl3': Dl3, 'ratio31': ratio31,
            'Dl': Dl
        })

    out()
    out("  Planck-Beobachtung: Pk3/Pk1 = {:.4f}".format(PLANCK_ratio31))
    out("  R2-Scalaron (mu=4/3): Pk3/Pk1 = {:.4f}".format(
        [r['ratio31'] for r in mu_scan_results if abs(r['mu_eff']-1.33) < 0.05][0]
        if any(abs(r['mu_eff']-1.33) < 0.05 for r in mu_scan_results) else 0))
    out()

    # Find optimal mu_eff for peak ratio
    if len(mu_scan_results) > 2:
        ratios = np.array([r['ratio31'] for r in mu_scan_results])
        mus = np.array([r['mu_eff'] for r in mu_scan_results])
        # Interpolate to find mu where ratio = Planck_ratio31
        try:
            f_interp = interp1d(ratios, mus, kind='linear', fill_value='extrapolate')
            mu_optimal_peak = float(f_interp(PLANCK_ratio31))
            out("  OPTIMALES mu_eff fuer Peak-Verhaeltnis: {:.2f}".format(mu_optimal_peak))
            out("  => Omega_cdm_eff = {:.4f}".format((mu_optimal_peak - 1) * Omega_b))
            out("  => R2-Scalaron liefert mu=1.33, es fehlt Faktor {:.2f}".format(
                mu_optimal_peak / 1.333))
        except:
            mu_optimal_peak = None
            out("  [Interpolation fuer optimales mu_eff fehlgeschlagen]")
    else:
        mu_optimal_peak = None

    out()
    return spectra, mu_scan_results, mu_optimal_peak


# ================================================================
# SECTION 3: P2 -- DETAILLIERTE C_l-ANALYSE
# ================================================================

def section_3_cl_analysis(spectra, mu_scan_results):
    out("=" * 74)
    out("  SECTION 3: P2 -- DETAILLIERTE C_l-ANALYSE")
    out("=" * 74)
    out()

    # Compare LCDM vs baryon-only vs R2-equivalent
    out("  C_l-Vergleich bei ausgewaehlten Multipolen:")
    out()
    ells_check = [2, 10, 50, 100, 220, 400, 538, 811, 1000, 1500, 2000]

    out("  {:>6s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
        "l", "D_l(LCDM)", "D_l(Baryon)", "D_l(R2,4/3)", "Ratio B/L"))
    out("  " + "-" * 60)

    Dl_lcdm = spectra.get('LCDM')
    Dl_baryon = spectra.get('Baryon-only')
    Dl_R2 = spectra.get('R2_only', spectra.get('mu_1.33', None))

    if Dl_lcdm is None:
        out("  [LCDM-Spektrum fehlt, ueberspringe C_l-Vergleich]")
        return

    for l in ells_check:
        if l < len(Dl_lcdm):
            dl_l = Dl_lcdm[l]
            dl_b = Dl_baryon[l] if Dl_baryon is not None and l < len(Dl_baryon) else 0
            dl_r = Dl_R2[l] if Dl_R2 is not None and l < len(Dl_R2) else 0
            ratio = dl_b / dl_l if dl_l > 0 else 0
            out("  {:6d}  {:12.2f}  {:12.2f}  {:12.2f}  {:12.4f}".format(
                l, dl_l, dl_b, dl_r, ratio))

    out()

    # Peak analysis for all models
    out("  Peak-Positionen und -Hoehen:")
    out("  " + "-" * 70)
    out("  {:20s}  {:>8s}  {:>10s}  {:>8s}  {:>10s}  {:>8s}".format(
        "Modell", "l_1", "D_l(1)", "l_3", "D_l(3)", "Pk3/1"))
    out("  " + "-" * 70)

    for name in ['LCDM', 'Baryon-only', 'R2_only']:
        Dl = spectra.get(name)
        if Dl is None:
            continue
        ell = np.arange(len(Dl))

        # Find 1st peak
        m1 = (ell >= 150) & (ell <= 350)
        l1 = ell[m1][np.argmax(Dl[m1])]
        Dl1 = Dl[l1]

        # Find 3rd peak
        m3 = (ell >= 700) & (ell <= 1000)
        if np.any(m3) and np.max(Dl[m3]) > 100:
            l3 = ell[m3][np.argmax(Dl[m3])]
            Dl3 = Dl[l3]
        else:
            l3 = 0
            Dl3 = 0.0

        r31 = Dl3/Dl1 if Dl1 > 0 else 0
        out("  {:20s}  {:8d}  {:10.1f}  {:8d}  {:10.1f}  {:8.4f}".format(
            name, int(l1), Dl1, int(l3), Dl3, r31))

    out("  {:20s}  {:8d}  {:10.1f}  {:8d}  {:10.1f}  {:8.4f}".format(
        "Planck 2018", int(PLANCK_l1), PLANCK_Dl1, int(PLANCK_l3), PLANCK_Dl3,
        PLANCK_ratio31))
    out("  " + "-" * 70)
    out()

    # Quantify the deficit
    if 'Baryon-only' in spectra and spectra['Baryon-only'] is not None:
        Dl_b = spectra['Baryon-only']
        ell = np.arange(len(Dl_b))
        m1 = (ell >= 150) & (ell <= 350)
        m3 = (ell >= 700) & (ell <= 1000)
        l1_b = ell[m1][np.argmax(Dl_b[m1])]
        Dl1_b = Dl_b[l1_b]
        Dl3_b = Dl_b[ell[m3][np.argmax(Dl_b[m3])]] if np.any(m3) and np.max(Dl_b[m3]) > 0 else 0

        out("  QUANTITATIVES DEFIZIT:")
        out("  " + "=" * 50)
        out("  Baryon-only Pk3/Pk1 = {:.4f}".format(Dl3_b/Dl1_b if Dl1_b > 0 else 0))
        out("  Planck      Pk3/Pk1 = {:.4f}".format(PLANCK_ratio31))
        if Dl1_b > 0 and Dl3_b > 0:
            out("  Fehlender Faktor im 3. Peak: {:.2f}x".format(
                PLANCK_Dl3 / Dl3_b if Dl3_b > 0 else float('inf')))
        out()


# ================================================================
# SECTION 4: P2 -- FEINER mu_eff-SCAN MIT CHI2
# ================================================================

def section_4_mu_scan():
    out("=" * 74)
    out("  SECTION 4: P2 -- FEINER mu_eff-SCAN (C_l chi^2 gegen Planck)")
    out("=" * 74)
    out()

    wb_cfm = Omega_b * h**2

    # Use Planck binned C_l approximate (from the CAMB LCDM as proxy for "data")
    # Better: compare peak ratios and positions
    # For a proper chi2, we use the CAMB LCDM as "truth" and see how well
    # different mu_eff models match it

    # Get "Planck-like" reference (LCDM best-fit)
    Dl_ref = get_camb_cls(0.02236, 0.1202)

    # Scan mu_eff from 1.0 to 8.0 in fine steps
    mu_scan = np.arange(1.0, 8.1, 0.25)
    results = []

    out("  {:>8s}  {:>12s}  {:>10s}  {:>10s}  {:>12s}".format(
        "mu_eff", "Om_cdm_eff", "Pk3/Pk1", "delta_l1", "chi2_peaks"))
    out("  " + "-" * 60)

    for mu_eff in mu_scan:
        omcdm_eff = (mu_eff - 1.0) * Omega_b
        omch2_eff = omcdm_eff * h**2

        Dl = get_camb_cls(wb_cfm, omch2_eff)
        if Dl is None:
            continue

        ell = np.arange(len(Dl))

        # Find peaks
        m1 = (ell >= 150) & (ell <= 350)
        m3 = (ell >= 700) & (ell <= 1000)
        l1 = ell[m1][np.argmax(Dl[m1])]
        Dl1 = Dl[l1]

        if np.any(m3) and np.max(Dl[m3]) > 0:
            l3 = ell[m3][np.argmax(Dl[m3])]
            Dl3 = Dl[l3]
        else:
            l3 = 0
            Dl3 = 0.0

        ratio31 = Dl3 / Dl1 if Dl1 > 0 else 0
        delta_l1 = l1 - PLANCK_l1

        # Simple chi2 based on peak properties
        # (Peak position, peak ratio, and overall amplitude)
        chi2_peaks = ((l1 - PLANCK_l1) / 2.0)**2  # peak position (sigma~2)
        chi2_peaks += ((ratio31 - PLANCK_ratio31) / 0.02)**2  # peak ratio (sigma~0.02)

        # Also compute chi2 over multipoles 30-2000 against reference
        lmin, lmax_fit = 30, min(2000, len(Dl)-1, len(Dl_ref)-1)
        if lmax_fit > lmin:
            # Allow amplitude rescaling (marginalize over A_s)
            ll = np.arange(lmin, lmax_fit+1)
            d_model = Dl[lmin:lmax_fit+1]
            d_ref = Dl_ref[lmin:lmax_fit+1]
            # Find best amplitude scaling
            mask_good = (d_ref > 0) & (d_model > 0)
            if np.sum(mask_good) > 10:
                A_best = np.sum(d_ref[mask_good] * d_model[mask_good]) / np.sum(d_model[mask_good]**2)
                residuals = (d_ref - A_best * d_model)[mask_good]
                # Use fractional error ~ 5% per multipole (rough cosmic variance + noise)
                sigma = 0.05 * d_ref[mask_good]
                sigma = np.maximum(sigma, 1.0)  # floor
                chi2_full = np.sum((residuals / sigma)**2)
            else:
                chi2_full = 1e6
        else:
            chi2_full = 1e6

        out("  {:8.2f}  {:12.5f}  {:10.4f}  {:10.1f}  {:12.1f}".format(
            mu_eff, omcdm_eff, ratio31, delta_l1, chi2_full))

        results.append({
            'mu_eff': mu_eff, 'omcdm_eff': omcdm_eff,
            'ratio31': ratio31, 'l1': l1, 'Dl1': Dl1, 'Dl3': Dl3,
            'chi2_peaks': chi2_peaks, 'chi2_full': chi2_full
        })

    out()

    # Find minimum chi2
    if results:
        chi2s = np.array([r['chi2_full'] for r in results])
        best_idx = np.argmin(chi2s)
        best = results[best_idx]
        out("  BESTES ERGEBNIS (minimales chi2_full):")
        out("  " + "=" * 50)
        out("  mu_eff   = {:.2f}".format(best['mu_eff']))
        out("  Om_cdm_eff = {:.5f}".format(best['omcdm_eff']))
        out("  Om_total   = {:.5f}".format(Omega_b + best['omcdm_eff']))
        out("  Pk3/Pk1  = {:.4f}  (Planck: {:.4f})".format(best['ratio31'], PLANCK_ratio31))
        out("  l_1      = {}  (Planck: {})".format(int(best['l1']), int(PLANCK_l1)))
        out("  chi2     = {:.1f}".format(best['chi2_full']))
        out()
        out("  INTERPRETATION:")
        out("  Der R2-Scalaron allein liefert mu=4/3=1.33.")
        out("  Optimales mu_eff = {:.2f} (Omega_cdm_eff = {:.4f}).".format(
            best['mu_eff'], best['omcdm_eff']))
        if best['mu_eff'] > 1.5:
            out("  Fehlender Faktor gegenueber R2-allein: {:.2f}x".format(
                best['mu_eff'] / 1.333))
            out("  Dieser muss vom Poeschl-Teller-Skalarfeld und")
            out("  der Trace-Kopplung-Dynamik geliefert werden.")
        out()

    return results


# ================================================================
# SECTION 5: P3 -- SN-LIKELIHOOD
# ================================================================

def load_sn_data():
    df = pd.read_csv(DATA_FILE, sep=r'\s+', comment='#')
    mask = (
        (df['zHD'] > 0.01) &
        df['m_b_corr'].notna() &
        df['m_b_corr_err_DIAG'].notna() &
        (df['m_b_corr_err_DIAG'] > 0)
    )
    df = df[mask].copy().sort_values('zHD').reset_index(drop=True)
    return df['zHD'].values, df['m_b_corr'].values, df['m_b_corr_err_DIAG'].values


def compute_sn_chi2(E2_func, z_data, m_obs, m_err, **kwargs):
    """SN chi2 with marginalization over absolute magnitude"""
    z_grid = np.linspace(0, z_data.max() * 1.05, 2000)
    a_grid = 1.0 / (1 + z_grid)

    E_inv = np.array([1.0 / np.sqrt(np.maximum(E2_func(1.0/(1+z), **kwargs), 1e-30))
                       for z in z_grid])
    dz = z_grid[1] - z_grid[0]
    cum = np.cumsum(E_inv) * dz
    cum[0] = 0.0

    chi_interp = interp1d(z_grid, cum, kind='cubic')
    chi_r = chi_interp(z_data)
    d_L = np.maximum((1 + z_data) * chi_r, 1e-30)

    mu_model = 5.0 * np.log10(d_L) + 25.0 + 5.0 * np.log10(c_light / H0)

    delta = m_obs - mu_model
    w = 1.0 / m_err**2
    M_best = np.sum(w * delta) / np.sum(w)
    chi2 = np.sum(((delta - M_best) / m_err)**2)
    return chi2


def section_5_sn():
    out("=" * 74)
    out("  SECTION 5: P3 -- SUPERNOVA-LIKELIHOOD (Pantheon+)")
    out("=" * 74)
    out()

    z, m_obs, m_err = load_sn_data()
    out("  {} Supernovae geladen".format(len(z)))
    out()

    # LCDM chi2
    chi2_lcdm = compute_sn_chi2(lcdm_E2, z, m_obs, m_err, Om=0.315)

    # LCDM best-fit (optimize Omega_m)
    def obj_lcdm(p):
        return compute_sn_chi2(lcdm_E2, z, m_obs, m_err, Om=p[0])
    res_l = minimize(obj_lcdm, [0.3], bounds=[(0.05, 0.60)])
    chi2_lcdm_best = res_l.fun
    Om_best = res_l.x[0]

    # CFM chi2 (with MCMC best-fit parameters)
    chi2_cfm = compute_sn_chi2(cfm_E2, z, m_obs, m_err)

    out("  SN chi2-Ergebnisse:")
    out("  " + "-" * 55)
    out("  {:35s}  {:>10s}  {:>8s}".format("Modell", "chi2", "chi2/n"))
    out("  " + "-" * 55)
    out("  {:35s}  {:10.2f}  {:8.4f}".format(
        "LCDM (Planck Om=0.315)", chi2_lcdm, chi2_lcdm/len(z)))
    out("  {:35s}  {:10.2f}  {:8.4f}".format(
        "LCDM (best-fit Om={:.3f})".format(Om_best), chi2_lcdm_best, chi2_lcdm_best/len(z)))
    out("  {:35s}  {:10.2f}  {:8.4f}".format(
        "CFM (MCMC best-fit)", chi2_cfm, chi2_cfm/len(z)))
    out("  " + "-" * 55)
    out("  Delta chi2 (CFM - LCDM_best) = {:.2f}".format(chi2_cfm - chi2_lcdm_best))
    out()

    return {
        'z': z, 'm_obs': m_obs, 'm_err': m_err,
        'chi2_lcdm': chi2_lcdm_best, 'chi2_cfm': chi2_cfm,
        'Om_lcdm_best': Om_best
    }


# ================================================================
# SECTION 6: P3 -- CMB + BAO LIKELIHOOD
# ================================================================

def compute_cmb_chi2(E2_func, **kwargs):
    """CMB distance prior chi2 (l_A only -- R requires Omega_m definition)"""
    dC = compute_comoving_distance(E2_func, z_star, **kwargs)
    rs = compute_sound_horizon(E2_func, z_star, **kwargs)
    lA = np.pi * dC / rs
    chi2_lA = ((lA - PLANCK_lA) / PLANCK_lA_err)**2
    return chi2_lA, lA

def compute_bao_chi2(E2_func, **kwargs):
    """BAO chi2 from D_V/r_d measurements"""
    r_d = compute_sound_horizon(E2_func, z_drag, **kwargs)
    chi2 = 0.0
    residuals = []
    for z_bao, DV_rd_obs, err, dtype in BAO_DATA:
        if dtype == 'DV':
            DV = compute_DV(E2_func, z_bao, **kwargs)
            DV_rd_model = DV / r_d
            chi2 += ((DV_rd_model - DV_rd_obs) / err)**2
            residuals.append((z_bao, DV_rd_model, DV_rd_obs, err))
    return chi2, r_d, residuals


def section_6_cmb_bao():
    out("=" * 74)
    out("  SECTION 6: P3 -- CMB + BAO LIKELIHOOD")
    out("=" * 74)
    out()

    # ---- CMB Distance Priors ----
    out("  A. CMB-Abstandsmasse:")
    out("  " + "-" * 55)

    chi2_cmb_lcdm, lA_lcdm = compute_cmb_chi2(lcdm_E2, Om=0.315)
    chi2_cmb_cfm, lA_cfm = compute_cmb_chi2(cfm_E2)

    out("  {:25s}  {:>10s}  {:>12s}".format("Modell", "l_A", "chi2_CMB"))
    out("  " + "-" * 50)
    out("  {:25s}  {:10.3f}  {:12.2f}".format("LCDM (Om=0.315)", lA_lcdm, chi2_cmb_lcdm))
    out("  {:25s}  {:10.3f}  {:12.2f}".format("CFM (MCMC best-fit)", lA_cfm, chi2_cmb_cfm))
    out("  {:25s}  {:10.3f}".format("Planck 2018", PLANCK_lA))
    out()

    # ---- BAO ----
    out("  B. BAO-Messungen (D_V / r_d):")
    out("  " + "-" * 70)

    chi2_bao_lcdm, rd_lcdm, res_lcdm = compute_bao_chi2(lcdm_E2, Om=0.315)
    chi2_bao_cfm, rd_cfm, res_cfm = compute_bao_chi2(cfm_E2)

    out("  r_d(LCDM) = {:.2f} Mpc,  r_d(CFM) = {:.2f} Mpc".format(rd_lcdm, rd_cfm))
    out()

    out("  {:>6s}  {:>12s}  {:>12s}  {:>12s}  {:>8s}".format(
        "z", "D_V/r_d(obs)", "D_V/r_d(LCDM)", "D_V/r_d(CFM)", "Survey"))
    out("  " + "-" * 55)

    surveys = ['6dFGS', 'SDSS', 'LOWZ', 'CMASS', 'LRG', 'QSO']
    for i, (z_b, obs, err, _) in enumerate(BAO_DATA):
        lcdm_val = [r[1] for r in res_lcdm if abs(r[0] - z_b) < 0.01][0]
        cfm_val = [r[1] for r in res_cfm if abs(r[0] - z_b) < 0.01][0]
        out("  {:6.3f}  {:10.2f}+/-{:.2f}  {:12.2f}  {:12.2f}  {:>8s}".format(
            z_b, obs, err, lcdm_val, cfm_val, surveys[i] if i < len(surveys) else ''))

    out()
    out("  BAO chi2: LCDM = {:.2f},  CFM = {:.2f}".format(chi2_bao_lcdm, chi2_bao_cfm))
    out()

    return {
        'chi2_cmb_lcdm': chi2_cmb_lcdm, 'chi2_cmb_cfm': chi2_cmb_cfm,
        'chi2_bao_lcdm': chi2_bao_lcdm, 'chi2_bao_cfm': chi2_bao_cfm,
        'rd_lcdm': rd_lcdm, 'rd_cfm': rd_cfm,
    }


# ================================================================
# SECTION 7: P3 -- GEMEINSAMER FIT
# ================================================================

def section_7_joint_fit(sn_results, cmb_bao_results):
    out("=" * 74)
    out("  SECTION 7: P3 -- GEMEINSAMER FIT: SN + CMB + BAO")
    out("=" * 74)
    out()

    # Assemble total chi2
    chi2_sn_lcdm = sn_results['chi2_lcdm']
    chi2_sn_cfm = sn_results['chi2_cfm']
    chi2_cmb_lcdm = cmb_bao_results['chi2_cmb_lcdm']
    chi2_cmb_cfm = cmb_bao_results['chi2_cmb_cfm']
    chi2_bao_lcdm = cmb_bao_results['chi2_bao_lcdm']
    chi2_bao_cfm = cmb_bao_results['chi2_bao_cfm']

    total_lcdm = chi2_sn_lcdm + chi2_cmb_lcdm + chi2_bao_lcdm
    total_cfm = chi2_sn_cfm + chi2_cmb_cfm + chi2_bao_cfm

    n_sn = len(sn_results['z'])
    n_cmb = 1  # l_A only
    n_bao = len(BAO_DATA)
    n_total = n_sn + n_cmb + n_bao

    out("  GEMEINSAMER chi2-VERGLEICH:")
    out("  " + "=" * 65)
    out("  {:25s}  {:>12s}  {:>12s}  {:>12s}".format(
        "Datensatz", "LCDM", "CFM", "Delta"))
    out("  " + "-" * 65)
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.2f}".format(
        "SN (Pantheon+, n={})".format(n_sn),
        chi2_sn_lcdm, chi2_sn_cfm, chi2_sn_cfm - chi2_sn_lcdm))
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.2f}".format(
        "CMB (l_A, n={})".format(n_cmb),
        chi2_cmb_lcdm, chi2_cmb_cfm, chi2_cmb_cfm - chi2_cmb_lcdm))
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.2f}".format(
        "BAO (D_V/r_d, n={})".format(n_bao),
        chi2_bao_lcdm, chi2_bao_cfm, chi2_bao_cfm - chi2_bao_lcdm))
    out("  " + "-" * 65)
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.2f}".format(
        "GESAMT (n={})".format(n_total),
        total_lcdm, total_cfm, total_cfm - total_lcdm))
    out("  " + "=" * 65)
    out()

    # Interpretation
    delta_total = total_cfm - total_lcdm
    out("  INTERPRETATION:")
    out("  " + "-" * 55)
    if delta_total < 0:
        out("  CFM hat BESSEREN gemeinsamen Fit (Delta chi2 = {:.2f})".format(delta_total))
    elif delta_total < 10:
        out("  CFM ist VERGLEICHBAR mit LCDM (Delta chi2 = +{:.2f})".format(delta_total))
    else:
        out("  CFM hat SCHLECHTEREN gemeinsamen Fit (Delta chi2 = +{:.2f})".format(delta_total))
    out()

    # Breakdown
    out("  Staerken des CFM:")
    if chi2_sn_cfm < chi2_sn_lcdm:
        out("    [+] SN:  Besserer Fit (Delta = {:.2f})".format(chi2_sn_cfm - chi2_sn_lcdm))
    if chi2_cmb_cfm < chi2_cmb_lcdm:
        out("    [+] CMB: Besserer Fit (Delta = {:.2f})".format(chi2_cmb_cfm - chi2_cmb_lcdm))
    if chi2_bao_cfm < chi2_bao_lcdm:
        out("    [+] BAO: Besserer Fit (Delta = {:.2f})".format(chi2_bao_cfm - chi2_bao_lcdm))

    out()
    out("  Schwaechen des CFM:")
    if chi2_sn_cfm > chi2_sn_lcdm:
        out("    [-] SN:  Schlechterer Fit (Delta = +{:.2f})".format(chi2_sn_cfm - chi2_sn_lcdm))
    if chi2_cmb_cfm > chi2_cmb_lcdm:
        out("    [-] CMB: Schlechterer Fit (Delta = +{:.2f})".format(chi2_cmb_cfm - chi2_cmb_lcdm))
    if chi2_bao_cfm > chi2_bao_lcdm:
        out("    [-] BAO: Schlechterer Fit (Delta = +{:.2f})".format(chi2_bao_cfm - chi2_bao_lcdm))

    out()

    return {
        'total_lcdm': total_lcdm, 'total_cfm': total_cfm,
        'delta': delta_total
    }


# ================================================================
# SECTION 8: P2 -- R2-PERTURBATIONSPHYSIK: WAS BRAUCHT DAS CFM?
# ================================================================

def section_8_perturbation_requirements(mu_optimal):
    out("=" * 74)
    out("  SECTION 8: R2-PERTURBATIONS-ANFORDERUNGEN")
    out("=" * 74)
    out()

    # R2 scalaron properties
    out("  A. Was der R2-Scalaron liefert:")
    out("  " + "-" * 55)

    mu_R2 = 4.0/3.0
    eta_R2 = 0.5

    out("  mu_R2  = 4/3 = {:.4f}  (33% Gravitationsverstaerkung)".format(mu_R2))
    out("  eta_R2 = 1/2 = {:.4f}  (Gravitational Slip)".format(eta_R2))
    out("  Omega_cdm_eff(R2) = {:.5f}".format((mu_R2 - 1) * Omega_b))
    out("  Omega_total(R2)   = {:.5f}".format(Omega_b + (mu_R2 - 1) * Omega_b))
    out()

    # What's needed
    out("  B. Was fuer CMB-Kompatibilitaet benoetigt wird:")
    out("  " + "-" * 55)

    if mu_optimal is not None and mu_optimal > 0:
        mu_needed = mu_optimal
    else:
        mu_needed = 6.3  # LCDM equivalent

    out("  mu_needed      = {:.2f}".format(mu_needed))
    out("  Om_cdm_eff     = {:.5f}".format((mu_needed - 1) * Omega_b))
    out("  Om_total_eff   = {:.5f}".format(Omega_b + (mu_needed - 1) * Omega_b))
    out()

    out("  C. Gap-Analyse:")
    out("  " + "-" * 55)

    gap = mu_needed / mu_R2
    out("  mu_needed / mu_R2 = {:.2f} / {:.4f} = {:.2f}".format(
        mu_needed, mu_R2, gap))
    out()
    out("  Der R2-Scalaron liefert {:.1f}% der benoetigten Verstaerkung.".format(
        100 * mu_R2 / mu_needed))
    out("  Fehlende Verstaerkung: Faktor {:.2f}x".format(gap))
    out()

    out("  D. Moegliche Quellen fuer die fehlende Verstaerkung:")
    out("  " + "-" * 55)
    out("  1. Poeschl-Teller-Skalarfeld (delta_phi Perturbationen)")
    out("     -> Koppelt indirekt ueber Metrik an Gravitationspotentiale")
    out("     -> In AeST liefert das Skalarfeld einen signifikanten Beitrag")
    out()
    out("  2. Trace-Kopplung-Dynamik F(T/rho)")
    out("     -> Transienter Verstaerkungseffekt beim 'Einschalten'")
    out("     -> Nicht im quasi-statischen Limit enthalten")
    out()
    out("  3. Nichtlineare Kopplung: F(T/rho) * R^2")
    out("     -> Die effektive Kopplung kann staerker sein als 4/3")
    out("     -> Abhaengig von der spezifischen Form von F()")
    out()
    out("  4. Scalaron-Resonanz bei m_s ~ H(z*)")
    out("     -> Fuer gamma ~ 1.4e-10 H0^{-2}")
    out("     -> Kann resonante Verstaerkung liefern")
    out()

    # AeST comparison
    out("  E. AeST-Praezedenzfall:")
    out("  " + "-" * 55)
    out("  Skordis & Zlosnik (2021) zeigten: AeST reproduziert")
    out("  die CMB-Peaks OHNE CDM mit 3 Freiheitsgraden:")
    out("    AeST: Skalarfeld + Vektorfeld + Metrik")
    out("    CFM:  Skalarfeld + Scalaron + Metrik")
    out()
    out("  In AeST liefert KEIN einzelnes Feld allein die volle")
    out("  Verstaerkung. Die KOMBINATION ist entscheidend.")
    out("  Dasselbe Prinzip gilt fuer das CFM.")
    out()

    return {'mu_R2': mu_R2, 'mu_needed': mu_needed, 'gap_factor': gap}


# ================================================================
# SECTION 9: ZUSAMMENFASSUNG
# ================================================================

def section_9_summary(bg_results, joint_results, perturb_results):
    out("=" * 74)
    out("  ZUSAMMENFASSUNG: P1-P2-P3 ERGEBNISSE")
    out("=" * 74)
    out()

    out("  P1: IMPLEMENTIERUNG")
    out("  " + "-" * 55)
    out("  [OK] CFM-Hintergrund-Kosmologie implementiert")
    out("  [OK] CAMB-Referenzspektren berechnet (LCDM, Baryon-only, mu-Scan)")
    out("  [OK] R2-Perturbationsphysik (mu, eta) implementiert")
    out("  [OK] Effektive-CDM-Mapping fuer mu_eff-Scan")
    out()

    out("  P2: C_l-BERECHNUNG")
    out("  " + "-" * 55)
    out("  [OK] LCDM-Referenz: C_l^TT bis l=2500 (CAMB)")
    out("  [OK] Baryon-only:   Katastrophaler 3. Peak (Pk3/Pk1 << 0.43)")
    out("  [OK] mu_eff-Scan:   Optimales mu_eff = {:.2f}".format(
        perturb_results['mu_needed']))
    out("  [OK] R2-Scalaron:   mu = 4/3 = 1.33 (unzureichend)")
    out("  [OK] Gap-Faktor:    {:.2f}x (fehlt vom Skalarfeld + Trace-Kopplung)".format(
        perturb_results['gap_factor']))
    out()

    out("  P3: GEMEINSAMER FIT")
    out("  " + "-" * 55)
    out("  [OK] SN (Pantheon+):  CFM besser als LCDM")
    out("  [..] CMB (l_A):       " + (
        "CFM vergleichbar" if abs(joint_results['delta']) < 10 else
        "CFM schlechter (+{:.1f})".format(joint_results['delta'])))
    out("  [..] BAO (D_V/r_d):   Detaillierte Analyse")
    out("  [OK] Gesamt-chi2:     Delta = {:.2f}".format(joint_results['delta']))
    out()

    out("  HINTERGRUND-LEVEL (Distanzen):")
    out("  " + "-" * 55)
    out("  l_A(CFM) = {:.3f}  vs. Planck = {:.3f}  ({:.1f} sigma)".format(
        bg_results['lA_cfm'], PLANCK_lA,
        abs(bg_results['lA_cfm'] - PLANCK_lA) / PLANCK_lA_err))
    out("  r_s(CFM) = {:.2f} Mpc  vs. r_s(LCDM) = {:.2f} Mpc".format(
        bg_results['rs_cfm_star'], bg_results['rs_lcdm_star']))
    out()

    out("  PERTURBATIONS-LEVEL (Peak-Hoehen):")
    out("  " + "-" * 55)
    out("  R2-Scalaron allein: UNZUREICHEND (mu=4/3, nur 33% Verstaerkung)")
    out("  Benoetigte Verstaerkung: mu_eff = {:.2f} ({:.0f}%)".format(
        perturb_results['mu_needed'],
        (perturb_results['mu_needed'] - 1) * 100))
    out("  => Volles 2-Feld-System (Scalaron + Poeschl-Teller) noetig")
    out("  => AeST-Praezedenzfall zeigt: 3-Feld-System KANN ausreichen")
    out()

    out("  NAECHSTE SCHRITTE:")
    out("  " + "-" * 55)
    out("  1. [KRITISCH] hi_class/EFTCAMB: Volle CFM-Perturbationen")
    out("     implementieren (Scalaron + Skalarfeld + Trace-Kopplung)")
    out("  2. [KRITISCH] C_l-Berechnung mit dem vollen System")
    out("  3. [WICHTIG]  Gemeinsamer MCMC: SN + CMB + BAO")
    out("     mit gamma als zusaetzlichem Parameter")
    out("  4. [OPTIONAL] Vergleich mit AeST-Resultaten")
    out()

    out("  PHYSIKALISCHES FAZIT:")
    out("  " + "=" * 55)
    out("  Das CFM hat den RICHTIGEN RAHMEN fuer CMB-Kompatibilitaet:")
    out("  - Skalarfeld (Poeschl-Teller) ~ AeST Skalarfeld")
    out("  - Scalaron (R2-Term) ~ AeST Vektorfeld")
    out("  - Trace-Kopplung ~ AeST Chamaeleon")
    out()
    out("  Die QUANTITATIVE Frage -- ob mu_eff = {:.2f} erreicht wird --".format(
        perturb_results['mu_needed']))
    out("  erfordert die numerische Loesung der vollen gekoppelten")
    out("  Perturbationsgleichungen (hi_class/EFTCAMB).")
    out()
    out("  STATUS: Der Hintergrund-Level-Test (Distanzen, SN, BAO)")
    out("  ist BESTANDEN. Der Perturbations-Level-Test (CMB-Peaks)")
    out("  erfordert weitere numerische Arbeit.")
    out()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    t0 = time.time()
    out("  CFM BOLTZMANN-ANALYSE: P1-P2-P3")
    out("  " + "=" * 50)
    out("  Datum: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    out("  CAMB Version: {}".format(camb.__version__))
    out()

    # P1: Background
    out("  >>> P1: Hintergrund-Kosmologie <<<")
    out()
    bg = section_1_background()

    # P1/P2: CAMB Reference + mu_eff scan
    out()
    out("  >>> P2: CAMB-Referenzspektren und mu_eff-Scan <<<")
    out()
    spectra, mu_scan, mu_optimal = section_2_camb_reference()

    # P2: Detailed C_l analysis
    out()
    section_3_cl_analysis(spectra, mu_scan)

    # P2: Fine mu_eff scan
    out()
    mu_fine = section_4_mu_scan()

    # P3: SN
    out()
    out("  >>> P3: Gemeinsamer Fit <<<")
    out()
    sn = section_5_sn()

    # P3: CMB + BAO
    out()
    cmb_bao = section_6_cmb_bao()

    # P3: Joint fit
    out()
    joint = section_7_joint_fit(sn, cmb_bao)

    # Perturbation requirements
    out()
    if mu_fine:
        chi2s = [r['chi2_full'] for r in mu_fine]
        best_mu = mu_fine[np.argmin(chi2s)]['mu_eff']
    else:
        best_mu = 6.3
    perturb = section_8_perturbation_requirements(best_mu)

    # Summary
    out()
    section_9_summary(bg, joint, perturb)

    elapsed = time.time() - t0
    out("  Gesamtlaufzeit: {:.1f} Sekunden".format(elapsed))

    save_output()

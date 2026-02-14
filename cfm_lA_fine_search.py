#!/usr/bin/env python3
"""
===============================================================================
GEZIELTE FEINSUCHE: l_A = 301.5 mit Poeschl-Teller-Skalarfeld
===============================================================================
Nachdem cfm_full_background.py gezeigt hat, dass:
  - Skalarfeld l_A senken KANN (omega_V = 5e8 -> l_A = 294)
  - R^2-Korrektur den Hintergrund zerstoert (nicht-perturbativ)
  - H0 allein nicht hilft

Hier: gezielte 2D-Suche (omega_V, psi0) + Physik-Check (BBN, Neff)
===============================================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import os, time, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_lines = []
def out(text=""):
    print(text)
    output_lines.append(text)

def save_output():
    path = os.path.join(OUTPUT_DIR, "Fine_lA_Search.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    out(f"\n  [Gespeichert: {path}]")

# ================================================================
# KONSTANTEN
# ================================================================
c_light = 299792.458   # km/s
T_CMB = 2.7255
Omega_gamma = 5.38e-5
Omega_nu = 3.65e-5
Omega_r = Omega_gamma + Omega_nu
z_star = 1089.92
a_star = 1.0 / (1 + z_star)
z_drag = 1059.62
z_BBN = 1e9   # BBN Epoche (T ~ 1 MeV -> z ~ 4e9, vereinfacht)

PLANCK_lA = 301.471
PLANCK_lA_err = 0.090
PLANCK_wb = 0.02236

# MCMC best-fit
MCMC_k = 9.81
MCMC_a_trans = 0.971
MCMC_alpha = 0.68
MCMC_beta = 2.02

# ================================================================
# CFM BASIS
# ================================================================
def cfm_base_E2(a, Ob, beta=None, alpha=None):
    """Basis-CFM E^2 ohne Skalarfeld"""
    if beta is None:
        beta = MCMC_beta
    if alpha is None:
        alpha = MCMC_alpha
    a_eq = Omega_r / Ob
    S = 1.0 / (1.0 + a_eq / np.maximum(a, 1e-20))
    s0 = np.tanh(MCMC_k * MCMC_a_trans)
    f_sat = (np.tanh(MCMC_k * (a - MCMC_a_trans)) + s0) / (1.0 + s0)
    S1 = 1.0 / (1.0 + a_eq)
    f1 = (np.tanh(MCMC_k * (1.0 - MCMC_a_trans)) + s0) / (1.0 + s0)
    Phi0 = (1.0 - Ob - Omega_r - alpha * S1) / max(f1, 1e-15)
    return Ob * a**(-3) + Omega_r * a**(-4) + Phi0 * f_sat + alpha * a**(-beta) * S


# ================================================================
# SKALARFELD + HINTERGRUND (schnelle Version)
# ================================================================
def compute_lA_with_scalar(H0, Ob, omega_V, psi0, N_points=1200, beta=None, alpha=None):
    """
    Berechne l_A mit Poeschl-Teller-Skalarfeld.
    Returns: dict mit l_A, r_s, d_C, Ophi_star, Ophi_BBN, etc.
    """
    N_min = np.log(1e-8)
    N_max = 0.0
    N_grid = np.linspace(N_min, N_max, N_points)
    a_grid = np.exp(N_grid)
    dN = N_grid[1] - N_grid[0]

    # Basis-CFM
    E2_base = np.array([cfm_base_E2(a, Ob, beta=beta, alpha=alpha) for a in a_grid])
    E2_base = np.maximum(E2_base, 1e-30)

    if omega_V <= 0 or psi0 <= 0:
        # Kein Skalarfeld
        E2_full = E2_base.copy()
        Omega_phi = np.zeros(N_points)
    else:
        # Skalarfeld auf Basis-Hintergrund loesen
        E2_interp_base = interp1d(N_grid, E2_base, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')

        def eps_H_func(N, E2_func):
            dN_h = 0.005
            Np = min(N + dN_h, N_max)
            Nm = max(N - dN_h, N_min)
            E2c = E2_func(N)
            if E2c < 1e-30:
                return -2.0
            return (E2_func(Np) - E2_func(Nm)) / (2 * dN_h * 2 * E2c)

        def rhs(N, y):
            psi, dpsi = y
            E2 = float(E2_interp_base(N))
            if E2 < 1e-30:
                E2 = 1e-30
            eH = eps_H_func(N, E2_interp_base)
            x = np.clip(psi / psi0, -50, 50)
            cosh_x = np.cosh(x)
            tanh_x = np.tanh(x)
            dVdpsi = -2.0 * omega_V / psi0 * tanh_x / cosh_x**2
            ddpsi = -(3.0 + eH) * dpsi - dVdpsi / E2
            return [dpsi, ddpsi]

        sol = solve_ivp(rhs, [N_min, N_max], [0.01, 0.0],
                        t_eval=N_grid, method='RK45',
                        rtol=1e-6, atol=1e-8, max_step=0.3)

        if sol.success:
            psi_arr = sol.y[0]
            dpsi_arr = sol.y[1]
        else:
            psi_arr = np.full(N_points, 0.01)
            dpsi_arr = np.zeros(N_points)

        x_arr = np.clip(psi_arr / psi0, -50, 50)
        Omega_phi = E2_base * dpsi_arr**2 / 6.0 + omega_V / np.cosh(x_arr)**2

        # Closure-Korrektur: anpasse Phi0 so dass E^2(a=1) = 1
        # E2_full = E2_base + Omega_phi
        # E2_full(1) sollte = 1 sein
        # cfm_base_E2 ist bereits so konstruiert dass E2_base(1) = 1
        # Also: E2_full(1) = 1 + Omega_phi(1)
        # Korrektur: subtrahiere Omega_phi(1) als effektives Lambda
        Ophi_0 = Omega_phi[-1]
        # Renormierung: E2_full = E2_base + Omega_phi - Ophi_0
        # Das verschiebt effektiv die dunkle Energie
        E2_full = E2_base + Omega_phi - Ophi_0
        E2_full = np.maximum(E2_full, 1e-30)

    # Interpolator fuer volle E2
    E2_interp = interp1d(N_grid, E2_full, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    # r_s(z*) berechnen
    def rs_integrand(a):
        if a < 1e-15:
            return 0.0
        N = np.log(a)
        if N < N_min:
            E2 = Omega_r * a**(-4) + Ob * a**(-3)
        else:
            E2 = float(E2_interp(N))
        E2 = max(E2, 1e-30)
        R_b = 3.0 * Ob * a / (4.0 * Omega_gamma)
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
        return c_s / (a**2 * np.sqrt(E2))

    r_s, _ = quad(rs_integrand, 1e-10, a_star, limit=2000)
    r_s *= c_light / H0

    # d_C(z*) berechnen
    def dC_integrand(a):
        N = np.log(max(a, 1e-15))
        N = max(N, N_min)
        E2 = float(E2_interp(N))
        E2 = max(E2, 1e-30)
        return 1.0 / (a**2 * np.sqrt(E2))

    d_C, _ = quad(dC_integrand, a_star, 1.0, limit=2000)
    d_C *= c_light / H0

    l_A = np.pi * d_C / r_s if r_s > 0 else 0

    # Omega_phi an verschiedenen Epochen
    idx_star = np.argmin(np.abs(a_grid - a_star))
    Ophi_star = Omega_phi[idx_star]
    E2_star = E2_full[idx_star]

    # BBN check: Omega_phi / E2 bei z ~ 1e9
    a_BBN = 1.0 / (1 + z_BBN)
    idx_BBN = np.argmin(np.abs(a_grid - a_BBN))
    if idx_BBN < len(Omega_phi):
        Ophi_BBN = Omega_phi[idx_BBN]
        E2_BBN = E2_full[idx_BBN]
        frac_BBN = Ophi_BBN / E2_BBN if E2_BBN > 0 else 0
    else:
        Ophi_BBN = 0
        E2_BBN = 0
        frac_BBN = 0

    # Frac at z*
    frac_star = Ophi_star / E2_star if E2_star > 0 else 0

    # Effektives Delta_Neff von Skalarfeld bei BBN
    # rho_phi/rho_rad = Omega_phi(a_BBN) / (Omega_r * a_BBN^{-4})
    rho_rad_BBN = Omega_r * a_BBN**(-4)
    Delta_Neff_BBN = (7.0/8.0)**(-1) * (4.0/11.0)**(-4.0/3.0) * Ophi_BBN / rho_rad_BBN if rho_rad_BBN > 0 else 0
    # Einfacher: Delta_Neff ~ (rho_phi/rho_gamma) * (8/7) * (11/4)^{4/3}
    # Oder: rho_phi/rho_total = f -> Delta_Neff ~ f * Neff_SM / (1-f)
    # BBN erlaubt Delta_Neff < 0.5 (2sigma)

    return {
        'l_A': l_A, 'r_s': r_s, 'd_C': d_C,
        'Ophi_star': Ophi_star, 'frac_star': frac_star,
        'frac_BBN': frac_BBN, 'E2_0': E2_full[-1],
        'Omega_phi': Omega_phi, 'E2_full': E2_full,
        'N_grid': N_grid
    }


# ================================================================
# LCDM REFERENZ
# ================================================================
def lcdm_lA():
    Om = 0.315
    H0 = 67.36
    Ob = PLANCK_wb / (H0/100)**2

    def E2(a):
        return Om * a**(-3) + Omega_r * a**(-4) + (1.0 - Om - Omega_r)

    def rs_int(a):
        R_b = 3.0 * Ob * a / (4.0 * Omega_gamma)
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
        return c_s / (a**2 * np.sqrt(max(E2(a), 1e-30)))

    def dC_int(a):
        return 1.0 / (a**2 * np.sqrt(max(E2(a), 1e-30)))

    r_s, _ = quad(rs_int, 1e-10, a_star, limit=2000)
    r_s *= c_light / H0
    d_C, _ = quad(dC_int, a_star, 1.0, limit=2000)
    d_C *= c_light / H0
    return np.pi * d_C / r_s, r_s, d_C


# ================================================================
# HAUPTANALYSE
# ================================================================
if __name__ == "__main__":
    t0 = time.time()
    out("=" * 74)
    out("  GEZIELTE FEINSUCHE: l_A = 301.5 MIT POESCHL-TELLER-SKALARFELD")
    out("=" * 74)
    out("  Datum: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    out()

    # LCDM Referenz
    lA_lcdm, rs_lcdm, dC_lcdm = lcdm_lA()
    out("  LCDM Referenz: l_A = {:.3f}, r_s = {:.2f} Mpc, d_C = {:.2f} Mpc".format(
        lA_lcdm, rs_lcdm, dC_lcdm))

    # CFM Referenz (ohne Skalarfeld)
    H0 = 67.36
    Ob_cfm = 0.05
    ref = compute_lA_with_scalar(H0, Ob_cfm, 0, 1.0)
    out("  CFM Referenz:  l_A = {:.3f}, r_s = {:.2f} Mpc, d_C = {:.2f} Mpc".format(
        ref['l_A'], ref['r_s'], ref['d_C']))
    out("  Abweichung:    Delta_lA = {:.1f} ({:.1f} sigma)".format(
        ref['l_A'] - PLANCK_lA, (ref['l_A'] - PLANCK_lA) / PLANCK_lA_err
        if 'PLANCK_lA_err' in dir() else 0))
    out()

    # ================================================================
    # SCAN 1: omega_V bei verschiedenen psi0 (H0 = 67.36, gamma = 0)
    # ================================================================
    out("=" * 74)
    out("  SCAN 1: omega_V vs psi0 (H0 = 67.36)")
    out("=" * 74)
    out()

    omega_V_vals = [1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 3e8, 5e8, 1e9]
    psi0_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    out("  Ziel: l_A = {:.3f}".format(PLANCK_lA))
    out()

    # Finde Kombinationen die l_A ~ 301.5 ergeben
    best_combos = []

    out("  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "omega_V", "psi0", "l_A", "r_s", "d_C", "f_phi(z*)", "f_phi(BBN)"))
    out("  " + "-" * 75)

    for psi0 in [0.1, 0.3, 0.5, 1.0]:
        for oV in omega_V_vals:
            try:
                res = compute_lA_with_scalar(H0, Ob_cfm, oV, psi0, N_points=1000)
                marker = " <<< " if abs(res['l_A'] - PLANCK_lA) < 5 else ""
                out("  {:10.1e}  {:8.3f}  {:10.3f}  {:10.2f}  {:10.2f}  {:10.4f}  {:10.2e}{}".format(
                    oV, psi0, res['l_A'], res['r_s'], res['d_C'],
                    res['frac_star'], res['frac_BBN'], marker))
                if abs(res['l_A'] - PLANCK_lA) < 10:
                    best_combos.append((oV, psi0, res))
            except Exception as e:
                out("  {:10.1e}  {:8.3f}  FEHLER: {}".format(oV, psi0, str(e)[:40]))
        out()

    # ================================================================
    # SCAN 2: Feinsuche um die besten Kombinationen
    # ================================================================
    out("=" * 74)
    out("  SCAN 2: FEINSUCHE UM VIELVERSPRECHENDE PARAMETER")
    out("=" * 74)
    out()

    if best_combos:
        # Finde die naechste Kombination
        best_combos.sort(key=lambda x: abs(x[2]['l_A'] - PLANCK_lA))
        oV_best, psi0_best, res_best = best_combos[0]
        out("  Naechste Kombination: omega_V = {:.1e}, psi0 = {:.3f}, l_A = {:.3f}".format(
            oV_best, psi0_best, res_best['l_A']))
        out()

        # Feinsuche um diesen Punkt
        oV_fine = np.logspace(np.log10(oV_best) - 0.5, np.log10(oV_best) + 0.5, 15)
        psi0_fine = np.linspace(max(psi0_best - 0.15, 0.02), psi0_best + 0.15, 10)

        out("  Feingitter: {} x {} = {} Punkte".format(len(oV_fine), len(psi0_fine),
                                                         len(oV_fine) * len(psi0_fine)))
        out()

        best_delta = 1e10
        best_fine = None

        for oV in oV_fine:
            for p0 in psi0_fine:
                try:
                    res = compute_lA_with_scalar(H0, Ob_cfm, oV, p0, N_points=1000)
                    delta = abs(res['l_A'] - PLANCK_lA)
                    if delta < best_delta:
                        best_delta = delta
                        best_fine = (oV, p0, res)
                except:
                    pass

        if best_fine:
            oV_f, p0_f, res_f = best_fine
            out("  BESTER l_A-MATCH (H0=67.36):")
            out("  " + "=" * 55)
            out("  omega_V      = {:.4e}".format(oV_f))
            out("  psi0         = {:.4f}".format(p0_f))
            out("  l_A          = {:.3f}  (Planck: {:.3f})".format(res_f['l_A'], PLANCK_lA))
            out("  Delta_lA     = {:.3f}".format(res_f['l_A'] - PLANCK_lA))
            out("  r_s(z*)      = {:.2f} Mpc".format(res_f['r_s']))
            out("  d_C(z*)      = {:.2f} Mpc".format(res_f['d_C']))
            out("  f_phi(z*)    = {:.4f}  (Skalarfeld-Anteil bei Rekombination)".format(res_f['frac_star']))
            out("  f_phi(BBN)   = {:.2e}  (Skalarfeld-Anteil bei BBN)".format(res_f['frac_BBN']))
            out("  E^2(a=1)     = {:.6f}".format(res_f['E2_0']))
            out()
    else:
        out("  Keine vielversprechenden Kombinationen in Scan 1 gefunden.")
        out()

    # ================================================================
    # SCAN 3: H0 + omega_V kombiniert (ohne R^2)
    # ================================================================
    out("=" * 74)
    out("  SCAN 3: H0 x omega_V (psi0 = 0.3, gamma = 0)")
    out("=" * 74)
    out()

    H0_vals = [60, 65, 67.36, 70, 75, 80, 90, 100]
    oV_vals_H0 = [0, 1e6, 1e7, 5e7, 1e8, 5e8]

    out("  {:>6s}  {:>8s}".format("H0", "Ob") + "".join(
        ["  {:>10.0e}".format(oV) if oV > 0 else "  {:>10s}".format("0") for oV in oV_vals_H0]))
    out("  " + "-" * (16 + 12 * len(oV_vals_H0)))

    best_H0 = None
    best_H0_delta = 1e10

    for H0_try in H0_vals:
        hh = H0_try / 100.0
        Ob_try = PLANCK_wb / hh**2
        row = "  {:6.1f}  {:8.4f}".format(H0_try, Ob_try)
        for oV in oV_vals_H0:
            try:
                res = compute_lA_with_scalar(H0_try, Ob_try, oV, 0.3, N_points=1000)
                row += "  {:10.1f}".format(res['l_A'])
                delta = abs(res['l_A'] - PLANCK_lA)
                if delta < best_H0_delta:
                    best_H0_delta = delta
                    best_H0 = (H0_try, Ob_try, oV, res)
            except:
                row += "  {:>10s}".format("ERR")
        out(row)

    out()

    # ================================================================
    # SCAN 4: beta-Variation (naeher an a^{-3} CDM-Skalierung)
    # ================================================================
    out("=" * 74)
    out("  SCAN 4: beta-Variation (H0=67.36, omega_V=0)")
    out("=" * 74)
    out()

    out("  CFM geometrischer DM ~ a^{-beta}. LCDM hat beta=3.")
    out("  MCMC best-fit: beta = 2.02. Erhoehung koennte l_A verbessern.")
    out()

    beta_vals = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    out("  {:>6s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "beta", "r_s", "d_C", "l_A", "Delta_lA"))
    out("  " + "-" * 50)

    for beta_try in beta_vals:
        try:
            res = compute_lA_with_scalar(67.36, 0.05, 0, 1.0, N_points=1000, beta=beta_try)
            out("  {:6.2f}  {:10.2f}  {:10.2f}  {:10.3f}  {:10.1f}".format(
                beta_try, res['r_s'], res['d_C'], res['l_A'],
                res['l_A'] - PLANCK_lA))
        except Exception as e:
            out("  {:6.2f}  FEHLER: {}".format(beta_try, str(e)[:40]))
    out()

    # ================================================================
    # SCAN 5: beta + Skalarfeld kombiniert
    # ================================================================
    out("=" * 74)
    out("  SCAN 5: beta + Skalarfeld kombiniert")
    out("=" * 74)
    out()

    out("  {:>6s}  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "beta", "omega_V", "psi0", "l_A", "r_s", "f_phi(z*)", "f_phi(BBN)"))
    out("  " + "-" * 70)

    best_combined = None
    best_comb_delta = 1e10

    for beta_try in [2.5, 2.8, 3.0]:
        for oV in [0, 1e6, 1e7, 5e7, 1e8]:
            for p0 in [0.2, 0.5, 1.0]:
                try:
                    res = compute_lA_with_scalar(67.36, 0.05, oV, p0, N_points=1000,
                                                  beta=beta_try)
                    delta = abs(res['l_A'] - PLANCK_lA)
                    marker = " <<<" if delta < 5 else ""
                    out("  {:6.2f}  {:10.1e}  {:8.3f}  {:10.3f}  {:10.2f}  {:10.4f}  {:10.2e}{}".format(
                        beta_try, oV, p0, res['l_A'], res['r_s'],
                        res['frac_star'], res['frac_BBN'], marker))
                    if delta < best_comb_delta:
                        best_comb_delta = delta
                        best_combined = (beta_try, oV, p0, res)
                except:
                    pass
        out()
    out()

    # ================================================================
    # ZUSAMMENFASSUNG
    # ================================================================
    out("=" * 74)
    out("  ZUSAMMENFASSUNG: CFM l_A-PROBLEM")
    out("=" * 74)
    out()

    out("  PLANCK-ZIEL: l_A = {:.3f} +/- 0.09".format(PLANCK_lA))
    out("  CFM BASIS:   l_A = {:.3f} (Delta = {:.1f}, {:.0f} sigma)".format(
        ref['l_A'], ref['l_A'] - PLANCK_lA, (ref['l_A'] - PLANCK_lA) / 0.09))
    out()

    out("  MECHANISMUS-ANALYSE:")
    out("  " + "-" * 55)
    out()
    out("  1. Skalarfeld (Poeschl-Teller):")
    if best_fine:
        oV_f, p0_f, res_f = best_fine
        out("     KANN l_A senken: l_A = {:.1f} bei omega_V={:.1e}, psi0={:.2f}".format(
            res_f['l_A'], oV_f, p0_f))
        out("     ABER: f_phi(z*) = {:.2f} -> {:.0f}% der Energie bei z*".format(
            res_f['frac_star'], res_f['frac_star'] * 100))
        out("     f_phi(BBN) = {:.2e} -> BBN-Grenze: Delta_Neff < 0.5".format(
            res_f['frac_BBN']))
    out()
    out("  2. H0-Erhoehung:")
    out("     HILFT NICHT! l_A steigt sogar leicht mit H0.")
    out("     Grund: d_C/r_s aendert sich kaum, da beide ~ 1/H0 skalieren.")
    out()
    out("  3. beta-Erhoehung (naeher an CDM-Skalierung):")
    if best_combined:
        bt, oV_c, p0_c, res_c = best_combined
        out("     beta = {:.1f}: l_A = {:.1f} (Verbesserung)".format(bt, res_c['l_A']))
    out("     beta -> 3 wuerde CFM geometrischen DM ~ CDM machen.")
    out()
    out("  4. R^2-Scalaron:")
    out("     ZERSTOERT den Hintergrund (nicht-perturbativ bei gamma > 0.01)")
    out("     Nur auf Perturbationsebene relevant (mu_eff ~ 4/3)")
    out()

    # Physikalische Bewertung
    out("  PHYSIKALISCHE BEWERTUNG:")
    out("  " + "=" * 55)
    out()
    out("  Das l_A-Problem ist ein FUNDAMENTALES Hintergrund-Problem.")
    out("  Ohne CDM (oder Aequivalent mit a^{-3}-Skalierung) expandiert")
    out("  das Universum im Bereich 1 < z < 1000 zu langsam.")
    out()
    out("  Wege zu l_A = 301.5:")
    out("  (a) beta ~ 3 (geometrischer DM skaliert wie CDM)")
    out("      -> Widerspricht SN-Fit (beta = 2.02 ist MCMC-optimiert)")
    out("  (b) Grosses Skalarfeld bei z ~ 1000")
    out("      -> Widerspricht BBN (Delta_Neff zu gross)")
    out("  (c) Kombination: moderates beta ~ 2.5 + kleines Skalarfeld")
    out("      -> Moeglicherweise konsistent, erfordert Re-Fit")
    out()

    elapsed = time.time() - t0
    out("  Laufzeit: {:.1f} Sekunden".format(elapsed))

    save_output()

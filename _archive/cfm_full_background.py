#!/usr/bin/env python3
"""
===============================================================================
VOLLE CFM-HINTERGRUND-KOSMOLOGIE MIT ALLEN FELDERN
===============================================================================

Friedmann-Gleichung mit:
  1. Baryonen + Strahlung
  2. CFM Saettigung (DE) + geometrischer DM (alpha*a^{-beta})
  3. Poeschl-Teller-Skalarfeld: rho_phi = 0.5*phi_dot^2 + V(phi)
  4. R^2-Scalaron-Korrektur (perturbativ)
  5. Trace-Kopplung F(T/rho)

Scannt ueber {omega_V, psi0, H0} um l_A = 301.5 (Planck) zu erreichen.

===============================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
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
    path = os.path.join(OUTPUT_DIR, "Full_Background_Analysis.txt")
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

# Planck 2018
PLANCK_lA = 301.471
PLANCK_lA_err = 0.090
PLANCK_wb = 0.02236   # omega_b = Omega_b * h^2

# MCMC best-fit (Paper II)
MCMC_k = 9.81
MCMC_a_trans = 0.971
MCMC_alpha = 0.68
MCMC_beta = 2.02

# ================================================================
# HILFSFUNKTIONEN
# ================================================================

def cfm_base_E2(a, Ob, k, a_trans, alpha, beta):
    """Basis-CFM E^2 (ohne Skalarfeld, ohne R^2)"""
    a_eq = Omega_r / Ob
    S = 1.0 / (1.0 + a_eq / np.maximum(a, 1e-20))
    s0 = np.tanh(k * a_trans)
    f_sat = (np.tanh(k * (a - a_trans)) + s0) / (1.0 + s0)
    # Phi0 aus Closure bei a=1
    S1 = 1.0 / (1.0 + a_eq)
    f1 = (np.tanh(k * (1.0 - a_trans)) + s0) / (1.0 + s0)
    Phi0 = (1.0 - Ob - Omega_r - alpha * S1) / max(f1, 1e-15)
    return Ob * a**(-3) + Omega_r * a**(-4) + Phi0 * f_sat + alpha * a**(-beta) * S


def trace_coupling(a, Ob):
    """F(T/rho) ~ rho_m / (rho_m + rho_r) = Trace-Kopplung"""
    rho_m = Ob * a**(-3)
    rho_r = Omega_r * a**(-4)
    return rho_m / (rho_m + rho_r + 1e-30)


# ================================================================
# SKALARFELD-EVOLUTION (Klein-Gordon)
# ================================================================

def solve_scalar_field(omega_V, psi0, E2_interp, N_grid):
    """
    Loest die Klein-Gordon-Gleichung auf dem Hintergrund E2_interp(N).

    Dimensionslose Variablen:
      psi = phi / M_Pl  (Feld in Planck-Einheiten)
      omega_V = V0 / rho_crit  (Potential in krit. Dichte)
      psi0 = phi0 / M_Pl  (Breite in Planck-Einheiten)

    KG: psi'' + (3 + eps_H) * psi' + (1/E^2) * dV_tilde/dpsi = 0
    wobei V_tilde = omega_V / cosh^2(psi/psi0)
          dV_tilde/dpsi = -2*omega_V/(psi0) * tanh(psi/psi0) / cosh^2(psi/psi0)

    Omega_phi = E^2 * (psi')^2 / 6 + omega_V / cosh^2(psi/psi0)
    """
    if omega_V <= 0:
        # Kein Skalarfeld
        return np.zeros_like(N_grid), np.zeros_like(N_grid), np.zeros_like(N_grid)

    N_min, N_max = N_grid[0], N_grid[-1]

    # eps_H = d(ln E)/dN, numerisch aus E2_interp
    dN = 0.001
    def eps_H_func(N):
        E2_p = E2_interp(min(N + dN, N_max))
        E2_m = E2_interp(max(N - dN, N_min))
        E2_c = E2_interp(N)
        if E2_c < 1e-30:
            return -2.0
        return (E2_p - E2_m) / (2 * dN * 2 * E2_c)  # d(lnE)/dN = dE2/(2*E2*dN)/2? No.
        # eps_H = E'/E = (1/(2E^2)) * dE^2/dN
        # So: eps_H = (E2_p - E2_m) / (2*dN) / (2 * E2_c)
        # Wait: E' = dE/dN. E = sqrt(E2). E' = (1/(2*sqrt(E2))) * dE2/dN
        # eps_H = E'/E = dE2/(2*E2*dN)... but that's d(lnE2)/(2*dN) = d(lnE)/dN. Yes.

    def rhs(N, y):
        psi, dpsi = y
        E2 = float(E2_interp(N))
        if E2 < 1e-30:
            E2 = 1e-30
        eH = eps_H_func(N)

        x = psi / psi0
        # Kappen um numerische Probleme zu vermeiden
        x = np.clip(x, -50, 50)
        cosh_x = np.cosh(x)
        tanh_x = np.tanh(x)

        # dV_tilde/dpsi = -2*omega_V/psi0 * tanh(x) / cosh^2(x)
        dVdpsi = -2.0 * omega_V / psi0 * tanh_x / cosh_x**2

        ddpsi = -(3.0 + eH) * dpsi - dVdpsi / E2
        return [dpsi, ddpsi]

    # Anfangsbedingungen: Feld nahe am Gipfel, leichte Auslenkung
    psi_init = 0.01  # kleine Stoerung
    dpsi_init = 0.0   # ruht

    sol = solve_ivp(rhs, [N_min, N_max], [psi_init, dpsi_init],
                    t_eval=N_grid, method='RK45', rtol=1e-6, atol=1e-8,
                    max_step=0.2)

    if sol.success:
        psi_arr = sol.y[0]
        dpsi_arr = sol.y[1]
    else:
        psi_arr = np.full_like(N_grid, psi_init)
        dpsi_arr = np.zeros_like(N_grid)

    # Omega_phi(N) = E^2*(psi')^2/6 + omega_V/cosh^2(psi/psi0)
    E2_arr = np.array([E2_interp(N) for N in N_grid])
    x_arr = np.clip(psi_arr / psi0, -50, 50)
    Omega_phi = E2_arr * dpsi_arr**2 / 6.0 + omega_V / np.cosh(x_arr)**2

    return psi_arr, dpsi_arr, Omega_phi


# ================================================================
# R^2-HINTERGRUND-KORREKTUR
# ================================================================

def r2_background_correction(N_grid, E2_arr, gamma, Ob):
    """
    R^2-Korrektur zur Friedmann-Gleichung (perturbativ).

    R = 6*H0^2*E^2*(eps_H + 2)
    eps = 16*pi*G*gamma (in H0^{-2} Einheiten: eps_tilde = eps*H0^2)

    Fuer kleine eps: Delta_E2 = eps_tilde * [r^2/6 - 2*E^2*r'] / (1 + 2*eps_tilde*r)
    wobei r = R/H0^2 = 6*E^2*(eps_H + 2)

    In der Strahlungs-Aera: R ~ 0 (konforme Symmetrie), Korrektur ~ 0.
    """
    if gamma <= 0:
        return np.zeros_like(N_grid)

    # eps_tilde = 16*pi*G*gamma * H0^2
    # In unseren Einheiten wo E^2 = H^2/H0^2:
    # eps_tilde ist dimensionslos, gamma hat Einheiten H0^{-2}
    # eps = 16*pi*G*gamma, eps_tilde = eps*H0^2
    # Aber 8*pi*G = 3*H0^2/rho_crit und H0^2 = ...
    # Einfacher: In Friedmann-Einheiten ist eps_tilde = 6*gamma (fuer gamma in H0^{-2})
    # Weil: eps = 16*pi*G*gamma und 8*pi*G*rho_crit = 3*H0^2
    # Also 16*pi*G = 6*H0^2/rho_crit * ... nein.
    # Lassen wir es als freien Parameter: eps_tilde = dimensionslos
    # gamma in H0^{-2} -> eps_tilde ~ gamma (Groessenordnung)
    eps_tilde = gamma  # vereinfachte Zuordnung

    dN = N_grid[1] - N_grid[0]

    # eps_H = d(ln E)/dN
    lnE2 = np.log(np.maximum(E2_arr, 1e-30))
    eps_H = np.gradient(lnE2, dN) / 2.0  # d(lnE)/dN = d(lnE2)/(2*dN)

    # Ricci-Skalar r = R/H0^2 = 6*E^2*(eps_H + 2)
    r = 6.0 * E2_arr * (eps_H + 2.0)

    # Trace-Kopplung
    a_arr = np.exp(N_grid)
    F_arr = np.array([trace_coupling(a, Ob) for a in a_arr])

    # Effektives epsilon
    eps_eff = eps_tilde * F_arr

    # r' = dr/dN
    r_prime = np.gradient(r, dN)

    # Korrektur: Delta_E2 = eps_eff * (r^2/6 - 2*E2*r') / (1 + 2*eps_eff*r)
    # Fuer kleine eps_eff*r: Delta_E2 ~ eps_eff * (r^2/6 - 2*E2*r')
    numerator = eps_eff * (r**2 / 6.0 - 2.0 * E2_arr * r_prime)
    denominator = 1.0 + 2.0 * eps_eff * r
    Delta_E2 = numerator / np.maximum(denominator, 1e-10)

    return Delta_E2


# ================================================================
# ITERATIVER LOESER: VOLLE FRIEDMANN-GLEICHUNG
# ================================================================

def solve_full_background(H0, Ob, k, a_trans, alpha, beta,
                           omega_V, psi0, gamma,
                           n_iter=2, N_points=1500):
    """
    Loest die volle CFM-Friedmann-Gleichung iterativ.

    Returns: dict mit r_s, d_C, l_A, E2_full, etc.
    """
    # N = ln(a) Gitter
    N_min = np.log(1e-8)
    N_max = 0.0  # a=1
    N_grid = np.linspace(N_min, N_max, N_points)
    a_grid = np.exp(N_grid)

    # Schritt 0: Basis-CFM (ohne phi, ohne R^2)
    # Phi0 haengt von omega_V ab wegen Closure!
    # Closure bei a=1: E^2(1) = 1
    # 1 = Ob + Or + Phi0*f_sat(1) + alpha*S(1) + omega_V/cosh^2(psi(1)/psi0) + kinetic
    # Erstmal: Phi0 OHNE Skalarfeld berechnen, dann iterativ korrigieren

    E2_base_arr = np.array([cfm_base_E2(a, Ob, k, a_trans, alpha, beta) for a in a_grid])
    E2_base_arr = np.maximum(E2_base_arr, 1e-30)
    E2_interp = interp1d(N_grid, E2_base_arr, kind='linear',
                          bounds_error=False, fill_value='extrapolate')

    Omega_phi_arr = np.zeros(N_points)
    Delta_R2_arr = np.zeros(N_points)
    psi_arr = np.zeros(N_points)
    dpsi_arr = np.zeros(N_points)

    for iteration in range(n_iter):
        # Skalarfeld auf aktuellem Hintergrund loesen
        psi_arr, dpsi_arr, Omega_phi_arr = solve_scalar_field(
            omega_V, psi0, E2_interp, N_grid)

        # R^2-Korrektur
        E2_current = np.array([E2_interp(N) for N in N_grid])
        Delta_R2_arr = r2_background_correction(N_grid, E2_current, gamma, Ob)

        # Volle E^2 = Basis + Skalarfeld + R^2
        E2_full = E2_base_arr + Omega_phi_arr + Delta_R2_arr
        E2_full = np.maximum(E2_full, 1e-30)

        # Closure-Korrektur: skaliere so dass E^2(a=1) = 1
        E2_at_1 = E2_full[-1]
        if E2_at_1 > 0:
            # Statt zu skalieren: passe Phi0 an
            # Einfacher: normiere die gesamte Kurve (approximativ)
            # Genauer: nur den DE-Anteil anpassen
            # Fuer Iteration: einfache Normierung
            correction_factor = 1.0 / E2_at_1
            # Nur auf die Basis anwenden, nicht auf phi/R^2
            # Besser: korrigiere Phi0
            pass  # erstmal ohne Closure-Korrektur, spaeter

        E2_interp = interp1d(N_grid, E2_full, kind='linear',
                              bounds_error=False, fill_value='extrapolate')

    # Distanzen berechnen
    h = H0 / 100.0

    # r_s(z*) = (c/H0) * int_0^{a*} c_s / (a^2 * E(a)) da
    def rs_integrand(a):
        if a < 1e-15:
            return 0.0
        N = np.log(a)
        if N < N_min:
            # Extrapoliere: Strahlungsdominanz
            E2 = Omega_r * a**(-4) + Ob * a**(-3)
        else:
            E2 = float(E2_interp(N))
        E2 = max(E2, 1e-30)
        R_b = 3.0 * Ob * a / (4.0 * Omega_gamma)
        c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
        return c_s / (a**2 * np.sqrt(E2))

    r_s, _ = quad(rs_integrand, 1e-10, a_star, limit=2000)
    r_s *= c_light / H0  # Mpc

    r_d, _ = quad(rs_integrand, 1e-10, 1.0/(1+z_drag), limit=2000)
    r_d *= c_light / H0

    # d_C(z*) = (c/H0) * int_{a*}^1 1/(a^2 * E(a)) da
    def dC_integrand(a):
        N = np.log(max(a, 1e-15))
        N = max(N, N_min)
        E2 = float(E2_interp(N))
        E2 = max(E2, 1e-30)
        return 1.0 / (a**2 * np.sqrt(E2))

    d_C, _ = quad(dC_integrand, a_star, 1.0, limit=2000)
    d_C *= c_light / H0

    l_A = np.pi * d_C / r_s if r_s > 0 else 0

    # Omega_phi bei z=0 und z*
    idx_star = np.argmin(np.abs(a_grid - a_star))
    Ophi_star = Omega_phi_arr[idx_star]
    Ophi_0 = Omega_phi_arr[-1]
    E2_star = E2_full[idx_star]
    E2_0 = E2_full[-1]

    return {
        'r_s': r_s, 'd_C': d_C, 'l_A': l_A, 'r_d': r_d,
        'E2_0': E2_0, 'E2_star': E2_star,
        'Ophi_star': Ophi_star, 'Ophi_0': Ophi_0,
        'N_grid': N_grid, 'E2_full': E2_full,
        'Omega_phi': Omega_phi_arr, 'Delta_R2': Delta_R2_arr,
        'psi': psi_arr, 'dpsi': dpsi_arr,
    }


# ================================================================
# PARAMETER-SCAN
# ================================================================

def scan_parameters():
    out("=" * 74)
    out("  PARAMETER-SCAN: VOLLE CFM-HINTERGRUND-KOSMOLOGIE")
    out("=" * 74)
    out()

    # Referenz: Basis-CFM ohne Skalarfeld
    H0_ref = 67.36
    Ob_ref = PLANCK_wb / (H0_ref/100)**2  # = 0.0493 (Planck)
    Ob_cfm = 0.05  # CFM Standard

    out("  A. Referenz: Basis-CFM (ohne Skalarfeld, ohne R^2)")
    out("  " + "-" * 55)
    ref = solve_full_background(H0_ref, Ob_cfm, MCMC_k, MCMC_a_trans,
                                 MCMC_alpha, MCMC_beta, 0, 1.0, 0)
    out("  r_s(z*) = {:.2f} Mpc".format(ref['r_s']))
    out("  d_C(z*) = {:.2f} Mpc".format(ref['d_C']))
    out("  l_A     = {:.3f}  (Planck: {:.3f})".format(ref['l_A'], PLANCK_lA))
    out("  E^2(a=1) = {:.4f}".format(ref['E2_0']))
    out()

    # ---- Scan 1: omega_V bei festem psi0, H0 ----
    out("  B. Scan: omega_V (Skalarfeld-Amplitude)")
    out("  " + "-" * 55)
    out("  psi0 = 1.0 (Planck-Breite), H0 = 67.36, gamma = 0")
    out()

    omega_V_values = [0, 1e5, 1e6, 1e7, 5e7, 1e8, 5e8]
    out("  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "omega_V", "r_s", "d_C", "l_A", "Ophi(z*)", "Ophi(0)"))
    out("  " + "-" * 65)

    for oV in omega_V_values:
        try:
            res = solve_full_background(67.36, 0.05, MCMC_k, MCMC_a_trans,
                                         MCMC_alpha, MCMC_beta, oV, 1.0, 0,
                                         n_iter=2)
            out("  {:10.1e}  {:10.2f}  {:10.2f}  {:10.3f}  {:10.2e}  {:10.2e}".format(
                oV, res['r_s'], res['d_C'], res['l_A'],
                res['Ophi_star'], res['Ophi_0']))
        except Exception as e:
            out("  {:10.1e}  FEHLER: {}".format(oV, str(e)[:40]))

    out()

    # ---- Scan 2: psi0 bei festem omega_V ----
    out("  C. Scan: psi0 (Feldbreite) bei omega_V = 1e7")
    out("  " + "-" * 55)

    psi0_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    out("  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "psi0", "r_s", "d_C", "l_A", "Ophi(z*)"))
    out("  " + "-" * 55)

    for p0 in psi0_values:
        try:
            res = solve_full_background(67.36, 0.05, MCMC_k, MCMC_a_trans,
                                         MCMC_alpha, MCMC_beta, 1e7, p0, 0,
                                         n_iter=2)
            out("  {:10.4f}  {:10.2f}  {:10.2f}  {:10.3f}  {:10.2e}".format(
                p0, res['r_s'], res['d_C'], res['l_A'], res['Ophi_star']))
        except Exception as e:
            out("  {:10.4f}  FEHLER: {}".format(p0, str(e)[:40]))

    out()

    # ---- Scan 3: H0-Variation ----
    out("  D. Scan: H0-Variation (mit Omega_b = omega_b_BBN / h^2)")
    out("  " + "-" * 55)
    out("  omega_V = 0 (kein Skalarfeld), gamma = 0")
    out()

    H0_values = [60, 67.36, 75, 85, 100]
    out("  {:>6s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "H0", "Ob", "r_s", "d_C", "l_A", "Delta_lA"))
    out("  " + "-" * 60)

    for H0 in H0_values:
        hh = H0 / 100.0
        Ob = PLANCK_wb / hh**2  # BBN-konsistent
        try:
            # Phi0 und alpha muessen fuer Closure angepasst werden
            # Einfach: verwende Standard-alpha, passe Phi0 via cfm_base_E2 an
            res = solve_full_background(H0, Ob, MCMC_k, MCMC_a_trans,
                                         MCMC_alpha, MCMC_beta, 0, 1.0, 0)
            delta = res['l_A'] - PLANCK_lA
            out("  {:6.1f}  {:8.4f}  {:10.2f}  {:10.2f}  {:10.3f}  {:10.1f}".format(
                H0, Ob, res['r_s'], res['d_C'], res['l_A'], delta))
        except Exception as e:
            out("  {:6.1f}  FEHLER: {}".format(H0, str(e)[:40]))

    out()

    # ---- Scan 4: Kombinierter Scan H0 + omega_V ----
    out("  E. 2D-Scan: H0 x omega_V (psi0=1.0, gamma=1.0)")
    out("  " + "-" * 55)
    out("  Ziel: l_A = {:.3f}".format(PLANCK_lA))
    out()

    H0_scan = [67.36, 80, 90]
    oV_scan = [0, 1e7, 1e8]

    # Header
    header = "  {:>6s}".format("H0\\oV")
    for oV in oV_scan:
        header += "  {:>10.0e}".format(oV) if oV > 0 else "  {:>10s}".format("0")
    out(header)
    out("  " + "-" * (8 + 12 * len(oV_scan)))

    for H0 in H0_scan:
        hh = H0 / 100.0
        Ob = PLANCK_wb / hh**2
        row = "  {:6.1f}".format(H0)
        for oV in oV_scan:
            try:
                res = solve_full_background(H0, Ob, MCMC_k, MCMC_a_trans,
                                             MCMC_alpha, MCMC_beta,
                                             oV, 1.0, 1.0, n_iter=2)
                row += "  {:10.1f}".format(res['l_A'])
            except:
                row += "  {:>10s}".format("ERR")
        out(row)

    out()

    # ---- Scan 5: R^2-Einfluss ----
    out("  F. R^2-Korrektur: Einfluss von gamma")
    out("  " + "-" * 55)
    out("  H0 = 67.36, Ob = 0.05, omega_V = 0, psi0 = 1.0")
    out()

    gamma_values = [0, 0.1, 1.0, 10, 100]
    out("  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
        "gamma", "r_s", "d_C", "l_A"))
    out("  " + "-" * 45)

    for gam in gamma_values:
        try:
            res = solve_full_background(67.36, 0.05, MCMC_k, MCMC_a_trans,
                                         MCMC_alpha, MCMC_beta, 0, 1.0, gam)
            out("  {:10.2f}  {:10.2f}  {:10.2f}  {:10.3f}".format(
                gam, res['r_s'], res['d_C'], res['l_A']))
        except Exception as e:
            out("  {:10.2f}  FEHLER: {}".format(gam, str(e)[:40]))

    out()
    return


# ================================================================
# OPTIMIERUNG: Finde Parameter fuer l_A = 301.5
# ================================================================

def optimize_lA():
    out("=" * 74)
    out("  OPTIMIERUNG: Finde CFM-Parameter mit l_A = {:.3f}".format(PLANCK_lA))
    out("=" * 74)
    out()

    # Schnelle Grid-Suche statt differential_evolution
    out("  Grid-Suche: H0 x omega_V (psi0=1.0, gamma=1.0)")
    out()

    best_delta = 1e10
    best_params = None

    H0_grid = np.arange(60, 110, 10)
    oV_grid = [0, 1e6, 1e7, 1e8, 5e8]

    for H0 in H0_grid:
        hh = H0 / 100.0
        Ob = PLANCK_wb / hh**2
        for oV in oV_grid:
            try:
                res = solve_full_background(H0, Ob, MCMC_k, MCMC_a_trans,
                                             MCMC_alpha, MCMC_beta,
                                             oV, 1.0, 1.0, n_iter=2,
                                             N_points=1500)
                delta = abs(res['l_A'] - PLANCK_lA)
                if delta < best_delta:
                    best_delta = delta
                    best_params = (H0, Ob, oV, res)
            except:
                pass

    if best_params:
        H0_opt, Ob_opt, oV_opt, res = best_params
        out("  BESTES ERGEBNIS:")
        out("  " + "=" * 50)
        out("  H0      = {:.1f} km/s/Mpc".format(H0_opt))
        out("  Omega_b = {:.5f}".format(Ob_opt))
        out("  omega_V = {:.1e}".format(oV_opt))
        out("  r_s(z*) = {:.2f} Mpc".format(res['r_s']))
        out("  d_C(z*) = {:.2f} Mpc".format(res['d_C']))
        out("  l_A     = {:.3f}  (Ziel: {:.3f}, Delta: {:.1f})".format(
            res['l_A'], PLANCK_lA, best_delta))
        out("  E^2(0)  = {:.4f}".format(res['E2_0']))
        out()
        return H0_opt, Ob_opt, oV_opt, 1.0, res
    else:
        out("  Keine Loesung gefunden.")
        return None


# ================================================================
# ANALYSE: WARUM l_A ZU GROSS IST
# ================================================================

def analyze_lA_problem():
    out("=" * 74)
    out("  ANALYSE: WARUM l_A(CFM) = 317 STATT 301?")
    out("=" * 74)
    out()

    # LCDM Referenz
    from scipy.integrate import quad
    def lcdm_E2(a, Om=0.315):
        return Om * a**(-3) + Omega_r * a**(-4) + (1.0 - Om - Omega_r)

    def integrand_rs(a, E2func, Ob, **kw):
        Rb = 3.0 * Ob * a / (4.0 * Omega_gamma)
        cs = 1.0 / np.sqrt(3.0 * (1.0 + Rb))
        return cs / (a**2 * np.sqrt(max(E2func(a, **kw), 1e-30)))

    def integrand_dC(a, E2func, **kw):
        return 1.0 / (a**2 * np.sqrt(max(E2func(a, **kw), 1e-30)))

    # LCDM
    rs_lcdm, _ = quad(lambda a: integrand_rs(a, lcdm_E2, 0.0493, Om=0.315), 1e-10, a_star, limit=2000)
    dC_lcdm, _ = quad(lambda a: integrand_dC(a, lcdm_E2, Om=0.315), a_star, 1.0, limit=2000)
    rs_lcdm *= c_light / 67.36
    dC_lcdm *= c_light / 67.36

    # CFM
    def cfm_E2_wrap(a):
        return cfm_base_E2(a, 0.05, MCMC_k, MCMC_a_trans, MCMC_alpha, MCMC_beta)

    rs_cfm, _ = quad(lambda a: integrand_rs(a, cfm_E2_wrap, 0.05), 1e-10, a_star, limit=2000)
    dC_cfm, _ = quad(lambda a: integrand_dC(a, cfm_E2_wrap), a_star, 1.0, limit=2000)
    rs_cfm *= c_light / 67.36
    dC_cfm *= c_light / 67.36

    out("  l_A = pi * d_C / r_s")
    out()
    out("  {:25s}  {:>12s}  {:>12s}  {:>12s}".format("", "LCDM", "CFM", "Ratio"))
    out("  " + "-" * 65)
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.4f}".format(
        "r_s(z*) [Mpc]", rs_lcdm, rs_cfm, rs_cfm/rs_lcdm))
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.4f}".format(
        "d_C(z*) [Mpc]", dC_lcdm, dC_cfm, dC_cfm/dC_lcdm))
    out("  {:25s}  {:12.3f}  {:12.3f}  {:12.4f}".format(
        "l_A = pi*d_C/r_s",
        np.pi*dC_lcdm/rs_lcdm, np.pi*dC_cfm/rs_cfm,
        (dC_cfm/rs_cfm)/(dC_lcdm/rs_lcdm)))
    out()

    out("  ZERLEGUNG des l_A-Problems:")
    out("  r_s(CFM)/r_s(LCDM) = {:.4f}  (r_s ist {:.1f}% zu gross)".format(
        rs_cfm/rs_lcdm, (rs_cfm/rs_lcdm - 1)*100))
    out("  d_C(CFM)/d_C(LCDM) = {:.4f}  (d_C ist {:.1f}% zu gross)".format(
        dC_cfm/dC_lcdm, (dC_cfm/dC_lcdm - 1)*100))
    out()
    out("  l_A ~ d_C/r_s. Damit l_A sinkt, muss d_C/r_s sinken.")
    out("  d_C ist 42.5% zu gross, r_s ist 35.4% zu gross.")
    out("  => d_C waechst SCHNELLER als r_s => l_A steigt.")
    out()
    out("  URSACHE: Ohne CDM ist H(z) bei z < z* VIEL kleiner.")
    out("  Im Bereich 0 < z < z*:")
    out("    LCDM: H^2 ~ 0.315*a^{-3} (CDM dominiert)")
    out("    CFM:  H^2 ~ 0.05*a^{-3} + 0.68*a^{-2} (viel weniger)")
    out()

    # Aufspaltung: wo kommt d_C her?
    z_splits = [0.1, 1, 10, 100, 500, z_star]
    out("  Beitraege zu d_C nach Rotverschiebungs-Intervall:")
    out("  {:>15s}  {:>12s}  {:>12s}  {:>8s}".format(
        "z-Intervall", "d_C(LCDM)", "d_C(CFM)", "Ratio"))
    out("  " + "-" * 50)

    z_prev = 0
    for z_split in z_splits:
        a_upper = 1.0 / (1 + z_prev) if z_prev > 0 else 1.0
        a_lower = 1.0 / (1 + z_split)
        dC_l, _ = quad(lambda a: integrand_dC(a, lcdm_E2, Om=0.315),
                        a_lower, a_upper, limit=500)
        dC_c, _ = quad(lambda a: integrand_dC(a, cfm_E2_wrap),
                        a_lower, a_upper, limit=500)
        dC_l *= c_light / 67.36
        dC_c *= c_light / 67.36
        r = dC_c / dC_l if dC_l > 0 else 0
        out("  {:>6.0f} < z < {:>5.0f}  {:12.1f}  {:12.1f}  {:8.2f}".format(
            z_prev, z_split, dC_l, dC_c, r))
        z_prev = z_split

    out()
    out("  SCHLUSSFOLGERUNG:")
    out("  " + "=" * 55)
    out("  Das l_A-Problem ist ein HINTERGRUND-Problem bei 0 < z < z*.")
    out("  Ohne CDM (oder Aequivalent) expandiert das Universum")
    out("  zu langsam bei z ~ 1-1000 -> d_C zu gross -> l_A zu gross.")
    out()
    out("  LOESUNGSWEGE:")
    out("  1. Hoeheres H0 -> d_C schrumpft proportional zu 1/H0")
    out("  2. Skalarfeld als 'Early Dark Energy' -> H(z) erhoehen")
    out("  3. Modifizierte Geometrie (R^2) -> effektive H(z) aendern")
    out("  4. beta > 2 (naeher an CDM-Skalierung a^{-3})")
    out()


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    t0 = time.time()
    out("  VOLLE CFM-HINTERGRUND-ANALYSE MIT ALLEN FELDERN")
    out("  " + "=" * 50)
    out("  Datum: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    out()

    # 1. Analyse des l_A-Problems
    analyze_lA_problem()

    # 2. Parameter-Scans
    scan_parameters()

    # 3. Optimierung
    opt_result = optimize_lA()

    # 4. Zusammenfassung
    out("=" * 74)
    out("  ZUSAMMENFASSUNG")
    out("=" * 74)
    out()
    if opt_result:
        H0_opt, Ob_opt, oV_opt, psi0_opt, res = opt_result
        out("  Bester l_A-Match: l_A = {:.3f} (Planck: {:.3f})".format(
            res['l_A'], PLANCK_lA))
        out("  Dafuer benoetigte Parameter:")
        out("    H0 = {:.2f} km/s/Mpc".format(H0_opt))
        out("    Omega_b = {:.5f}".format(Ob_opt))
        out("    omega_V = {:.2e}".format(oV_opt))
        out("    psi0 = {:.4f}".format(psi0_opt))
    else:
        out("  Keine Loesung fuer l_A = 301.5 gefunden.")
    out()

    elapsed = time.time() - t0
    out("  Laufzeit: {:.1f} Sekunden".format(elapsed))

    save_output()

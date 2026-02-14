#!/usr/bin/env python3
"""
=============================================================================
CFM "Baryon-Only" Test: Vereinigung mit MOND
=============================================================================
Hypothese: Wenn MOND gilt, gibt es keine Dunkle Materie.
=> Omega_m = Omega_b ~ 0.05 (nur Baryonen)
=> Kann CFM die Pantheon+ Daten trotzdem fitten?

Drei Szenarien:
  A) Omega_m = 0.05 fest, CFM-Parameter frei
  B) Omega_m in [0.03, 0.07] (Baryonen-Band), CFM-Parameter frei
  C) Erweitertes CFM: Omega_Phi(a) mit zusätzlichem Materieterm
     (geometrischer Effekt kompensiert fehlende DM)

Autor: Lukas Geiger (mit Claude Opus 4.6)
Datum: Februar 2026
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Pantheon+SH0ES.dat")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_MIN = 0.01
N_GRID = 2000
C_LIGHT = 299792.458

# ==========================================================================
# DATEN
# ==========================================================================

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
    print(f"  {len(z)} Supernovae geladen (z = {z.min():.4f} bis {z.max():.4f})")
    return z, m_obs, m_err


# ==========================================================================
# DISTANZ-BERECHNUNGEN
# ==========================================================================

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


def omega_phi_tanh(a, Phi0, k, a_trans):
    s = np.tanh(k * a_trans)
    return Phi0 * (np.tanh(k * (a - a_trans)) + s) / (1.0 + s)


def omega_phi_extended(a, Phi0, k, a_trans, alpha, beta):
    """
    Erweitertes CFM: Geometrischer Effekt mit Materiekompensation.

    Omega_Phi(a) = Phi0 * [tanh(k*(a-at)) + s]/(1+s) + alpha * a^(-beta)

    Der zweite Term (alpha * a^(-beta)) simuliert einen geometrischen
    "Materie-ähnlichen" Beitrag, der bei kleinen a dominiert und für
    grosse a verschwindet. In einem MOND-Universum könnte dieser Term
    die Rolle der Dunklen Materie auf kosmologischen Skalen übernehmen.
    """
    s = np.tanh(k * a_trans)
    phi_de = Phi0 * (np.tanh(k * (a - a_trans)) + s) / (1.0 + s)
    phi_dm = alpha * a**(-beta)  # "geometrische Dunkle Materie"
    return phi_de + phi_dm


def distance_modulus_cfm(z_data, Omega_m, Phi0, k, a_trans, phi_func=omega_phi_tanh, extra_params=None):
    zg = _z_grid(z_data.max())
    ag = 1.0 / (1.0 + zg)
    if extra_params is not None:
        Omega_Phi = phi_func(ag, Phi0, k, a_trans, *extra_params)
    else:
        Omega_Phi = phi_func(ag, Phi0, k, a_trans)
    E2 = Omega_m * (1 + zg)**3 + Omega_Phi
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


def phi0_from_flatness(Omega_m, k, a_trans):
    s = np.tanh(k * a_trans)
    num = (1.0 - Omega_m) * (1.0 + s)
    den = np.tanh(k * (1.0 - a_trans)) + s
    if abs(den) < 1e-15:
        return 1e10
    return num / den


# ==========================================================================
# TEST A: Omega_m = 0.05 fest (reine Baryonen)
# ==========================================================================

def test_A_fixed_baryon(z, m_obs, m_err, Om_fixed=0.05):
    """CFM mit festem Omega_m = Omega_b (Baryon-Only)."""
    print("\n" + "="*70)
    print(f"  TEST A: CFM Baryon-Only (Omega_m = {Om_fixed} fest)")
    print("="*70)

    def objective(p):
        kk, at = p
        P0 = phi0_from_flatness(Om_fixed, kk, at)
        if P0 < 0 or P0 > 10.0:
            return 1e10
        try:
            mu = distance_modulus_cfm(z, Om_fixed, P0, kk, at)
            if np.any(np.isnan(mu)):
                return 1e10
            return chi2_marginalized(mu, m_obs, m_err)[0]
        except:
            return 1e10

    bounds = [(0.1, 100.0), (0.05, 0.95)]
    res = differential_evolution(objective, bounds, seed=42, maxiter=500,
                                 tol=1e-8, popsize=30, mutation=(0.5, 1.5),
                                 recombination=0.9, polish=True)

    kk, at = res.x
    P0 = phi0_from_flatness(Om_fixed, kk, at)
    mu_th = distance_modulus_cfm(z, Om_fixed, P0, kk, at)
    chi2, M = chi2_marginalized(mu_th, m_obs, m_err)

    n = len(z)
    k_eff = 3  # k, a_trans, M (Omega_m ist fest)
    aic = chi2 + 2 * k_eff
    bic = chi2 + k_eff * np.log(n)

    OPhi_today = omega_phi_tanh(np.array([1.0]), P0, kk, at)[0]

    result = {
        'name': 'CFM_baryon_fixed',
        'Omega_m': Om_fixed,
        'Phi0': P0,
        'k': kk,
        'a_trans': at,
        'z_trans': 1.0/at - 1.0,
        'Omega_Phi_today': OPhi_today,
        'Omega_total': Om_fixed + OPhi_today,
        'M': M,
        'chi2': chi2,
        'dof': n - k_eff,
        'chi2_dof': chi2 / (n - k_eff),
        'aic': aic,
        'bic': bic,
        'n_params': k_eff,
    }

    print(f"  Omega_m  = {Om_fixed:.4f} (fest)")
    print(f"  Phi0     = {P0:.4f}")
    print(f"  k        = {kk:.4f}")
    print(f"  a_trans   = {at:.4f}  (z_trans = {1.0/at-1:.2f})")
    print(f"  Omega_Phi(z=0) = {OPhi_today:.4f}")
    print(f"  Omega_total    = {Om_fixed + OPhi_today:.6f}")
    print(f"  M        = {M:.4f}")
    print(f"  chi2     = {chi2:.2f}")
    print(f"  chi2/dof = {chi2/(n-k_eff):.4f}  (dof={n-k_eff})")
    print(f"  AIC      = {aic:.2f}")
    print(f"  BIC      = {bic:.2f}")

    return result


# ==========================================================================
# TEST B: Omega_m in Baryonen-Band [0.03, 0.07]
# ==========================================================================

def test_B_baryon_band(z, m_obs, m_err):
    """CFM mit Omega_m im Baryonen-Band."""
    print("\n" + "="*70)
    print("  TEST B: CFM Baryonen-Band (0.03 <= Omega_m <= 0.07)")
    print("="*70)

    def objective(p):
        Om, kk, at = p
        P0 = phi0_from_flatness(Om, kk, at)
        if P0 < 0 or P0 > 10.0:
            return 1e10
        try:
            mu = distance_modulus_cfm(z, Om, P0, kk, at)
            if np.any(np.isnan(mu)):
                return 1e10
            return chi2_marginalized(mu, m_obs, m_err)[0]
        except:
            return 1e10

    bounds = [(0.03, 0.07), (0.1, 100.0), (0.05, 0.95)]
    res = differential_evolution(objective, bounds, seed=42, maxiter=500,
                                 tol=1e-8, popsize=30, mutation=(0.5, 1.5),
                                 recombination=0.9, polish=True)

    Om, kk, at = res.x
    P0 = phi0_from_flatness(Om, kk, at)
    mu_th = distance_modulus_cfm(z, Om, P0, kk, at)
    chi2, M = chi2_marginalized(mu_th, m_obs, m_err)

    n = len(z)
    k_eff = 4
    aic = chi2 + 2 * k_eff
    bic = chi2 + k_eff * np.log(n)

    OPhi_today = omega_phi_tanh(np.array([1.0]), P0, kk, at)[0]

    result = {
        'name': 'CFM_baryon_band',
        'Omega_m': Om,
        'Phi0': P0,
        'k': kk,
        'a_trans': at,
        'z_trans': 1.0/at - 1.0,
        'Omega_Phi_today': OPhi_today,
        'Omega_total': Om + OPhi_today,
        'M': M,
        'chi2': chi2,
        'dof': n - k_eff,
        'chi2_dof': chi2 / (n - k_eff),
        'aic': aic,
        'bic': bic,
        'n_params': k_eff,
    }

    print(f"  Omega_m  = {Om:.4f}")
    print(f"  Phi0     = {P0:.4f}")
    print(f"  k        = {kk:.4f}")
    print(f"  a_trans   = {at:.4f}  (z_trans = {1.0/at-1:.2f})")
    print(f"  Omega_Phi(z=0) = {OPhi_today:.4f}")
    print(f"  Omega_total    = {Om + OPhi_today:.6f}")
    print(f"  M        = {M:.4f}")
    print(f"  chi2     = {chi2:.2f}")
    print(f"  chi2/dof = {chi2/(n-k_eff):.4f}  (dof={n-k_eff})")
    print(f"  AIC      = {aic:.2f}")
    print(f"  BIC      = {bic:.2f}")

    return result


# ==========================================================================
# TEST C: Erweitertes CFM mit geometrischem DM-Ersatz
# ==========================================================================

def test_C_extended_cfm(z, m_obs, m_err, Om_fixed=0.05):
    """
    Erweitertes CFM: Omega_Phi hat einen zusätzlichen Term, der wie
    'geometrische Dunkle Materie' wirkt (skaliert ~a^(-beta)).
    """
    print("\n" + "="*70)
    print(f"  TEST C: Erweitertes CFM (Omega_m = {Om_fixed}, + geom. DM-Term)")
    print("="*70)

    def objective(p):
        kk, at, alpha, beta = p
        # Phi0 so, dass Omega_total(a=1) = 1
        # Omega_Phi(1) = Phi0 * f(1) + alpha * 1^(-beta)
        # => Phi0 = (1 - Om - alpha) / f(1)
        s = np.tanh(kk * at)
        f_at_1 = (np.tanh(kk * (1.0 - at)) + s) / (1.0 + s)
        if abs(f_at_1) < 1e-15:
            return 1e10
        P0 = (1.0 - Om_fixed - alpha) / f_at_1
        if P0 < -5.0 or P0 > 10.0:
            return 1e10
        try:
            mu = distance_modulus_cfm(z, Om_fixed, P0, kk, at,
                                       omega_phi_extended, extra_params=(alpha, beta))
            if np.any(np.isnan(mu)):
                return 1e10
            return chi2_marginalized(mu, m_obs, m_err)[0]
        except:
            return 1e10

    # alpha: Amplitude des DM-Ersatzterms
    # beta: Skalierung (beta~3 = materieähnlich, beta~2 = Strahlung)
    bounds = [(0.1, 100.0), (0.05, 0.95), (0.01, 0.50), (0.5, 4.0)]
    res = differential_evolution(objective, bounds, seed=42, maxiter=500,
                                 tol=1e-8, popsize=30, mutation=(0.5, 1.5),
                                 recombination=0.9, polish=True)

    kk, at, alpha, beta = res.x
    s = np.tanh(kk * at)
    f_at_1 = (np.tanh(kk * (1.0 - at)) + s) / (1.0 + s)
    P0 = (1.0 - Om_fixed - alpha) / f_at_1

    mu_th = distance_modulus_cfm(z, Om_fixed, P0, kk, at, omega_phi_extended,
                                  extra_params=(alpha, beta))
    chi2, M = chi2_marginalized(mu_th, m_obs, m_err)

    n = len(z)
    k_eff = 5  # k, a_trans, alpha, beta, M
    aic = chi2 + 2 * k_eff
    bic = chi2 + k_eff * np.log(n)

    # Omega_Phi(z=0) = gesamter geometrischer Beitrag
    OPhi_today = omega_phi_extended(np.array([1.0]), P0, kk, at, alpha, beta)[0]

    result = {
        'name': 'CFM_extended_MOND',
        'Omega_m': Om_fixed,
        'Phi0': P0,
        'k': kk,
        'a_trans': at,
        'z_trans': 1.0/at - 1.0,
        'alpha': alpha,
        'beta': beta,
        'Omega_Phi_today': OPhi_today,
        'Omega_total': Om_fixed + OPhi_today,
        'M': M,
        'chi2': chi2,
        'dof': n - k_eff,
        'chi2_dof': chi2 / (n - k_eff),
        'aic': aic,
        'bic': bic,
        'n_params': k_eff,
    }

    print(f"  Omega_m  = {Om_fixed:.4f} (fest, nur Baryonen)")
    print(f"  Phi0     = {P0:.4f}  (DE-Term)")
    print(f"  k        = {kk:.4f}")
    print(f"  a_trans   = {at:.4f}  (z_trans = {1.0/at-1:.2f})")
    print(f"  alpha    = {alpha:.4f}  (geom. DM-Amplitude)")
    print(f"  beta     = {beta:.4f}  (geom. DM-Skalierung)")
    print(f"  Omega_Phi(z=0) = {OPhi_today:.4f}")
    print(f"  Omega_total    = {Om_fixed + OPhi_today:.6f}")
    print(f"  M        = {M:.4f}")
    print(f"  chi2     = {chi2:.2f}")
    print(f"  chi2/dof = {chi2/(n-k_eff):.4f}  (dof={n-k_eff})")
    print(f"  AIC      = {aic:.2f}")
    print(f"  BIC      = {bic:.2f}")

    return result


# ==========================================================================
# REFERENZ-FITS
# ==========================================================================

def fit_lcdm(z, m_obs, m_err):
    print("\n" + "="*70)
    print("  REFERENZ: LCDM (Standard)")
    print("="*70)

    def objective(p):
        mu = distance_modulus_lcdm(z, p[0])
        return chi2_marginalized(mu, m_obs, m_err)[0]

    res = differential_evolution(objective, [(0.05, 0.60)], seed=42,
                                 maxiter=100, tol=1e-8, polish=True)
    mu_th = distance_modulus_lcdm(z, res.x[0])
    chi2, M = chi2_marginalized(mu_th, m_obs, m_err)

    n = len(z)
    k = 2
    result = {
        'name': 'LCDM',
        'Omega_m': res.x[0],
        'M': M,
        'chi2': chi2,
        'dof': n - k,
        'chi2_dof': chi2 / (n - k),
        'aic': chi2 + 2*k,
        'bic': chi2 + k*np.log(n),
        'n_params': k,
    }

    print(f"  Omega_m = {res.x[0]:.4f}, M = {M:.4f}")
    print(f"  chi2 = {chi2:.2f}, chi2/dof = {chi2/(n-k):.4f}")
    print(f"  AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")
    return result


def fit_cfm_standard(z, m_obs, m_err):
    print("\n" + "="*70)
    print("  REFERENZ: CFM Standard (Omega_m frei)")
    print("="*70)

    def objective(p):
        Om, kk, at = p
        P0 = phi0_from_flatness(Om, kk, at)
        if P0 < 0 or P0 > 5.0:
            return 1e10
        mu = distance_modulus_cfm(z, Om, P0, kk, at)
        if np.any(np.isnan(mu)):
            return 1e10
        return chi2_marginalized(mu, m_obs, m_err)[0]

    bounds = [(0.10, 0.50), (0.5, 50.0), (0.20, 0.75)]
    res = differential_evolution(objective, bounds, seed=42, maxiter=300,
                                 tol=1e-8, popsize=20, polish=True)

    Om, kk, at = res.x
    P0 = phi0_from_flatness(Om, kk, at)
    mu_th = distance_modulus_cfm(z, Om, P0, kk, at)
    chi2, M = chi2_marginalized(mu_th, m_obs, m_err)

    n = len(z)
    k_eff = 4
    OPhi_today = omega_phi_tanh(np.array([1.0]), P0, kk, at)[0]

    result = {
        'name': 'CFM_standard',
        'Omega_m': Om,
        'Phi0': P0,
        'k': kk,
        'a_trans': at,
        'z_trans': 1.0/at - 1.0,
        'Omega_Phi_today': OPhi_today,
        'M': M,
        'chi2': chi2,
        'dof': n - k_eff,
        'chi2_dof': chi2 / (n - k_eff),
        'aic': chi2 + 2*k_eff,
        'bic': chi2 + k_eff*np.log(n),
        'n_params': k_eff,
    }

    print(f"  Omega_m = {Om:.4f}, k = {kk:.2f}, a_trans = {at:.4f}")
    print(f"  Phi0 = {P0:.4f}, Omega_Phi(z=0) = {OPhi_today:.4f}")
    print(f"  chi2 = {chi2:.2f}, chi2/dof = {chi2/(n-k_eff):.4f}")
    print(f"  AIC = {result['aic']:.2f}, BIC = {result['bic']:.2f}")
    return result


# ==========================================================================
# VISUALISIERUNG
# ==========================================================================

def plot_results(z, m_obs, m_err, lcdm, cfm_std, testA, testB, testC):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: Hubble-Residuen ---
    ax = axes[0, 0]
    mu_lcdm = distance_modulus_lcdm(z, lcdm['Omega_m'])
    residuals_lcdm = m_obs - mu_lcdm - lcdm['M']

    ax.scatter(z, residuals_lcdm, s=1, alpha=0.3, c='gray', label='Daten (LCDM-Residuen)')

    # CFM Standard
    P0 = cfm_std['Phi0']
    mu_cfm = distance_modulus_cfm(z, cfm_std['Omega_m'], P0, cfm_std['k'], cfm_std['a_trans'])
    res_cfm = m_obs - mu_cfm - cfm_std['M']
    zs = np.argsort(z)

    # Baryon-Only
    P0_A = testA['Phi0']
    mu_A = distance_modulus_cfm(z, testA['Omega_m'], P0_A, testA['k'], testA['a_trans'])
    res_A = m_obs - mu_A - testA['M']

    # Extended
    mu_C = distance_modulus_cfm(z, testC['Omega_m'], testC['Phi0'], testC['k'], testC['a_trans'],
                                 omega_phi_extended, extra_params=(testC['alpha'], testC['beta']))
    res_C = m_obs - mu_C - testC['M']

    # Binned residuals
    zbins = np.logspace(np.log10(z.min()), np.log10(z.max()), 25)
    for label, res, color, ls in [
        ('LCDM', residuals_lcdm, 'black', '-'),
        (f"CFM (Om={cfm_std['Omega_m']:.2f})", res_cfm, '#E91E63', '-'),
        (f"Baryon-Only (Om={testA['Omega_m']:.2f})", res_A, '#2196F3', '--'),
        (f"Extended (Om={testC['Omega_m']:.2f})", res_C, '#4CAF50', '-.'),
    ]:
        bin_means = []
        bin_centers = []
        for i in range(len(zbins)-1):
            mask = (z >= zbins[i]) & (z < zbins[i+1])
            if mask.sum() > 5:
                bin_means.append(np.mean(res[mask]))
                bin_centers.append(np.sqrt(zbins[i] * zbins[i+1]))
        ax.plot(bin_centers, bin_means, ls, color=color, lw=2, label=label)

    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Redshift z')
    ax.set_ylabel('Residuen (mag)')
    ax.set_title('Hubble-Residuen vs. LCDM')
    ax.legend(fontsize=8)
    ax.set_xscale('log')
    ax.set_ylim(-0.3, 0.3)

    # --- Panel 2: Omega_Phi(a) Vergleich ---
    ax = axes[0, 1]
    a = np.linspace(0.01, 1.5, 500)

    # Standard CFM
    OPhi_std = omega_phi_tanh(a, cfm_std['Phi0'], cfm_std['k'], cfm_std['a_trans'])
    ax.plot(a, OPhi_std, '-', color='#E91E63', lw=2.5,
            label=f"CFM Standard ($\\Omega_m$={cfm_std['Omega_m']:.3f})")

    # Baryon-Only
    OPhi_A = omega_phi_tanh(a, testA['Phi0'], testA['k'], testA['a_trans'])
    ax.plot(a, OPhi_A, '--', color='#2196F3', lw=2.5,
            label=f"Baryon-Only ($\\Omega_m$={testA['Omega_m']:.3f})")

    # Extended
    OPhi_C = omega_phi_extended(a, testC['Phi0'], testC['k'], testC['a_trans'],
                                 testC['alpha'], testC['beta'])
    ax.plot(a, OPhi_C, '-.', color='#4CAF50', lw=2.5,
            label=f"Extended ($\\Omega_m$={testC['Omega_m']:.3f})")

    ax.axhline(0.75, color='gray', ls=':', lw=1, alpha=0.5, label='$\\Omega_\\Lambda$ (LCDM)')
    ax.axvline(1.0, color='black', ls=':', lw=0.5, alpha=0.3)
    ax.set_xlabel('Skalenfaktor $a$')
    ax.set_ylabel('$\\Omega_\\Phi(a)$')
    ax.set_title('Geometrisches Potential: Standard vs. Baryon-Only')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-0.2, 2.5)

    # --- Panel 3: Chi2 Vergleich ---
    ax = axes[1, 0]
    models = ['LCDM', 'CFM\nStandard', 'Baryon\nFixed', 'Baryon\nBand', 'Extended\nCFM+MOND']
    chi2s = [lcdm['chi2'], cfm_std['chi2'], testA['chi2'], testB['chi2'], testC['chi2']]
    aics = [lcdm['aic'], cfm_std['aic'], testA['aic'], testB['aic'], testC['aic']]
    colors = ['gray', '#E91E63', '#2196F3', '#03A9F4', '#4CAF50']

    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos - 0.15, chi2s, 0.3, color=colors, alpha=0.8, label='$\\chi^2$')
    bars2 = ax.bar(x_pos + 0.15, aics, 0.3, color=colors, alpha=0.4, label='AIC')

    for i, (c, a_val) in enumerate(zip(chi2s, aics)):
        ax.text(i - 0.15, c + 5, f'{c:.0f}', ha='center', fontsize=8)
        ax.text(i + 0.15, a_val + 5, f'{a_val:.0f}', ha='center', fontsize=8, alpha=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel('$\\chi^2$ / AIC')
    ax.set_title('Modellvergleich: $\\chi^2$ und AIC')
    ax.legend()

    # --- Panel 4: Zusammenfassung ---
    ax = axes[1, 1]
    ax.axis('off')

    summary = []
    summary.append("ERGEBNIS: Baryon-Only CFM + MOND Test")
    summary.append("=" * 48)
    summary.append("")
    summary.append(f"{'Modell':<22} {'chi2':>8} {'AIC':>8} {'BIC':>8} {'Params':>6}")
    summary.append("-" * 56)

    for name, r in [('LCDM', lcdm), ('CFM Standard', cfm_std),
                     ('Baryon Fixed', testA), ('Baryon Band', testB),
                     ('Extended CFM', testC)]:
        summary.append(f"{name:<22} {r['chi2']:>8.1f} {r['aic']:>8.1f} {r['bic']:>8.1f} {r['n_params']:>6d}")

    summary.append("")
    summary.append(f"Delta chi2 (Baryon vs LCDM): {testA['chi2'] - lcdm['chi2']:+.1f}")
    summary.append(f"Delta AIC  (Baryon vs LCDM): {testA['aic'] - lcdm['aic']:+.1f}")
    summary.append(f"Delta chi2 (Extended vs LCDM): {testC['chi2'] - lcdm['chi2']:+.1f}")
    summary.append(f"Delta AIC  (Extended vs LCDM): {testC['aic'] - lcdm['aic']:+.1f}")

    text = "\n".join(summary)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('CFM + MOND: Baryon-Only Universe Test gegen Pantheon+',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'CFM_MOND_Baryon_Only_Test.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\n  Plot gespeichert: {outpath}")
    plt.close()


# ==========================================================================
# ERGEBNIS-REPORT
# ==========================================================================

def write_report(lcdm, cfm_std, testA, testB, testC):
    outpath = os.path.join(OUTPUT_DIR, 'CFM_MOND_Baryon_Only_Results.txt')

    lines = []
    lines.append("=" * 72)
    lines.append("CFM + MOND: BARYON-ONLY UNIVERSE TEST")
    lines.append("=" * 72)
    lines.append(f"Datum: Februar 2026")
    lines.append(f"Datensatz: Pantheon+ (Scolnic et al. 2022)")
    lines.append(f"Hypothese: Wenn MOND gilt, ist Omega_m = Omega_b ~ 0.05")
    lines.append("")

    lines.append("-" * 72)
    lines.append("REFERENZ: LCDM")
    lines.append("-" * 72)
    lines.append(f"  Omega_m = {lcdm['Omega_m']:.4f}")
    lines.append(f"  chi2    = {lcdm['chi2']:.2f}")
    lines.append(f"  AIC     = {lcdm['aic']:.2f}")
    lines.append(f"  BIC     = {lcdm['bic']:.2f}")
    lines.append(f"  Params  = {lcdm['n_params']}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("REFERENZ: CFM Standard (Omega_m frei)")
    lines.append("-" * 72)
    lines.append(f"  Omega_m = {cfm_std['Omega_m']:.4f}")
    lines.append(f"  k       = {cfm_std['k']:.4f}")
    lines.append(f"  a_trans  = {cfm_std['a_trans']:.4f}")
    lines.append(f"  Phi0    = {cfm_std['Phi0']:.4f}")
    lines.append(f"  chi2    = {cfm_std['chi2']:.2f}")
    lines.append(f"  AIC     = {cfm_std['aic']:.2f}")
    lines.append(f"  BIC     = {cfm_std['bic']:.2f}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("TEST A: CFM Baryon-Only (Omega_m = 0.05 fest)")
    lines.append("-" * 72)
    lines.append(f"  Omega_m = {testA['Omega_m']:.4f} (fest)")
    lines.append(f"  k       = {testA['k']:.4f}")
    lines.append(f"  a_trans  = {testA['a_trans']:.4f}  (z_trans = {testA['z_trans']:.2f})")
    lines.append(f"  Phi0    = {testA['Phi0']:.4f}")
    lines.append(f"  Omega_Phi(z=0) = {testA['Omega_Phi_today']:.4f}")
    lines.append(f"  Omega_total    = {testA['Omega_total']:.6f}")
    lines.append(f"  chi2    = {testA['chi2']:.2f}")
    lines.append(f"  AIC     = {testA['aic']:.2f}")
    lines.append(f"  BIC     = {testA['bic']:.2f}")
    lines.append(f"  Delta chi2 vs LCDM = {testA['chi2'] - lcdm['chi2']:+.2f}")
    lines.append(f"  Delta AIC  vs LCDM = {testA['aic'] - lcdm['aic']:+.2f}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("TEST B: CFM Baryonen-Band (0.03 <= Omega_m <= 0.07)")
    lines.append("-" * 72)
    lines.append(f"  Omega_m = {testB['Omega_m']:.4f}")
    lines.append(f"  k       = {testB['k']:.4f}")
    lines.append(f"  a_trans  = {testB['a_trans']:.4f}  (z_trans = {testB['z_trans']:.2f})")
    lines.append(f"  Phi0    = {testB['Phi0']:.4f}")
    lines.append(f"  Omega_Phi(z=0) = {testB['Omega_Phi_today']:.4f}")
    lines.append(f"  Omega_total    = {testB['Omega_total']:.6f}")
    lines.append(f"  chi2    = {testB['chi2']:.2f}")
    lines.append(f"  AIC     = {testB['aic']:.2f}")
    lines.append(f"  BIC     = {testB['bic']:.2f}")
    lines.append(f"  Delta chi2 vs LCDM = {testB['chi2'] - lcdm['chi2']:+.2f}")
    lines.append(f"  Delta AIC  vs LCDM = {testB['aic'] - lcdm['aic']:+.2f}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("TEST C: Erweitertes CFM (geom. DM-Ersatz)")
    lines.append("-" * 72)
    lines.append(f"  Omega_m = {testC['Omega_m']:.4f} (fest, nur Baryonen)")
    lines.append(f"  k       = {testC['k']:.4f}")
    lines.append(f"  a_trans  = {testC['a_trans']:.4f}  (z_trans = {testC['z_trans']:.2f})")
    lines.append(f"  Phi0    = {testC['Phi0']:.4f}  (DE-Anteil)")
    lines.append(f"  alpha   = {testC['alpha']:.4f}  (geom. DM-Amplitude)")
    lines.append(f"  beta    = {testC['beta']:.4f}  (geom. DM-Skalierung)")
    lines.append(f"  Omega_Phi(z=0) = {testC['Omega_Phi_today']:.4f}")
    lines.append(f"  Omega_total    = {testC['Omega_total']:.6f}")
    lines.append(f"  chi2    = {testC['chi2']:.2f}")
    lines.append(f"  AIC     = {testC['aic']:.2f}")
    lines.append(f"  BIC     = {testC['bic']:.2f}")
    lines.append(f"  Delta chi2 vs LCDM = {testC['chi2'] - lcdm['chi2']:+.2f}")
    lines.append(f"  Delta AIC  vs LCDM = {testC['aic'] - lcdm['aic']:+.2f}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("ZUSAMMENFASSUNG")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"{'Modell':<25} {'chi2':>8} {'AIC':>8} {'BIC':>8} {'dchi2':>8} {'dAIC':>8}")
    lines.append("-" * 72)
    for name, r in [('LCDM', lcdm), ('CFM Standard', cfm_std),
                     ('CFM Baryon Fixed', testA), ('CFM Baryon Band', testB),
                     ('CFM Extended+MOND', testC)]:
        dc = r['chi2'] - lcdm['chi2']
        da = r['aic'] - lcdm['aic']
        lines.append(f"{name:<25} {r['chi2']:>8.1f} {r['aic']:>8.1f} {r['bic']:>8.1f} {dc:>+8.1f} {da:>+8.1f}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("BEWERTUNG")
    lines.append("=" * 72)

    # Automatische Bewertung
    dchi2_A = testA['chi2'] - lcdm['chi2']
    dchi2_C = testC['chi2'] - lcdm['chi2']

    if dchi2_A < 50:
        lines.append("")
        lines.append("  ERGEBNIS: Das einfache Baryon-Only CFM (Test A) liefert einen")
        lines.append(f"  chi2-Unterschied von {dchi2_A:+.1f} gegenueber LCDM.")
        if dchi2_A < 0:
            lines.append("  => SENSATION: Baryon-Only CFM fittet BESSER als LCDM!")
        elif dchi2_A < 20:
            lines.append("  => VIELVERSPRECHEND: Der Fit degradiert nur moderat.")
        else:
            lines.append("  => MODERAT: Der Fit degradiert, aber bleibt im Rahmen.")
    else:
        lines.append("")
        lines.append(f"  ERGEBNIS: Das einfache Baryon-Only CFM (Test A) degradiert stark")
        lines.append(f"  (Delta chi2 = {dchi2_A:+.1f}).")

    lines.append("")
    if dchi2_C < 0:
        lines.append("  Das erweiterte CFM (Test C) mit geometrischem DM-Ersatz")
        lines.append(f"  fittet BESSER als LCDM (Delta chi2 = {dchi2_C:+.1f})!")
        lines.append("  => Die Vereinigung von CFM + MOND ist prinzipiell MOEGLICH.")
    elif dchi2_C < 20:
        lines.append("  Das erweiterte CFM (Test C) mit geometrischem DM-Ersatz")
        lines.append(f"  zeigt nur moderate Degradation (Delta chi2 = {dchi2_C:+.1f}).")
        lines.append("  => Die Vereinigung verdient weitere Untersuchung.")
    else:
        lines.append("  Das erweiterte CFM (Test C) mit geometrischem DM-Ersatz")
        lines.append(f"  degradiert erheblich (Delta chi2 = {dchi2_C:+.1f}).")

    lines.append("")

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n  Report gespeichert: {outpath}")
    return '\n'.join(lines)


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  CFM + MOND: BARYON-ONLY UNIVERSE TEST")
    print("  Hypothese: Kein dunkler Sektor (keine DE, keine DM)")
    print("=" * 72)

    # Daten laden
    print("\n[1/7] Lade Pantheon+ Daten...")
    z, m_obs, m_err = load_data()

    # Referenz-Fits
    print("\n[2/7] Referenz: LCDM...")
    lcdm = fit_lcdm(z, m_obs, m_err)

    print("\n[3/7] Referenz: CFM Standard...")
    cfm_std = fit_cfm_standard(z, m_obs, m_err)

    # Baryon-Only Tests
    print("\n[4/7] Test A: Baryon-Only (Omega_m = 0.05 fest)...")
    testA = test_A_fixed_baryon(z, m_obs, m_err, Om_fixed=0.05)

    print("\n[5/7] Test B: Baryonen-Band (0.03 <= Omega_m <= 0.07)...")
    testB = test_B_baryon_band(z, m_obs, m_err)

    print("\n[6/7] Test C: Erweitertes CFM + geom. DM-Ersatz...")
    testC = test_C_extended_cfm(z, m_obs, m_err, Om_fixed=0.05)

    # Ergebnisse
    print("\n[7/7] Erstelle Visualisierung und Report...")
    plot_results(z, m_obs, m_err, lcdm, cfm_std, testA, testB, testC)
    report = write_report(lcdm, cfm_std, testA, testB, testC)

    print("\n" + report)
    print("\n  FERTIG.")

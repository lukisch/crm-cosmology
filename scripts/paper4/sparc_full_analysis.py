#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFM Paper IV: Full SPARC Analysis (175 Galaxies)
=================================================
Tests the CFM prediction a_0 = cH_0/(2*pi) against the full SPARC database.

Three runs:
  Run A: a_0 free per galaxy (McGaugh interpolation)
  Run B: a_0 = cH_0/(2*pi) fixed (CFM prediction, 0 free parameters for a_0)
  Run C: a_0 global free (1 parameter)

Reads real SPARC rotation curve data from Lelli, McGaugh & Schombert (2016).

Author: L. Geiger / Claude Code
Date: 2026-02-22
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from pathlib import Path
import time
import os
import sys
import json
import subprocess

# ============================================================
# Physical constants
# ============================================================
G = 6.67430e-11          # m^3 kg^-1 s^-2
c_light = 2.99792458e8   # m/s
KPC = 3.0856775814e19    # m
MSUN = 1.98892e30        # kg
MPC = KPC * 1e3
H0_PLANCK = 67.36e3 / MPC  # s^-1 (Planck 2018)
A0_CFM = c_light * H0_PLANCK / (2 * np.pi)  # ~1.042e-10 m/s^2
A0_OBS = 1.20e-10        # m/s^2 (McGaugh+2016)

UPSILON_DISK = 0.5       # M/L at 3.6um (Schombert+2014)
UPSILON_BUL = 0.7        # Bulge M/L at 3.6um

# Paths
SPARC_DIR = Path("/home/cfm-cosmology/data/sparc/rotmod")
SPARC_TABLE = Path("/home/cfm-cosmology/data/sparc/SPARC_Lelli2016c.mrt")
RESULTS_DIR = Path("/home/cfm-cosmology/results/paper4/sparc")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Telegram
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "595767047")


def send_telegram(msg):
    """Send a Telegram notification."""
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            "-d", f"chat_id={TELEGRAM_CHAT}",
            "-d", f"text=[Paper IV SPARC] {msg}"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


# ============================================================
# SPARC data parser
# ============================================================
def parse_sparc_galaxy(filepath):
    """
    Parse a single SPARC rotation curve file.

    Columns: R [kpc], Vobs [km/s], errV [km/s], Vgas [km/s],
             Vdisk [km/s], Vbul [km/s], SBdisk [L/pc^2], SBbul [L/pc^2]
    """
    data = {'R': [], 'Vobs': [], 'errV': [], 'Vgas': [],
            'Vdisk': [], 'Vbul': [], 'SBdisk': [], 'SBbul': []}
    distance = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Distance'):
                parts = line.split('=')
                if len(parts) == 2:
                    distance = float(parts[1].strip().split()[0])
                continue
            if line.startswith('#') or len(line) == 0:
                continue

            parts = line.split()
            if len(parts) >= 6:
                data['R'].append(float(parts[0]))
                data['Vobs'].append(float(parts[1]))
                data['errV'].append(float(parts[2]))
                data['Vgas'].append(float(parts[3]))
                data['Vdisk'].append(float(parts[4]))
                data['Vbul'].append(float(parts[5]))
                if len(parts) >= 7:
                    data['SBdisk'].append(float(parts[6]))
                else:
                    data['SBdisk'].append(0.0)
                if len(parts) >= 8:
                    data['SBbul'].append(float(parts[7]))
                else:
                    data['SBbul'].append(0.0)

    for key in data:
        data[key] = np.array(data[key])

    data['distance_Mpc'] = distance
    data['n_pts'] = len(data['R'])
    return data


def parse_sparc_table(filepath):
    """Parse the SPARC galaxy properties table (MRT format)."""
    galaxies = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find data start (after the last separator line)
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---') and i > 10:
            data_start = i + 1

    for line in lines[data_start:]:
        line = line.rstrip()
        if len(line) < 90:
            continue

        try:
            name = line[0:11].strip()
            hubble_type = int(line[11:13].strip()) if line[11:13].strip() else 0
            dist = float(line[13:19].strip()) if line[13:19].strip() else 0
            inc = float(line[26:30].strip()) if line[26:30].strip() else 0
            lum_36 = float(line[34:41].strip()) if line[34:41].strip() else 0
            r_eff = float(line[48:53].strip()) if line[48:53].strip() else 0
            r_disk = float(line[53:61].strip()) if line[53:61].strip() else 0
            m_hi = float(line[74:81].strip()) if line[74:81].strip() else 0
            v_flat = float(line[86:91].strip()) if line[86:91].strip() else 0
            quality = int(line[96:99].strip()) if line[96:99].strip() else 3

            galaxies[name] = {
                'hubble_type': hubble_type,
                'distance_Mpc': dist,
                'inclination': inc,
                'luminosity_1e9Lsun': lum_36,
                'R_eff_kpc': r_eff,
                'R_disk_kpc': r_disk,
                'M_HI_1e9Msun': m_hi,
                'V_flat': v_flat,
                'quality': quality,
            }
        except (ValueError, IndexError):
            continue

    return galaxies


def load_all_sparc():
    """Load all SPARC galaxies: rotation curves + properties."""
    props = parse_sparc_table(SPARC_TABLE)
    galaxy_files = sorted(SPARC_DIR.glob("*_rotmod.dat"))

    galaxies = []
    for fpath in galaxy_files:
        name = fpath.stem.replace("_rotmod", "")
        data = parse_sparc_galaxy(fpath)

        if data['n_pts'] < 5:
            continue

        if name in props:
            data.update(props[name])

        data['name'] = name
        data['file'] = str(fpath)
        galaxies.append(data)

    return galaxies


# ============================================================
# Physics: McGaugh interpolation function
# ============================================================
def mcgaugh_interpolation(g_bar, a0):
    """McGaugh et al. (2016) RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))"""
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-30)
    return g_bar / denom


def compute_g_bar(R_kpc, Vgas, Vdisk, Vbul, Upsilon_disk=UPSILON_DISK, Upsilon_bul=UPSILON_BUL):
    """Compute baryonic acceleration from SPARC velocity components."""
    R_m = R_kpc * KPC
    V_gas_sq = np.sign(Vgas) * Vgas**2
    V_disk_sq = np.sign(Vdisk) * Vdisk**2
    V_bul_sq = np.sign(Vbul) * Vbul**2
    V_bar_sq = V_gas_sq + Upsilon_disk * V_disk_sq + Upsilon_bul * V_bul_sq
    g_bar = V_bar_sq * 1e6 / R_m
    return g_bar, V_bar_sq


def chi2_galaxy(params, R_kpc, Vobs, errV, Vgas, Vdisk, Vbul, a0, fit_a0=False):
    """Chi^2 for a single galaxy."""
    Y_scale = 10**(params[0])

    if fit_a0:
        a0_use = params[1]
        if a0_use < 1e-12 or a0_use > 1e-8:
            return 1e10
    else:
        a0_use = a0

    R_m = R_kpc * KPC
    g_bar, _ = compute_g_bar(R_kpc, Vgas, Vdisk, Vbul,
                              Upsilon_disk=UPSILON_DISK * Y_scale,
                              Upsilon_bul=UPSILON_BUL)

    pos = g_bar > 0
    if np.sum(pos) < 3:
        return 1e10

    g_obs_model = mcgaugh_interpolation(g_bar[pos], a0_use)
    V_model = np.sqrt(np.maximum(g_obs_model * R_m[pos], 0)) / 1e3
    V_obs_pos = Vobs[pos]
    errV_pos = np.maximum(errV[pos], 1.0)

    return np.sum(((V_obs_pos - V_model) / errV_pos)**2)


def fit_galaxy(R_kpc, Vobs, errV, Vgas, Vdisk, Vbul, a0, fit_a0=False):
    """Fit a single galaxy rotation curve."""
    if fit_a0:
        x0 = [0.0, A0_OBS]
        result = minimize(chi2_galaxy, x0,
                          args=(R_kpc, Vobs, errV, Vgas, Vdisk, Vbul, a0, True),
                          method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
        a0_fit = result.x[1]
    else:
        x0 = [0.0]
        result = minimize(chi2_galaxy, x0,
                          args=(R_kpc, Vobs, errV, Vgas, Vdisk, Vbul, a0, False),
                          method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6})
        a0_fit = a0

    chi2 = result.fun
    ndf = max(len(Vobs) - len(result.x), 1)

    return {
        'chi2': chi2, 'ndf': ndf, 'chi2_red': chi2 / ndf,
        'Upsilon_scale': 10**(result.x[0]),
        'a0_fit': a0_fit, 'success': result.success,
    }


# ============================================================
# Main analysis
# ============================================================
def main():
    t_start = time.time()

    print("=" * 70)
    print("CFM Paper IV: Full SPARC Analysis (175 Galaxies)")
    print("=" * 70)
    print(f"a0(CFM)  = c*H0/(2*pi) = {A0_CFM:.4e} m/s^2")
    print(f"a0(obs)  = {A0_OBS:.4e} m/s^2")
    print(f"Ratio    = {A0_CFM/A0_OBS:.4f}")
    print()

    send_telegram(f"Analyse gestartet. a0_CFM={A0_CFM:.3e}")

    # Load
    print("Loading SPARC data ...")
    galaxies = load_all_sparc()
    good_galaxies = [g for g in galaxies if g['n_pts'] >= 5]
    print(f"  Loaded {len(good_galaxies)} galaxies")

    # === Run A ===
    print("\n" + "-" * 70)
    print("RUN A: a0 free per galaxy")
    print("-" * 70)

    results_A = {}
    chi2_total_A = 0
    ndf_total_A = 0

    for i, gal in enumerate(good_galaxies):
        try:
            fit = fit_galaxy(gal['R'], gal['Vobs'], gal['errV'],
                             gal['Vgas'], gal['Vdisk'], gal['Vbul'],
                             a0=A0_OBS, fit_a0=True)
            results_A[gal['name']] = fit
            chi2_total_A += fit['chi2']
            ndf_total_A += fit['ndf']
        except Exception as e:
            results_A[gal['name']] = {'chi2': 0, 'ndf': 0, 'chi2_red': np.nan,
                                       'a0_fit': np.nan, 'Upsilon_scale': np.nan}

        if (i + 1) % 50 == 0:
            msg = f"Run A: {i+1}/{len(good_galaxies)} done. chi2/dof={chi2_total_A/max(ndf_total_A,1):.3f}"
            print(f"  {msg}")
            send_telegram(msg)

    print(f"  Run A: chi2/dof = {chi2_total_A/max(ndf_total_A,1):.3f}")
    a0_vals_A = [results_A[n]['a0_fit'] for n in results_A
                 if np.isfinite(results_A[n].get('a0_fit', np.nan))]

    # === Run B ===
    print("\n" + "-" * 70)
    print(f"RUN B: CFM a0 = {A0_CFM:.4e} FIXED")
    print("-" * 70)

    results_B = {}
    chi2_total_B = 0
    ndf_total_B = 0

    for i, gal in enumerate(good_galaxies):
        try:
            fit = fit_galaxy(gal['R'], gal['Vobs'], gal['errV'],
                             gal['Vgas'], gal['Vdisk'], gal['Vbul'],
                             a0=A0_CFM, fit_a0=False)
            results_B[gal['name']] = fit
            chi2_total_B += fit['chi2']
            ndf_total_B += fit['ndf']
        except Exception:
            results_B[gal['name']] = {'chi2': 0, 'ndf': 0, 'chi2_red': np.nan,
                                       'Upsilon_scale': np.nan}

        if (i + 1) % 50 == 0:
            msg = f"Run B: {i+1}/{len(good_galaxies)} done. chi2/dof={chi2_total_B/max(ndf_total_B,1):.3f}"
            print(f"  {msg}")
            send_telegram(msg)

    print(f"  Run B: chi2/dof = {chi2_total_B/max(ndf_total_B,1):.3f}")

    # === Run C ===
    print("\n" + "-" * 70)
    print("RUN C: global a0 free")
    print("-" * 70)

    send_telegram("Run C: Optimizing global a0...")

    def total_chi2_global(log_a0):
        a0_try = 10**log_a0
        chi2 = 0
        for gal in good_galaxies:
            try:
                fit = fit_galaxy(gal['R'], gal['Vobs'], gal['errV'],
                                 gal['Vgas'], gal['Vdisk'], gal['Vbul'],
                                 a0=a0_try, fit_a0=False)
                chi2 += fit['chi2']
            except Exception:
                pass
        return chi2

    res_opt = minimize_scalar(total_chi2_global, bounds=(-10.5, -9.0), method='bounded',
                              options={'xatol': 0.001})
    a0_best = 10**res_opt.x

    results_C = {}
    chi2_total_C = 0
    ndf_total_C = 0
    for gal in good_galaxies:
        try:
            fit = fit_galaxy(gal['R'], gal['Vobs'], gal['errV'],
                             gal['Vgas'], gal['Vdisk'], gal['Vbul'],
                             a0=a0_best, fit_a0=False)
            results_C[gal['name']] = fit
            chi2_total_C += fit['chi2']
            ndf_total_C += fit['ndf']
        except Exception:
            pass

    print(f"  Best a0 = {a0_best:.4e}, chi2/dof = {chi2_total_C/max(ndf_total_C,1):.3f}")

    # === Comparison ===
    chi2_red_A = chi2_total_A / max(ndf_total_A, 1)
    chi2_red_B = chi2_total_B / max(ndf_total_B, 1)
    chi2_red_C = chi2_total_C / max(ndf_total_C, 1)
    delta_BA = chi2_total_B - chi2_total_A
    delta_BC = chi2_total_B - chi2_total_C

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"  Run A (free a0):  chi2/dof = {chi2_red_A:.3f}")
    print(f"  Run B (CFM fix):  chi2/dof = {chi2_red_B:.3f}")
    print(f"  Run C (global):   chi2/dof = {chi2_red_C:.3f}")
    print(f"  Delta(B-A) = {delta_BA:+.1f}, Delta(B-C) = {delta_BC:+.1f}")
    print(f"  Best a0 = {a0_best*1e10:.3f} x 10^-10")

    comparison_msg = (
        f"ERGEBNIS:\n"
        f"A(frei): {chi2_red_A:.3f}\n"
        f"B(CFM):  {chi2_red_B:.3f}\n"
        f"C(glob): {chi2_red_C:.3f}\n"
        f"Best a0={a0_best*1e10:.3f}e-10"
    )
    send_telegram(comparison_msg)

    # === Plots ===
    print("\nGenerating plots ...")

    # RAR Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    g_bar_all = []
    g_obs_all = []
    for gal in good_galaxies:
        g_bar, _ = compute_g_bar(gal['R'], gal['Vgas'], gal['Vdisk'], gal['Vbul'])
        g_obs = gal['Vobs']**2 * 1e6 / (gal['R'] * KPC)
        pos = (g_bar > 0) & (g_obs > 0)
        g_bar_all.extend(g_bar[pos])
        g_obs_all.extend(g_obs[pos])

    g_bar_all = np.array(g_bar_all)
    g_obs_all = np.array(g_obs_all)

    ax = axes[0]
    ax.scatter(np.log10(g_bar_all), np.log10(g_obs_all), s=0.5, alpha=0.2, c='gray')
    gr = np.logspace(-14, -8, 200)
    ax.plot(np.log10(gr), np.log10(mcgaugh_interpolation(gr, A0_OBS)),
            'g-', lw=2, label=f'MOND (a0={A0_OBS:.2e})')
    ax.plot(np.log10(gr), np.log10(mcgaugh_interpolation(gr, A0_CFM)),
            'r--', lw=2, label=f'CFM (a0={A0_CFM:.2e})')
    ax.plot(np.log10(gr), np.log10(gr), 'k:', lw=1, alpha=0.5)
    ax.set_xlabel(r'$\log_{10}(g_{\rm bar})$ [m/s$^2$]')
    ax.set_ylabel(r'$\log_{10}(g_{\rm obs})$ [m/s$^2$]')
    ax.set_title('RAR')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(-13, -8.5); ax.set_ylim(-13, -8.5); ax.set_aspect('equal')

    ax = axes[1]
    res_mond = np.log10(g_obs_all) - np.log10(mcgaugh_interpolation(g_bar_all, A0_OBS))
    res_cfm = np.log10(g_obs_all) - np.log10(mcgaugh_interpolation(g_bar_all, A0_CFM))
    ax.scatter(np.log10(g_bar_all), res_mond, s=0.3, alpha=0.15, c='green', label='MOND')
    ax.scatter(np.log10(g_bar_all), res_cfm, s=0.3, alpha=0.15, c='red', label='CFM')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel(r'$\log_{10}(g_{\rm bar})$')
    ax.set_ylabel('Residuals')
    ax.set_title('RAR Residuals')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(-0.5, 0.5)

    ax = axes[2]
    a0_valid = [v for v in a0_vals_A if 0.1e-10 < v < 5e-10]
    if a0_valid:
        ax.hist(np.array(a0_valid) * 1e10, bins=30, alpha=0.6, color='blue')
    ax.axvline(A0_CFM * 1e10, color='red', ls='-', lw=2, label=f'CFM: {A0_CFM*1e10:.3f}')
    ax.axvline(A0_OBS * 1e10, color='green', ls='--', lw=2, label=f'Obs: {A0_OBS*1e10:.2f}')
    ax.axvline(a0_best * 1e10, color='purple', ls=':', lw=2, label=f'Best: {a0_best*1e10:.3f}')
    ax.set_xlabel(r'$a_0$ [$10^{-10}$ m/s$^2$]')
    ax.set_title('a0 Distribution (Run A)')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f'SPARC ({len(good_galaxies)} gal): A={chi2_red_A:.2f}, B(CFM)={chi2_red_B:.2f}, C={chi2_red_C:.2f}')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'rar_full_sparc.png', dpi=300)
    plt.close(fig)

    # BTFR
    fig, ax = plt.subplots(figsize=(9, 7))
    M_bar_arr = []; V_flat_arr = []; q_arr = []
    for gal in good_galaxies:
        L = gal.get('luminosity_1e9Lsun', 0)
        MHI = gal.get('M_HI_1e9Msun', 0)
        Vf = gal.get('V_flat', 0)
        q = gal.get('quality', 3)
        if L > 0 and Vf > 0:
            M_bar_arr.append(L * 1e9 * UPSILON_DISK + MHI * 1e9 * 1.33)
            V_flat_arr.append(Vf)
            q_arr.append(q)
    M_bar_arr = np.array(M_bar_arr); V_flat_arr = np.array(V_flat_arr); q_arr = np.array(q_arr)

    for q, col in [(1, 'blue'), (2, 'orange'), (3, 'gray')]:
        m = q_arr == q
        if np.any(m):
            ax.scatter(np.log10(M_bar_arr[m] * MSUN), np.log10(V_flat_arr[m]),
                       s=20, c=col, alpha=0.7, label=f'Q={q}', edgecolors='k', linewidths=0.3)

    M_range = np.logspace(7, 12, 100)
    ax.plot(np.log10(M_range * MSUN), np.log10((G * M_range * MSUN * A0_OBS)**0.25 / 1e3),
            'g--', lw=2.5, label=f'MOND')
    ax.plot(np.log10(M_range * MSUN), np.log10((G * M_range * MSUN * A0_CFM)**0.25 / 1e3),
            'r-', lw=2.5, label=f'CFM')
    ax.set_xlabel(r'$\log_{10}(M_{\rm bar})$ [kg]')
    ax.set_ylabel(r'$\log_{10}(V_{\rm flat})$ [km/s]')
    ax.set_title('BTFR (SPARC)')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'btfr_sparc.png', dpi=300)
    plt.close(fig)

    # chi2 comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    chi2_A_arr = []; chi2_B_arr = []
    for n in results_A:
        if n in results_B:
            cA = results_A[n].get('chi2_red', np.nan)
            cB = results_B[n].get('chi2_red', np.nan)
            if np.isfinite(cA) and np.isfinite(cB):
                chi2_A_arr.append(cA); chi2_B_arr.append(cB)
    axes[0].scatter(chi2_A_arr, chi2_B_arr, s=10, alpha=0.6)
    axes[0].plot([0, 20], [0, 20], 'k--', lw=1)
    axes[0].set_xlabel('chi2_red (A)'); axes[0].set_ylabel('chi2_red (B)')
    axes[0].set_xlim(0, 15); axes[0].set_ylim(0, 15); axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    delta_arr = [chi2_B_arr[i] - chi2_A_arr[i] for i in range(len(chi2_A_arr))]
    axes[1].hist(delta_arr, bins=40, alpha=0.7, color='steelblue')
    axes[1].axvline(0, color='k', ls='--')
    axes[1].axvline(np.median(delta_arr), color='red', ls='-', lw=2,
                    label=f'Median={np.median(delta_arr):.2f}')
    axes[1].set_xlabel('Delta chi2_red (B-A)'); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'chi2_comparison.png', dpi=300)
    plt.close(fig)

    # Save JSON
    summary = {
        'n_galaxies': len(good_galaxies),
        'a0_CFM': A0_CFM, 'a0_obs': A0_OBS, 'a0_best_global': a0_best,
        'run_A': {'chi2': chi2_total_A, 'ndf': ndf_total_A, 'chi2_red': chi2_red_A,
                  'a0_median': float(np.median(a0_vals_A)) if a0_vals_A else None},
        'run_B': {'chi2': chi2_total_B, 'ndf': ndf_total_B, 'chi2_red': chi2_red_B},
        'run_C': {'chi2': chi2_total_C, 'ndf': ndf_total_C, 'chi2_red': chi2_red_C,
                  'a0_best': a0_best},
        'delta_chi2_BA': delta_BA, 'delta_chi2_BC': delta_BC,
    }
    with open(RESULTS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # LaTeX table
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{CFM SPARC Analysis}\n\\label{tab:sparc}\n"
    latex += "\\begin{tabular}{lccc}\n\\hline\n"
    latex += "Run & $a_0$ [$10^{-10}$] & $\\chi^2/\\mathrm{dof}$ & $N_{\\mathrm{dof}}$ \\\\\n\\hline\n"
    latex += f"A (free) & variable & {chi2_red_A:.3f} & {ndf_total_A} \\\\\n"
    latex += f"B (CFM) & {A0_CFM*1e10:.3f} & {chi2_red_B:.3f} & {ndf_total_B} \\\\\n"
    latex += f"C (global) & {a0_best*1e10:.3f} & {chi2_red_C:.3f} & {ndf_total_C} \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    with open(RESULTS_DIR / 'summary_table.tex', 'w') as f:
        f.write(latex)

    dt = time.time() - t_start
    print(f"\nRuntime: {dt:.1f}s ({dt/60:.1f} min)")

    send_telegram(
        f"FERTIG! {dt/60:.1f}min\n"
        f"A={chi2_red_A:.3f}, B(CFM)={chi2_red_B:.3f}, C={chi2_red_C:.3f}\n"
        f"Best a0={a0_best*1e10:.3f}e-10"
    )


if __name__ == "__main__":
    main()

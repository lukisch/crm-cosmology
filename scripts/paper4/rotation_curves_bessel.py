#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFM Paper IV: Improved Rotation Curves with Bessel Functions (Aufgabe 7)
=========================================================================
Computes V(r) from BVP solution using exact Bessel function formulation
for the exponential disk potential (Freeman 1970).

For 6 galaxy masses: V_CFM(r) vs V_MOND(r) vs V_Newton(r)

Author: L. Geiger / Claude Code
Date: 2026-02-22
"""

import numpy as np
from scipy.special import i0, i1, k0, k1
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import subprocess

# ============================================================
# Constants
# ============================================================
G = 6.67430e-11
c_light = 2.99792458e8
KPC = 3.0856775814e19
MSUN = 1.98892e30
MPC = KPC * 1e3
H0 = 67.36e3 / MPC
A0 = c_light * H0 / (2 * np.pi)

RESULTS_DIR = Path("/home/cfm-cosmology/results/paper4/rotcurves")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_TOKEN = "***TELEGRAM_TOKEN_REMOVED***"
TELEGRAM_CHAT = "595767047"


def send_telegram(msg):
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            "-d", f"chat_id={TELEGRAM_CHAT}",
            "-d", f"text=[Paper IV RotCurves] {msg}"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


# ============================================================
# Exact exponential disk rotation curve (Freeman 1970)
# ============================================================
def V_disk_freeman(r, M_disk, R_d):
    """
    Exact rotation velocity from an exponential disk using
    Freeman (1970) formula with Bessel functions:

    V^2(r) = 4*pi*G*Sigma_0*R_d * y^2 * [I0(y)*K0(y) - I1(y)*K1(y)]
    where y = r/(2*R_d) and Sigma_0 = M_disk/(2*pi*R_d^2)
    """
    y = r / (2 * R_d)
    y = np.maximum(y, 1e-10)

    # Bessel functions (handle overflow)
    y_safe = np.minimum(y, 500)

    bessel_term = i0(y_safe) * k0(y_safe) - i1(y_safe) * k1(y_safe)

    Sigma_0 = M_disk / (2 * np.pi * R_d**2)
    V2 = 4 * np.pi * G * Sigma_0 * R_d * y**2 * bessel_term

    return np.sqrt(np.maximum(V2, 0))


def g_disk_freeman(r, M_disk, R_d):
    """Gravitational acceleration from exponential disk."""
    V = V_disk_freeman(r, M_disk, R_d)
    return V**2 / np.maximum(r, 1e-10 * R_d)


def mcgaugh_rar(g_bar, a0):
    """McGaugh RAR interpolation."""
    x = np.sqrt(np.maximum(g_bar / a0, 1e-30))
    denom = np.maximum(1.0 - np.exp(-x), 1e-30)
    return g_bar / denom


# ============================================================
# Galaxy models
# ============================================================
def galaxy_model(M_star, R_d, f_gas=0.2, R_gas_factor=2.0):
    """Create galaxy parameters."""
    M_gas = M_star * f_gas
    R_gas = R_d * R_gas_factor
    return M_star, R_d, M_gas, R_gas


def compute_rotation_curve(r, M_star, R_d, M_gas, R_gas):
    """Compute baryonic, MOND, and Newton rotation curves."""
    # Baryonic acceleration
    g_disk = g_disk_freeman(r, M_star, R_d)
    g_gas = g_disk_freeman(r, M_gas, R_gas)
    g_bar = g_disk + g_gas

    # Newton
    V_newton = np.sqrt(np.maximum(g_bar * r, 0)) / 1e3

    # MOND (McGaugh interpolation)
    g_mond = mcgaugh_rar(g_bar, 1.2e-10)
    V_mond = np.sqrt(np.maximum(g_mond * r, 0)) / 1e3

    # CFM (a0 = cH0/2pi)
    g_cfm = mcgaugh_rar(g_bar, A0)
    V_cfm = np.sqrt(np.maximum(g_cfm * r, 0)) / 1e3

    # Deep MOND limit
    V_deep_mond = (G * (M_star + M_gas) * 1.2e-10)**0.25 / 1e3 * np.ones_like(r)
    V_deep_cfm = (G * (M_star + M_gas) * A0)**0.25 / 1e3 * np.ones_like(r)

    return {
        'g_bar': g_bar, 'g_mond': g_mond, 'g_cfm': g_cfm,
        'V_newton': V_newton, 'V_mond': V_mond, 'V_cfm': V_cfm,
        'V_deep_mond': V_deep_mond, 'V_deep_cfm': V_deep_cfm,
    }


def main():
    t_start = time.time()

    print("=" * 70)
    print("CFM Paper IV: Improved Rotation Curves (Bessel)")
    print("=" * 70)

    send_telegram("Bessel-Rotationskurven gestartet")

    # 6 galaxy models
    galaxies = [
        {'label': r'$10^9 M_\odot$ (dwarf)', 'logM': 9.0, 'R_d_kpc': 0.5, 'f_gas': 0.5},
        {'label': r'$10^{9.5} M_\odot$', 'logM': 9.5, 'R_d_kpc': 0.8, 'f_gas': 0.4},
        {'label': r'$10^{10} M_\odot$', 'logM': 10.0, 'R_d_kpc': 1.5, 'f_gas': 0.3},
        {'label': r'$10^{10.5} M_\odot$ (MW-like)', 'logM': 10.5, 'R_d_kpc': 2.5, 'f_gas': 0.2},
        {'label': r'$10^{11} M_\odot$ (large spiral)', 'logM': 11.0, 'R_d_kpc': 3.5, 'f_gas': 0.15},
        {'label': r'$10^{12} M_\odot$ (massive)', 'logM': 12.0, 'R_d_kpc': 8.0, 'f_gas': 0.1},
    ]

    # ================================================================
    # 6-panel rotation curve plot
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    all_results = []

    for idx, gal in enumerate(galaxies):
        ax = axes[idx // 3, idx % 3]

        M_star = 10**gal['logM'] * MSUN
        R_d = gal['R_d_kpc'] * KPC
        M_star, R_d_m, M_gas, R_gas = galaxy_model(M_star, R_d, f_gas=gal['f_gas'])

        r_max_kpc = max(10 * gal['R_d_kpc'], 50)
        r_kpc = np.linspace(0.1, r_max_kpc, 500)
        r = r_kpc * KPC

        rc = compute_rotation_curve(r, M_star, R_d_m, M_gas, R_gas)
        all_results.append(rc)

        ax.plot(r_kpc, rc['V_newton'], 'b--', lw=1.5, label='Newton (baryon)')
        ax.plot(r_kpc, rc['V_mond'], 'g-', lw=2, label=r'MOND ($a_0$=1.2)')
        ax.plot(r_kpc, rc['V_cfm'], 'r-', lw=2, label=r'CFM ($a_0$=cH$_0$/2$\pi$)')
        ax.axhline(rc['V_deep_mond'][0], color='green', ls=':', lw=1, alpha=0.5)
        ax.axhline(rc['V_deep_cfm'][0], color='red', ls=':', lw=1, alpha=0.5)

        ax.set_title(gal['label'], fontsize=11)
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('V [km/s]')
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, r_max_kpc)
        ax.set_ylim(0, None)

        # Print summary
        V_flat_mond = rc['V_deep_mond'][0]
        V_flat_cfm = rc['V_deep_cfm'][0]
        print(f"  M=10^{gal['logM']}: V_flat(MOND)={V_flat_mond:.1f}, V_flat(CFM)={V_flat_cfm:.1f}, "
              f"ratio={V_flat_cfm/V_flat_mond:.4f}")

    fig.suptitle('CFM Paper IV: Rotation Curves (Freeman 1970 Bessel)', fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'rotation_curves_bessel.png', dpi=300)
    plt.close(fig)

    # ================================================================
    # Relative difference plot
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (gal, rc) in enumerate(zip(galaxies, all_results)):
        ax = axes[idx // 3, idx % 3]

        # (V_CFM - V_MOND) / V_MOND as percentage
        V_mond = rc['V_mond']
        V_cfm = rc['V_cfm']

        pos = V_mond > 1  # above 1 km/s
        diff_pct = (V_cfm[pos] - V_mond[pos]) / V_mond[pos] * 100

        r_kpc = np.linspace(0.1, max(10 * gal['R_d_kpc'], 50), 500)
        ax.plot(r_kpc[pos], diff_pct, 'r-', lw=2)
        ax.axhline(0, color='k', ls='--', lw=1)
        expected_diff = ((A0 / 1.2e-10)**0.25 - 1) * 100
        ax.axhline(-expected_diff, color='gray', ls=':', lw=1,
                    label=f'Expected: {-expected_diff:.1f}%')

        ax.set_title(gal['label'], fontsize=11)
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel(r'$(V_{\rm CFM} - V_{\rm MOND})/V_{\rm MOND}$ [%]')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Relative Difference CFM vs MOND', fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'cfm_vs_mond_diff.png', dpi=300)
    plt.close(fig)

    # ================================================================
    # Tully-Fisher comparison
    # ================================================================
    fig, ax = plt.subplots(figsize=(9, 7))

    M_range = np.logspace(7, 12.5, 200) * MSUN
    V_tf_mond = (G * M_range * 1.2e-10)**0.25 / 1e3
    V_tf_cfm = (G * M_range * A0)**0.25 / 1e3

    ax.plot(np.log10(M_range / MSUN), np.log10(V_tf_mond), 'g-', lw=2.5,
            label=r'MOND ($a_0 = 1.2 \times 10^{-10}$)')
    ax.plot(np.log10(M_range / MSUN), np.log10(V_tf_cfm), 'r--', lw=2.5,
            label=r'CFM ($a_0 = cH_0/2\pi$)')

    # Mark the 6 galaxies
    for gal in galaxies:
        M = 10**gal['logM'] * (1 + gal['f_gas'])  # total baryonic
        V_cfm = (G * M * MSUN * A0)**0.25 / 1e3
        ax.plot(np.log10(M), np.log10(V_cfm), 'ro', ms=10)

    ax.set_xlabel(r'$\log_{10}(M_{\rm bar}/M_\odot)$', fontsize=13)
    ax.set_ylabel(r'$\log_{10}(V_{\rm flat}$ [km/s])', fontsize=13)
    ax.set_title('Baryonic Tully-Fisher Relation', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'btfr_bessel.png', dpi=300)
    plt.close(fig)

    dt = time.time() - t_start
    print(f"\nRuntime: {dt:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")
    send_telegram(f"Bessel-RotCurves FERTIG ({dt:.0f}s)")


if __name__ == "__main__":
    main()

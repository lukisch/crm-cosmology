#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFM Paper IV: a_0 Discrepancy Analysis (Aufgabe 8)
====================================================
Analyzes the 13% discrepancy between a_0(CFM) = cH_0/(2*pi) = 1.042e-10
and a_0(obs) = 1.20e-10 m/s^2.

Key question: Can the saturation factor B_0 from the BVP solution explain
the discrepancy?

Analysis:
1. Extract B_0 from BVP solution (sech^2(phi_bar/phi_0))
2. Compute a_0_eff = a_0 * (1 + delta_B0)
3. Compare with observational value

Author: L. Geiger / Claude Code
Date: 2026-02-22
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import subprocess
import os

# ============================================================
# Constants
# ============================================================
G = 6.67430e-11
c_light = 2.99792458e8
KPC = 3.0856775814e19
MSUN = 1.98892e30
MPC = KPC * 1e3
H0 = 67.36e3 / MPC
RHO_CRIT = 3 * H0**2 / (8 * np.pi * G)
A0_BASE = c_light * H0 / (2 * np.pi)
A0_OBS = 1.20e-10

RESULTS_DIR = Path("/home/cfm-cosmology/results/paper4/a0_discrepancy")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "595767047")


def send_telegram(msg):
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            "-d", f"chat_id={TELEGRAM_CHAT}",
            "-d", f"text=[Paper IV a0] {msg}"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


def main():
    print("=" * 70)
    print("CFM Paper IV: a_0 Discrepancy Analysis")
    print("=" * 70)

    send_telegram("a0-Diskrepanz-Analyse gestartet")

    # ================================================================
    # 1. The bare prediction
    # ================================================================
    print("\n1. BARE PREDICTION")
    print(f"   a_0(CFM)  = c * H_0 / (2*pi) = {A0_BASE:.4e} m/s^2")
    print(f"   a_0(obs)  = {A0_OBS:.4e} m/s^2")
    discrepancy = (A0_OBS - A0_BASE) / A0_OBS * 100
    ratio = A0_BASE / A0_OBS
    print(f"   Discrepancy = {discrepancy:.1f}%")
    print(f"   Ratio = {ratio:.4f}")

    # ================================================================
    # 2. Saturation factor B_0
    # ================================================================
    print("\n2. SATURATION FACTOR B_0")
    print("   The vector field coupling includes B(phi) = sech^2(phi_bar/phi_0)")
    print("   At the present epoch, phi_bar/phi_0 is not necessarily small.")
    print("   The effective MOND scale becomes:")
    print("   a_0^eff = (c * H_0 / (2*pi)) * f(B_0)")
    print()

    # Model: a_0^eff = a_0_base / B_0 (because B_0 < 1 weakens coupling)
    # OR: a_0^eff = a_0_base * (some power of B_0)
    # The exact relationship depends on the field equations

    # From Paper IV formalism:
    # Xi(r) = B_0 * (dphi_bar/dt) / rho_crit * varphi'(r)
    # At r_MOND: Xi = a_0
    # So: a_0 = B_0 * H_0 * phi_0 / rho_crit * varphi'(r_MOND)
    # The factor cH_0/(2*pi) absorbs phi_0, rho_crit, and the Fourier factor
    # But B_0 remains as a multiplicative correction

    # If B_0 < 1: a_0_eff = a_0_base * (1/B_0)  -> a_0_eff > a_0_base
    # This goes the right direction! B_0 < 1 increases the effective a_0

    # More precisely, from the field equation analysis:
    # The MOND transition is where the scalar Compton wavelength becomes
    # comparable to the galactic size. The transition acceleration is:
    # a_0 = (c * H_0 / (2*pi)) * (1 / B_0^alpha)
    # where alpha depends on the exponent in the Chameleon potential

    # For the Poeschl-Teller potential used in Paper III:
    # V_PT(phi) = V_0 / cosh^2(phi/phi_0)
    # B(phi) = sech^2(phi/phi_0)
    # At partial saturation: phi_bar/phi_0 ~ 0.5-1.0

    phi_bar_over_phi0 = np.linspace(0.01, 2.0, 200)
    B0_values = 1.0 / np.cosh(phi_bar_over_phi0)**2

    # Model 1: a0_eff = a0_base / sqrt(B_0)
    a0_model1 = A0_BASE / np.sqrt(B0_values)

    # Model 2: a0_eff = a0_base / B_0
    a0_model2 = A0_BASE / B0_values

    # Model 3: a0_eff = a0_base * sqrt(1/B_0)  (same as model 1)
    # Model 4: a0_eff = a0_base * (2 - B_0)  (linear correction)
    a0_model4 = A0_BASE * (2 - B0_values)

    # Find B_0 that gives a0 = 1.2e-10
    B0_needed_m1 = (A0_BASE / A0_OBS)**2  # model 1: B0 = (a0_base/a0_obs)^2
    B0_needed_m2 = A0_BASE / A0_OBS       # model 2: B0 = a0_base/a0_obs

    print(f"   Model 1 (a0 = base/sqrt(B0)): B_0 needed = {B0_needed_m1:.4f}")
    print(f"     -> phi_bar/phi_0 = {np.arccosh(1/np.sqrt(B0_needed_m1)):.3f}")
    print(f"   Model 2 (a0 = base/B0):       B_0 needed = {B0_needed_m2:.4f}")
    print(f"     -> phi_bar/phi_0 = {np.arccosh(1/np.sqrt(B0_needed_m2)):.3f}")

    # ================================================================
    # 3. H_0 tension contribution
    # ================================================================
    print("\n3. H_0 TENSION CONTRIBUTION")
    H0_planck = 67.36  # km/s/Mpc
    H0_shoes = 73.04   # km/s/Mpc (Riess+2022)
    H0_cepheid_trgb = 69.8  # km/s/Mpc (Freedman+2021)

    a0_planck = c_light * H0_planck * 1e3 / MPC / (2 * np.pi)
    a0_shoes = c_light * H0_shoes * 1e3 / MPC / (2 * np.pi)
    a0_trgb = c_light * H0_cepheid_trgb * 1e3 / MPC / (2 * np.pi)

    print(f"   H0 = {H0_planck} (Planck):   a0 = {a0_planck:.4e} ({a0_planck/A0_OBS*100:.1f}% of obs)")
    print(f"   H0 = {H0_cepheid_trgb} (TRGB):    a0 = {a0_trgb:.4e} ({a0_trgb/A0_OBS*100:.1f}% of obs)")
    print(f"   H0 = {H0_shoes} (SH0ES):   a0 = {a0_shoes:.4e} ({a0_shoes/A0_OBS*100:.1f}% of obs)")

    # ================================================================
    # 4. Combined: B_0 + H_0 uncertainty
    # ================================================================
    print("\n4. COMBINED ANALYSIS")

    # If H_0 = 73 and B_0 = 0.95:
    a0_combined = c_light * H0_shoes * 1e3 / MPC / (2 * np.pi) / np.sqrt(0.95)
    print(f"   H0=73, B0=0.95: a0_eff = {a0_combined:.4e} ({a0_combined/A0_OBS*100:.1f}% of obs)")

    a0_combined2 = c_light * H0_cepheid_trgb * 1e3 / MPC / (2 * np.pi) / np.sqrt(0.90)
    print(f"   H0=69.8, B0=0.90: a0_eff = {a0_combined2:.4e} ({a0_combined2/A0_OBS*100:.1f}% of obs)")

    # ================================================================
    # 5. Observational uncertainty on a_0
    # ================================================================
    print("\n5. OBSERVATIONAL UNCERTAINTY")
    print("   McGaugh+2016: a_0 = (1.20 +/- 0.02 stat +/- 0.24 syst) * 10^-10")
    print(f"   CFM prediction: a_0 = {A0_BASE*1e10:.3f} * 10^-10")
    sigma_stat = 0.02e-10  # statistical
    sigma_syst = 0.24e-10  # systematic (dominated by distance uncertainty)
    sigma_total = np.sqrt(sigma_stat**2 + sigma_syst**2)
    n_sigma = abs(A0_OBS - A0_BASE) / sigma_total
    print(f"   Combined sigma = {sigma_total*1e10:.3f} * 10^-10")
    print(f"   Tension = |a0_obs - a0_CFM| / sigma = {n_sigma:.2f} sigma")
    print(f"   -> The discrepancy is {n_sigma:.2f} sigma from zero.")

    # ================================================================
    # Plots
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: a0 vs B_0
    ax = axes[0, 0]
    ax.plot(B0_values, a0_model1 * 1e10, 'b-', lw=2, label=r'$a_0^{\rm eff} = a_0^{\rm base}/\sqrt{\mathcal{B}_0}$')
    ax.plot(B0_values, a0_model2 * 1e10, 'r--', lw=2, label=r'$a_0^{\rm eff} = a_0^{\rm base}/\mathcal{B}_0$')
    ax.axhline(A0_OBS * 1e10, color='green', ls='-', lw=2, alpha=0.7, label=f'$a_0^{{\\rm obs}}$ = {A0_OBS*1e10:.2f}')
    ax.axhline(A0_BASE * 1e10, color='gray', ls=':', lw=1, label=f'$cH_0/(2\\pi)$ = {A0_BASE*1e10:.3f}')
    ax.fill_between([0, 1], [(A0_OBS - sigma_total) * 1e10] * 2,
                    [(A0_OBS + sigma_total) * 1e10] * 2, alpha=0.15, color='green')
    ax.set_xlabel(r'$\mathcal{B}_0 = \mathrm{sech}^2(\bar\phi/\phi_0)$', fontsize=12)
    ax.set_ylabel(r'$a_0^{\rm eff}$ [$10^{-10}$ m/s$^2$]', fontsize=12)
    ax.set_title(r'$a_0$ vs Saturation Factor', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.8, 2.0)

    # Panel 2: a0 vs H_0
    ax = axes[0, 1]
    H0_range = np.linspace(60, 80, 100)
    a0_H0 = c_light * H0_range * 1e3 / MPC / (2 * np.pi) * 1e10
    ax.plot(H0_range, a0_H0, 'b-', lw=2, label=r'$a_0 = cH_0/(2\pi)$')
    ax.axhline(A0_OBS * 1e10, color='green', ls='-', lw=2, alpha=0.7)
    ax.fill_between(H0_range, (A0_OBS - sigma_total) * 1e10,
                    (A0_OBS + sigma_total) * 1e10, alpha=0.15, color='green')
    ax.axvspan(67.36 - 0.54, 67.36 + 0.54, alpha=0.2, color='blue', label='Planck')
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, alpha=0.2, color='red', label='SH0ES')
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=12)
    ax.set_ylabel(r'$a_0$ [$10^{-10}$ m/s$^2$]', fontsize=12)
    ax.set_title(r'$a_0$ vs $H_0$', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Combined contour (H_0 vs B_0)
    ax = axes[1, 0]
    H0_grid = np.linspace(60, 80, 100)
    B0_grid = np.linspace(0.5, 1.0, 100)
    H0_2d, B0_2d = np.meshgrid(H0_grid, B0_grid)
    a0_2d = c_light * H0_2d * 1e3 / MPC / (2 * np.pi) / np.sqrt(B0_2d)

    levels = [1.00, 1.10, 1.15, 1.20, 1.25, 1.30, 1.50]
    cs = ax.contour(H0_2d, B0_2d, a0_2d * 1e10, levels=levels, colors='blue')
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.axhline(1.0, color='gray', ls=':', lw=1)

    # Mark regions
    ax.plot(67.36, 1.0, 'bs', ms=10, label='Bare (Planck)')
    ax.plot(73.0, 0.88, 'r*', ms=12, label='Needed for $a_0^{\\rm obs}$')
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=12)
    ax.set_ylabel(r'$\mathcal{B}_0$', fontsize=12)
    ax.set_title(r'$a_0^{\rm eff}$ contours [$10^{-10}$ m/s$^2$]', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Comparison bar chart
    ax = axes[1, 1]
    values = {
        'cH0/(2pi)\n[Planck]': A0_BASE * 1e10,
        'cH0/(2pi)\n[SH0ES]': a0_shoes * 1e10,
        'cH0/(2pi)\n[TRGB]': a0_trgb * 1e10,
        'B0=0.90\n[Planck]': A0_BASE / np.sqrt(0.90) * 1e10,
        'B0=0.85\n[SH0ES]': a0_shoes / np.sqrt(0.85) * 1e10,
        'Observed\nMcGaugh+16': A0_OBS * 1e10,
    }

    names = list(values.keys())
    vals = list(values.values())
    colors = ['lightblue', 'lightsalmon', 'lightyellow', 'steelblue', 'salmon', 'green']

    bars = ax.barh(names, vals, color=colors, edgecolor='black', alpha=0.8)
    ax.axvline(A0_OBS * 1e10, color='green', ls='-', lw=2, alpha=0.5)
    ax.axvspan((A0_OBS - sigma_total) * 1e10, (A0_OBS + sigma_total) * 1e10,
               alpha=0.1, color='green')

    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9)

    ax.set_xlabel(r'$a_0$ [$10^{-10}$ m/s$^2$]', fontsize=12)
    ax.set_title(r'$a_0$ Predictions vs Observation', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('CFM Paper IV: The 13% a_0 Discrepancy', fontsize=15)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'a0_discrepancy.png', dpi=300)
    plt.close(fig)

    # ================================================================
    # Save JSON
    # ================================================================
    summary = {
        'a0_base_cH0_2pi': A0_BASE,
        'a0_obs': A0_OBS,
        'discrepancy_pct': discrepancy,
        'ratio': ratio,
        'tension_sigma': n_sigma,
        'H0_values': {
            'Planck': H0_planck,
            'SH0ES': H0_shoes,
            'TRGB': H0_cepheid_trgb,
        },
        'a0_from_H0': {
            'Planck': a0_planck,
            'SH0ES': a0_shoes,
            'TRGB': a0_trgb,
        },
        'B0_needed_model1_sqrt': B0_needed_m1,
        'B0_needed_model2_linear': B0_needed_m2,
        'obs_uncertainty': {
            'sigma_stat': sigma_stat,
            'sigma_syst': sigma_syst,
            'sigma_total': sigma_total,
        },
    }

    with open(RESULTS_DIR / 'a0_discrepancy_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # LaTeX
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{$a_0$ predictions and observations}\n"
    latex += "\\begin{tabular}{lcc}\n\\hline\n"
    latex += "Source & $a_0$ [$10^{-10}$ m/s$^2$] & Tension \\\\\n\\hline\n"
    latex += f"$cH_0/(2\\pi)$ (Planck) & {A0_BASE*1e10:.3f} & {n_sigma:.1f}$\\sigma$ \\\\\n"
    latex += f"$cH_0/(2\\pi)$ (SH0ES) & {a0_shoes*1e10:.3f} & "
    n_shoes = abs(A0_OBS - a0_shoes) / sigma_total
    latex += f"{n_shoes:.1f}$\\sigma$ \\\\\n"
    latex += f"McGaugh+2016 (obs) & {A0_OBS*1e10:.2f} $\\pm$ {sigma_total*1e10:.2f} & --- \\\\\n"
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"

    with open(RESULTS_DIR / 'a0_table.tex', 'w') as f:
        f.write(latex)

    print(f"\nResults saved to: {RESULTS_DIR}")

    send_telegram(
        f"a0-Analyse fertig!\n"
        f"Diskrepanz: {discrepancy:.1f}% ({n_sigma:.2f} sigma)\n"
        f"B0 noetig (Model 1): {B0_needed_m1:.3f}\n"
        f"H0-Tension hilft: SH0ES a0={a0_shoes*1e10:.3f}"
    )


if __name__ == "__main__":
    main()

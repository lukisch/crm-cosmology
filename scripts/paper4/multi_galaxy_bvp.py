#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFM Paper IV: Multi-Galaxy BVP Analysis (Aufgabe 6)
=====================================================
Applies BVP solver v5 to 6 galaxy masses: 10^9 to 10^12 Msun.
Confirms MOND attractor (slope ~0.5), determines r_MOND(M),
extracts emergent mu(x), and compares with McGaugh interpolation.

Author: L. Geiger / Claude Code
Date: 2026-02-22
"""

import numpy as np
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

RESULTS_DIR = Path("/home/cfm-cosmology/results/paper4/bvp")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_TOKEN = "***TELEGRAM_TOKEN_REMOVED***"
TELEGRAM_CHAT = "595767047"


def send_telegram(msg):
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            "-d", f"chat_id={TELEGRAM_CHAT}",
            "-d", f"text=[Paper IV BVP] {msg}"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


# ============================================================
# Galaxy model: Plummer sphere
# ============================================================
def rho_plummer(r, M, a):
    return (3 * M / (4 * np.pi * a**3)) * (1 + (r / a)**2)**(-2.5)


def enclosed_mass_plummer(r, M, a):
    x = r / a
    return M * x**3 / (1 + x**2)**1.5


def g_newton(r, M, a):
    Me = enclosed_mass_plummer(r, M, a)
    return G * Me / np.maximum(r**2, (0.01 * a)**2)


# ============================================================
# Coupling functions
# ============================================================
def sech2(x):
    x = np.clip(x, -50, 50)
    return 1.0 / np.cosh(x)**2


def B_func(phi, phi0):
    return sech2(phi / phi0)


def sigma_func(rho, rho_screen):
    return np.exp(-rho / rho_screen)


def f_screen(g_obs, a0, eta):
    ratio = np.clip(a0 / np.maximum(g_obs, 1e-20), 0, 1e8)
    return 1.0 / (1.0 + ratio**eta)


# ============================================================
# Sparse tridiagonal solver
# ============================================================
def build_scalar_matrix(r, dr, m2_eff):
    N = len(r)
    inv_dr2 = 1.0 / dr**2
    inv_2dr = 0.5 / dr
    main = np.full(N, -2.0 * inv_dr2) - m2_eff
    lower = np.ones(N - 1) * inv_dr2
    upper = np.ones(N - 1) * inv_dr2
    for i in range(1, N - 1):
        coeff = inv_2dr / r[i]
        lower[i - 1] -= coeff
        upper[i] += coeff
    main[0] = 1.0; upper[0] = -1.0
    main[-1] = 1.0; lower[-1] = 0.0
    return diags([lower, main, upper], offsets=[-1, 0, 1], format='csc')


# ============================================================
# Core v5 solver
# ============================================================
def solve_cfm_v5(M_gal, r_s, beta=1.0/3.0, r_max_kpc=200.0, N=2000,
                 epsilon=1.0, n_grad=0.5, eta=1.0, kappa=1.0,
                 lambda_steps=None, omega=0.15, omega_g=0.10,
                 max_iter=3000, tol=1e-9, verbose=False):
    if lambda_steps is None:
        lambda_steps = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3,
                        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    r_max = r_max_kpc * KPC
    dr = r_max / N
    r = (np.arange(N) + 0.5) * dr

    phi0 = beta * G * M_gal / (r_s * c_light**2)
    r_mond = np.sqrt(G * M_gal / A0)
    m_factor = 0.3
    m_gal = m_factor / r_mond
    phi_ref = beta * G * M_gal / (r_s**2 * c_light**2)

    rho = rho_plummer(r, M_gal, r_s)
    rho_screen = rho_plummer(r_mond, M_gal, r_s)
    sigma = sigma_func(rho, rho_screen)
    phi_min = phi0 * rho_screen / (rho + rho_screen)
    phi_bg = phi0
    gN = g_newton(r, M_gal, r_s)
    m2_base = m_gal**2 * (1.0 + rho / rho_screen)
    r_core = 0.3 * r_s
    m2_floor = (kappa / (r + r_core))**2
    m2_base = np.maximum(m2_base, m2_floor)

    phi = phi_min.copy()
    g_obs = gN.copy()
    Xi = np.zeros(N)
    convergence_log = []

    for lam in lambda_steps:
        converged = False
        for it in range(max_iter):
            phi_old = phi.copy()
            g_obs_old = g_obs.copy()

            dphi = np.zeros(N)
            dphi[1:-1] = (phi[2:] - phi[:-2]) / (2 * dr)
            dphi[0] = 0.0
            dphi[-1] = (phi[-1] - phi[-2]) / dr

            B = B_func(phi, phi0)
            dphi_norm = np.abs(dphi) / np.maximum(phi_ref, 1e-50)
            Xi = epsilon * A0 * B * sigma * dphi_norm**n_grad * np.sign(dphi)

            g_obs_new = gN + Xi
            g_obs = (1 - omega_g) * g_obs + omega_g * g_obs_new
            g_obs = np.maximum(g_obs, 1e-20)

            f_scr = f_screen(np.abs(g_obs), A0, eta)
            m2_eff = m2_base * ((1.0 - lam) + lam * f_scr)
            m2_min = m_gal**2 * 0.001
            m2_eff = np.maximum(m2_eff, m2_min)

            A = build_scalar_matrix(r, dr, m2_eff)
            rhs = -m2_eff * phi_min
            rhs[0] = 0.0
            rhs[-1] = phi_bg
            phi_new = spsolve(A, rhs)
            phi = (1 - omega) * phi + omega * phi_new

            err_phi = np.max(np.abs(phi - phi_old)) / (np.max(np.abs(phi)) + 1e-50)
            err_g = np.max(np.abs(g_obs - g_obs_old)) / (np.max(np.abs(g_obs)) + 1e-50)
            if max(err_phi, err_g) < tol:
                converged = True
                break

        V_flat_arr = np.sqrt(np.maximum(g_obs * r, 0)) / 1e3
        idx_outer = (r / KPC > 30) & (r / KPC < 150)
        V_flat = np.mean(V_flat_arr[idx_outer]) if np.any(idx_outer) else 0
        V_N = np.mean(np.sqrt(np.maximum(gN[idx_outer] * r[idx_outer], 0))) / 1e3 if np.any(idx_outer) else 0

        convergence_log.append({
            'lambda': lam, 'V_flat': V_flat, 'V_N': V_N,
            'converged': converged, 'iters': it,
        })

    # RAR slope
    r_kpc = r / KPC
    mask = (r_kpc > 0.5) & (gN > 0) & (g_obs > 0) & (gN < A0) & (gN > 1e-14)
    if np.sum(mask) > 5:
        slope = np.polyfit(np.log10(gN[mask]), np.log10(g_obs[mask]), 1)[0]
    else:
        slope = np.nan

    # mu(x) function
    x_arr = gN / A0
    mu_arr = g_obs / np.maximum(gN, 1e-30)

    # r_MOND: where g_obs ~ a_0
    diffs = np.abs(g_obs - A0)
    r_mond_obs = r[np.argmin(diffs)] / KPC

    Xi_eff = g_obs - gN

    return {
        'r': r, 'r_kpc': r_kpc, 'gN': gN, 'g_obs': g_obs, 'Xi': Xi_eff,
        'phi': phi, 'f_screen': f_scr, 'm2_eff': m2_eff,
        'slope': slope, 'V_flat': V_flat, 'V_N': V_N, 'r_mond_kpc': r_mond_obs,
        'r_mond_theory': r_mond / KPC,
        'x_mu': x_arr, 'mu': mu_arr,
        'M_gal': M_gal, 'r_s': r_s,
        'convergence_log': convergence_log,
    }


# ============================================================
# Main: Multi-Galaxy BVP
# ============================================================
def main():
    t_start = time.time()

    print("=" * 70)
    print("CFM Paper IV: Multi-Galaxy BVP Analysis")
    print("=" * 70)
    print(f"a0 = {A0:.4e} m/s^2")
    print()

    send_telegram("Multi-Galaxy BVP gestartet (6 Massen: 10^9 - 10^12 Msun)")

    # Galaxy masses and scale radii
    log_masses = [9.0, 9.5, 10.0, 10.5, 11.0, 12.0]
    scale_radii_kpc = [0.5, 0.8, 1.5, 2.5, 3.5, 8.0]  # Approximate scaling

    results = []

    print(f"\n{'log(M)':>8s} {'slope':>8s} {'V_flat':>10s} {'V_N':>10s} {'enh':>8s} "
          f"{'r_MOND':>8s} {'r_MOND_th':>10s} {'status':>8s}")
    print("-" * 80)

    for i, (logM, rs_kpc) in enumerate(zip(log_masses, scale_radii_kpc)):
        M_gal = 10**logM * MSUN
        r_s = rs_kpc * KPC
        r_max = max(200, 10 * np.sqrt(G * M_gal / A0) / KPC)

        t0 = time.time()
        res = solve_cfm_v5(
            M_gal=M_gal, r_s=r_s,
            r_max_kpc=min(r_max, 500), N=2000,
            epsilon=1.0, n_grad=0.5, eta=1.0, kappa=1.0,
            lambda_steps=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0],
            omega=0.15, omega_g=0.10,
            max_iter=3000, tol=1e-9,
            verbose=False,
        )
        dt = time.time() - t0
        results.append(res)

        enh = res['V_flat'] / max(res['V_N'], 0.01)
        status = "OK" if res['convergence_log'][-1]['converged'] else "FAIL"
        print(f"{logM:8.1f} {res['slope']:8.3f} {res['V_flat']:10.1f} {res['V_N']:10.1f} "
              f"{enh:8.2f} {res['r_mond_kpc']:8.1f} {res['r_mond_theory']:10.1f} {status:>8s} ({dt:.0f}s)")

        send_telegram(f"M=10^{logM:.1f}: slope={res['slope']:.3f}, V_flat={res['V_flat']:.1f} km/s, "
                      f"r_MOND={res['r_mond_kpc']:.1f} kpc ({dt:.0f}s)")

    # ================================================================
    # Plot 1: 6-panel rotation curves
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (res, logM) in enumerate(zip(results, log_masses)):
        ax = axes[idx // 3, idx % 3]
        r = res['r_kpc']
        mask = (r > 0.3) & (r < 150)

        V_N = np.sqrt(np.maximum(res['gN'] * res['r'], 0)) / 1e3
        V_obs = np.sqrt(np.maximum(res['g_obs'] * res['r'], 0)) / 1e3
        V_mond = np.sqrt(np.maximum(np.sqrt(res['gN'] * A0) * res['r'], 0)) / 1e3

        ax.plot(r[mask], V_N[mask], 'b--', lw=1.5, label='Newton')
        ax.plot(r[mask], V_obs[mask], 'r-', lw=2, label='CFM')
        ax.plot(r[mask], V_mond[mask], 'g:', lw=1.5, label='MOND target')
        ax.set_title(f'$M = 10^{{{logM:.1f}}}$ $M_\\odot$ (slope={res["slope"]:.3f})')
        ax.set_xlabel('r [kpc]')
        ax.set_ylabel('V [km/s]')
        if idx == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)

    fig.suptitle('CFM v5: Multi-Galaxy Rotation Curves', fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'multi_galaxy_rc.png', dpi=300)
    plt.close(fig)

    # ================================================================
    # Plot 2: 6-panel RAR
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    gr = np.logspace(-14, -8, 200)

    for idx, (res, logM) in enumerate(zip(results, log_masses)):
        ax = axes[idx // 3, idx % 3]
        r = res['r_kpc']
        gN = res['gN']; go = res['g_obs']
        pos = (r > 0.5) & (gN > 1e-14) & (go > 0)

        ax.scatter(np.log10(gN[pos]), np.log10(go[pos]),
                   c=r[pos], cmap='viridis', s=5, alpha=0.7)
        ax.plot(np.log10(gr), np.log10(gr), 'k--', lw=1, alpha=0.5)
        ax.plot(np.log10(gr), np.log10(np.sqrt(gr * A0)), 'g:', lw=1.5)
        ax.set_title(f'$M = 10^{{{logM:.1f}}}$ (slope={res["slope"]:.3f})')
        ax.set_xlabel(r'$\log_{10}(g_N)$')
        ax.set_ylabel(r'$\log_{10}(g_{obs})$')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    fig.suptitle('RAR: Multi-Galaxy', fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'multi_galaxy_rar.png', dpi=300)
    plt.close(fig)

    # ================================================================
    # Plot 3: mu(x) comparison
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x_th = np.logspace(-3, 2, 200)
    mu_mcgaugh = 1.0 / (1.0 - np.exp(-np.sqrt(x_th)))
    mu_simple = (1 + x_th) / x_th

    ax.plot(np.log10(x_th), mu_mcgaugh, 'k-', lw=2, label='McGaugh (2016)')
    ax.plot(np.log10(x_th), mu_simple, 'gray', ls=':', lw=1, label='simple IF')

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    for res, logM, col in zip(results, log_masses, colors):
        r = res['r_kpc']
        x = res['x_mu']
        mu = res['mu']
        mask = (r > 1.0) & (x > 1e-3) & (x < 100) & (mu > 0.5) & (mu < 10)
        if np.sum(mask) > 5:
            ax.scatter(np.log10(x[mask]), mu[mask], s=3, alpha=0.5, c=[col],
                       label=f'$10^{{{logM:.0f}}}$')

    ax.set_xlabel(r'$\log_{10}(x = g_N/a_0)$')
    ax.set_ylabel(r'$\nu(y) = g_{obs}/g_N$')
    ax.set_title('Emergent interpolation function')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 2)
    ax.set_ylim(0.8, 10)
    ax.set_yscale('log')

    # Panel 2: r_MOND scaling
    ax = axes[1]
    M_arr = [10**lm for lm in log_masses]
    r_mond_obs = [res['r_mond_kpc'] for res in results]
    r_mond_th = [res['r_mond_theory'] for res in results]

    ax.scatter(log_masses, r_mond_obs, s=100, c='red', zorder=5, label='BVP result')
    ax.scatter(log_masses, r_mond_th, s=60, c='green', marker='^', zorder=5, label=r'$\sqrt{GM/a_0}$')

    M_range = np.logspace(8.5, 12.5, 100)
    r_th_line = np.sqrt(G * M_range * MSUN / A0) / KPC
    ax.plot(np.log10(M_range), r_th_line, 'g--', lw=1.5, alpha=0.7)

    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
    ax.set_ylabel(r'$r_{\rm MOND}$ [kpc]')
    ax.set_title('MOND transition radius')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.suptitle('CFM: Emergent MOND Properties', fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'mu_and_rmond.png', dpi=300)
    plt.close(fig)

    # ================================================================
    # Save JSON
    # ================================================================
    summary = {
        'a0': A0,
        'masses': log_masses,
        'results': [{
            'log_M': lm,
            'slope': res['slope'],
            'V_flat': res['V_flat'],
            'V_N': res['V_N'],
            'enhancement': res['V_flat'] / max(res['V_N'], 0.01),
            'r_mond_kpc': res['r_mond_kpc'],
            'r_mond_theory_kpc': res['r_mond_theory'],
        } for lm, res in zip(log_masses, results)]
    }

    with open(RESULTS_DIR / 'multi_galaxy_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    dt_total = time.time() - t_start
    print(f"\nTotal runtime: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print(f"Results saved to: {RESULTS_DIR}")

    slopes = [res['slope'] for res in results]
    send_telegram(
        f"BVP FERTIG! {dt_total/60:.1f} min\n"
        f"Slopes: {', '.join(f'{s:.3f}' for s in slopes)}\n"
        f"Median slope: {np.nanmedian(slopes):.3f}"
    )


if __name__ == "__main__":
    main()

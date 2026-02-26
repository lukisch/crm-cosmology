#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFM Paper IV: Extract mu(x) from BVP Solutions
===============================================
Laueft den BVP-Solver v5 fuer 20 Galaxiemassen (10^8 bis 10^12.5 Msun)
und extrahiert die emergente CRM-native Interpolationsfunktion mu(x).

Vergleich mit:
  (a) McGaugh (2016):  mu = 1 / (1 - exp(-sqrt(x)))
  (b) Simple IF:       mu = (1 + x) / x   (aequivalent zu nu = 1 + 1/x)
  (c) CRM-native:      mu = x / (1 - exp(-x^alpha))  mit freiem alpha

Aufruf auf Hetzner-Server:
  nice -n 10 python extract_mu_from_bvp.py

Author: L. Geiger / Claude Code
Date: 2026-02-26
"""

import sys
import json
import time
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import aus bestehendem BVP-Solver v5
# ---------------------------------------------------------------------------
sys.path.insert(0, '/home/cfm-cosmology/scripts/paper4')
from multi_galaxy_bvp import solve_cfm_v5, A0, G, MSUN, KPC

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("/home/cfm-cosmology/results/paper4/mu_extraction")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_TOKEN = "7952992531:AAH_z_IlLcc5pl0HsBSJxSG9XtgX1jUiJFc"
TELEGRAM_CHAT  = "595767047"

# 20 Massen von 10^8 bis 10^12.5 in 0.25-dex Schritten
LOG_MASSES = np.arange(8.0, 12.75, 0.25).tolist()  # [8.0, 8.25, ..., 12.5]

TIMEOUT_PER_GALAXY = 600  # Sekunden


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def send_telegram(msg: str) -> None:
    """Sendet eine Telegram-Nachricht (fire-and-forget, ignoriert Fehler)."""
    try:
        subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                "-d", f"chat_id={TELEGRAM_CHAT}",
                "-d", f"text=[mu(x) Extraktion] {msg}",
            ],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def scale_radius_kpc(log_m: float) -> float:
    """
    Skalierungsrelation fuer Plummer-Skalenradius:
      r_s ~ 1.5 * (M / 10^10)^0.3  kpc

    Kalibriert auf Beobachtungsdaten (McGaugh et al. 2016):
      log M = 8  ->  r_s ~ 0.38 kpc
      log M = 10 ->  r_s ~ 1.50 kpc
      log M = 12 ->  r_s ~ 5.96 kpc
    """
    return 1.5 * (10 ** (log_m - 10.0)) ** 0.3


def r_max_kpc(log_m: float, factor: float = 20.0) -> float:
    """
    Setzt r_max auf max(200, factor * r_MOND), damit auch massearme
    Galaxien gut ins MOND-Regime kommen.
    """
    M = 10**log_m * MSUN
    r_mond = np.sqrt(G * M / A0) / KPC
    return float(np.clip(factor * r_mond, 200.0, 800.0))


# ---------------------------------------------------------------------------
# Analytische Interpolationsfunktionen (nu = g_obs / g_N als Funktion von x)
# ---------------------------------------------------------------------------

def nu_mcgaugh(x):
    """McGaugh (2016): nu = 1 / (1 - exp(-sqrt(x)))"""
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.clip(x, 1e-10, None))))


def nu_simple(x):
    """Einfache IF: nu = 1 + 1/x  (aequivalent zu mu = x/(1+x))"""
    return 1.0 + 1.0 / np.clip(x, 1e-10, None)


def nu_crm(x, alpha):
    """
    CRM-native Ansatz: nu(x) = 1 / (1 - exp(-x^alpha))
    (entspricht mu(x) = x / (1 - exp(-x^alpha)))
    Fuer alpha = 0.5 konvergiert das gegen McGaugh.
    """
    xa = np.clip(x, 1e-10, None) ** np.clip(alpha, 0.01, 5.0)
    return 1.0 / (1.0 - np.exp(-np.clip(xa, 0.0, 700.0)))


# ---------------------------------------------------------------------------
# Fit-Routine
# ---------------------------------------------------------------------------

def fit_nu(x_data: np.ndarray, nu_data: np.ndarray, nu_sigma: np.ndarray):
    """
    Fittet die drei analytischen Formen an die kombinierten Datenpunkte.
    Gibt ein Dict mit Parametern, chi^2 und dof zurueck.
    """
    results = {}

    # --- (a) McGaugh: kein freier Parameter, direkte chi^2-Auswertung ---
    nu_pred_mc = nu_mcgaugh(x_data)
    residuals_mc = (nu_data - nu_pred_mc) / nu_sigma
    chi2_mc = float(np.sum(residuals_mc**2))
    dof_mc = len(x_data)
    results['mcgaugh'] = {
        'params': {},
        'chi2': chi2_mc,
        'dof': dof_mc,
        'chi2_red': chi2_mc / max(dof_mc, 1),
        'p_value': float(1.0 - chi2_dist.cdf(chi2_mc, dof_mc)),
    }

    # --- (b) Simple IF: kein freier Parameter ---
    nu_pred_si = nu_simple(x_data)
    residuals_si = (nu_data - nu_pred_si) / nu_sigma
    chi2_si = float(np.sum(residuals_si**2))
    dof_si = len(x_data)
    results['simple'] = {
        'params': {},
        'chi2': chi2_si,
        'dof': dof_si,
        'chi2_red': chi2_si / max(dof_si, 1),
        'p_value': float(1.0 - chi2_dist.cdf(chi2_si, dof_si)),
    }

    # --- (c) CRM-native: freier Parameter alpha ---
    try:
        popt, pcov = curve_fit(
            nu_crm, x_data, nu_data,
            p0=[0.5],
            sigma=nu_sigma,
            absolute_sigma=True,
            bounds=([0.05], [5.0]),
            maxfev=10000,
        )
        alpha_fit = float(popt[0])
        alpha_err = float(np.sqrt(pcov[0, 0])) if pcov is not None else np.nan
        nu_pred_crm = nu_crm(x_data, alpha_fit)
        residuals_crm = (nu_data - nu_pred_crm) / nu_sigma
        chi2_crm = float(np.sum(residuals_crm**2))
        dof_crm = len(x_data) - 1
        results['crm_native'] = {
            'params': {'alpha': alpha_fit, 'alpha_err': alpha_err},
            'chi2': chi2_crm,
            'dof': dof_crm,
            'chi2_red': chi2_crm / max(dof_crm, 1),
            'p_value': float(1.0 - chi2_dist.cdf(chi2_crm, max(dof_crm, 1))),
        }
    except Exception as e:
        results['crm_native'] = {
            'params': {'alpha': np.nan, 'alpha_err': np.nan},
            'chi2': np.nan,
            'dof': len(x_data) - 1,
            'chi2_red': np.nan,
            'p_value': np.nan,
            'error': str(e),
        }

    return results


# ---------------------------------------------------------------------------
# Datenpunkte aus BVP-Loesung extrahieren
# ---------------------------------------------------------------------------

def extract_mu_points(res: dict, r_min_kpc: float = 0.5) -> dict:
    """
    Extrahiert mu(x) = g_obs/g_N als Funktion von x = g_N/a0
    aus einem solve_cfm_v5()-Ergebnis.

    Qualitaetsschnitte:
      - r > r_min_kpc (Kernbereich meiden)
      - x in [1e-4, 100]   (MOND-Uebergangsbereich)
      - g_obs > 0, g_N > 1e-16
      - nu in [1.0, 50]    (physikalisch sinnvoll)
    """
    r_kpc = res['r_kpc']
    gN    = res['gN']
    g_obs = res['g_obs']

    x_raw  = gN / A0
    nu_raw = g_obs / np.maximum(gN, 1e-30)

    mask = (
        (r_kpc > r_min_kpc) &
        (gN > 1e-16) &
        (g_obs > 0) &
        (x_raw > 1e-4) &
        (x_raw < 100.0) &
        (nu_raw > 1.0) &
        (nu_raw < 50.0)
    )

    return {
        'x':  x_raw[mask],
        'nu': nu_raw[mask],
        'r':  r_kpc[mask],
        'n_points': int(np.sum(mask)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_mu_plot(
    all_x: np.ndarray,
    all_nu: np.ndarray,
    per_galaxy: list,
    log_masses: list,
    fit_results: dict,
    out_path: Path,
) -> None:
    """Hauptplot: mu(x) mit Datenpunkten und analytischen Fits."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- linkes Panel: log-log mu(x) ----
    ax = axes[0]

    x_th = np.logspace(-4, 2, 400)
    ax.plot(x_th, nu_mcgaugh(x_th),  'k-',  lw=2.5, label='McGaugh (2016)', zorder=5)
    ax.plot(x_th, nu_simple(x_th),   'b--', lw=1.8, label='Simple IF', zorder=5)

    alpha_crm = fit_results.get('crm_native', {}).get('params', {}).get('alpha', np.nan)
    if np.isfinite(alpha_crm):
        ax.plot(x_th, nu_crm(x_th, alpha_crm), 'r-.',
                lw=2.0,
                label=fr'CRM-native ($\alpha={alpha_crm:.3f}$)',
                zorder=5)

    # Datenpunkte nach Galaxiemasse einfarben
    cmap   = plt.cm.plasma
    colors = cmap(np.linspace(0.1, 0.9, len(per_galaxy)))

    for pts, logM, col in zip(per_galaxy, log_masses, colors):
        if pts['n_points'] > 0:
            ax.scatter(
                pts['x'], pts['nu'],
                s=2, alpha=0.35, color=col,
                label=f'$10^{{{logM:.2f}}}$' if logM in [8.0, 9.0, 10.0, 11.0, 12.0, 12.5] else None,
                rasterized=True,
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$x = g_\mathrm{N}/a_0$', fontsize=13)
    ax.set_ylabel(r'$\nu(x) = g_\mathrm{obs}/g_\mathrm{N}$', fontsize=13)
    ax.set_title('Emergente CRM-Interpolationsfunktion', fontsize=12)
    ax.set_xlim(1e-4, 100)
    ax.set_ylim(0.9, 100)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=7, ncol=2, loc='upper right')

    # ---- rechtes Panel: Residuen relativ zu McGaugh ----
    ax2 = axes[1]

    # Median-Binning fuer saubere Residuen
    x_bins = np.logspace(-4, 2, 60)
    bin_idx = np.digitize(all_x, x_bins)
    bin_centers, bin_res_mc, bin_res_si, bin_res_crm = [], [], [], []

    for bi in range(1, len(x_bins)):
        sel = bin_idx == bi
        if np.sum(sel) < 5:
            continue
        xc    = np.median(all_x[sel])
        nu_m  = np.median(all_nu[sel])
        bin_centers.append(xc)
        bin_res_mc.append(nu_m / nu_mcgaugh(xc) - 1.0)
        bin_res_si.append(nu_m / nu_simple(xc)   - 1.0)
        if np.isfinite(alpha_crm):
            bin_res_crm.append(nu_m / nu_crm(xc, alpha_crm) - 1.0)

    bin_centers = np.array(bin_centers)
    bin_res_mc  = np.array(bin_res_mc)
    bin_res_si  = np.array(bin_res_si)

    ax2.axhline(0, color='k', lw=1, ls='--', alpha=0.5)
    ax2.plot(bin_centers, bin_res_mc, 'k-o', ms=4, lw=1.5,
             label='vs. McGaugh')
    ax2.plot(bin_centers, bin_res_si, 'b-s', ms=4, lw=1.5,
             label='vs. Simple')
    if np.isfinite(alpha_crm) and len(bin_res_crm) > 0:
        ax2.plot(bin_centers, np.array(bin_res_crm), 'r-^', ms=4, lw=1.5,
                 label=fr'vs. CRM ($\alpha={alpha_crm:.3f}$)')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'$x = g_\mathrm{N}/a_0$', fontsize=13)
    ax2.set_ylabel(r'$\nu_\mathrm{BVP}/\nu_\mathrm{fit} - 1$', fontsize=13)
    ax2.set_title('Residuen (Median-Bins)', fontsize=12)
    ax2.set_xlim(1e-4, 100)
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=9)

    # Chi^2-Tabelle als Textbox
    txt_lines = [r'$\chi^2_\mathrm{red}$:']
    for key, label in [('mcgaugh', 'McGaugh'), ('simple', 'Simple'), ('crm_native', 'CRM')]:
        if key in fit_results:
            cr = fit_results[key].get('chi2_red', np.nan)
            txt_lines.append(f'  {label}: {cr:.3f}' if np.isfinite(cr) else f'  {label}: N/A')
    ax2.text(0.02, 0.97, '\n'.join(txt_lines),
             transform=ax2.transAxes,
             verticalalignment='top',
             fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(
        f'CFM BVP: mu(x)-Extraktion  |  {len(per_galaxy)} Galaxiemassen '
        fr'($10^{{8}}$–$10^{{12.5}}\,M_\odot$)',
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Gespeichert: {out_path}")


def make_summary_plot(
    per_galaxy: list,
    log_masses: list,
    galaxy_meta: list,
    out_path: Path,
) -> None:
    """
    4-Panel-Ueberblick:
      (1) Anzahl Datenpunkte pro Galaxie
      (2) Slope (RAR log-log Steigung) vs. log M
      (3) V_flat vs. log M
      (4) r_MOND vs. log M
    """
    n_pts   = [pts['n_points'] for pts in per_galaxy]
    slopes  = [m.get('slope',       np.nan) for m in galaxy_meta]
    v_flat  = [m.get('V_flat',      np.nan) for m in galaxy_meta]
    r_mond  = [m.get('r_mond_kpc',  np.nan) for m in galaxy_meta]
    r_th    = [m.get('r_mond_th',   np.nan) for m in galaxy_meta]
    log_m   = np.array(log_masses)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(log_m, n_pts, width=0.2, color='steelblue', alpha=0.8)
    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
    ax.set_ylabel('Datenpunkte')
    ax.set_title('mu(x)-Punkte pro Galaxie')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    ax.scatter(log_m, slopes, s=60, c='tomato', zorder=5)
    ax.axhline(0.5, color='k', ls='--', lw=1.5, label='MOND-Ziel (0.5)')
    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
    ax.set_ylabel('RAR-Steigung')
    ax.set_title('RAR log-log Steigung')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(log_m, v_flat, s=60, c='darkorange', zorder=5, label='CFM BVP')
    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
    ax.set_ylabel(r'$V_\mathrm{flat}$ [km/s]')
    ax.set_title('Flachrotationsgeschwindigkeit')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()

    ax = axes[1, 1]
    ax.scatter(log_m, r_mond, s=60, c='crimson', zorder=5, label='BVP-Ergebnis')
    ax.scatter(log_m, r_th,   s=40, c='forestgreen', marker='^', zorder=5,
               label=r'$\sqrt{GM/a_0}$')
    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
    ax.set_ylabel(r'$r_\mathrm{MOND}$ [kpc]')
    ax.set_title('MOND-Uebergangsradius')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('CFM BVP: 20-Galaxien-Ueberblick', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Gespeichert: {out_path}")


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 72)
    print("CFM Paper IV: mu(x)-Extraktion aus BVP-Loesungen")
    print("=" * 72)
    print(f"a0 = {A0:.6e} m/s^2")
    print(f"Galaxiemassen: {len(LOG_MASSES)} Stufen von 10^{LOG_MASSES[0]} "
          f"bis 10^{LOG_MASSES[-1]} (0.25 dex)")
    print(f"Ergebnisse: {RESULTS_DIR}")
    print()

    send_telegram(
        f"START: mu(x)-Extraktion, {len(LOG_MASSES)} Galaxiemassen "
        f"(10^{LOG_MASSES[0]:.2f} bis 10^{LOG_MASSES[-1]:.2f})"
    )

    # ------------------------------------------------------------------
    # Phase 1: BVP-Solver fuer jede Galaxiemasse ausfuehren
    # ------------------------------------------------------------------
    per_galaxy  = []   # Extrahierte mu(x)-Punkte
    galaxy_meta = []   # Metadaten (slope, V_flat, ...)
    failed      = []

    header = (f"{'log(M)':>7s} {'r_s':>6s} {'r_max':>7s} {'N_pts':>6s} "
              f"{'slope':>7s} {'V_flat':>8s} {'r_MOND':>8s} "
              f"{'status':>8s} {'t[s]':>6s}")
    print(header)
    print("-" * len(header))

    for logM in LOG_MASSES:
        M_gal  = 10**logM * MSUN
        rs_kpc = scale_radius_kpc(logM)
        rmax   = r_max_kpc(logM)
        r_s    = rs_kpc * KPC

        t0 = time.time()
        try:
            res = solve_cfm_v5(
                M_gal=M_gal,
                r_s=r_s,
                r_max_kpc=rmax,
                N=2000,
                epsilon=1.0,
                n_grad=0.5,
                eta=1.0,
                kappa=1.0,
                lambda_steps=[0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0],
                omega=0.15,
                omega_g=0.10,
                max_iter=3000,
                tol=1e-9,
                verbose=False,
            )
            dt = time.time() - t0

            # Konvergenz pruefen
            last = res['convergence_log'][-1]
            converged = last['converged']
            status = "OK" if converged else "WARN"

            # mu(x)-Punkte extrahieren
            pts = extract_mu_points(res)
            per_galaxy.append(pts)

            meta = {
                'log_M':       logM,
                'r_s_kpc':     rs_kpc,
                'r_max_kpc':   rmax,
                'slope':       float(res['slope']),
                'V_flat':      float(res['V_flat']),
                'V_N':         float(res['V_N']),
                'r_mond_kpc':  float(res['r_mond_kpc']),
                'r_mond_th':   float(res['r_mond_theory']),
                'n_mu_points': pts['n_points'],
                'converged':   bool(converged),
                'runtime_s':   float(dt),
            }
            galaxy_meta.append(meta)

            print(f"{logM:7.2f} {rs_kpc:6.2f} {rmax:7.0f} {pts['n_points']:6d} "
                  f"{res['slope']:7.3f} {res['V_flat']:8.1f} "
                  f"{res['r_mond_kpc']:8.1f} {status:>8s} {dt:6.0f}")

            if dt < TIMEOUT_PER_GALAXY:
                send_telegram(
                    f"M=10^{logM:.2f}: slope={res['slope']:.3f}, "
                    f"V_flat={res['V_flat']:.1f} km/s, "
                    f"N_mu={pts['n_points']}, {dt:.0f}s, {status}"
                )

        except Exception as exc:
            dt = time.time() - t0
            print(f"{logM:7.2f} {rs_kpc:6.2f} {rmax:7.0f} {'---':>6s} "
                  f"{'---':>7s} {'---':>8s} {'---':>8s} {'FAIL':>8s} {dt:6.0f}")
            print(f"  [FEHLER] {exc}")
            failed.append({'log_M': logM, 'error': str(exc)})
            per_galaxy.append({'x': np.array([]), 'nu': np.array([]), 'r': np.array([]),
                                'n_points': 0})
            galaxy_meta.append({'log_M': logM, 'converged': False, 'runtime_s': dt,
                                 'error': str(exc)})
            send_telegram(f"FAIL M=10^{logM:.2f}: {str(exc)[:80]}")

    # ------------------------------------------------------------------
    # Phase 2: Kombiniere alle Datenpunkte
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Phase 2: Kombiniere mu(x)-Daten und fitte analytische Formen")

    all_x  = np.concatenate([pts['x']  for pts in per_galaxy if pts['n_points'] > 0])
    all_nu = np.concatenate([pts['nu'] for pts in per_galaxy if pts['n_points'] > 0])

    print(f"  Gesamtzahl Datenpunkte: {len(all_x)}")

    # Logarithmisch aequidistante Bins fuer repraesentativen Fit
    # (Verhindert, dass dichte Regionen den Fit dominieren)
    x_bins    = np.logspace(np.log10(max(all_x.min(), 1e-4)), np.log10(min(all_x.max(), 100)), 100)
    bin_idx   = np.digitize(all_x, x_bins)
    x_binned, nu_binned, nu_sigma_binned = [], [], []

    for bi in range(1, len(x_bins)):
        sel = bin_idx == bi
        if np.sum(sel) < 3:
            continue
        xc  = np.median(all_x[sel])
        nuc = np.median(all_nu[sel])
        # Sigma: MAD (robust) oder Standardfehler des Medians
        mad = np.median(np.abs(all_nu[sel] - nuc))
        sig = max(mad * 1.4826 / np.sqrt(np.sum(sel)), nuc * 0.01)  # min 1% relativer Fehler
        x_binned.append(xc)
        nu_binned.append(nuc)
        nu_sigma_binned.append(sig)

    x_fit    = np.array(x_binned)
    nu_fit   = np.array(nu_binned)
    sig_fit  = np.array(nu_sigma_binned)

    print(f"  Fit-Bins (median): {len(x_fit)}")

    # ------------------------------------------------------------------
    # Phase 3: Analytische Fits
    # ------------------------------------------------------------------
    print("\nFitte analytische Interpolationsfunktionen:")
    fit_results = fit_nu(x_fit, nu_fit, sig_fit)

    for key, label in [('mcgaugh', 'McGaugh'), ('simple', 'Simple IF'), ('crm_native', 'CRM-native')]:
        fr = fit_results.get(key, {})
        chi2_r = fr.get('chi2_red', np.nan)
        pval   = fr.get('p_value',   np.nan)
        params = fr.get('params', {})
        param_str = ', '.join(f'{k}={v:.4f}' for k, v in params.items()
                              if not k.endswith('_err')) if params else 'keine'
        print(f"  [{label}]: chi2_red={chi2_r:.4f}, p={pval:.3f}  params={param_str}")

    alpha_crm = fit_results.get('crm_native', {}).get('params', {}).get('alpha', np.nan)
    alpha_err = fit_results.get('crm_native', {}).get('params', {}).get('alpha_err', np.nan)
    print(f"\n  CRM alpha = {alpha_crm:.4f} +/- {alpha_err:.4f}"
          f"  (MOND erwartet ~0.5)")

    # ------------------------------------------------------------------
    # Phase 4: Tabellarische mu(x)
    # ------------------------------------------------------------------
    x_tab  = np.logspace(-4, 2, 200)
    mu_tab = {
        'x':          x_tab.tolist(),
        'nu_mcgaugh': nu_mcgaugh(x_tab).tolist(),
        'nu_simple':  nu_simple(x_tab).tolist(),
        'nu_crm':     (nu_crm(x_tab, alpha_crm).tolist()
                       if np.isfinite(alpha_crm) else [None] * len(x_tab)),
        'nu_bvp_median': [],
        'x_bvp_bins':    [],
    }
    # BVP-Medianwerte in Tabelle
    for xc, nuc in zip(x_fit, nu_fit):
        mu_tab['x_bvp_bins'].append(float(xc))
        mu_tab['nu_bvp_median'].append(float(nuc))

    # ------------------------------------------------------------------
    # Phase 5: JSON-Output
    # ------------------------------------------------------------------
    output = {
        'meta': {
            'script':       'extract_mu_from_bvp.py',
            'date':         time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'n_galaxies':   len(LOG_MASSES),
            'log_m_range':  [LOG_MASSES[0], LOG_MASSES[-1]],
            'a0':           float(A0),
            'n_total_pts':  int(len(all_x)),
            'n_fit_bins':   int(len(x_fit)),
        },
        'galaxies':    galaxy_meta,
        'fit_results': fit_results,
        'mu_table':    mu_tab,
        'failed':      failed,
    }

    json_path = RESULTS_DIR / 'mu_extraction_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=lambda v: None if v != v else v)
    print(f"\n  [JSON] Gespeichert: {json_path}")

    # ------------------------------------------------------------------
    # Phase 6: Plots
    # ------------------------------------------------------------------
    print("\nErstelle Plots...")
    make_mu_plot(
        all_x, all_nu,
        per_galaxy, LOG_MASSES,
        fit_results,
        RESULTS_DIR / 'mu_extraction_plot.png',
    )
    make_summary_plot(
        per_galaxy, LOG_MASSES, galaxy_meta,
        RESULTS_DIR / 'mu_extraction_summary.png',
    )

    # ------------------------------------------------------------------
    # Abschluss
    # ------------------------------------------------------------------
    dt_total = time.time() - t_start
    n_ok     = sum(1 for m in galaxy_meta if m.get('converged', False))

    print(f"\n{'='*72}")
    print(f"Fertig: {n_ok}/{len(LOG_MASSES)} Galaxien konvergiert")
    print(f"CRM alpha = {alpha_crm:.4f} +/- {alpha_err:.4f}  "
          f"(McGaugh-Limit: 0.5)")
    print(f"chi2_red:  McGaugh={fit_results['mcgaugh']['chi2_red']:.4f}  "
          f"Simple={fit_results['simple']['chi2_red']:.4f}  "
          f"CRM={fit_results.get('crm_native', {}).get('chi2_red', float('nan')):.4f}")
    print(f"Laufzeit: {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print(f"Ergebnisse: {RESULTS_DIR}")

    send_telegram(
        f"FERTIG! {n_ok}/{len(LOG_MASSES)} OK, {dt_total/60:.1f} min\n"
        f"CRM alpha={alpha_crm:.4f}+/-{alpha_err:.4f} (erw. 0.5)\n"
        f"chi2_red: McGaugh={fit_results['mcgaugh']['chi2_red']:.3f}, "
        f"Simple={fit_results['simple']['chi2_red']:.3f}, "
        f"CRM={fit_results.get('crm_native', {}).get('chi2_red', float('nan')):.3f}"
    )


if __name__ == "__main__":
    main()

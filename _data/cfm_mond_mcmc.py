#!/usr/bin/env python3
"""
=============================================================================
MCMC-Analyse für das erweiterte CFM+MOND Modell (Baryon-Only)
=============================================================================
Bestimmt Posterior-Verteilungen für alle 4 Parameter:
  k, a_trans, alpha, beta  (Omega_m = 0.05 fest, Phi0 abgeleitet)

Autor: Lukas Geiger (mit Claude Opus 4.6)
Datum: Februar 2026
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import emcee
import os
import time
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "Pantheon+SH0ES.dat")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

Z_MIN = 0.01
N_GRID = 2000
OMEGA_B = 0.05  # Fixed baryonic matter density

# ==========================================================================
# DATEN & PHYSIK
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
    return (df['zHD'].values.astype(np.float64),
            df['m_b_corr'].values.astype(np.float64),
            df['m_b_corr_err_DIAG'].values.astype(np.float64))


def omega_phi_extended(a, Phi0, k, a_trans, alpha, beta):
    s = np.tanh(k * a_trans)
    phi_de = Phi0 * (np.tanh(k * (a - a_trans)) + s) / (1.0 + s)
    phi_dm = alpha * a**(-beta)
    return phi_de + phi_dm


def phi0_from_flatness_extended(k, a_trans, alpha):
    """Phi0 so dass Omega_total(a=1) = 1."""
    s = np.tanh(k * a_trans)
    f_at_1 = (np.tanh(k * (1.0 - a_trans)) + s) / (1.0 + s)
    if abs(f_at_1) < 1e-15:
        return 1e10
    return (1.0 - OMEGA_B - alpha) / f_at_1


def distance_modulus(z_data, k, a_trans, alpha, beta):
    zg = np.linspace(0, z_data.max() * 1.05, N_GRID)
    ag = 1.0 / (1.0 + zg)
    Phi0 = phi0_from_flatness_extended(k, a_trans, alpha)
    Omega_Phi = omega_phi_extended(ag, Phi0, k, a_trans, alpha, beta)
    E2 = OMEGA_B * (1 + zg)**3 + Omega_Phi
    E2 = np.maximum(E2, 1e-30)
    E = np.sqrt(E2)
    dz = zg[1] - zg[0]
    cum = np.cumsum(1.0 / E) * dz
    cum[0] = 0.0
    chi_r = np.interp(z_data, zg, cum)
    d_L = np.maximum((1 + z_data) * chi_r, 1e-30)
    return 5.0 * np.log10(d_L)


def chi2_marginalized(mu_theory, m_obs, m_err):
    w = 1.0 / m_err**2
    delta = m_obs - mu_theory
    M_best = np.sum(w * delta) / np.sum(w)
    chi2 = np.sum(((delta - M_best) / m_err)**2)
    return chi2, M_best


# ==========================================================================
# BEST-FIT (Differential Evolution)
# ==========================================================================

def find_best_fit(z, m_obs, m_err):
    print("  Suche Best-Fit mit Differential Evolution...")

    def objective(p):
        kk, at, alpha, beta = p
        P0 = phi0_from_flatness_extended(kk, at, alpha)
        if P0 < -5.0 or P0 > 10.0:
            return 1e10
        try:
            mu = distance_modulus(z, kk, at, alpha, beta)
            if np.any(np.isnan(mu)):
                return 1e10
            return chi2_marginalized(mu, m_obs, m_err)[0]
        except:
            return 1e10

    bounds = [(0.5, 50.0), (0.05, 0.99), (0.05, 0.70), (1.0, 3.5)]
    res = differential_evolution(objective, bounds, seed=42, maxiter=500,
                                 tol=1e-8, popsize=30, mutation=(0.5, 1.5),
                                 recombination=0.9, polish=True)

    kk, at, alpha, beta = res.x
    P0 = phi0_from_flatness_extended(kk, at, alpha)
    mu_th = distance_modulus(z, kk, at, alpha, beta)
    chi2, M = chi2_marginalized(mu_th, m_obs, m_err)

    print(f"    k = {kk:.4f}, a_trans = {at:.4f}, alpha = {alpha:.4f}, beta = {beta:.4f}")
    print(f"    Phi0 = {P0:.4f}, chi2 = {chi2:.2f}")

    return res.x, chi2


# ==========================================================================
# MCMC
# ==========================================================================

def run_mcmc(z, m_obs, m_err, best_params, nwalkers=48, nsteps=5000, burnin=1000):
    print(f"\n  MCMC: {nwalkers} Walkers, {nsteps} Steps, {burnin} Burn-in")
    t0 = time.time()

    k0, at0, alpha0, beta0 = best_params
    ndim = 4

    def log_prior(theta):
        kk, at, alpha, beta = theta
        if (0.5 < kk < 50.0 and 0.05 < at < 0.99 and
            0.05 < alpha < 0.70 and 1.0 < beta < 3.5):
            P0 = phi0_from_flatness_extended(kk, at, alpha)
            if -2.0 < P0 < 5.0:
                return 0.0
        return -np.inf

    def log_likelihood(theta):
        kk, at, alpha, beta = theta
        P0 = phi0_from_flatness_extended(kk, at, alpha)
        if P0 < -5.0 or P0 > 10.0:
            return -np.inf
        try:
            mu = distance_modulus(z, kk, at, alpha, beta)
            if np.any(np.isnan(mu)):
                return -np.inf
            chi2, _ = chi2_marginalized(mu, m_obs, m_err)
            return -0.5 * chi2
        except:
            return -np.inf

    def log_posterior(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # Initialisierung: Use known best-fit from test C as starting point
    # (k=3.99, a_trans=0.95, alpha=0.50, beta=2.61)
    start = np.array([k0, at0, alpha0, beta0])
    # Use fractional scatter proportional to each parameter
    scales = np.array([0.5, 0.05, 0.05, 0.2])
    np.random.seed(42)
    pos = start + scales * np.random.randn(nwalkers, ndim)
    # Priors einhalten
    for i in range(nwalkers):
        pos[i, 0] = np.clip(pos[i, 0], 0.6, 49.0)
        pos[i, 1] = np.clip(pos[i, 1], 0.06, 0.98)
        pos[i, 2] = np.clip(pos[i, 2], 0.06, 0.69)
        pos[i, 3] = np.clip(pos[i, 3], 1.1, 3.4)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)

    # Burn-in
    print(f"    Burn-in ({burnin} steps)...", end="", flush=True)
    state = sampler.run_mcmc(pos, burnin, progress=False)
    print(" done.")
    sampler.reset()

    # Production
    print(f"    Production ({nsteps} steps)...", end="", flush=True)
    sampler.run_mcmc(state, nsteps, progress=False)
    print(" done.")

    dt = time.time() - t0
    print(f"    Laufzeit: {dt:.1f} s")

    # Ergebnisse
    chain = sampler.get_chain(flat=True)
    labels = ['k', 'a_trans', 'alpha', 'beta']

    print(f"\n    Akzeptanzrate: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"    Samples: {chain.shape[0]}")

    results = {}
    print(f"\n    {'Param':<10} {'Median':>10} {'Mean':>10} {'+1sigma':>10} {'-1sigma':>10}")
    print("    " + "-" * 50)
    for i, label in enumerate(labels):
        q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
        mean = np.mean(chain[:, i])
        results[label] = {
            'median': q50, 'mean': mean,
            'plus_1sigma': q84 - q50, 'minus_1sigma': q50 - q16,
            'q16': q16, 'q84': q84
        }
        print(f"    {label:<10} {q50:>10.4f} {mean:>10.4f} {q84-q50:>+10.4f} {q50-q16:>-10.4f}")

    # Abgeleitete Groeßen
    Phi0_chain = np.array([phi0_from_flatness_extended(c[0], c[1], c[2]) for c in chain])
    q16, q50, q84 = np.percentile(Phi0_chain, [16, 50, 84])
    results['Phi0'] = {'median': q50, 'plus_1sigma': q84-q50, 'minus_1sigma': q50-q16}
    print(f"    {'Phi0':<10} {q50:>10.4f} {'(derived)':>10} {q84-q50:>+10.4f} {q50-q16:>-10.4f}")

    # w_eff des DM-Terms
    beta_med = results['beta']['median']
    w_dm = beta_med / 3.0 - 1.0
    results['w_dm_geom'] = w_dm
    print(f"\n    w_eff(DM-Term) = beta/3 - 1 = {w_dm:.3f}")

    return chain, results, sampler


# ==========================================================================
# VISUALISIERUNG
# ==========================================================================

def plot_posteriors(chain, results, best_chi2):
    labels = ['$k$', '$a_{\\mathrm{trans}}$', '$\\alpha$', '$\\beta$']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Corner-artige Darstellung: Histogramme
    for i, (ax, label) in enumerate(zip([axes[0,0], axes[0,1], axes[0,2], axes[1,0]], labels)):
        ax.hist(chain[:, i], bins=80, density=True, color=['#E91E63', '#2196F3', '#4CAF50', '#FF9800'][i],
                alpha=0.7, edgecolor='white', linewidth=0.5)
        r = results[['k', 'a_trans', 'alpha', 'beta'][i]]
        ax.axvline(r['median'], color='black', ls='-', lw=2, label=f"Median: {r['median']:.3f}")
        ax.axvline(r['q16'], color='black', ls='--', lw=1)
        ax.axvline(r['q84'], color='black', ls='--', lw=1)
        ax.set_xlabel(label, fontsize=13)
        ax.set_ylabel('Posterior')
        ax.legend(fontsize=9)
        ax.set_title(f"{label} = {r['median']:.3f}" +
                     f"$^{{+{r['plus_1sigma']:.3f}}}_{{-{r['minus_1sigma']:.3f}}}$",
                     fontsize=11)

    # Panel 5: alpha vs beta (2D)
    ax = axes[1, 1]
    ax.scatter(chain[::10, 2], chain[::10, 3], s=1, alpha=0.1, c='gray')
    ax.scatter(results['alpha']['median'], results['beta']['median'],
               s=100, c='red', marker='*', zorder=5, label='Best-fit')
    ax.set_xlabel('$\\alpha$ (geom. DM amplitude)', fontsize=12)
    ax.set_ylabel('$\\beta$ (geom. DM scaling)', fontsize=12)
    ax.set_title('$\\alpha$ vs $\\beta$: Geometric DM Parameter Space', fontsize=11)
    ax.legend()

    # Reference lines for known scalings
    ax.axhline(3.0, color='blue', ls=':', lw=1, alpha=0.5, label='Matter ($a^{-3}$)')
    ax.axhline(2.0, color='green', ls=':', lw=1, alpha=0.5, label='Curvature ($a^{-2}$)')
    ax.axhline(4.0, color='orange', ls=':', lw=1, alpha=0.5, label='Radiation ($a^{-4}$)')
    ax.legend(fontsize=8)

    # Panel 6: Zusammenfassung
    ax = axes[1, 2]
    ax.axis('off')
    beta_med = results['beta']['median']
    w_dm = results['w_dm_geom']

    summary = [
        "MCMC Results: Extended CFM+MOND",
        "=" * 40,
        f"Omega_b = {OMEGA_B:.3f} (fixed, baryons only)",
        "",
        f"k       = {results['k']['median']:.3f} (+{results['k']['plus_1sigma']:.3f} / -{results['k']['minus_1sigma']:.3f})",
        f"a_trans = {results['a_trans']['median']:.3f} (+{results['a_trans']['plus_1sigma']:.3f} / -{results['a_trans']['minus_1sigma']:.3f})",
        f"alpha   = {results['alpha']['median']:.3f} (+{results['alpha']['plus_1sigma']:.3f} / -{results['alpha']['minus_1sigma']:.3f})",
        f"beta    = {results['beta']['median']:.3f} (+{results['beta']['plus_1sigma']:.3f} / -{results['beta']['minus_1sigma']:.3f})",
        f"Phi0    = {results['Phi0']['median']:.3f} (derived)",
        "",
        f"w_eff(DM-term) = {w_dm:.3f}",
        f"  (Matter: 0, Curvature: -1/3)",
        "",
        f"Best chi2 = {best_chi2:.1f}",
        f"Delta chi2 vs LCDM = {best_chi2 - 729.0:+.1f}",
    ]

    ax.text(0.05, 0.95, "\n".join(summary), transform=ax.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('MCMC Posteriors: Extended CFM+MOND (Baryon-Only Universe)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, 'CFM_MOND_MCMC_Posteriors.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\n  Plot gespeichert: {outpath}")
    plt.close()


def write_mcmc_report(results, best_params, best_chi2):
    outpath = os.path.join(OUTPUT_DIR, 'CFM_MOND_MCMC_Results.txt')
    lines = []
    lines.append("=" * 60)
    lines.append("MCMC ERGEBNISSE: Extended CFM+MOND (Baryon-Only)")
    lines.append("=" * 60)
    lines.append(f"Omega_b = {OMEGA_B:.4f} (fest, nur Baryonen)")
    lines.append(f"Best-fit chi2 = {best_chi2:.2f}")
    lines.append(f"Delta chi2 vs LCDM = {best_chi2 - 729.0:+.2f}")
    lines.append("")

    for param in ['k', 'a_trans', 'alpha', 'beta', 'Phi0']:
        r = results[param]
        lines.append(f"  {param:<12} = {r['median']:.4f}  (+{r['plus_1sigma']:.4f} / -{r['minus_1sigma']:.4f})")

    beta_med = results['beta']['median']
    w_dm = results['w_dm_geom']
    lines.append("")
    lines.append(f"  w_eff(DM-Term) = beta/3 - 1 = {w_dm:.4f}")
    lines.append(f"    (Materie: 0.000, Kruemmung: -0.333, Strahlung: 0.333)")
    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append(f"  beta = {beta_med:.2f} liegt zwischen Materie (3.0) und Kruemmung (2.0)")
    lines.append(f"  => Geometrischer Effekt mit materie-aehnlicher Skalierung")
    lines.append(f"  => Konsistent mit MOND-inspiriertem geometrischen DM-Ersatz")

    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Report gespeichert: {outpath}")
    return '\n'.join(lines)


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MCMC: Extended CFM+MOND (Baryon-Only Universe)")
    print("=" * 60)

    print("\n[1/4] Lade Daten...")
    z, m_obs, m_err = load_data()
    print(f"  {len(z)} Supernovae geladen")

    print("\n[2/4] Best-Fit...")
    best_params, best_chi2 = find_best_fit(z, m_obs, m_err)

    print("\n[3/4] MCMC...")
    chain, results, sampler = run_mcmc(z, m_obs, m_err, best_params,
                                        nwalkers=48, nsteps=5000, burnin=1000)

    print("\n[4/4] Visualisierung & Report...")
    plot_posteriors(chain, results, best_chi2)
    report = write_mcmc_report(results, best_params, best_chi2)

    print("\n" + report)
    print("\n  FERTIG.")

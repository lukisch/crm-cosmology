#!/usr/bin/env python3
"""
Analyze cfm_fR MCMC results from run_full_mcmc.py.
Reads /tmp/cfm_fR_mcmc_results.npz and produces:
  1. Summary statistics (mean, std, 68% CI for each parameter)
  2. Detection significance for alpha_M_0 > 0
  3. Corner plot (if matplotlib available)
  4. LaTeX-formatted table for Paper III
"""
import numpy as np
import sys

# Load results
try:
    data = np.load('/tmp/cfm_fR_mcmc_results.npz', allow_pickle=True)
except FileNotFoundError:
    print("ERROR: /tmp/cfm_fR_mcmc_results.npz not found.")
    print("Run run_full_mcmc.py first.")
    sys.exit(1)

chain = data['chain']
log_prob = data['log_prob']
param_names = list(data['param_names'])
best_params = data['best_params']
best_chi2 = float(data['best_chi2'])

print("=" * 80)
print("cfm_fR MCMC ANALYSIS")
print("=" * 80)
print(f"\nChain shape: {chain.shape} ({chain.shape[0]} samples)")
print(f"Parameters: {param_names}")

# ================================================================
# 1. SUMMARY STATISTICS
# ================================================================
print(f"\n{'='*80}")
print("MARGINALIZED CONSTRAINTS (68% CI)")
print(f"{'='*80}")
print(f"{'Parameter':>12s} {'Mean':>12s} {'Std':>10s} {'Median':>12s} "
      f"{'16%':>10s} {'84%':>10s}")
print("-" * 70)

results = {}
for i, name in enumerate(param_names):
    vals = chain[:, i]
    q16, q50, q84 = np.percentile(vals, [16, 50, 84])
    mean = np.mean(vals)
    std = np.std(vals)
    results[name] = {
        'mean': mean, 'std': std,
        'q16': q16, 'q50': q50, 'q84': q84,
        'up': q84 - q50, 'down': q50 - q16
    }
    print(f"  {name:>12s} {mean:>12.6f} {std:>10.6f} "
          f"{q50:>12.6f} {q16:>10.6f} {q84:>10.6f}")

# ================================================================
# 2. BEST-FIT
# ================================================================
print(f"\n{'='*80}")
print("BEST-FIT PARAMETERS")
print(f"{'='*80}")
best_idx = np.argmax(log_prob)
best = chain[best_idx]
chi2_best = -2 * log_prob[best_idx]
print(f"Best chi2 = {chi2_best:.1f} (dchi2 = {chi2_best - 6628.8:.1f} vs LCDM)")
for i, name in enumerate(param_names):
    print(f"  {name:>12s} = {best[i]:.6f}")

# ================================================================
# 3. DETECTION SIGNIFICANCE
# ================================================================
print(f"\n{'='*80}")
print("DETECTION SIGNIFICANCE")
print(f"{'='*80}")

aM0 = chain[:, 0]
frac_above = np.mean(aM0 > 1e-6)
mean_aM0 = np.mean(aM0)
std_aM0 = np.std(aM0)

print(f"P(alpha_M_0 > 0) = {frac_above:.4f}")
if std_aM0 > 0:
    sigma = mean_aM0 / std_aM0
    print(f"alpha_M_0 detection: {sigma:.2f} sigma")
    print(f"  mean = {mean_aM0:.6f}")
    print(f"  std  = {std_aM0:.6f}")

# Upper limit
q95 = np.percentile(aM0, 95)
q99 = np.percentile(aM0, 99)
print(f"  95% upper limit: alpha_M_0 < {q95:.6f}")
print(f"  99% upper limit: alpha_M_0 < {q99:.6f}")

# n_exp
n_chain = chain[:, 1]
print(f"\nn_exp constraint: {np.mean(n_chain):.3f} +/- {np.std(n_chain):.3f}")
print(f"  68% CI: [{np.percentile(n_chain, 16):.3f}, {np.percentile(n_chain, 84):.3f}]")

# ================================================================
# 4. DERIVED QUANTITIES
# ================================================================
print(f"\n{'='*80}")
print("DERIVED QUANTITIES")
print(f"{'='*80}")

h = 0.6732
omega_b = 0.02237
omega_cdm_chain = chain[:, 2]
As_chain = np.exp(chain[:, 3]) * 1e-10
aM_today = chain[:, 0] * chain[:, 1] / (1 + chain[:, 0])
Omega_m_chain = (omega_b + omega_cdm_chain) / h**2

print(f"Omega_m: {np.mean(Omega_m_chain):.4f} +/- {np.std(Omega_m_chain):.4f}")
print(f"A_s (1e-9): {np.mean(As_chain)*1e9:.4f} +/- {np.std(As_chain)*1e9:.4f}")
print(f"alpha_M(a=1): {np.mean(aM_today):.6f} +/- {np.std(aM_today):.6f}")

# ================================================================
# 5. CORRELATION MATRIX
# ================================================================
print(f"\n{'='*80}")
print("CORRELATION MATRIX")
print(f"{'='*80}")
corr = np.corrcoef(chain.T)
print(f"{'':>12s}", end='')
for name in param_names:
    print(f" {name:>10s}", end='')
print()
for i, name in enumerate(param_names):
    print(f"  {name:>12s}", end='')
    for j in range(len(param_names)):
        print(f" {corr[i,j]:>10.3f}", end='')
    print()

# ================================================================
# 6. LATEX TABLE FOR PAPER III
# ================================================================
print(f"\n{'='*80}")
print("LATEX TABLE (for Paper III)")
print(f"{'='*80}")
print(r"""
\begin{table}[h]
\centering
\caption{Marginalized posterior constraints from the cfm\_fR MCMC analysis
(24 walkers, 400 production steps, TT+TE+EE against 6{,}405 Planck data points).}
\label{tab:mcmc_results}
\begin{tabular}{lccc}
\toprule
\textbf{Parameter} & \textbf{Best-fit} & \textbf{Mean $\pm$ std} & \textbf{68\% CI} \\
\midrule""")

latex_names = {
    'alpha_M_0': r'$\alpha_{M,0}$',
    'n_exp': r'$n$',
    'omega_cdm': r'$\omega_{\mathrm{cdm}}$',
    'logAs': r'$\ln(10^{10}A_s)$',
    'n_s': r'$n_s$'
}

for i, name in enumerate(param_names):
    r = results[name]
    ln = latex_names.get(name, name)
    print(f"{ln} & ${best[i]:.5f}$ & ${r['mean']:.5f} \\pm {r['std']:.5f}$ "
          f"& ${r['q50']:.5f}^{{+{r['up']:.5f}}}_{{-{r['down']:.5f}}}$ \\\\")

print(r"""\midrule
\multicolumn{4}{c}{\textit{Derived}} \\""")
print(f"$\\chi^2_{{\\mathrm{{best}}}}$ & ${chi2_best:.1f}$ & --- & --- \\\\")
print(f"$\\Delta\\chi^2$ vs.\\ $\\Lambda$CDM & ${chi2_best - 6628.8:.1f}$ & --- & --- \\\\")
if std_aM0 > 0:
    print(f"$\\alpha_{{M,0}} > 0$ significance & ${mean_aM0/std_aM0:.1f}\\sigma$ & --- & --- \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ================================================================
# 7. CORNER PLOT (if matplotlib available)
# ================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 5, figsize=(14, 14))
    labels = [r'$\alpha_{M,0}$', r'$n$', r'$\omega_{cdm}$',
              r'$\ln(10^{10}A_s)$', r'$n_s$']

    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(chain[:, i], bins=50, density=True,
                        color='steelblue', alpha=0.7)
                ax.axvline(best[i], color='red', ls='--', lw=1)
            else:
                ax.scatter(chain[::10, j], chain[::10, i],
                           c=log_prob[::10], s=1, alpha=0.3, cmap='viridis')
                ax.plot(best[j], best[i], 'r+', ms=10, mew=2)
            if i == 4:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])

    plt.tight_layout()
    plt.savefig('/tmp/cfm_fR_mcmc_corner.png', dpi=150)
    print("\nCorner plot saved to /tmp/cfm_fR_mcmc_corner.png")
except ImportError:
    print("\n(matplotlib not available, skipping corner plot)")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")

#!/usr/bin/env python3
"""
Plot: chi2-sigma8 Tradeoff und alpha_M-Konvergenz.
Zwei Subplots: Links chi2 vs sigma8, Rechts alpha_M scan.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ================================================================
# LEFT: chi2 vs sigma8 tradeoff for all models
# ================================================================

# LCDM reference
lcdm_chi2 = 2539.5
lcdm_s8 = 0.811

# propto_omega data (cM, dchi2, sigma8)
po_data = [
    (0.0002, -0.1, 0.826),
    (0.0003, -0.2, 0.833),
    (0.0005, -0.5, 0.849),
    (0.0007, -0.7, 0.865),
    (0.0010, -1.2, 0.891),
    (0.0015, -1.6, 0.937),
    (0.0020, -0.5, 0.987),
]

# propto_scale data
ps_data = [
    (0.0001, -0.3, 0.824),
    (0.0002, -0.6, 0.837),
    (0.0003, -0.9, 0.851),
    (0.0005, -1.6, 0.880),
    (0.0007, -2.3, 0.910),
    (0.0010, -2.9, 0.960),
]

# constant_alphas std f(R)
ca_data = [
    (1e-5,  -0.07, 0.814),
    (5e-5,  -0.27, 0.826),
    (1e-4,  -0.29, 0.842),
    (1.5e-4, -0.08, 0.859),
    (2e-4,  +0.48, 0.876),
    (3e-4,  +3.18, 0.914),
    (5e-4, +25.51, 1.001),
]

# Plot propto_omega
po_s8 = [d[2] for d in po_data]
po_dchi2 = [d[1] for d in po_data]
ax1.plot(po_s8, po_dchi2, 'o-', color='#2196F3', markersize=8, linewidth=2, label='propto_omega (CFM)')
for d in po_data:
    ax1.annotate(f'cM={d[0]}', (d[2], d[1]), fontsize=6, textcoords="offset points", xytext=(5, 5))

# Plot propto_scale
ps_s8 = [d[2] for d in ps_data]
ps_dchi2 = [d[1] for d in ps_data]
ax1.plot(ps_s8, ps_dchi2, 's-', color='#FF5722', markersize=8, linewidth=2, label='propto_scale (CFM)')
for d in ps_data:
    ax1.annotate(f'cM={d[0]}', (d[2], d[1]), fontsize=6, textcoords="offset points", xytext=(5, 5))

# Plot constant_alphas
ca_s8 = [d[2] for d in ca_data if d[1] < 10]
ca_dchi2 = [d[1] for d in ca_data if d[1] < 10]
ax1.plot(ca_s8, ca_dchi2, 'D-', color='#4CAF50', markersize=8, linewidth=2, label='const_alphas (std fR)')

# LCDM reference
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='LCDM')
ax1.axvline(x=0.811, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Shaded regions
ax1.axhspan(-10, 0, alpha=0.1, color='green')
ax1.axvspan(0.811*0.95, 0.811*1.05, alpha=0.1, color='blue')

ax1.set_xlabel(r'$\sigma_8$', fontsize=14)
ax1.set_ylabel(r'$\Delta\chi^2$ vs LCDM', fontsize=14)
ax1.set_title(r'$\chi^2$-$\sigma_8$ Tradeoff (Planck TT)', fontsize=14)
ax1.legend(fontsize=10, loc='upper left')
ax1.set_xlim(0.80, 1.00)
ax1.set_ylim(-4, 5)
ax1.grid(True, alpha=0.3)
ax1.text(0.85, -3.5, r'$\Delta\chi^2 < 0$: besser als LCDM', fontsize=9, color='green')
ax1.text(0.815, 4, r'Planck $\sigma_8 \pm 5\%$', fontsize=8, color='blue', alpha=0.7)

# ================================================================
# RIGHT: alpha_M convergence test
# ================================================================

# CFM relation data
cfm_aM = [1e-15, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
cfm_dchi2 = [-0.00, -0.01, -0.00, +0.00, +0.04, -0.01, +0.03, +0.23, +0.75, +2.67, +5.80, +16.11]
cfm_s8 = [0.8108, 0.8108, 0.8108, 0.8108, 0.8108, 0.8109, 0.8118, 0.8157, 0.8207, 0.8311, 0.8421, 0.8660]

# std f(R) relation data
fr_aM = [1e-15, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
fr_dchi2 = [-0.01, -0.01, -0.00, +0.00, +0.04, -0.02, -0.07, -0.27, -0.29, +0.48, +3.18, +25.51]
fr_s8 = [0.8108, 0.8108, 0.8108, 0.8108, 0.8108, 0.8111, 0.8138, 0.8260, 0.8419, 0.8761, 0.9138, 1.0014]

ax2.semilogx(cfm_aM, cfm_dchi2, 'o-', color='#2196F3', markersize=6, linewidth=2,
             label=r'CFM: $\alpha_B = -\alpha_M/2$')
ax2.semilogx(fr_aM, fr_dchi2, 's-', color='#FF5722', markersize=6, linewidth=2,
             label=r'Std f(R): $\alpha_B = -\alpha_M$')

ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax2.axhspan(-1, 0, alpha=0.1, color='green')

# Mark the optimal point
ax2.plot(1e-4, -0.29, '*', color='red', markersize=15, zorder=5)
ax2.annotate('Optimum\naM=1e-4\ndchi2=-0.29', (1e-4, -0.29), fontsize=8,
             textcoords="offset points", xytext=(15, -20), color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

ax2.set_xlabel(r'$\alpha_M$ (constant_alphas)', fontsize=14)
ax2.set_ylabel(r'$\Delta\chi^2$ vs LCDM', fontsize=14)
ax2.set_title(r'$\alpha_M \to 0$ Konvergenz', fontsize=14)
ax2.legend(fontsize=10)
ax2.set_xlim(1e-16, 1e-3)
ax2.set_ylim(-1, 8)
ax2.grid(True, alpha=0.3)
ax2.text(1e-13, 6, r'$\alpha_M < 10^{-6}$:' + '\nnumerisch = LCDM', fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('/tmp/cfm_tradeoff_convergence.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_tradeoff_convergence.png")

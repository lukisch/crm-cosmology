#!/usr/bin/env python3
"""Plot f*sigma8(z) for LCDM and CFM models vs observations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# z values
z = [0.0, 0.1, 0.15, 0.2, 0.3, 0.38, 0.51, 0.61, 0.70, 0.80, 1.00, 1.50, 2.00]

# Model predictions
lcdm =     [0.426, 0.450, 0.458, 0.464, 0.473, 0.475, 0.473, 0.468, 0.461, 0.452, 0.431, 0.374, 0.324]
po_0002 =  [0.467, 0.484, 0.489, 0.493, 0.496, 0.495, 0.488, 0.480, 0.471, 0.460, 0.436, 0.376, 0.325]
po_0005 =  [0.533, 0.539, 0.539, 0.538, 0.532, 0.525, 0.511, 0.498, 0.486, 0.472, 0.443, 0.378, 0.326]
po_001 =   [0.659, 0.642, 0.631, 0.620, 0.597, 0.579, 0.550, 0.529, 0.511, 0.492, 0.456, 0.383, 0.328]
ps_0003 =  [0.518, 0.528, 0.530, 0.531, 0.529, 0.524, 0.513, 0.502, 0.490, 0.477, 0.449, 0.383, 0.329]
ps_001 =   [0.797, 0.757, 0.737, 0.718, 0.683, 0.656, 0.617, 0.588, 0.564, 0.539, 0.494, 0.405, 0.341]

# Observations
obs = [
    (0.02, 0.428, 0.048, "ALFALFA"),
    (0.10, 0.370, 0.130, "6dFGS"),
    (0.15, 0.490, 0.145, "2MTF"),
    (0.38, 0.497, 0.045, "BOSS LOWZ"),
    (0.51, 0.458, 0.038, "BOSS CMASS"),
    (0.61, 0.436, 0.034, "BOSS CMASS"),
    (0.77, 0.490, 0.180, "VIPERS"),
    (0.85, 0.450, 0.110, "DESI LRG"),
    (1.40, 0.482, 0.116, "FastSound"),
]

# LEFT: f*sigma8 vs z
ax1.plot(z, lcdm, 'k-', linewidth=2.5, label=r'$\Lambda$CDM', zorder=3)
ax1.plot(z, po_0002, '-', color='#2196F3', linewidth=2, label=r'pO $c_M$=0.0002')
ax1.plot(z, po_0005, '--', color='#2196F3', linewidth=1.5, label=r'pO $c_M$=0.0005')
ax1.plot(z, po_001, ':', color='#2196F3', linewidth=1.5, label=r'pO $c_M$=0.001')
ax1.plot(z, ps_0003, '-', color='#FF5722', linewidth=2, label=r'pS $c_M$=0.0003')

# Plot observations
obs_z = [o[0] for o in obs]
obs_v = [o[1] for o in obs]
obs_e = [o[2] for o in obs]
ax1.errorbar(obs_z, obs_v, yerr=obs_e, fmt='o', color='gray', markersize=6,
             capsize=3, linewidth=1, label='RSD data', zorder=5)
for o in obs:
    ax1.annotate(o[3], (o[0], o[1]), fontsize=6, textcoords="offset points",
                 xytext=(5, 5), color='gray')

ax1.set_xlabel('Redshift z', fontsize=13)
ax1.set_ylabel(r'$f\sigma_8(z)$', fontsize=13)
ax1.set_title(r'Growth Rate $f\sigma_8$ vs Observations', fontsize=13)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_xlim(-0.05, 2.1)
ax1.set_ylim(0.25, 0.85)
ax1.grid(True, alpha=0.3)

# RIGHT: S8 comparison
surveys = ['Planck+ACT\n+SPT', 'KiDS\nLegacy', 'DES Y6\n3x2pt', 'HSC Y3',
           'eROSITA\nCluster', r'CFM pO' + '\ncM=0.0002', r'CFM pO' + '\ncM=0.0005']
s8_vals = [0.836, 0.815, 0.789, 0.776, 0.860, 0.847, 0.871]
s8_errs = [0.013, 0.021, 0.012, 0.033, 0.010, 0.0, 0.0]
colors = ['#333333', '#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#2196F3', '#2196F3']

for i, (s, v, e, c) in enumerate(zip(surveys, s8_vals, s8_errs, colors)):
    if e > 0:
        ax2.errorbar(i, v, yerr=e, fmt='o', color=c, markersize=10, capsize=5,
                     linewidth=2, zorder=5)
    else:
        ax2.plot(i, v, '*', color=c, markersize=15, zorder=5)

ax2.set_xticks(range(len(surveys)))
ax2.set_xticklabels(surveys, fontsize=8)
ax2.set_ylabel(r'$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$', fontsize=13)
ax2.set_title(r'$S_8$ Comparison (2026)', fontsize=13)

# Reference bands
ax2.axhspan(0.836-0.013, 0.836+0.013, alpha=0.15, color='gray', label='Planck 1σ')
ax2.axhline(y=0.836, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax2.set_ylim(0.72, 0.92)
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('/tmp/cfm_fsigma8_s8.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_fsigma8_s8.png")

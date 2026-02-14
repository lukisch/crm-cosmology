#!/usr/bin/env python3
"""Plot 2D chi2 contour from grid scan results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
data = np.load('/tmp/cfm_grid_results.npz')
omch2_grid = data['omch2_grid']
aM_grid = data['aM_grid']
results = data['results']  # (omch2, aM, chi2, s8, th)

# Reshape into 2D grid
n_om = len(omch2_grid)
n_am = len(aM_grid)
chi2_2d = np.full((n_om, n_am), np.nan)
s8_2d = np.full((n_om, n_am), np.nan)

for i, omch2 in enumerate(omch2_grid):
    for j, aM in enumerate(aM_grid):
        idx = i * n_am + j
        if results[idx, 2] < 1e9:
            chi2_2d[i, j] = results[idx, 2]
            s8_2d[i, j] = results[idx, 3]

chi2_min = np.nanmin(chi2_2d)
dchi2 = chi2_2d - chi2_min

# Skip aM=0 column (crashes)
aM_plot = aM_grid[1:]
dchi2_plot = dchi2[:, 1:]
s8_plot = s8_2d[:, 1:]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Delta chi2 contour
ax = axes[0]
# Use only valid points for contour
X, Y = np.meshgrid(aM_plot * 1000, omch2_grid)  # aM in units of 10^-3
levels = [2.30, 6.18, 11.83]  # 1sigma, 2sigma, 3sigma for 2 params
cs = ax.contour(X, Y, dchi2_plot, levels=levels,
                colors=['blue', 'green', 'red'], linewidths=[2, 1.5, 1])
ax.clabel(cs, fmt={2.30: '68%', 6.18: '95%', 11.83: '99.7%'}, fontsize=10)

# Also show filled contour
cf = ax.contourf(X, Y, dchi2_plot, levels=[0, 2.30, 6.18, 11.83, 50],
                 colors=['#0066ff40', '#00ff0030', '#ff000020', '#ffffff00'])

# Mark minimum
best_idx = np.unravel_index(np.nanargmin(dchi2_plot), dchi2_plot.shape)
ax.plot(aM_plot[best_idx[1]] * 1000, omch2_grid[best_idx[0]],
        'k*', markersize=15, label=f'Best fit')
ax.set_xlabel(r'$\alpha_M \times 10^3$', fontsize=13)
ax.set_ylabel(r'$\omega_{cdm}$', fontsize=13)
ax.set_title(r'$\Delta\chi^2$ contours (CFM + constant $\alpha_M$)', fontsize=13)
ax.legend(fontsize=11)

# Right: sigma8 contour
ax2 = axes[1]
cs2 = ax2.contour(X, Y, s8_plot,
                  levels=[0.80, 0.811, 0.83, 0.85, 0.87, 0.90],
                  colors='gray', linewidths=1)
ax2.clabel(cs2, fmt='%.3f', fontsize=9)

# Overlay chi2 contour
ax2.contour(X, Y, dchi2_plot, levels=[2.30, 6.18],
            colors=['blue', 'green'], linewidths=[2, 1.5], linestyles='--')

# Planck sigma8 band
ax2.axhline(y=0, color='white')  # dummy

ax2.plot(aM_plot[best_idx[1]] * 1000, omch2_grid[best_idx[0]],
         'k*', markersize=15, label=f'Best fit ($\\sigma_8$={s8_plot[best_idx]:.3f})')
ax2.set_xlabel(r'$\alpha_M \times 10^3$', fontsize=13)
ax2.set_ylabel(r'$\omega_{cdm}$', fontsize=13)
ax2.set_title(r'$\sigma_8$ contours + $\Delta\chi^2$ (dashed)', fontsize=13)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/tmp/cfm_contour.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_contour.png")

# Also create 1D marginalized plots
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))

# 1D: omega_cdm (marginalize over alpha_M)
ax3 = axes2[0]
chi2_marg_om = np.nanmin(dchi2_plot, axis=1)
ax3.plot(omch2_grid, chi2_marg_om, 'b-o', linewidth=2)
ax3.axhline(1, color='gray', linestyle=':', linewidth=0.5)
ax3.axhline(4, color='gray', linestyle=':', linewidth=0.5)
ax3.set_xlabel(r'$\omega_{cdm}$', fontsize=13)
ax3.set_ylabel(r'$\Delta\chi^2_{min}$', fontsize=13)
ax3.set_title(r'Profile likelihood: $\omega_{cdm}$', fontsize=12)
ax3.set_ylim(-1, 20)

# 1D: alpha_M (marginalize over omega_cdm)
ax4 = axes2[1]
chi2_marg_am = np.nanmin(dchi2_plot, axis=0)
ax4.plot(aM_plot * 1000, chi2_marg_am, 'r-o', linewidth=2)
ax4.axhline(1, color='gray', linestyle=':', linewidth=0.5)
ax4.axhline(4, color='gray', linestyle=':', linewidth=0.5)
ax4.set_xlabel(r'$\alpha_M \times 10^3$', fontsize=13)
ax4.set_ylabel(r'$\Delta\chi^2_{min}$', fontsize=13)
ax4.set_title(r'Profile likelihood: $\alpha_M$', fontsize=12)
ax4.set_ylim(-1, 20)

plt.tight_layout()
plt.savefig('/tmp/cfm_profile.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_profile.png")

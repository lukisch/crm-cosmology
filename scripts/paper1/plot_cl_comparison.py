#!/usr/bin/env python3
"""
Generate publication-quality C_l comparison plots.
Saves as PNG since matplotlib.pyplot may not have display.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load Planck data
planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]
planck_err_lo = planck_data[:, 2]
planck_err_hi = planck_data[:, 3]
planck_err = (planck_err_lo + planck_err_hi) / 2.0

def compute_spectrum(name, extra_params, As=2.1e-9, ns=0.9649):
    cosmo = Class()
    base = {
        'output': 'tCl,pCl,lCl,mPk',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': 0.673,
        'T_cmb': 2.7255,
        'omega_b': 0.02237,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'tau_reio': 0.0544,
        'A_s': As,
        'n_s': ns,
    }
    base.update(extra_params)
    cosmo.set(base)
    cosmo.compute()
    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    T = 2.7255e6
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
    s8 = cosmo.sigma8()
    th = cosmo.theta_s_100() / 100.0
    cosmo.struct_cleanup()
    cosmo.empty()
    return ell, Dl, s8, th

# Compute models
print("Computing LCDM...")
ell_lcdm, Dl_lcdm, s8_lcdm, th_lcdm = compute_spectrum(
    "LCDM", {'omega_cdm': 0.1200, 'h': 0.6736}, As=2.1e-9, ns=0.9649)

print("Computing CFM+const_alphas (optimized)...")
ell_ca, Dl_ca, s8_ca, th_ca = compute_spectrum(
    "CFM+const", {
        'omega_cdm': 0.1143,
        'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
        'gravity_model': 'constant_alphas',
        'parameters_smg': '0.0, -0.00025, 0.0005, 0.0, 1.0',
        'expansion_model': 'lcdm', 'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }, As=2.05e-9, ns=0.97)

print("Computing CFM+cfm_fR (n=0.01, optimized)...")
ell_fr, Dl_fr, s8_fr, th_fr = compute_spectrum(
    "CFM+fR", {
        'omega_cdm': 0.1138,
        'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
        'gravity_model': 'cfm_fR',
        'parameters_smg': '0.06, 0.01, 1.0',
        'expansion_model': 'lcdm', 'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }, As=2.05e-9, ns=0.97)

print("Computing CFM Basis (no MG)...")
ell_b, Dl_b, s8_b, th_b = compute_spectrum(
    "CFM basis", {'omega_cdm': 0.1066}, As=2.05e-9, ns=0.97)

# ============================================================
# Figure 1: Full C_l comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1],
                                gridspec_kw={'hspace': 0.05})

# Top panel: D_l spectra
mask = (planck_ell >= 2) & (planck_ell <= 2500)
ax1.errorbar(planck_ell[mask], planck_Dl[mask],
             yerr=[planck_err_lo[mask], planck_err_hi[mask]],
             fmt='.', color='gray', alpha=0.3, markersize=1, label='Planck 2018 TT')

ax1.plot(ell_lcdm, Dl_lcdm, 'k-', linewidth=1.5, label=f'LCDM ($\\sigma_8$={s8_lcdm:.3f})')
ax1.plot(ell_ca, Dl_ca, 'b-', linewidth=1.2, alpha=0.8,
         label=f'CFM+const ($\\sigma_8$={s8_ca:.3f}, $\\chi^2/n$=1.09)')
ax1.plot(ell_fr, Dl_fr, 'r--', linewidth=1.2, alpha=0.8,
         label=f'CFM+$f(R)$ n=0.01 ($\\sigma_8$={s8_fr:.3f}, $\\chi^2/n$=1.10)')
ax1.plot(ell_b, Dl_b, 'g:', linewidth=1.0, alpha=0.6,
         label=f'CFM basis ($\\sigma_8$={s8_b:.3f})')

ax1.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi$ [$\mu K^2$]', fontsize=14)
ax1.set_xlim(2, 2500)
ax1.set_ylim(0, 6500)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_title('CMB TT Power Spectrum: CFM Models vs Planck 2018', fontsize=14)
ax1.tick_params(labelbottom=False)

# Bottom panel: Residuals (model - Planck) / sigma
mask_res = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)
ell_res = planck_ell[mask_res]

for model_ell, model_Dl, color, ls, label in [
    (ell_lcdm, Dl_lcdm, 'k', '-', 'LCDM'),
    (ell_ca, Dl_ca, 'b', '-', 'CFM+const'),
    (ell_fr, Dl_fr, 'r', '--', 'CFM+f(R)'),
]:
    interp_Dl = np.interp(planck_ell, model_ell, model_Dl)
    residual = (interp_Dl[mask_res] - planck_Dl[mask_res]) / planck_err[mask_res]
    # Smooth for visibility (running average over 20 points)
    kernel = np.ones(20) / 20
    smooth_res = np.convolve(residual, kernel, mode='same')
    ax2.plot(ell_res, smooth_res, color=color, linestyle=ls, linewidth=1.2, label=label)

ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axhline(1, color='gray', linewidth=0.3, linestyle=':')
ax2.axhline(-1, color='gray', linewidth=0.3, linestyle=':')
ax2.set_xlabel(r'Multipole $\ell$', fontsize=14)
ax2.set_ylabel(r'Residual [$\sigma$]', fontsize=14)
ax2.set_xlim(2, 2500)
ax2.set_ylim(-3, 3)
ax2.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('/tmp/cfm_cl_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_cl_comparison.png")

# ============================================================
# Figure 2: Zoom on first 3 peaks
# ============================================================
fig2, ax3 = plt.subplots(1, 1, figsize=(12, 6))

mask_zoom = (planck_ell >= 100) & (planck_ell <= 1000)
ax3.errorbar(planck_ell[mask_zoom], planck_Dl[mask_zoom],
             yerr=[planck_err_lo[mask_zoom], planck_err_hi[mask_zoom]],
             fmt='o', color='gray', alpha=0.4, markersize=3, label='Planck 2018')

ax3.plot(ell_lcdm, Dl_lcdm, 'k-', linewidth=2, label='LCDM (Planck best-fit)')
ax3.plot(ell_ca, Dl_ca, 'b-', linewidth=1.5, label='CFM + constant $\\alpha_M$')
ax3.plot(ell_fr, Dl_fr, 'r--', linewidth=1.5, label='CFM + $f(R)$ (n=0.01)')
ax3.plot(ell_b, Dl_b, 'g:', linewidth=1.5, alpha=0.7, label='CFM basis (no MG)')

ax3.set_xlabel(r'Multipole $\ell$', fontsize=14)
ax3.set_ylabel(r'$D_\ell$ [$\mu K^2$]', fontsize=14)
ax3.set_xlim(100, 1000)
ax3.set_ylim(0, 6500)
ax3.legend(fontsize=11)
ax3.set_title('CMB TT Peaks 1-3: CFM vs LCDM vs Planck 2018', fontsize=14)

plt.tight_layout()
plt.savefig('/tmp/cfm_cl_peaks.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_cl_peaks.png")

# ============================================================
# Figure 3: sigma8 vs chi2 summary
# ============================================================
fig3, ax4 = plt.subplots(1, 1, figsize=(8, 6))

# Data points
models_data = [
    ('LCDM', 1.028, 0.811, 'k', 's', 14),
    ('CFM+const\n($\\alpha_M$=const)', 1.089, 0.832, 'b', 'o', 12),
    ('CFM+$f(R)$\nn=0.01', 1.103, 0.847, 'r', 'D', 12),
    ('CFM+$f(R)$\nn=0.1', 1.413, 1.359, 'orange', '^', 12),
    ('CFM basis\n(no MG)', 2.10, 0.751, 'g', 'v', 12),
]

for name, chi2n, s8, color, marker, ms in models_data:
    ax4.scatter(chi2n, s8, color=color, marker=marker, s=ms**2, zorder=5)
    ax4.annotate(name, (chi2n, s8), textcoords="offset points",
                 xytext=(10, 5), fontsize=9)

# Planck reference lines
ax4.axhline(0.811, color='gray', linewidth=0.5, linestyle='--', label='Planck $\\sigma_8$=0.811')
ax4.axhspan(0.811-0.006, 0.811+0.006, alpha=0.1, color='gray')
ax4.axvline(1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

ax4.set_xlabel(r'$\chi^2 / n_{pts}$ (Planck TT, $\ell$=30-2500)', fontsize=13)
ax4.set_ylabel(r'$\sigma_8$', fontsize=13)
ax4.set_title(r'Model Quality: $\sigma_8$ vs $\chi^2$/n', fontsize=14)
ax4.set_xlim(0.9, 2.3)
ax4.set_ylim(0.7, 1.0)
ax4.legend(fontsize=10)

plt.tight_layout()
plt.savefig('/tmp/cfm_sigma8_chi2.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/cfm_sigma8_chi2.png")

print("\nAll plots saved to /tmp/")

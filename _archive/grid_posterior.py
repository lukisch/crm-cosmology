#!/usr/bin/env python3
"""
Grid-based posterior approximation for CFM+constant_alphas.
2D scan of (omega_cdm, alpha_M) with fixed A_s=2.05e-9, n_s=0.97.
Much faster than MCMC: ~30s x 100 points = 50 min.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

# Load Planck data
planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]
planck_err = (planck_data[:, 2] + planck_data[:, 3]) / 2.0
mask_fit = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)
npts = mask_fit.sum()

def compute_chi2(omch2, aM, As=2.05e-9, ns=0.97):
    aB = -aM / 2.0
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': 0.673,
        'T_cmb': 2.7255,
        'omega_b': 0.02237,
        'omega_cdm': omch2,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'tau_reio': 0.0544,
        'A_s': As,
        'n_s': ns,
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'constant_alphas',
        'parameters_smg': f'0.0, {aB}, {aM}, 0.0, 1.0',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }
    cosmo.set(params)
    try:
        cosmo.compute()
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return 1e10, float('nan'), float('nan')

    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6
        Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
        try: s8 = cosmo.sigma8()
        except: s8 = float('nan')
        try: th = cosmo.theta_s_100() / 100.0
        except: th = float('nan')
        cosmo.struct_cleanup()
        cosmo.empty()
        model_Dl = np.interp(planck_ell, ell, Dl)
        res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2 = float(np.sum(res**2))
        return chi2, s8, th
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return 1e10, float('nan'), float('nan')

# ============================================================
# 2D Grid: omega_cdm x alpha_M
# ============================================================
omch2_grid = np.array([0.1120, 0.1125, 0.1130, 0.1135, 0.1138, 0.1140,
                        0.1143, 0.1145, 0.1150, 0.1155])
aM_grid = np.array([0.0000, 0.0002, 0.0004, 0.0005, 0.0006, 0.0008,
                     0.0010, 0.0012, 0.0015, 0.0020])

print("=" * 100)
print("2D GRID POSTERIOR: omega_cdm x alpha_M (constant_alphas)")
print(f"Grid: {len(omch2_grid)} x {len(aM_grid)} = {len(omch2_grid)*len(aM_grid)} points")
print(f"Fixed: A_s=2.05e-9, n_s=0.97, omega_b=0.02237, h=0.673")
print("=" * 100)
sys.stdout.flush()

# Header
print(f"{'omch2':>8s} {'alpha_M':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s} | {'dchi2':>8s}")
print("-" * 80)
sys.stdout.flush()

results = []
chi2_min = 1e10

for omch2 in omch2_grid:
    for aM in aM_grid:
        chi2, s8, th = compute_chi2(omch2, aM)
        chi2n = chi2 / npts if chi2 < 1e9 else 999
        if chi2 < chi2_min:
            chi2_min = chi2
        results.append((omch2, aM, chi2, s8, th))
        marker = ""
        if chi2n < 1.1:
            marker = " ***"
        elif chi2n < 1.15:
            marker = " **"
        elif chi2n < 1.2:
            marker = " *"
        print(f"{omch2:>8.4f} {aM:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f} | {chi2-chi2_min:>+8.1f}{marker}")
        sys.stdout.flush()

# Summary
print(f"\n{'='*100}")
print(f"MINIMUM chi2 = {chi2_min:.1f} (chi2/n = {chi2_min/npts:.3f})")
print(f"{'='*100}")

# Find best point
best = min(results, key=lambda x: x[2])
print(f"Best: omega_cdm={best[0]:.4f}, alpha_M={best[1]:.5f}, sigma8={best[3]:.4f}, theta_s={best[4]:.6f}")

# Delta chi2 contours
print(f"\nDelta_chi2 contours (1sigma=1, 2sigma=4, 3sigma=9 for 1 param):")
print(f"{'omch2':>8s} {'alpha_M':>8s} {'dchi2':>8s} {'level':>8s}")
print("-" * 40)
for omch2, aM, chi2, s8, th in sorted(results, key=lambda x: x[2]):
    dchi2 = chi2 - chi2_min
    if dchi2 < 10:
        if dchi2 < 1:
            level = "< 1sig"
        elif dchi2 < 4:
            level = "1-2sig"
        elif dchi2 < 9:
            level = "2-3sig"
        else:
            level = "> 3sig"
        print(f"{omch2:>8.4f} {aM:>8.5f} {dchi2:>8.1f} {level:>8s}")

# Save for plotting
np.savez('/tmp/cfm_grid_results.npz',
         omch2_grid=omch2_grid, aM_grid=aM_grid,
         results=np.array([(r[0], r[1], r[2], r[3], r[4]) for r in results]))
print("\nResults saved to /tmp/cfm_grid_results.npz")

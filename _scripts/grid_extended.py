#!/usr/bin/env python3
"""
Extended grid scan: higher omega_cdm range to find true minimum.
Previous grid stopped at 0.1155 where chi2 was still decreasing.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

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

# Extended grid: focus on higher omega_cdm
omch2_grid = np.array([0.1145, 0.1150, 0.1155, 0.1160, 0.1165, 0.1170,
                        0.1175, 0.1180, 0.1190, 0.1200])
aM_grid = np.array([0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0015])

print("=" * 100)
print("EXTENDED GRID: omega_cdm up to LCDM value (0.1200)")
print("=" * 100)
print(f"{'omch2':>8s} {'alpha_M':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s} | {'dl1 est':>8s}")
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
        marker = " ***" if chi2n < 1.07 else (" **" if chi2n < 1.08 else (" *" if chi2n < 1.1 else ""))
        print(f"{omch2:>8.4f} {aM:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f} | {chi2-chi2_min:>+8.1f}{marker}")
        sys.stdout.flush()

print(f"\nMINIMUM chi2 = {chi2_min:.1f} (chi2/n = {chi2_min/npts:.3f})")
best = min(results, key=lambda x: x[2])
print(f"Best: omega_cdm={best[0]:.4f}, alpha_M={best[1]:.5f}")
print(f"      sigma8={best[3]:.4f}, theta_s={best[4]:.6f}")

# Profile likelihood
print(f"\nProfile omega_cdm:")
for omch2 in omch2_grid:
    pts = [(r[2], r[1]) for r in results if r[0] == omch2 and r[2] < 1e9]
    if pts:
        best_chi2, best_aM = min(pts)
        dchi2 = best_chi2 - chi2_min
        print(f"  omch2={omch2:.4f}: min_chi2/n={best_chi2/npts:.4f} (dchi2={dchi2:+.1f}), best_aM={best_aM:.5f}")

print(f"\nProfile alpha_M:")
for aM in aM_grid:
    pts = [(r[2], r[0]) for r in results if r[1] == aM and r[2] < 1e9]
    if pts:
        best_chi2, best_omch2 = min(pts)
        dchi2 = best_chi2 - chi2_min
        print(f"  aM={aM:.5f}: min_chi2/n={best_chi2/npts:.4f} (dchi2={dchi2:+.1f}), best_omch2={best_omch2:.4f}")

print(f"\nPlanck 2018: sigma8=0.811, 100*theta_s=1.0411, omega_cdm=0.1200")

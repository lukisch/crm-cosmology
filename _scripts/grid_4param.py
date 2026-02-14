#!/usr/bin/env python3
"""
4-parameter grid scan: h, omega_b, omega_cdm, alpha_M
Strategy:
  Phase 1: Scan h to match theta_s = 0.010411 (Planck)
  Phase 2: 2D grid (omega_cdm x alpha_M) at optimal h
  Phase 3: omega_b fine-tuning at best point
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

def compute_chi2(omch2, aM, h=0.673, omb=0.02237, As=2.05e-9, ns=0.97):
    aB = -aM / 2.0
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': h,
        'T_cmb': 2.7255,
        'omega_b': omb,
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
        return 1e10, float('nan'), float('nan'), float('nan')
    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6
        Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
        try: s8 = cosmo.sigma8()
        except: s8 = float('nan')
        try: th = cosmo.theta_s_100() / 100.0
        except: th = float('nan')
        try: H0 = cosmo.Hubble(0) * 299792.458  # km/s/Mpc
        except: H0 = float('nan')
        cosmo.struct_cleanup()
        cosmo.empty()
        model_Dl = np.interp(planck_ell, ell, Dl)
        res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2 = float(np.sum(res**2))
        return chi2, s8, th, H0
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return 1e10, float('nan'), float('nan'), float('nan')

# ================================================================
# PHASE 1: h-scan to find theta_s match
# ================================================================
print("=" * 110)
print("PHASE 1: h-SCAN (theta_s matching)")
print("Fixed: omch2=0.1165, aM=0.0008, omega_b=0.02237, A_s=2.05e-9, n_s=0.97")
print("Target: theta_s = 0.010411 (Planck 2018)")
print("=" * 110)
print(f"{'h':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s} {'H0':>8s} | {'dtheta%':>8s}")
print("-" * 85)
sys.stdout.flush()

h_grid = [0.660, 0.665, 0.670, 0.673, 0.676, 0.680, 0.685, 0.690, 0.695, 0.700]
h_results = []

for h in h_grid:
    chi2, s8, th, H0 = compute_chi2(0.1165, 0.0008, h=h)
    chi2n = chi2 / npts if chi2 < 1e9 else 999
    dth_pct = (th - 0.010411) / 0.010411 * 100 if not np.isnan(th) else float('nan')
    h_results.append((h, chi2, s8, th, H0, dth_pct))
    print(f"{h:>8.3f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f} {H0:>8.2f} | {dth_pct:>+8.3f}%")
    sys.stdout.flush()

# Find h that minimizes |dtheta|
valid_h = [(abs(r[5]), r[0], r[1], r[2], r[3], r[4]) for r in h_results if not np.isnan(r[5])]
best_h_entry = min(valid_h, key=lambda x: x[0])
best_h = best_h_entry[1]

# Interpolate for exact theta_s match
h_vals = np.array([r[0] for r in h_results if not np.isnan(r[3])])
th_vals = np.array([r[3] for r in h_results if not np.isnan(r[3])])
# Linear interpolation to find h where theta_s = 0.010411
try:
    h_interp = np.interp(0.010411, th_vals[::-1], h_vals[::-1])  # theta_s decreases with h
    print(f"\nInterpolated h for theta_s=0.010411: h = {h_interp:.4f}")
except:
    h_interp = best_h
    print(f"\nInterpolation failed, using best grid h = {best_h:.4f}")

# Use the interpolated h (rounded to reasonable precision)
h_opt = round(h_interp, 4)
print(f"Using h_opt = {h_opt:.4f} for Phase 2")
sys.stdout.flush()

# ================================================================
# PHASE 2: 2D grid (omega_cdm x alpha_M) at optimal h
# ================================================================
print(f"\n{'=' * 110}")
print(f"PHASE 2: 2D GRID at h={h_opt:.4f}")
print(f"{'=' * 110}")
print(f"{'omch2':>8s} {'alpha_M':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s} {'H0':>8s} | {'dchi2':>8s}")
print("-" * 90)
sys.stdout.flush()

omch2_grid = np.array([0.1120, 0.1130, 0.1140, 0.1150, 0.1160, 0.1170, 0.1180, 0.1200])
aM_grid = np.array([0.0004, 0.0006, 0.0008, 0.0010, 0.0012, 0.0015])

results_p2 = []
chi2_min = 1e10

for omch2 in omch2_grid:
    for aM in aM_grid:
        chi2, s8, th, H0 = compute_chi2(omch2, aM, h=h_opt)
        chi2n = chi2 / npts if chi2 < 1e9 else 999
        if chi2 < chi2_min:
            chi2_min = chi2
        results_p2.append((omch2, aM, chi2, s8, th, H0))
        marker = " ***" if chi2n < 1.05 else (" **" if chi2n < 1.06 else (" *" if chi2n < 1.07 else ""))
        print(f"{omch2:>8.4f} {aM:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f} {H0:>8.2f} | {chi2-chi2_min:>+8.1f}{marker}")
        sys.stdout.flush()

best_p2 = min(results_p2, key=lambda x: x[2])
print(f"\nPHASE 2 BEST: omch2={best_p2[0]:.4f}, aM={best_p2[1]:.5f}")
print(f"  chi2={best_p2[2]:.1f} (chi2/n={best_p2[2]/npts:.4f})")
print(f"  sigma8={best_p2[3]:.4f}, theta_s={best_p2[4]:.6f}, H0={best_p2[5]:.2f}")
sys.stdout.flush()

# ================================================================
# PHASE 3: omega_b fine-tuning at best point
# ================================================================
best_omch2 = best_p2[0]
best_aM = best_p2[1]

print(f"\n{'=' * 110}")
print(f"PHASE 3: omega_b SCAN at h={h_opt:.4f}, omch2={best_omch2:.4f}, aM={best_aM:.5f}")
print(f"{'=' * 110}")
print(f"{'omega_b':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s} {'H0':>8s} | {'dchi2':>8s}")
print("-" * 85)
sys.stdout.flush()

omb_grid = [0.02180, 0.02200, 0.02220, 0.02237, 0.02250, 0.02260, 0.02280, 0.02300]
results_p3 = []

for omb in omb_grid:
    chi2, s8, th, H0 = compute_chi2(best_omch2, best_aM, h=h_opt, omb=omb)
    chi2n = chi2 / npts if chi2 < 1e9 else 999
    results_p3.append((omb, chi2, s8, th, H0))
    marker = " ***" if chi2n < 1.05 else (" **" if chi2n < 1.06 else (" *" if chi2n < 1.07 else ""))
    print(f"{omb:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f} {H0:>8.2f} | {chi2-chi2_min:>+8.1f}{marker}")
    sys.stdout.flush()

best_p3 = min(results_p3, key=lambda x: x[1])
print(f"\nPHASE 3 BEST: omega_b={best_p3[0]:.5f}")
print(f"  chi2={best_p3[1]:.1f} (chi2/n={best_p3[1]/npts:.4f})")
print(f"  sigma8={best_p3[2]:.4f}, theta_s={best_p3[3]:.6f}, H0={best_p3[4]:.2f}")

# ================================================================
# PHASE 4: Final refinement - small grid around best point
# ================================================================
best_omb = best_p3[0]

print(f"\n{'=' * 110}")
print(f"PHASE 4: FINAL REFINEMENT around optimum")
print(f"h={h_opt:.4f}, omega_b={best_omb:.5f}")
print(f"{'=' * 110}")
print(f"{'omch2':>8s} {'alpha_M':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s} {'H0':>8s} | {'dchi2':>8s}")
print("-" * 90)
sys.stdout.flush()

# Fine grid around best
omch2_fine = np.arange(best_omch2 - 0.0015, best_omch2 + 0.0020, 0.0005)
aM_fine = np.array([best_aM - 0.0002, best_aM, best_aM + 0.0002, best_aM + 0.0004])

results_p4 = []
chi2_min_final = 1e10

for omch2 in omch2_fine:
    for aM in aM_fine:
        if aM < 0.0002:
            continue  # avoid crash near 0
        chi2, s8, th, H0 = compute_chi2(omch2, aM, h=h_opt, omb=best_omb)
        chi2n = chi2 / npts if chi2 < 1e9 else 999
        if chi2 < chi2_min_final:
            chi2_min_final = chi2
        results_p4.append((omch2, aM, chi2, s8, th, H0))
        marker = " ***" if chi2n < 1.04 else (" **" if chi2n < 1.05 else (" *" if chi2n < 1.06 else ""))
        print(f"{omch2:>8.4f} {aM:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f} {H0:>8.2f} | {chi2-chi2_min_final:>+8.1f}{marker}")
        sys.stdout.flush()

# ================================================================
# FINAL SUMMARY
# ================================================================
all_results = results_p2 + [(best_omch2, best_aM, r[1], r[2], r[3], r[4]) for r in results_p3]
for r in results_p4:
    all_results.append(r)

overall_best = min(all_results, key=lambda x: x[2])

print(f"\n{'=' * 110}")
print(f"FINAL SUMMARY")
print(f"{'=' * 110}")
print(f"h = {h_opt:.4f}")
print(f"omega_b = {best_omb:.5f}")
print(f"omega_cdm = {overall_best[0]:.4f}")
print(f"alpha_M = {overall_best[1]:.5f}")
print(f"chi2 = {overall_best[2]:.1f} (chi2/n = {overall_best[2]/npts:.4f})")
print(f"sigma8 = {overall_best[3]:.4f}")
print(f"theta_s = {overall_best[4]:.6f} (Planck: 0.010411)")
print(f"H0 = {overall_best[5]:.2f} km/s/Mpc")
print(f"dtheta_s = {(overall_best[4] - 0.010411)/0.010411*100:+.3f}%")
print(f"dsigma8 = {(overall_best[3] - 0.811)/0.811*100:+.1f}%")
print(f"\nPlanck 2018 reference: sigma8=0.811, theta_s=0.010411, H0=67.3, omega_cdm=0.1200")

# Also run LCDM at the same h for fair comparison
print(f"\n--- LCDM at h={h_opt:.4f} for comparison ---")
sys.stdout.flush()
# LCDM: no SMG
cosmo = Class()
lcdm_params = {
    'output': 'tCl,pCl,lCl,mPk',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'h': h_opt,
    'T_cmb': 2.7255,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'tau_reio': 0.0544,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
}
cosmo.set(lcdm_params)
cosmo.compute()
cls_lcdm = cosmo.lensed_cl(2500)
ell_lcdm = np.arange(2, 2501)
T = 2.7255e6
Dl_lcdm = cls_lcdm['tt'][2:2501] * ell_lcdm * (ell_lcdm + 1) / (2 * np.pi) * T**2
s8_lcdm = cosmo.sigma8()
th_lcdm = cosmo.theta_s_100() / 100.0
H0_lcdm = cosmo.Hubble(0) * 299792.458
model_Dl_lcdm = np.interp(planck_ell, ell_lcdm, Dl_lcdm)
res_lcdm = (model_Dl_lcdm[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
chi2_lcdm = float(np.sum(res_lcdm**2))
cosmo.struct_cleanup()
cosmo.empty()
print(f"LCDM: chi2={chi2_lcdm:.1f} (chi2/n={chi2_lcdm/npts:.4f})")
print(f"  sigma8={s8_lcdm:.4f}, theta_s={th_lcdm:.6f}, H0={H0_lcdm:.2f}")
print(f"\nCFM advantage over LCDM: dchi2 = {overall_best[2] - chi2_lcdm:+.1f}")

#!/usr/bin/env python3
"""
3D joint grid scan: h x omega_cdm x alpha_M
Phase 1: Coarse 3D grid (3 x 7 x 5 = 105 points)
Phase 2: Fine refinement around minimum (20-30 points)
Also: A_s/n_s optimization at best point
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np
import time

# Load Planck data
planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]
planck_err = (planck_data[:, 2] + planck_data[:, 3]) / 2.0
mask_fit = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)
npts = mask_fit.sum()
print(f"Planck TT data: {len(planck_ell)} points, {npts} in fit range")

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

def compute_lcdm(h=0.6732, omb=0.02237, omch2=0.1200, As=2.1e-9, ns=0.9649):
    """Standard LCDM without SMG for comparison."""
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
    }
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    T = 2.7255e6
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
    s8 = cosmo.sigma8()
    th = cosmo.theta_s_100() / 100.0
    cosmo.struct_cleanup()
    cosmo.empty()
    model_Dl = np.interp(planck_ell, ell, Dl)
    res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
    chi2 = float(np.sum(res**2))
    return chi2, s8, th

# ================================================================
# LCDM REFERENCE
# ================================================================
print("=" * 110)
print("LCDM REFERENCE (Planck 2018 best-fit)")
chi2_lcdm, s8_lcdm, th_lcdm = compute_lcdm()
print(f"chi2={chi2_lcdm:.1f} (chi2/n={chi2_lcdm/npts:.4f}), sigma8={s8_lcdm:.4f}, theta_s={th_lcdm:.6f}")
print("=" * 110)
sys.stdout.flush()

# ================================================================
# PHASE 1: 3D COARSE GRID
# ================================================================
h_grid = np.array([0.674, 0.678, 0.682, 0.686])
omch2_grid = np.array([0.1130, 0.1145, 0.1160, 0.1175, 0.1190, 0.1200])
aM_grid = np.array([0.0004, 0.0006, 0.0008, 0.0010, 0.0012])

total_pts = len(h_grid) * len(omch2_grid) * len(aM_grid)
print(f"\nPHASE 1: 3D COARSE GRID ({len(h_grid)} x {len(omch2_grid)} x {len(aM_grid)} = {total_pts} points)")
print(f"{'h':>7s} {'omch2':>8s} {'aM':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'s8':>7s} {'theta':>10s} | {'dt%':>7s} {'ds8%':>7s}")
print("-" * 95)
sys.stdout.flush()

results = []
chi2_min = 1e10
t0 = time.time()

for ih, h in enumerate(h_grid):
    for iom, omch2 in enumerate(omch2_grid):
        for iam, aM in enumerate(aM_grid):
            idx = ih * len(omch2_grid) * len(aM_grid) + iom * len(aM_grid) + iam + 1
            chi2, s8, th = compute_chi2(omch2, aM, h=h)
            chi2n = chi2 / npts if chi2 < 1e9 else 999
            if chi2 < chi2_min:
                chi2_min = chi2
            dth = (th - 0.010411) / 0.010411 * 100 if not np.isnan(th) else float('nan')
            ds8 = (s8 - 0.811) / 0.811 * 100 if not np.isnan(s8) else float('nan')
            results.append((h, omch2, aM, chi2, s8, th))
            marker = " ***" if chi2n < 1.05 else (" **" if chi2n < 1.06 else (" *" if chi2n < 1.08 else ""))
            print(f"{h:>7.3f} {omch2:>8.4f} {aM:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>7.4f} {th:>10.6f} | {dth:>+7.2f} {ds8:>+7.1f}{marker}")
            if idx % 10 == 0:
                elapsed = time.time() - t0
                rate = elapsed / idx
                remaining = rate * (total_pts - idx)
                print(f"  [{idx}/{total_pts}] elapsed={elapsed:.0f}s, remaining~{remaining:.0f}s")
            sys.stdout.flush()

# Best point from coarse grid
best = min(results, key=lambda x: x[2])
print(f"\n{'='*110}")
print(f"PHASE 1 BEST: h={best[0]:.3f}, omch2={best[1]:.4f}, aM={best[2]:.5f}")
print(f"  chi2={best[3]:.1f} (chi2/n={best[3]/npts:.4f})")
print(f"  sigma8={best[4]:.4f}, theta_s={best[5]:.6f}")
dth = (best[5] - 0.010411) / 0.010411 * 100
print(f"  dtheta_s = {dth:+.3f}%, dsigma8 = {(best[4]-0.811)/0.811*100:+.1f}%")
sys.stdout.flush()

# Profile: best at each h
print(f"\nProfile h (best chi2 at each h):")
for h in h_grid:
    pts = [(r[3], r[1], r[2], r[4], r[5]) for r in results if r[0] == h and r[3] < 1e9]
    if pts:
        best_chi2, best_omch2, best_aM, best_s8, best_th = min(pts)
        dth_pct = (best_th - 0.010411) / 0.010411 * 100
        print(f"  h={h:.3f}: chi2/n={best_chi2/npts:.4f}, omch2={best_omch2:.4f}, aM={best_aM:.5f}, "
              f"s8={best_s8:.4f}, theta={best_th:.6f} ({dth_pct:+.2f}%)")

# ================================================================
# PHASE 2: REFINEMENT around best point
# ================================================================
h_best = best[0]
omch2_best = best[1]
aM_best = best[2]

# Fine grid: h steps of 0.001, omch2 steps of 0.0005, aM steps of 0.0001
h_fine = np.arange(h_best - 0.003, h_best + 0.004, 0.001)
omch2_fine = np.arange(omch2_best - 0.0010, omch2_best + 0.0015, 0.0005)
aM_fine = np.array([aM_best - 0.0002, aM_best - 0.0001, aM_best, aM_best + 0.0001, aM_best + 0.0002])
aM_fine = aM_fine[aM_fine >= 0.0002]  # avoid crash near 0

n_fine = len(h_fine) * len(omch2_fine) * len(aM_fine)
print(f"\nPHASE 2: FINE GRID ({len(h_fine)} x {len(omch2_fine)} x {len(aM_fine)} = {n_fine} points)")
print(f"{'h':>7s} {'omch2':>8s} {'aM':>8s} | {'chi2':>10s} {'chi2/n':>8s} {'s8':>7s} {'theta':>10s} | {'dt%':>7s} {'ds8%':>7s}")
print("-" * 95)
sys.stdout.flush()

results_fine = []
chi2_min_fine = 1e10

for h in h_fine:
    for omch2 in omch2_fine:
        for aM in aM_fine:
            chi2, s8, th = compute_chi2(omch2, aM, h=h)
            chi2n = chi2 / npts if chi2 < 1e9 else 999
            if chi2 < chi2_min_fine:
                chi2_min_fine = chi2
            dth = (th - 0.010411) / 0.010411 * 100 if not np.isnan(th) else float('nan')
            ds8 = (s8 - 0.811) / 0.811 * 100 if not np.isnan(s8) else float('nan')
            results_fine.append((h, omch2, aM, chi2, s8, th))
            marker = " ***" if chi2n < 1.04 else (" **" if chi2n < 1.05 else (" *" if chi2n < 1.06 else ""))
            print(f"{h:>7.3f} {omch2:>8.4f} {aM:>8.5f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>7.4f} {th:>10.6f} | {dth:>+7.2f} {ds8:>+7.1f}{marker}")
            sys.stdout.flush()

best_fine = min(results_fine, key=lambda x: x[2])
print(f"\nPHASE 2 BEST: h={best_fine[0]:.4f}, omch2={best_fine[1]:.4f}, aM={best_fine[2]:.5f}")
print(f"  chi2={best_fine[3]:.1f} (chi2/n={best_fine[3]/npts:.4f})")
print(f"  sigma8={best_fine[4]:.4f}, theta_s={best_fine[5]:.6f}")

# ================================================================
# PHASE 3: A_s / n_s optimization at best point
# ================================================================
h_opt = best_fine[0]
omch2_opt = best_fine[1]
aM_opt = best_fine[2]

print(f"\n{'='*110}")
print(f"PHASE 3: A_s/n_s OPTIMIZATION at h={h_opt:.4f}, omch2={omch2_opt:.4f}, aM={aM_opt:.5f}")
print(f"{'='*110}")
print(f"{'As':>10s} {'ns':>7s} | {'chi2':>10s} {'chi2/n':>8s} {'s8':>7s} {'theta':>10s}")
print("-" * 70)
sys.stdout.flush()

As_grid = [1.95e-9, 2.00e-9, 2.05e-9, 2.10e-9, 2.15e-9, 2.20e-9]
ns_grid = [0.960, 0.965, 0.970, 0.975, 0.980]

results_As = []
for As in As_grid:
    for ns in ns_grid:
        chi2, s8, th = compute_chi2(omch2_opt, aM_opt, h=h_opt, As=As, ns=ns)
        chi2n = chi2 / npts if chi2 < 1e9 else 999
        results_As.append((As, ns, chi2, s8, th))
        marker = " ***" if chi2n < 1.04 else (" **" if chi2n < 1.05 else "")
        print(f"{As:>10.3e} {ns:>7.3f} | {chi2:>10.1f} {chi2n:>8.3f} {s8:>7.4f} {th:>10.6f}{marker}")
        sys.stdout.flush()

best_As = min(results_As, key=lambda x: x[2])
print(f"\nBest A_s={best_As[0]:.3e}, n_s={best_As[1]:.3f}")
print(f"  chi2={best_As[2]:.1f} (chi2/n={best_As[2]/npts:.4f})")
print(f"  sigma8={best_As[3]:.4f}, theta_s={best_As[4]:.6f}")

# ================================================================
# FINAL SUMMARY
# ================================================================
# Run once more at absolute best with optimized A_s/n_s
As_opt = best_As[0]
ns_opt = best_As[1]
chi2_final, s8_final, th_final = compute_chi2(omch2_opt, aM_opt, h=h_opt, As=As_opt, ns=ns_opt)

print(f"\n{'='*110}")
print(f"FINAL OPTIMAL CFM (constant_alphas)")
print(f"{'='*110}")
print(f"h           = {h_opt:.4f}")
print(f"omega_b     = 0.02237 (Planck prior)")
print(f"omega_cdm   = {omch2_opt:.4f}")
print(f"alpha_M     = {aM_opt:.5f}")
print(f"A_s         = {As_opt:.3e}")
print(f"n_s         = {ns_opt:.3f}")
print(f"tau_reio    = 0.0544 (Planck prior)")
print(f"")
print(f"chi2        = {chi2_final:.1f} (chi2/n = {chi2_final/npts:.4f})")
print(f"sigma8      = {s8_final:.4f}")
print(f"theta_s     = {th_final:.6f}")
print(f"dtheta_s    = {(th_final - 0.010411)/0.010411*100:+.3f}%")
print(f"dsigma8     = {(s8_final - 0.811)/0.811*100:+.1f}%")
print(f"")
print(f"LCDM reference: chi2={chi2_lcdm:.1f} (chi2/n={chi2_lcdm/npts:.4f}), s8={s8_lcdm:.4f}, th={th_lcdm:.6f}")
print(f"Delta chi2 (CFM - LCDM) = {chi2_final - chi2_lcdm:+.1f}")

# Summarize: which h best matches theta_s?
print(f"\n--- theta_s diagnostic ---")
all_res = results + results_fine
for h in sorted(set(r[0] for r in all_res)):
    pts = [(abs((r[5]-0.010411)/0.010411*100), r[3], r[1], r[2], r[4], r[5]) for r in all_res
           if r[0] == h and r[3] < 1e9 and not np.isnan(r[5])]
    if pts:
        # Best theta_s match at this h
        best_th_match = min(pts, key=lambda x: x[0])
        # Best chi2 at this h
        best_chi2_match = min(pts, key=lambda x: x[1])
        print(f"  h={h:.4f}: best_theta_match={best_th_match[5]:.6f} ({best_th_match[0]:+.2f}%, chi2/n={best_th_match[1]/npts:.3f}), "
              f"best_chi2={best_chi2_match[1]/npts:.3f} (theta={best_chi2_match[5]:.6f})")

# Effective parameters
Omega_m = (omch2_opt + 0.02237) / h_opt**2
print(f"\n--- Derived parameters ---")
print(f"Omega_m     = {Omega_m:.4f} (Planck: 0.315)")
print(f"Omega_cdm   = {omch2_opt/h_opt**2:.4f} (Planck: 0.265)")
print(f"H0          = {h_opt*100:.1f} km/s/Mpc (Planck: 67.3)")
print(f"alpha_B     = {-aM_opt/2:.5f}")

#!/usr/bin/env python3
"""
Fine grid scan for propto_omega with CFM f(R) relation.
Scan (cM, omega_cdm) at Planck-optimal A_s=2.1e-9, n_s=0.9649.
Also test h variation for theta_s matching.
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

def compute(grav, cM, h=0.6732, omch2=0.1200, As=2.1e-9, ns=0.9649):
    cB = -cM / 2.0
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
        'h': h, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': omch2,
        'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
        'tau_reio': 0.0544, 'A_s': As, 'n_s': ns,
        'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
        'gravity_model': grav,
        'parameters_smg': f'0.0, {cB}, {cM}, 0.0, 1.0',
        'expansion_model': 'lcdm', 'expansion_smg': '0.5',
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
        return None
    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6
        Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
        try: s8 = cosmo.sigma8()
        except: s8 = float('nan')
        try: th = cosmo.theta_s_100() / 100.0
        except: th = float('nan')
        cosmo.struct_cleanup(); cosmo.empty()
        model_Dl = np.interp(planck_ell, ell, Dl)
        res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2 = float(np.sum(res**2))
        return chi2, s8, th
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# LCDM ref
cosmo = Class()
cosmo.set({
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
    'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
})
cosmo.compute()
cls_ref = cosmo.lensed_cl(2500)
ell = np.arange(2, 2501)
T = 2.7255e6
Dl_ref = cls_ref['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
s8_ref = cosmo.sigma8()
cosmo.struct_cleanup(); cosmo.empty()
res_ref = (np.interp(planck_ell, ell, Dl_ref)[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
chi2_ref = float(np.sum(res_ref**2))
print(f"LCDM: chi2={chi2_ref:.1f}, s8={s8_ref:.4f}")

# ================================================================
# GRID 1: propto_omega, (cM, omega_cdm) at h=0.6732
# ================================================================
print(f"\n{'='*100}")
print("GRID 1: propto_omega CFM, h=0.6732, A_s=2.1e-9, n_s=0.9649")
print(f"{'cM':>8s} {'omch2':>8s} | {'chi2':>8s} {'dchi2':>8s} {'s8':>7s} {'ds8%':>7s} {'theta':>10s}")
print("-" * 75)
sys.stdout.flush()

cM_grid = [0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002]
omch2_grid = [0.1150, 0.1170, 0.1190, 0.1200, 0.1210, 0.1220]

best = None
for cM in cM_grid:
    for omch2 in omch2_grid:
        r = compute('propto_omega', cM, omch2=omch2)
        if r is None:
            print(f"{cM:>8.4f} {omch2:>8.4f} | {'FAIL':>8s}")
            sys.stdout.flush()
            continue
        chi2, s8, th = r
        dchi2 = chi2 - chi2_ref
        ds8 = (s8 - 0.811) / 0.811 * 100
        m = " ***" if dchi2 < 0 and abs(ds8) < 5 else (" **" if dchi2 < 0 and abs(ds8) < 10 else (" *" if dchi2 < 0 else ""))
        print(f"{cM:>8.4f} {omch2:>8.4f} | {chi2:>8.1f} {dchi2:>+8.1f} {s8:>7.4f} {ds8:>+7.1f} {th:>10.6f}{m}")
        sys.stdout.flush()
        if best is None or chi2 < best[2]:
            best = (cM, omch2, chi2, s8, th)

if best:
    print(f"\nBest: cM={best[0]:.4f}, omch2={best[1]:.4f}, chi2={best[2]:.1f} ({best[2]-chi2_ref:+.1f}), s8={best[3]:.4f}")

# ================================================================
# GRID 2: propto_scale, (cM, omega_cdm) at h=0.6732
# ================================================================
print(f"\n{'='*100}")
print("GRID 2: propto_scale CFM, h=0.6732")
print(f"{'cM':>8s} {'omch2':>8s} | {'chi2':>8s} {'dchi2':>8s} {'s8':>7s} {'ds8%':>7s} {'theta':>10s}")
print("-" * 75)
sys.stdout.flush()

cM_grid_ps = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
best_ps = None
for cM in cM_grid_ps:
    for omch2 in omch2_grid:
        r = compute('propto_scale', cM, omch2=omch2)
        if r is None:
            print(f"{cM:>8.4f} {omch2:>8.4f} | {'FAIL':>8s}")
            sys.stdout.flush()
            continue
        chi2, s8, th = r
        dchi2 = chi2 - chi2_ref
        ds8 = (s8 - 0.811) / 0.811 * 100
        m = " ***" if dchi2 < 0 and abs(ds8) < 5 else (" **" if dchi2 < 0 and abs(ds8) < 10 else (" *" if dchi2 < 0 else ""))
        print(f"{cM:>8.4f} {omch2:>8.4f} | {chi2:>8.1f} {dchi2:>+8.1f} {s8:>7.4f} {ds8:>+7.1f} {th:>10.6f}{m}")
        sys.stdout.flush()
        if best_ps is None or chi2 < best_ps[2]:
            best_ps = (cM, omch2, chi2, s8, th)

if best_ps:
    print(f"\nBest: cM={best_ps[0]:.4f}, omch2={best_ps[1]:.4f}, chi2={best_ps[2]:.1f} ({best_ps[2]-chi2_ref:+.1f}), s8={best_ps[3]:.4f}")

# ================================================================
# GRID 3: Fine propto_omega grid around best point
# ================================================================
if best:
    cM_b, omch2_b = best[0], best[1]
    print(f"\n{'='*100}")
    print(f"GRID 3: FINE propto_omega around cM={cM_b:.4f}, omch2={omch2_b:.4f}")
    print(f"{'cM':>8s} {'omch2':>8s} | {'chi2':>8s} {'dchi2':>8s} {'s8':>7s} {'ds8%':>7s}")
    print("-" * 65)
    sys.stdout.flush()

    cM_fine = np.arange(max(0.0001, cM_b - 0.0003), cM_b + 0.0004, 0.0001)
    omch2_fine = np.arange(omch2_b - 0.0010, omch2_b + 0.0015, 0.0005)

    best_fine = None
    for cM in cM_fine:
        for omch2 in omch2_fine:
            r = compute('propto_omega', cM, omch2=omch2)
            if r is None:
                continue
            chi2, s8, th = r
            dchi2 = chi2 - chi2_ref
            ds8 = (s8 - 0.811) / 0.811 * 100
            m = " ***" if dchi2 < 0 and abs(ds8) < 5 else (" **" if dchi2 < 0 and abs(ds8) < 10 else (" *" if dchi2 < 0 else ""))
            print(f"{cM:>8.4f} {omch2:>8.4f} | {chi2:>8.1f} {dchi2:>+8.1f} {s8:>7.4f} {ds8:>+7.1f}{m}")
            sys.stdout.flush()
            if best_fine is None or chi2 < best_fine[2]:
                best_fine = (cM, omch2, chi2, s8, th)

    if best_fine:
        ds8 = (best_fine[3]-0.811)/0.811*100
        print(f"\nFINE BEST: cM={best_fine[0]:.4f}, omch2={best_fine[1]:.4f}")
        print(f"  chi2={best_fine[2]:.1f} ({best_fine[2]-chi2_ref:+.1f}), s8={best_fine[3]:.4f} ({ds8:+.1f}%)")

# ================================================================
# SUMMARY TABLE
# ================================================================
print(f"\n{'='*100}")
print("FINAL SUMMARY: Best points per sigma8 category")
print(f"{'='*100}")
print(f"LCDM:           chi2={chi2_ref:.1f}, sigma8={s8_ref:.4f}")
if best:
    print(f"propto_omega:   cM={best[0]:.4f}, omch2={best[1]:.4f}, chi2={best[2]:.1f} ({best[2]-chi2_ref:+.1f}), s8={best[3]:.4f}")
if best_ps:
    print(f"propto_scale:   cM={best_ps[0]:.4f}, omch2={best_ps[1]:.4f}, chi2={best_ps[2]:.1f} ({best_ps[2]-chi2_ref:+.1f}), s8={best_ps[3]:.4f}")

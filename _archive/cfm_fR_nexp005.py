#!/usr/bin/env python3
"""
Deep dive: cfm_fR with n_exp=0.05 (nearly constant alpha_M).
This profile showed sigma8=0.97 which is the best so far.
Now optimize aM0 and omch2 to get l1~220, r31~0.4295, sigma8~0.81.
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

def find_peak(ell, Dl, lmin, lmax):
    mask = (ell >= lmin) & (ell <= lmax)
    e = ell[mask]; d = Dl[mask]
    if len(d) == 0 or np.any(np.isnan(d)):
        return float('nan'), float('nan')
    idx = np.argmax(d)
    if idx == 0 or idx == len(d)-1:
        return float(e[idx]), float(d[idx])
    x = e[idx-1:idx+2].astype(float); y = d[idx-1:idx+2]
    a_c = (y[2] - 2*y[1] + y[0]) / 2.0
    b_c = (y[2] - y[0]) / 2.0
    if abs(a_c) < 1e-30:
        return float(x[1]), float(y[1])
    return float(x[1] - b_c/(2*a_c)), float(y[1] - b_c*b_c/(4*a_c))

def run_model(omch2, aM0, n_exp, gravity_model='cfm_fR', params_smg=None, As=2.05e-9, ns=0.97):
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
        'gravity_model': gravity_model,
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }
    if params_smg:
        params['parameters_smg'] = params_smg
    else:
        params['parameters_smg'] = f'{aM0}, {n_exp}, 1.0'
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
        if np.any(np.isnan(Dl)) or np.any(np.isinf(Dl)):
            cosmo.struct_cleanup(); cosmo.empty()
            return None
        l1, D1 = find_peak(ell, Dl, 150, 300)
        l3, D3 = find_peak(ell, Dl, 700, 900)
        r31 = D3/D1 if D1 > 0 else float('nan')
        try: s8 = cosmo.sigma8()
        except: s8 = float('nan')
        try: th = cosmo.theta_s_100() / 100.0
        except: th = float('nan')
        model_Dl = np.interp(planck_ell, ell, Dl)
        res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2n = float(np.sum(res**2)) / int(mask_fit.sum())
        cosmo.struct_cleanup(); cosmo.empty()
        return (l1, r31, s8, chi2n, th)
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

print("=" * 115)
print("cfm_fR DEEP DIVE: n_exp=0.05 PROFILE")
print("Previous best: n_exp=0.05, aM0=0.021, omch2=0.1136 -> sigma8=0.969")
print("=" * 115)
sys.stdout.flush()

# Part 1: Reference constant_alphas at the KNOWN working point
print("\n--- Reference: constant_alphas (aM=0.0005, omch2=0.1143) ---")
sys.stdout.flush()
ref = run_model(0.1143, 0, 0, gravity_model='constant_alphas',
                params_smg='0.0, -0.00025, 0.0005, 0.0, 1.0')
if ref:
    l1, r31, s8, chi2n, th = ref
    print(f"  l1={l1:.2f}, r31={r31:.4f}, sigma8={s8:.4f}, chi2/n={chi2n:.3f}, theta_s={th:.6f}")
else:
    print("  FAILED")
sys.stdout.flush()

# Part 2: cfm_fR n_exp=0.05, varying aM0 at omch2=0.1136
print(f"\n--- Part 2: cfm_fR n_exp=0.05, omch2=0.1136, varying aM0 ---")
print(f"{'aM0':>8s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s} {'theta_s':>10s} | {'aM_eff':>8s}")
print("-" * 75)
sys.stdout.flush()

for aM0 in [0.005, 0.010, 0.015, 0.018, 0.020, 0.021, 0.025, 0.030]:
    n_exp = 0.05
    aM_eff = aM0 * n_exp * 1.0 / (1.0 + aM0 * 1.0)  # alpha_M(a=1)
    res = run_model(0.1136, aM0, n_exp)
    if res:
        l1, r31, s8, chi2n, th = res
        ok = abs(l1-220) < 1 and s8 < 0.9
        print(f"{aM0:>8.4f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f} {th:>10.6f} | {aM_eff:>8.5f}{'  ***' if ok else ''}")
    else:
        print(f"{aM0:>8.4f} | FAILED")
    sys.stdout.flush()

# Part 3: cfm_fR n_exp=0.05, fix aM0=0.021, varying omch2
print(f"\n--- Part 3: cfm_fR n_exp=0.05, aM0=0.021, varying omch2 ---")
print(f"{'omch2':>8s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s} {'theta_s':>10s}")
print("-" * 65)
sys.stdout.flush()

for omch2 in [0.1120, 0.1125, 0.1130, 0.1135, 0.1136, 0.1140, 0.1143]:
    res = run_model(omch2, 0.021, 0.05)
    if res:
        l1, r31, s8, chi2n, th = res
        ok = abs(l1-220) < 1 and abs(r31-0.4295)/0.4295 < 0.02
        print(f"{omch2:>8.4f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f} {th:>10.6f}{'  ***' if ok else ''}")
    else:
        print(f"{omch2:>8.4f} | FAILED")
    sys.stdout.flush()

# Part 4: Very small n_exp (approaching constant)
print(f"\n--- Part 4: Very small n_exp (approaching constant alpha_M) ---")
print(f"{'n_exp':>8s} {'aM0':>8s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s} | {'aM(a=1)':>8s} {'aM(a=0.001)':>12s}")
print("-" * 85)
sys.stdout.flush()

for n_exp, aM0 in [(0.01, 0.100), (0.02, 0.050), (0.03, 0.035),
                    (0.04, 0.026), (0.05, 0.021), (0.07, 0.015),
                    (0.01, 0.050), (0.02, 0.025), (0.03, 0.020)]:
    aM_1 = aM0 * n_exp * 1.0 / (1.0 + aM0 * 1.0)
    aM_001 = aM0 * n_exp * (0.001**n_exp) / (1.0 + aM0 * (0.001**n_exp))
    res = run_model(0.1136, aM0, n_exp)
    if res:
        l1, r31, s8, chi2n, th = res
        ok = abs(l1-220) < 1 and s8 < 0.9
        print(f"{n_exp:>8.3f} {aM0:>8.4f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f} | {aM_1:>8.5f} {aM_001:>12.7f}{'  ***' if ok else ''}")
    else:
        print(f"{n_exp:>8.3f} {aM0:>8.4f} | FAILED")
    sys.stdout.flush()

print(f"\nPlanck 2018 reference: l1=220, r31=0.4295, sigma8=0.811, 100*theta_s=1.0411")

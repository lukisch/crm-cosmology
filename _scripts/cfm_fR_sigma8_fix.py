#!/usr/bin/env python3
"""
Investigate cfm_fR sigma8 problem with robust error handling.
Avoid extreme parameters that cause segfaults.
"""
import sys, signal
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

# Load Planck data
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
    a = (y[2] - 2*y[1] + y[0]) / 2.0
    b = (y[2] - y[0]) / 2.0
    if abs(a) < 1e-30:
        return float(x[1]), float(y[1])
    return float(x[1] - b/(2*a)), float(y[1] - b*b/(4*a))

def run_cfm_fR(omch2, aM0, n_exp, As=2.05e-9, ns=0.97):
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
        'gravity_model': 'cfm_fR',
        'parameters_smg': f'{aM0}, {n_exp}, 1.0',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except:
            pass
        return None

    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T_cmb_muK = 2.7255e6
        Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T_cmb_muK**2

        if np.any(np.isnan(Dl)) or np.any(np.isinf(Dl)):
            cosmo.struct_cleanup()
            cosmo.empty()
            return None

        l1, D1 = find_peak(ell, Dl, 150, 300)
        l3, D3 = find_peak(ell, Dl, 700, 900)
        r31 = D3/D1 if D1 > 0 else float('nan')

        try:
            sigma8 = cosmo.sigma8()
        except:
            sigma8 = float('nan')

        # Chi2
        model_Dl = np.interp(planck_ell, ell, Dl)
        residual = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2n = float(np.sum(residual**2)) / int(mask_fit.sum())

        cosmo.struct_cleanup()
        cosmo.empty()
        return (l1, r31, sigma8, chi2n)
    except Exception as e:
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except:
            pass
        return None

print("=" * 110)
print("cfm_fR SIGMA8 INVESTIGATION")
print("Goal: Find l1~220, r31~0.4295, sigma8<0.9")
print("=" * 110)
sys.stdout.flush()

# Strategy 1: Vary n_exp (steeper profiles limit late-time growth)
print("\n--- Strategy 1: Vary n_exp with aM0=0.021, omch2=0.1136 ---")
print(f"{'n_exp':>8s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s} | {'aM(a=1)':>8s} {'aM(a=0.5)':>10s}")
print("-" * 75)
sys.stdout.flush()

for n_exp in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    aM0 = 0.021
    # Compute alpha_M at a=1 and a=0.5 for reference
    aM_1 = aM0 * n_exp * 1.0 / (1.0 + aM0 * 1.0)
    aM_05 = aM0 * n_exp * (0.5**n_exp) / (1.0 + aM0 * (0.5**n_exp))
    res = run_cfm_fR(0.1136, aM0, n_exp)
    if res:
        l1, r31, s8, chi2n = res
        marker = " ***" if abs(l1-220) < 1 and s8 < 0.9 else ""
        print(f"{n_exp:>8.2f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f} | {aM_1:>8.5f} {aM_05:>10.5f}{marker}")
    else:
        print(f"{n_exp:>8.2f} | FAILED                      | {aM_1:>8.5f} {aM_05:>10.5f}")
    sys.stdout.flush()

# Strategy 2: Smaller aM0 with const_alphas comparison
print("\n--- Strategy 2: Compare with constant_alphas at same omch2 ---")
print(f"{'model':>15s} {'omch2':>8s} {'aM0':>8s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s}")
print("-" * 72)
sys.stdout.flush()

# constant_alphas reference (already known to work well)
cosmo = Class()
params = {
    'output': 'tCl,pCl,lCl,mPk',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'h': 0.673,
    'T_cmb': 2.7255,
    'omega_b': 0.02237,
    'omega_cdm': 0.1143,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'tau_reio': 0.0544,
    'A_s': 2.05e-9,
    'n_s': 0.97,
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'Omega_smg': -1,
    'gravity_model': 'constant_alphas',
    'parameters_smg': '0.0, -0.00025, 0.0005, 0.0, 1.0',
    'expansion_model': 'lcdm',
    'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
}
cosmo.set(params)
cosmo.compute()
cls = cosmo.lensed_cl(2500)
ell_arr = np.arange(2, 2501)
T_cmb_muK = 2.7255e6
Dl_ca = cls['tt'][2:2501] * ell_arr * (ell_arr + 1) / (2 * np.pi) * T_cmb_muK**2
l1_ca, D1_ca = find_peak(ell_arr, Dl_ca, 150, 300)
l3_ca, D3_ca = find_peak(ell_arr, Dl_ca, 700, 900)
r31_ca = D3_ca/D1_ca
s8_ca = cosmo.sigma8()
model_Dl_ca = np.interp(planck_ell, ell_arr, Dl_ca)
res_ca = (model_Dl_ca[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
chi2n_ca = float(np.sum(res_ca**2)) / int(mask_fit.sum())
cosmo.struct_cleanup()
cosmo.empty()
print(f"{'const_alphas':>15s} {'0.1143':>8s} {'0.0005':>8s} | {l1_ca:>7.2f} {r31_ca:>8.4f} {s8_ca:>8.4f} {chi2n_ca:>8.3f}")
sys.stdout.flush()

# cfm_fR with smaller aM0 values
for omch2, aM0 in [(0.1143, 0.005), (0.1143, 0.003), (0.1143, 0.001),
                    (0.1140, 0.005), (0.1140, 0.003),
                    (0.1136, 0.010), (0.1136, 0.005)]:
    res = run_cfm_fR(omch2, aM0, 0.1)
    if res:
        l1, r31, s8, chi2n = res
        marker = " ***" if s8 < 0.9 and chi2n < 1.2 else ""
        print(f"{'cfm_fR':>15s} {omch2:>8.4f} {aM0:>8.4f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f}{marker}")
    else:
        print(f"{'cfm_fR':>15s} {omch2:>8.4f} {aM0:>8.4f} | FAILED")
    sys.stdout.flush()

# Strategy 3: n_exp=1 with larger aM0 (more physical: alpha_M ~ a, negligible at recomb)
print("\n--- Strategy 3: n_exp=1 (alpha_M ~ aM0*a at early times) ---")
print(f"{'omch2':>8s} {'aM0':>8s} {'n_exp':>6s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s} | {'aM(a=1)':>8s}")
print("-" * 80)
sys.stdout.flush()

for omch2, aM0 in [(0.1143, 0.001), (0.1143, 0.003), (0.1143, 0.005),
                    (0.1143, 0.010), (0.1143, 0.020), (0.1143, 0.050)]:
    n_exp = 1.0
    aM_1 = aM0 * n_exp * 1.0 / (1.0 + aM0 * 1.0)
    res = run_cfm_fR(omch2, aM0, n_exp)
    if res:
        l1, r31, s8, chi2n = res
        marker = " ***" if abs(l1-220) < 1 and s8 < 0.9 else ""
        print(f"{omch2:>8.4f} {aM0:>8.4f} {n_exp:>6.1f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f} | {aM_1:>8.5f}{marker}")
    else:
        print(f"{omch2:>8.4f} {aM0:>8.4f} {n_exp:>6.1f} | FAILED                      | {aM_1:>8.5f}")
    sys.stdout.flush()

print(f"\nPlanck 2018 reference: l1=220.0, r31=0.4295, sigma8=0.811")
print(f"Note: A_s=2.05e-9, n_s=0.97 used for all runs (optimized for const_alphas)")

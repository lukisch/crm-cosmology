#!/usr/bin/env python3
"""Use classy Python wrapper for precise cfm_fR parameter optimization"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

def find_peak(ell, Dl, lmin, lmax):
    mask = (ell >= lmin) & (ell <= lmax)
    e = ell[mask]; d = Dl[mask]
    idx = np.argmax(d)
    if idx == 0 or idx == len(d)-1:
        return e[idx], d[idx]
    x = e[idx-1:idx+2]; y = d[idx-1:idx+2]
    a = (y[2] - 2*y[1] + y[0]) / 2.0
    b = (y[2] - y[0]) / 2.0
    if abs(a) < 1e-30:
        return x[1], y[1]
    return x[1] - b/(2*a), y[1] - b*b/(4*a)

def run_model(omch2, model, params_smg, expansion_smg='0.5'):
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl',
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
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': model,
        'parameters_smg': params_smg,
        'expansion_model': 'lcdm',
        'expansion_smg': expansion_smg,
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        cosmo.struct_cleanup()
        cosmo.empty()
        return None

    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi)

    l1, Dl1 = find_peak(ell, Dl, 150, 300)
    l3, Dl3 = find_peak(ell, Dl, 700, 900)
    r31 = Dl3/Dl1 if Dl1 > 0 else float('nan')

    # Also get theta_s
    try:
        theta_s = cosmo.theta_s_100() / 100.0
    except:
        theta_s = float('nan')

    cosmo.struct_cleanup()
    cosmo.empty()
    return (l1, r31, theta_s)

print("=== Precise cfm_fR Optimization with classy ===")
print(f"{'model':>10s} {'omch2':>8s} {'aM0':>8s} {'n':>6s} | {'l1':>7s} {'r31':>8s} {'theta_s':>10s} | {'dl1':>6s} {'dr31%':>7s} {'dth%':>7s}")
print("=" * 95)

# Phase 1: cfm_fR n=0.1, aM0=0.02 with different omch2
print("--- cfm_fR n=0.1, aM0=0.02, varying omch2 ---")
for omch2 in [0.1100, 0.1110, 0.1120, 0.1130, 0.1135, 0.1140, 0.1143]:
    res = run_model(omch2, 'cfm_fR', '0.02, 0.1, 1.0')
    if res:
        l1, r31, theta_s = res
        marker = " ***" if abs(l1-220) < 0.3 and abs(r31-0.4295)/0.4295 < 0.005 else ""
        print(f"{'cfm_fR':>10s} {omch2:>8.4f} {'0.02':>8s} {'0.1':>6s} | {l1:>7.2f} {r31:>8.4f} {theta_s:>10.6f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}% {(theta_s-0.010411)/0.010411*100:>+7.2f}%{marker}")
    else:
        print(f"{'cfm_fR':>10s} {omch2:>8.4f} {'0.02':>8s} {'0.1':>6s} | FAILED")

print()

# Phase 2: cfm_fR n=0.01, aM0=0.06 with different omch2
print("--- cfm_fR n=0.01, aM0=0.06, varying omch2 ---")
for omch2 in [0.1100, 0.1110, 0.1120, 0.1130, 0.1135, 0.1140, 0.1143]:
    res = run_model(omch2, 'cfm_fR', '0.06, 0.01, 1.0')
    if res:
        l1, r31, theta_s = res
        marker = " ***" if abs(l1-220) < 0.3 and abs(r31-0.4295)/0.4295 < 0.005 else ""
        print(f"{'cfm_fR':>10s} {omch2:>8.4f} {'0.06':>8s} {'0.01':>6s} | {l1:>7.2f} {r31:>8.4f} {theta_s:>10.6f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}% {(theta_s-0.010411)/0.010411*100:>+7.2f}%{marker}")
    else:
        print(f"{'cfm_fR':>10s} {omch2:>8.4f} {'0.06':>8s} {'0.01':>6s} | FAILED")

print()

# Phase 3: Fine-tune best candidate with small aM0 variations
print("--- Fine-tuning best candidate ---")
best_combos = [
    (0.1120, 0.020, 0.1),
    (0.1125, 0.020, 0.1),
    (0.1130, 0.019, 0.1),
    (0.1130, 0.020, 0.1),
    (0.1130, 0.021, 0.1),
    (0.1115, 0.060, 0.01),
    (0.1120, 0.060, 0.01),
    (0.1125, 0.060, 0.01),
]

for omch2, aM0, n_exp in best_combos:
    params_str = f"{aM0}, {n_exp}, 1.0"
    res = run_model(omch2, 'cfm_fR', params_str)
    if res:
        l1, r31, theta_s = res
        marker = " ***" if abs(l1-220) < 0.3 and abs(r31-0.4295)/0.4295 < 0.005 else ""
        print(f"{'cfm_fR':>10s} {omch2:>8.4f} {aM0:>8.4f} {n_exp:>6.2f} | {l1:>7.2f} {r31:>8.4f} {theta_s:>10.6f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}% {(theta_s-0.010411)/0.010411*100:>+7.2f}%{marker}")
    else:
        print(f"{'cfm_fR':>10s} {omch2:>8.4f} {aM0:>8.4f} {n_exp:>6.2f} | FAILED")

print(f"\nPlanck 2018 reference: l1=220.0, r31=0.4295, theta_s=0.010411")

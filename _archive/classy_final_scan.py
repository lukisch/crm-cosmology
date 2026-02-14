#!/usr/bin/env python3
"""Final precision scan to find exact cfm_fR sweet spot"""
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

def run_model(omch2, aM0, n_exp):
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
        cosmo.struct_cleanup()
        cosmo.empty()
        return None

    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi)

    l1, Dl1 = find_peak(ell, Dl, 150, 300)
    l2, Dl2 = find_peak(ell, Dl, 400, 650)
    l3, Dl3 = find_peak(ell, Dl, 700, 900)
    r31 = Dl3/Dl1 if Dl1 > 0 else float('nan')

    try:
        theta_s = cosmo.theta_s_100() / 100.0
        rs_d = cosmo.rs_drag()
    except:
        theta_s = float('nan')
        rs_d = float('nan')

    cosmo.struct_cleanup()
    cosmo.empty()
    return (l1, l2, l3, r31, theta_s, rs_d)

# Planck 2018 references
P_l1, P_r31, P_theta = 220.0, 0.4295, 0.010411

print("=" * 110)
print("FINAL PRECISION SCAN: cfm_fR Sweet-Spot Optimization")
print("=" * 110)
print(f"{'omch2':>8s} {'aM0':>8s} {'n':>5s} | {'l1':>7s} {'l2':>5s} {'l3':>5s} {'r31':>8s} {'theta_s':>10s} {'rs_d':>8s} | {'dl1':>6s} {'dr31%':>7s} {'dth%':>7s}")
print("-" * 110)

# Ultra-fine grid around the best region
results = []
for omch2 in [0.1128, 0.1130, 0.1132, 0.1134, 0.1135, 0.1136, 0.1138]:
    for aM0 in [0.019, 0.0195, 0.020, 0.0205, 0.021]:
        res = run_model(omch2, aM0, 0.1)
        if res:
            l1, l2, l3, r31, theta_s, rs_d = res
            dl1 = l1 - P_l1
            dr31 = (r31 - P_r31) / P_r31 * 100
            dth = (theta_s - P_theta) / P_theta * 100
            chi2 = dl1**2 + (dr31/0.5)**2  # combined chi2 with ~0.5% r31 weight
            results.append((chi2, omch2, aM0, l1, l2, l3, r31, theta_s, rs_d, dl1, dr31, dth))
            marker = " ***" if abs(dl1) < 0.2 and abs(dr31) < 0.3 else ""
            print(f"{omch2:>8.4f} {aM0:>8.4f} {'0.1':>5s} | {l1:>7.2f} {l2:>5.0f} {l3:>5.0f} {r31:>8.4f} {theta_s:>10.6f} {rs_d:>8.2f} | {dl1:>+6.2f} {dr31:>+7.2f}% {dth:>+7.2f}%{marker}")
        else:
            print(f"{omch2:>8.4f} {aM0:>8.4f} {'0.1':>5s} | FAILED")

# Sort by chi2 and print best 5
print("\n" + "=" * 110)
print("TOP 5 BEST MATCHES (sorted by combined chi2):")
print("=" * 110)
results.sort(key=lambda x: x[0])
for i, (chi2, omch2, aM0, l1, l2, l3, r31, theta_s, rs_d, dl1, dr31, dth) in enumerate(results[:5]):
    print(f"#{i+1}: omch2={omch2:.4f}, aM0={aM0:.4f}, n=0.1")
    print(f"     l1={l1:.2f} (dl={dl1:+.2f}), r31={r31:.4f} (dr={dr31:+.2f}%), theta_s={theta_s:.6f} (dth={dth:+.2f}%)")
    print(f"     l2={l2:.0f}, l3={l3:.0f}, rs_d={rs_d:.2f} Mpc, chi2={chi2:.4f}")

print(f"\nPlanck 2018: l1=220.0, r31=0.4295, theta_s*100=1.0411, rs_d=147.09 Mpc")

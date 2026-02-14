#!/usr/bin/env python3
"""Fine scan of cfm_fR parameter space near the sweet spot"""
import subprocess, numpy as np, os

CLASS = "/home/hi_class/class"
TEMPLATE = """
output = tCl,pCl,lCl
l_max_scalars = 2500
lensing = yes
root = /tmp/cfm_fine_

h = 0.673
T_cmb = 2.7255
omega_b = 0.02237
omega_cdm = {omch2}
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
tau_reio = 0.0544

Omega_Lambda = 0
Omega_fld = 0
Omega_smg = -1

gravity_model = {model}
parameters_smg = {params}
expansion_model = lcdm
expansion_smg = 0.5

method_qs_smg = quasi_static
skip_stability_tests_smg = yes
pert_qs_ic_tolerance_test_smg = -1
"""

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

def run_class(omch2, model, params):
    ini = TEMPLATE.format(omch2=omch2, model=model, params=params)
    ini_file = "/tmp/cfm_fine_run.ini"
    with open(ini_file, 'w') as f:
        f.write(ini)

    try:
        result = subprocess.run([CLASS, ini_file], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

    cl_file = "/tmp/cfm_fine_00_cl_lensed.dat"
    if not os.path.exists(cl_file):
        return "FAILED"

    data = np.loadtxt(cl_file)
    ell = data[:,0]; TT = data[:,1]
    l1, Dl1 = find_peak(ell, TT, 150, 300)
    l3, Dl3 = find_peak(ell, TT, 700, 900)
    r31 = Dl3/Dl1 if Dl1 > 0 else float('nan')

    for f in [cl_file, cl_file.replace("_lensed", "")]:
        if os.path.exists(f):
            os.remove(f)

    return (l1, r31)

print("=== Fine Scan: cfm_fR around sweet spot ===")
print(f"{'model':>15s} {'aM0':>8s} {'n_exp':>6s} {'omch2':>8s} | {'l1':>7s} {'r31':>8s} | {'dl1':>6s} {'dr31%':>7s}")
print("-" * 85)

# 1. Fine scan of cfm_fR at n=0.01 varying aM0
for aM0 in [0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.12, 0.15]:
    params = f"{aM0}, 0.01, 1.0"
    res = run_class(0.1143, "cfm_fR", params)
    if isinstance(res, tuple):
        l1, r31 = res
        marker = " ***" if abs(l1-220) < 0.3 else ""
        print(f"{'cfm_fR':>15s} {aM0:>8.3f} {'0.01':>6s} {'0.1143':>8s} | {l1:>7.2f} {r31:>8.4f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}%{marker}")
    else:
        print(f"{'cfm_fR':>15s} {aM0:>8.3f} {'0.01':>6s} {'0.1143':>8s} | {res}")

print("-" * 85)

# 2. Fine scan of cfm_fR at n=0.1 varying aM0 (near 0.01-0.02)
for aM0 in [0.012, 0.015, 0.018, 0.020, 0.025, 0.03]:
    params = f"{aM0}, 0.1, 1.0"
    res = run_class(0.1143, "cfm_fR", params)
    if isinstance(res, tuple):
        l1, r31 = res
        marker = " ***" if abs(l1-220) < 0.3 else ""
        print(f"{'cfm_fR':>15s} {aM0:>8.4f} {'0.1':>6s} {'0.1143':>8s} | {l1:>7.2f} {r31:>8.4f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}%{marker}")
    else:
        print(f"{'cfm_fR':>15s} {aM0:>8.4f} {'0.1':>6s} {'0.1143':>8s} | {res}")

print("-" * 85)

# 3. Once we find the best aM0, vary omch2 to tune r31
# First try: constant_alphas reference with different omch2
for omch2 in [0.1100, 0.1120, 0.1143, 0.1160, 0.1180]:
    params = f"0.0, -0.00025, 0.0005, 0.0, 1.0"
    res = run_class(omch2, "constant_alphas", params)
    if isinstance(res, tuple):
        l1, r31 = res
        marker = " ***" if abs(l1-220) < 0.3 and abs(r31-0.4295)/0.4295 < 0.005 else ""
        print(f"{'const_a':>15s} {'0.0005':>8s} {'-':>6s} {omch2:>8.4f} | {l1:>7.2f} {r31:>8.4f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}%{marker}")
    else:
        print(f"{'const_a':>15s} {'0.0005':>8s} {'-':>6s} {omch2:>8.4f} | {res}")

print(f"\nPlanck reference: l1=220.0, r31=0.4295")

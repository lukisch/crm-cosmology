#!/usr/bin/env python3
"""Scan cfm_fR parameter space with proper Omega_smg activation"""
import subprocess, numpy as np, os

CLASS = "/home/hi_class/class"
TEMPLATE = """
output = tCl,pCl,lCl
l_max_scalars = 2500
lensing = yes
root = /tmp/cfm_scan2_

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
    return x[1] - b/(2*a), y[1] - b*b/(4*a)

def run_class(omch2, model, params):
    ini = TEMPLATE.format(omch2=omch2, model=model, params=params)
    ini_file = "/tmp/cfm_scan2_run.ini"
    with open(ini_file, 'w') as f:
        f.write(ini)

    try:
        result = subprocess.run([CLASS, ini_file], capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None

    cl_file = "/tmp/cfm_scan2_00_cl_lensed.dat"
    if not os.path.exists(cl_file):
        stderr_tail = result.stderr[-200:] if result.stderr else ""
        return f"FAILED: {stderr_tail}"

    data = np.loadtxt(cl_file)
    ell = data[:,0]; TT = data[:,1]
    l1, Dl1 = find_peak(ell, TT, 150, 300)
    l3, Dl3 = find_peak(ell, TT, 700, 900)
    r31 = Dl3/Dl1

    for f in [cl_file, cl_file.replace("_lensed", "")]:
        if os.path.exists(f):
            os.remove(f)

    return (l1, r31)

# Header
print("=== cfm_fR vs constant_alphas Scan ===")
print(f"{'model':>25s} {'params':>35s} | {'l1':>7s} {'r31':>8s} | {'dl1':>6s} {'dr31%':>7s}")
print("-" * 100)

# Reference: constant_alphas at different aM values
for aM in [0.0003, 0.0005, 0.0007, 0.001]:
    aB = -aM/2
    params = f"0.0, {aB}, {aM}, 0.0, 1.0"
    res = run_class(0.1143, "constant_alphas", params)
    if isinstance(res, tuple):
        l1, r31 = res
        print(f"{'constant_alphas':>25s} {'aM='+str(aM):>35s} | {l1:>7.2f} {r31:>8.4f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}%")
    else:
        print(f"{'constant_alphas':>25s} {'aM='+str(aM):>35s} | {res}")

print("-" * 100)

# cfm_fR scans: vary aM_0 and n_exp
for n_exp in [0.01, 0.1, 0.5, 1.0, 2.0]:
    for aM_0 in [0.001, 0.01, 0.1, 1.0, 10.0]:
        params = f"{aM_0}, {n_exp}, 1.0"
        res = run_class(0.1143, "cfm_fR", params)
        if isinstance(res, tuple):
            l1, r31 = res
            marker = ""
            if abs(l1-220) < 0.5:
                marker = " <-- l1 MATCH"
            print(f"{'cfm_fR':>25s} {'aM0='+str(aM_0)+' n='+str(n_exp):>35s} | {l1:>7.2f} {r31:>8.4f} | {l1-220:>+6.2f} {(r31-0.4295)/0.4295*100:>+7.2f}%{marker}")
        else:
            print(f"{'cfm_fR':>25s} {'aM0='+str(aM_0)+' n='+str(n_exp):>35s} | {res}")

print(f"\nPlanck reference: l1=220, r31=0.4295")

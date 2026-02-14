#!/usr/bin/env python3
"""Scan cfm_fR parameter space in hi_class"""
import subprocess, numpy as np, os, sys

CLASS = "/home/hi_class/class"
TEMPLATE = """
output = tCl,pCl,lCl
l_max_scalars = 2500
lensing = yes
root = /tmp/cfm_scan_

h = 0.673
T_cmb = 2.7255
omega_b = 0.02237
omega_cdm = {omch2}
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
tau_reio = 0.0544

gravity_model = cfm_fR
parameters_smg = {aM}, {n_exp}, 1.0
expansion_model = lcdm
expansion_smg = 0.5
"""

def run_class(omch2, aM, n_exp):
    ini = TEMPLATE.format(omch2=omch2, aM=aM, n_exp=n_exp)
    ini_file = "/tmp/cfm_scan_run.ini"
    with open(ini_file, 'w') as f:
        f.write(ini)

    result = subprocess.run([CLASS, ini_file], capture_output=True, text=True, timeout=300)

    cl_file = "/tmp/cfm_scan_00_cl_lensed.dat"
    if not os.path.exists(cl_file):
        return None

    data = np.loadtxt(cl_file)
    ell = data[:,0]
    TT = data[:,1]

    m1 = (ell > 150) & (ell < 300)
    l1 = int(ell[m1][np.argmax(TT[m1])])
    Dl1 = TT[m1][np.argmax(TT[m1])]

    m3 = (ell > 700) & (ell < 900)
    l3 = int(ell[m3][np.argmax(TT[m3])])
    Dl3 = TT[m3][np.argmax(TT[m3])]

    r31 = Dl3/Dl1

    # Clean up
    for f in [cl_file, cl_file.replace("_lensed", "")]:
        if os.path.exists(f):
            os.remove(f)

    return l1, r31, l3

# Parameter scan
print("=== cfm_fR Parameter Scan ===")
print(f"{'omch2':>8s} {'aM':>8s} {'n_exp':>6s} | {'l1':>4s} {'r31':>8s} {'l3':>4s} | {'dl1':>4s} {'dr31%':>7s}")
print("-" * 70)

scan_params = [
    # omch2, aM, n_exp
    (0.1143, 0.001,  1.0),
    (0.1143, 0.0015, 1.0),
    (0.1143, 0.002,  1.0),
    (0.1143, 0.003,  1.0),
    (0.1143, 0.001,  0.5),
    (0.1143, 0.002,  0.5),
    (0.1143, 0.003,  0.5),
    (0.1143, 0.005,  0.5),
    (0.1143, 0.001,  2.0),
    (0.1143, 0.003,  2.0),
    (0.1143, 0.005,  2.0),
    (0.1143, 0.01,   2.0),
]

for omch2, aM, n_exp in scan_params:
    try:
        res = run_class(omch2, aM, n_exp)
        if res is None:
            print(f"{omch2:>8.4f} {aM:>8.4f} {n_exp:>6.1f} | FAILED")
        else:
            l1, r31, l3 = res
            dl1 = l1 - 220
            dr31_pct = (r31 - 0.4295) / 0.4295 * 100
            marker = " <-- EXACT" if l1 == 220 and abs(dr31_pct) < 0.5 else ""
            print(f"{omch2:>8.4f} {aM:>8.4f} {n_exp:>6.1f} | {l1:>4d} {r31:>8.4f} {l3:>4d} | {dl1:>+4d} {dr31_pct:>+7.2f}%{marker}")
    except Exception as e:
        print(f"{omch2:>8.4f} {aM:>8.4f} {n_exp:>6.1f} | ERROR: {e}")

print("\nPlanck reference: l1=220, r31=0.4295")

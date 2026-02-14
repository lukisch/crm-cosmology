#!/usr/bin/env python3
"""Compare LCDM, constant_alphas, and cfm_fR models"""
import subprocess, numpy as np, os

CLASS = "/home/hi_class/class"
BASE = """
output = tCl,pCl,lCl
l_max_scalars = 2500
lensing = yes

h = 0.673
T_cmb = 2.7255
omega_b = 0.02237
omega_cdm = 0.1143
N_ur = 2.0328
N_ncdm = 1
m_ncdm = 0.06
tau_reio = 0.0544
"""

configs = {
    "LCDM": BASE,
    "const_aM0.0007": BASE + """
gravity_model = constant_alphas
parameters_smg = 0.0, -0.00035, 0.0007, 0.0, 1.0
expansion_model = lcdm
expansion_smg = 0.5
""",
    "const_aM0.01": BASE + """
gravity_model = constant_alphas
parameters_smg = 0.0, -0.005, 0.01, 0.0, 1.0
expansion_model = lcdm
expansion_smg = 0.5
""",
    "cfm_fR_0.0007": BASE + """
gravity_model = cfm_fR
parameters_smg = 0.0007, 1.0, 1.0
expansion_model = lcdm
expansion_smg = 0.5
""",
    "cfm_fR_0.1": BASE + """
gravity_model = cfm_fR
parameters_smg = 0.1, 1.0, 1.0
expansion_model = lcdm
expansion_smg = 0.5
""",
}

def find_peak_precise(ell, Dl, lmin, lmax):
    mask = (ell >= lmin) & (ell <= lmax)
    e = ell[mask]; d = Dl[mask]
    idx = np.argmax(d)
    if idx == 0 or idx == len(d)-1:
        return e[idx], d[idx]
    x = e[idx-1:idx+2]; y = d[idx-1:idx+2]
    a = (y[2] - 2*y[1] + y[0]) / 2.0
    b = (y[2] - y[0]) / 2.0
    x_peak = x[1] - b/(2*a)
    y_peak = y[1] - b*b/(4*a)
    return x_peak, y_peak

results = {}
for name, ini_content in configs.items():
    root = f"/tmp/cmp_{name}_"
    ini_content_full = ini_content + f"\nroot = {root}\n"
    ini_file = f"/tmp/cmp_{name}.ini"
    with open(ini_file, 'w') as f:
        f.write(ini_content_full)

    result = subprocess.run([CLASS, ini_file], capture_output=True, text=True, timeout=300)
    cl_file = f"{root}00_cl_lensed.dat"
    if not os.path.exists(cl_file):
        print(f"{name}: FAILED - {result.stderr[-200:]}")
        continue

    data = np.loadtxt(cl_file)
    ell = data[:,0]; TT = data[:,1]
    l1, Dl1 = find_peak_precise(ell, TT, 150, 300)
    l3, Dl3 = find_peak_precise(ell, TT, 700, 900)
    r31 = Dl3/Dl1
    results[name] = (ell, TT, l1, Dl1, l3, Dl3, r31)
    print(f"{name:25s}: l1={l1:.2f}  r31={r31:.4f}  Dl1={Dl1:.6e}")

# Compare spectra
if len(results) >= 2:
    print("\n=== Spectrum Differences ===")
    ref_name = "LCDM"
    if ref_name in results:
        ref_TT = results[ref_name][1]
        for name in results:
            if name == ref_name:
                continue
            TT = results[name][1]
            maxdiff = np.max(np.abs(TT - ref_TT))
            reldiff = np.max(np.abs(TT - ref_TT) / (np.abs(ref_TT) + 1e-30))
            print(f"  {name} vs LCDM: max_abs_diff={maxdiff:.6e}, max_rel_diff={reldiff:.6e}")

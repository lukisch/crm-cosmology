#!/usr/bin/env python3
"""
Full C_l comparison: cfm_fR vs LCDM vs Planck 2018 data
Workflow Steps 1-3: Compute spectra, extract peaks, compare with Planck
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

# ============================================================
# STEP 1: Load Planck 2018 TT data
# ============================================================
print("=" * 80)
print("STEP 1: Loading Planck 2018 TT power spectrum data")
print("=" * 80)

planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]  # D_l = l(l+1)/(2pi) * C_l in muK^2
planck_err_lo = planck_data[:, 2]
planck_err_hi = planck_data[:, 3]
print(f"  Planck data: ell = {planck_ell[0]} to {planck_ell[-1]}, {len(planck_ell)} points")

# ============================================================
# STEP 2: Compute theoretical spectra
# ============================================================
print("\n" + "=" * 80)
print("STEP 2: Computing theoretical CMB spectra with hi_class/classy")
print("=" * 80)

def compute_spectrum(name, extra_params=None):
    """Compute lensed TT spectrum and derived quantities"""
    cosmo = Class()
    base_params = {
        'output': 'tCl,pCl,lCl,mPk',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': 0.673,
        'T_cmb': 2.7255,
        'omega_b': 0.02237,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'tau_reio': 0.0544,
    }
    if extra_params:
        base_params.update(extra_params)

    cosmo.set(base_params)
    cosmo.compute()

    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    # Convert to D_l in muK^2
    T_cmb_muK = 2.7255e6  # muK
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T_cmb_muK**2

    # Derived quantities
    theta_s = cosmo.theta_s_100() / 100.0
    rs_d = cosmo.rs_drag()
    H0 = cosmo.Hubble(0) * 299792.458  # km/s/Mpc
    age = cosmo.age()
    sigma8 = cosmo.sigma8()

    cosmo.struct_cleanup()
    cosmo.empty()

    return ell, Dl, theta_s, rs_d, H0, age, sigma8

def find_peak(ell, Dl, lmin, lmax):
    mask = (ell >= lmin) & (ell <= lmax)
    e = ell[mask]; d = Dl[mask]
    idx = np.argmax(d)
    if idx == 0 or idx == len(d)-1:
        return e[idx], d[idx]
    x = e[idx-1:idx+2].astype(float); y = d[idx-1:idx+2]
    a = (y[2] - 2*y[1] + y[0]) / 2.0
    b = (y[2] - y[0]) / 2.0
    if abs(a) < 1e-30:
        return x[1], y[1]
    return x[1] - b/(2*a), y[1] - b*b/(4*a)

models = {}

# 2a: Standard LCDM (Planck 2018 best-fit parameters)
print("\n  Computing LCDM (Planck 2018 best-fit)...")
ell_lcdm, Dl_lcdm, th_lcdm, rs_lcdm, H0_lcdm, age_lcdm, s8_lcdm = compute_spectrum(
    "LCDM",
    {'omega_cdm': 0.1200, 'h': 0.6736, 'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649}
)
models['LCDM'] = (ell_lcdm, Dl_lcdm, th_lcdm, rs_lcdm, H0_lcdm, age_lcdm, s8_lcdm)
print(f"    theta_s={th_lcdm:.6f}, rs_d={rs_lcdm:.2f}, H0={H0_lcdm:.2f}")

# 2b: CFM Basis (no MG)
print("  Computing CFM Basis (omch2=0.1066, no MG)...")
ell_cfm0, Dl_cfm0, th_cfm0, rs_cfm0, H0_cfm0, age_cfm0, s8_cfm0 = compute_spectrum(
    "CFM_basis",
    {'omega_cdm': 0.1066}
)
models['CFM_basis'] = (ell_cfm0, Dl_cfm0, th_cfm0, rs_cfm0, H0_cfm0, age_cfm0, s8_cfm0)
print(f"    theta_s={th_cfm0:.6f}, rs_d={rs_cfm0:.2f}, H0={H0_cfm0:.2f}")

# 2c: CFM + cfm_fR (BEST MODEL)
print("  Computing CFM + cfm_fR (omch2=0.1136, aM0=0.021, n=0.1)...")
ell_cfmR, Dl_cfmR, th_cfmR, rs_cfmR, H0_cfmR, age_cfmR, s8_cfmR = compute_spectrum(
    "CFM_fR",
    {
        'omega_cdm': 0.1136,
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'cfm_fR',
        'parameters_smg': '0.021, 0.1, 1.0',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }
)
models['CFM_fR'] = (ell_cfmR, Dl_cfmR, th_cfmR, rs_cfmR, H0_cfmR, age_cfmR, s8_cfmR)
print(f"    theta_s={th_cfmR:.6f}, rs_d={rs_cfmR:.2f}, H0={H0_cfmR:.2f}")

# 2d: CFM + constant_alphas (previous best)
print("  Computing CFM + constant_alphas (omch2=0.1143, aM=0.0007)...")
ell_cfmC, Dl_cfmC, th_cfmC, rs_cfmC, H0_cfmC, age_cfmC, s8_cfmC = compute_spectrum(
    "CFM_const",
    {
        'omega_cdm': 0.1143,
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
)
models['CFM_const'] = (ell_cfmC, Dl_cfmC, th_cfmC, rs_cfmC, H0_cfmC, age_cfmC, s8_cfmC)
print(f"    theta_s={th_cfmC:.6f}, rs_d={rs_cfmC:.2f}, H0={H0_cfmC:.2f}")

# ============================================================
# STEP 3: Peak analysis and comparison
# ============================================================
print("\n" + "=" * 80)
print("STEP 3: Peak analysis")
print("=" * 80)

peak_ranges = [(150, 300), (400, 650), (700, 900), (1050, 1250), (1350, 1550)]
peak_names = ['l1', 'l2', 'l3', 'l4', 'l5']

# Planck data peaks
print("\nPlanck 2018 data peaks:")
for name, (lmin, lmax) in zip(peak_names, peak_ranges):
    mask = (planck_ell >= lmin) & (planck_ell <= lmax)
    if mask.sum() > 0:
        idx = np.argmax(planck_Dl[mask])
        l_p = planck_ell[mask][idx]
        D_p = planck_Dl[mask][idx]
        print(f"  {name} = {l_p}, D_l = {D_p:.1f} muK^2")

print("\nModel peak positions and ratios:")
print(f"{'Model':>15s} | {'l1':>7s} {'l2':>5s} {'l3':>5s} {'r31':>8s} {'r21':>8s} | {'theta_s':>10s} {'rs_d':>8s} {'H0':>6s} {'sigma8':>7s}")
print("-" * 95)

for mname, (ell, Dl, th, rs, H0, age, s8) in models.items():
    peaks = []
    for lmin, lmax in peak_ranges[:3]:
        lp, Dp = find_peak(ell, Dl, lmin, lmax)
        peaks.append((lp, Dp))
    l1, D1 = peaks[0]
    l2, D2 = peaks[1]
    l3, D3 = peaks[2]
    r31 = D3/D1
    r21 = D2/D1
    print(f"{mname:>15s} | {l1:>7.2f} {l2:>5.0f} {l3:>5.0f} {r31:>8.4f} {r21:>8.4f} | {th:>10.6f} {rs:>8.2f} {H0:>6.2f} {s8:>7.4f}")

# ============================================================
# STEP 4: Chi-squared analysis against Planck data
# ============================================================
print("\n" + "=" * 80)
print("STEP 4: Chi-squared analysis against Planck 2018 TT data")
print("=" * 80)

# Use symmetric error for chi2
planck_err = (planck_err_lo + planck_err_hi) / 2.0

for mname, (ell, Dl, th, rs, H0, age, s8) in models.items():
    # Interpolate model to Planck ell values
    model_Dl_at_planck = np.interp(planck_ell, ell, Dl)

    # Chi2 in different ell ranges
    for lmin, lmax, label in [(30, 2500, "full"), (100, 800, "peaks1-2"), (100, 1500, "peaks1-4")]:
        mask = (planck_ell >= lmin) & (planck_ell <= lmax) & (planck_err > 0)
        residual = (model_Dl_at_planck[mask] - planck_Dl[mask]) / planck_err[mask]
        chi2 = np.sum(residual**2)
        npts = mask.sum()
        chi2_red = chi2 / npts if npts > 0 else 0
        print(f"  {mname:>15s} [{label:>10s}]: chi2={chi2:>10.1f}, npts={npts:>5d}, chi2/npts={chi2_red:>6.2f}")

# ============================================================
# STEP 5: Residual analysis
# ============================================================
print("\n" + "=" * 80)
print("STEP 5: Residual analysis (model - Planck) / sigma")
print("=" * 80)

for mname, (ell, Dl, th, rs, H0, age, s8) in models.items():
    model_Dl_at_planck = np.interp(planck_ell, ell, Dl)
    mask = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)
    residual = (model_Dl_at_planck[mask] - planck_Dl[mask]) / planck_err[mask]

    print(f"\n  {mname}:")
    print(f"    Mean residual:    {np.mean(residual):>+8.3f} sigma")
    print(f"    RMS residual:     {np.sqrt(np.mean(residual**2)):>8.3f} sigma")
    print(f"    Max |residual|:   {np.max(np.abs(residual)):>8.3f} sigma at l={planck_ell[mask][np.argmax(np.abs(residual))]}")

    # Residuals in specific peak regions
    for pname, (lmin, lmax) in zip(['Peak1', 'Peak2', 'Peak3'], [(180, 260), (450, 600), (750, 870)]):
        pmask = mask & (planck_ell >= lmin) & (planck_ell <= lmax)
        if pmask.sum() > 0:
            pres = (model_Dl_at_planck[pmask] - planck_Dl[pmask]) / planck_err[pmask]
            print(f"    {pname} region ({lmin}-{lmax}): mean={np.mean(pres):>+6.2f}s, rms={np.sqrt(np.mean(pres**2)):>5.2f}s")

# ============================================================
# STEP 6: Summary table
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY: CFM f(R) vs LCDM vs Planck 2018")
print("=" * 80)

# Planck reference values
P = {'l1': 220.0, 'r31': 0.4295, 'theta_s': 0.010411, 'rs_d': 147.09, 'H0': 67.36}

print(f"\n{'Observable':>20s} {'Planck 2018':>12s} {'LCDM':>12s} {'CFM basis':>12s} {'CFM+fR':>12s} {'CFM+const':>12s}")
print("-" * 82)

for mname_list, data_list in [
    (['LCDM', 'CFM_basis', 'CFM_fR', 'CFM_const'],
     [models['LCDM'], models['CFM_basis'], models['CFM_fR'], models['CFM_const']])
]:
    vals = {}
    for mn, (ell, Dl, th, rs, H0, age, s8) in zip(mname_list, data_list):
        l1, D1 = find_peak(ell, Dl, 150, 300)
        l3, D3 = find_peak(ell, Dl, 700, 900)
        vals[mn] = {'l1': l1, 'r31': D3/D1, 'theta_s': th, 'rs_d': rs, 'H0': H0, 'sigma8': s8}

    for obs, planck_val, fmt in [
        ('l1', 220.0, '.1f'),
        ('r31', 0.4295, '.4f'),
        ('100*theta_s', 1.0411, '.4f'),
        ('rs_d [Mpc]', 147.09, '.2f'),
        ('H0 [km/s/Mpc]', 67.36, '.2f'),
        ('sigma8', 0.811, '.3f'),
    ]:
        key = obs.split('[')[0].strip().replace('100*', '')
        if key == 'theta_s':
            row = f"{obs:>20s} {planck_val:>12{fmt}}"
            for mn in mname_list:
                v = vals[mn][key] * 100
                row += f" {v:>12{fmt}}"
        elif key in vals[mname_list[0]]:
            row = f"{obs:>20s} {planck_val:>12{fmt}}"
            for mn in mname_list:
                row += f" {vals[mn][key]:>12{fmt}}"
        else:
            continue
        print(row)

print(f"\nPlanck 2018: theta_s*100 = 1.04110 +/- 0.00031, rs_d = 147.09 +/- 0.26 Mpc")
print(f"Note: A_s and n_s not fitted here; using default hi_class values for CFM models")

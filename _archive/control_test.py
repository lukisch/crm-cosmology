#!/usr/bin/env python3
"""
Control test: Compare LCDM via standard CLASS vs SMG module.
Also test different alpha_K values to see if D regularization helps.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

# Load Planck data
planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]
planck_err = (planck_data[:, 2] + planck_data[:, 3]) / 2.0
mask_fit = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)
npts = mask_fit.sum()
print(f"Planck data: {len(planck_ell)} points, {npts} in fit range")
print(f"ell range: {planck_ell.min()} - {planck_ell.max()}")
print(f"Mean error: {planck_err[mask_fit].mean():.1f}")

base_params = {
    'output': 'tCl,pCl,lCl,mPk',
    'l_max_scalars': 2500,
    'lensing': 'yes',
    'h': 0.6732,
    'T_cmb': 2.7255,
    'omega_b': 0.02237,
    'omega_cdm': 0.1200,
    'N_ur': 2.0328,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'tau_reio': 0.0544,
    'A_s': 2.1e-9,
    'n_s': 0.9649,
}

def evaluate(params, label):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        print(f"  {label}: FAILED - {e}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return
    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    T = 2.7255e6
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
    try: s8 = cosmo.sigma8()
    except: s8 = float('nan')
    try: th = cosmo.theta_s_100() / 100.0
    except: th = float('nan')
    cosmo.struct_cleanup()
    cosmo.empty()
    model_Dl = np.interp(planck_ell, ell, Dl)
    res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
    chi2 = float(np.sum(res**2))
    print(f"  {label}: chi2={chi2:.1f} (chi2/n={chi2/npts:.4f}), s8={s8:.4f}, th={th:.6f}")
    return chi2, s8, th

# ============================================================
# TEST 1: Standard LCDM (no SMG)
# ============================================================
print("\n=== TEST 1: Standard LCDM (CLASS without SMG) ===")
evaluate(base_params.copy(), "Standard LCDM")

# ============================================================
# TEST 2: LCDM through SMG with tiny alpha_M
# ============================================================
print("\n=== TEST 2: SMG with tiny alpha_M (should ≈ LCDM) ===")
for aM in [0.0001, 0.0002, 0.0005, 0.001]:
    aB = -aM / 2.0
    smg_params = base_params.copy()
    smg_params.update({
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'constant_alphas',
        'parameters_smg': f'0.0, {aB}, {aM}, 0.0, 1.0',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    })
    evaluate(smg_params, f"SMG aM={aM:.4f} (aK=0)")

# ============================================================
# TEST 3: SMG with non-zero alpha_K (regularized D)
# ============================================================
print("\n=== TEST 3: SMG with non-zero alpha_K (D regularization) ===")
aM = 0.0008
aB = -aM / 2.0
for aK in [0.0, 0.001, 0.01, 0.1, 1.0]:
    smg_params = base_params.copy()
    smg_params.update({
        'omega_cdm': 0.1165,  # CFM optimal
        'A_s': 2.05e-9,
        'n_s': 0.97,
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'constant_alphas',
        'parameters_smg': f'{aK}, {aB}, {aM}, 0.0, 1.0',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    })
    D = aK + 3./2. * aB**2
    Geff = 1 + 2 * aM**2 / D if D > 0 else float('inf')
    evaluate(smg_params, f"aK={aK:.3f} (D={D:.2e}, Geff/G={Geff:.2f})")

# ============================================================
# TEST 4: propto_omega model (proper f(R) parametrization)
# ============================================================
print("\n=== TEST 4: propto_omega (proper f(R)) ===")
for f0 in [0.001, 0.01, 0.1]:
    smg_params = base_params.copy()
    smg_params.update({
        'omega_cdm': 0.1165,
        'A_s': 2.05e-9,
        'n_s': 0.97,
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'propto_omega',
        'parameters_smg': f'{f0}',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    })
    evaluate(smg_params, f"propto_omega f0={f0}")

# ============================================================
# TEST 5: CFM best point from previous session for validation
# ============================================================
print("\n=== TEST 5: Previous best CFM point (omch2=0.1165, aM=0.0008) ===")
smg_params = base_params.copy()
smg_params.update({
    'h': 0.673,
    'omega_cdm': 0.1165,
    'A_s': 2.05e-9,
    'n_s': 0.97,
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'Omega_smg': -1,
    'gravity_model': 'constant_alphas',
    'parameters_smg': f'0.0, -0.0004, 0.0008, 0.0, 1.0',
    'expansion_model': 'lcdm',
    'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
})
evaluate(smg_params, "Previous best (h=0.673)")

# TEST 5b: with h=0.682 (theta_s closer)
smg_params['h'] = 0.682
smg_params['parameters_smg'] = '0.0, -0.0004, 0.0008, 0.0, 1.0'
evaluate(smg_params, "Previous best (h=0.682)")

# ============================================================
# Planck data diagnostics
# ============================================================
print(f"\n=== Planck Data Diagnostics ===")
print(f"Total data points: {len(planck_ell)}")
print(f"Points in fit range (30 <= ell <= 2500, err > 0): {npts}")
print(f"First 5 data points: ell={planck_ell[:5]}, Dl={planck_Dl[:5]}, err={planck_err[:5]}")
print(f"Last 5 data points: ell={planck_ell[-5:]}, Dl={planck_Dl[-5:]}, err={planck_err[-5:]}")
print(f"Error bar range: {planck_err[mask_fit].min():.1f} - {planck_err[mask_fit].max():.1f}")

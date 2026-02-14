#!/usr/bin/env python3
"""
Optimize A_s and n_s for CFM models against Planck 2018 TT data.
Key insight: Previous runs used default A_s/n_s, not optimized values.
This can significantly improve chi2 and may help with sigma8.
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

# Only use l >= 30 for chi2 (cosmic variance dominated below)
mask_fit = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)

def compute_chi2(extra_params, As, ns):
    """Compute chi2 against Planck TT data for given model + A_s + n_s"""
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
        'A_s': As,
        'n_s': ns,
    }
    base_params.update(extra_params)
    cosmo.set(base_params)
    try:
        cosmo.compute()
    except Exception as e:
        cosmo.struct_cleanup()
        cosmo.empty()
        return 1e10, 0, 0

    cls = cosmo.lensed_cl(2500)
    ell = np.arange(2, 2501)
    T_cmb_muK = 2.7255e6
    Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T_cmb_muK**2

    try:
        sigma8 = cosmo.sigma8()
    except:
        sigma8 = float('nan')

    try:
        theta_s = cosmo.theta_s_100() / 100.0
    except:
        theta_s = float('nan')

    cosmo.struct_cleanup()
    cosmo.empty()

    # Interpolate to Planck ell values
    model_Dl = np.interp(planck_ell, ell, Dl)
    residual = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
    chi2 = np.sum(residual**2)
    npts = mask_fit.sum()

    return chi2 / npts, sigma8, theta_s

# ============================================================
# Model configurations
# ============================================================
models = {
    'LCDM': {
        'omega_cdm': 0.1200,
        'h': 0.6736,
    },
    'CFM+const_alphas': {
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
    },
    'CFM+cfm_fR': {
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
    },
}

# ============================================================
# A_s / n_s Grid Search
# ============================================================
print("=" * 100)
print("OPTIMIZATION: A_s and n_s for each model against Planck 2018 TT")
print("=" * 100)

# Planck best-fit: A_s = 2.1e-9, n_s = 0.9649
As_grid = [1.8e-9, 1.9e-9, 2.0e-9, 2.05e-9, 2.1e-9, 2.15e-9, 2.2e-9, 2.3e-9]
ns_grid = [0.950, 0.960, 0.9649, 0.970, 0.975, 0.980]

for mname, extra_params in models.items():
    print(f"\n{'='*80}")
    print(f"  Model: {mname}")
    print(f"{'='*80}")
    print(f"  {'A_s':>10s} {'n_s':>6s} | {'chi2/n':>8s} {'sigma8':>8s} {'theta_s':>10s}")
    print(f"  {'-'*55}")

    best = (1e10, None, None, None, None)
    for As in As_grid:
        for ns in ns_grid:
            chi2n, s8, th = compute_chi2(extra_params, As, ns)
            marker = ""
            if chi2n < best[0]:
                best = (chi2n, As, ns, s8, th)
                marker = " <-- BEST"
            # Only print promising ones
            if chi2n < 2.0 or marker:
                print(f"  {As:>10.3e} {ns:>6.4f} | {chi2n:>8.3f} {s8:>8.4f} {th:>10.6f}{marker}")

    chi2n, As, ns, s8, th = best
    print(f"\n  OPTIMAL: A_s={As:.3e}, n_s={ns:.4f}")
    print(f"           chi2/npts={chi2n:.3f}, sigma8={s8:.4f}, theta_s={th:.6f}")

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

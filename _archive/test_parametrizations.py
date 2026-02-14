#!/usr/bin/env python3
"""
Compare different gravity parametrizations for CMB fit:
1. constant_alphas with alpha_B = -alpha_M/2 (CFM derivation)
2. constant_alphas with alpha_B = -alpha_M (standard f(R))
3. propto_omega with f(R) relations (alphas ∝ Omega_DE)
4. propto_scale with f(R) relations (alphas ∝ a)
Test which gives best chi2 fit to Planck TT.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]
planck_err = (planck_data[:, 2] + planck_data[:, 3]) / 2.0
mask_fit = (planck_ell >= 30) & (planck_ell <= 2500) & (planck_err > 0)
npts = mask_fit.sum()

base_smg = {
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
    'Omega_Lambda': 0,
    'Omega_fld': 0,
    'Omega_smg': -1,
    'expansion_model': 'lcdm',
    'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
}

def evaluate(params, label):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        emsg = str(e)[:200]
        print(f"  {label}: FAILED - {emsg}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None
    try:
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
        dth = (th - 0.010411) / 0.010411 * 100 if not np.isnan(th) else float('nan')
        ds8 = (s8 - 0.811) / 0.811 * 100 if not np.isnan(s8) else float('nan')
        print(f"  {label}: chi2={chi2:.1f} ({chi2/npts:.3f}), s8={s8:.4f} ({ds8:+.1f}%), th={th:.6f} ({dth:+.2f}%)")
        return chi2, s8, th
    except Exception as e:
        print(f"  {label}: POST-COMPUTE FAIL - {str(e)[:100]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# ============================================================
# LCDM reference
# ============================================================
print("=" * 100)
print("LCDM REFERENCE")
cosmo = Class()
cosmo.set({k: v for k, v in base_smg.items() if k not in [
    'Omega_Lambda', 'Omega_fld', 'Omega_smg', 'expansion_model', 'expansion_smg',
    'gravity_model', 'parameters_smg', 'method_qs_smg', 'skip_stability_tests_smg',
    'pert_qs_ic_tolerance_test_smg']})
cosmo.compute()
cls_ref = cosmo.lensed_cl(2500)
ell = np.arange(2, 2501)
T = 2.7255e6
Dl_ref = cls_ref['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
s8_ref = cosmo.sigma8()
th_ref = cosmo.theta_s_100() / 100.0
cosmo.struct_cleanup(); cosmo.empty()
model_Dl_ref = np.interp(planck_ell, ell, Dl_ref)
res_ref = (model_Dl_ref[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
chi2_ref = float(np.sum(res_ref**2))
print(f"  LCDM: chi2={chi2_ref:.1f} ({chi2_ref/npts:.3f}), s8={s8_ref:.4f}, th={th_ref:.6f}")
sys.stdout.flush()

# ============================================================
# TEST A: constant_alphas, alpha_B = -alpha_M/2 (CFM)
# ============================================================
print("\n" + "=" * 100)
print("TEST A: constant_alphas, alpha_B = -alpha_M/2 (CFM derivation)")
print("=" * 100)
for c_M in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]:
    c_B = -c_M / 2.0
    p = base_smg.copy()
    p['gravity_model'] = 'constant_alphas'
    p['parameters_smg'] = f'0.0, {c_B}, {c_M}, 0.0, 1.0'
    evaluate(p, f"cA: aM={c_M:.4f}, aB={c_B:.5f}")
    sys.stdout.flush()

# ============================================================
# TEST B: constant_alphas, alpha_B = -alpha_M (standard f(R))
# ============================================================
print("\n" + "=" * 100)
print("TEST B: constant_alphas, alpha_B = -alpha_M (standard f(R))")
print("=" * 100)
for c_M in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]:
    c_B = -c_M
    p = base_smg.copy()
    p['gravity_model'] = 'constant_alphas'
    p['parameters_smg'] = f'0.0, {c_B}, {c_M}, 0.0, 1.0'
    evaluate(p, f"cA-fR: aM={c_M:.4f}, aB={c_B:.5f}")
    sys.stdout.flush()

# ============================================================
# TEST C: propto_omega, alpha_B = -alpha_M/2 (CFM)
# ============================================================
print("\n" + "=" * 100)
print("TEST C: propto_omega, alpha_B = -alpha_M/2 (CFM, alphas ∝ Omega_DE)")
print("=" * 100)
for c_M in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    c_B = -c_M / 2.0
    p = base_smg.copy()
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'0.0, {c_B}, {c_M}, 0.0, 1.0'
    evaluate(p, f"pO-CFM: cM={c_M:.3f}")
    sys.stdout.flush()

# ============================================================
# TEST D: propto_omega, alpha_B = -alpha_M (standard f(R))
# ============================================================
print("\n" + "=" * 100)
print("TEST D: propto_omega, alpha_B = -alpha_M (standard f(R), alphas ∝ Omega_DE)")
print("=" * 100)
for c_M in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    c_B = -c_M
    p = base_smg.copy()
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'0.0, {c_B}, {c_M}, 0.0, 1.0'
    evaluate(p, f"pO-fR: cM={c_M:.3f}")
    sys.stdout.flush()

# ============================================================
# TEST E: propto_scale, alpha_B = -alpha_M/2 (CFM, alphas ∝ a)
# ============================================================
print("\n" + "=" * 100)
print("TEST E: propto_scale, alpha_B = -alpha_M/2 (CFM, alphas ∝ a)")
print("=" * 100)
for c_M in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    c_B = -c_M / 2.0
    p = base_smg.copy()
    p['gravity_model'] = 'propto_scale'
    p['parameters_smg'] = f'0.0, {c_B}, {c_M}, 0.0, 1.0'
    evaluate(p, f"pS-CFM: cM={c_M:.3f}")
    sys.stdout.flush()

# ============================================================
# TEST F: propto_omega with non-zero c_K
# ============================================================
print("\n" + "=" * 100)
print("TEST F: propto_omega with c_K (D regularization)")
print("=" * 100)
c_M = 0.1  # moderate value
for c_K in [0.0, 0.01, 0.1, 1.0, 10.0]:
    c_B = -c_M / 2.0
    p = base_smg.copy()
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'{c_K}, {c_B}, {c_M}, 0.0, 1.0'
    # At today: Omega_smg ~0.69
    D_today = c_K * 0.69 + 1.5 * (c_B * 0.69)**2
    evaluate(p, f"pO: cK={c_K:.1f}, cM=0.1 (D~{D_today:.3f})")
    sys.stdout.flush()

# ============================================================
# TEST G: Best CFM at reduced omega_cdm with propto_omega
# ============================================================
print("\n" + "=" * 100)
print("TEST G: propto_omega at CFM-preferred omega_cdm=0.1165")
print("=" * 100)
for c_M in [0.001, 0.005, 0.01, 0.05, 0.1, 0.3]:
    c_B = -c_M / 2.0
    p = base_smg.copy()
    p['omega_cdm'] = 0.1165
    p['A_s'] = 2.05e-9
    p['n_s'] = 0.97
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'0.0, {c_B}, {c_M}, 0.0, 1.0'
    evaluate(p, f"pO-CFM165: cM={c_M:.3f}")
    sys.stdout.flush()

print("\n" + "=" * 100)
print("SUMMARY: delta chi2 relative to LCDM")
print(f"LCDM chi2 = {chi2_ref:.1f}")
print("=" * 100)

#!/usr/bin/env python3
"""
Compute f*sigma8(z) for LCDM and CFM models using hi_class.
Also compute Sigma (gravitational lensing parameter) and mu (growth).
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

z_vals = [0.0, 0.1, 0.15, 0.2, 0.3, 0.38, 0.51, 0.61, 0.7, 0.8, 1.0, 1.5, 2.0]

def compute_fsigma8(params_dict, label):
    cosmo = Class()
    cosmo.set(params_dict)
    try:
        cosmo.compute()
    except Exception as e:
        print(f"{label}: FAILED compute - {str(e)[:80]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

    try:
        s8_0 = cosmo.sigma8()
        h = cosmo.h()
        results = []
        for z in z_vals:
            try:
                s8z = cosmo.sigma(8.0/h, z)
                # Growth rate f = d ln D / d ln a ~ Omega_m(z)^0.55 for LCDM
                # Better: numerical derivative
                dz = 0.005
                s8p = cosmo.sigma(8.0/h, z + dz)
                if z > dz:
                    s8m = cosmo.sigma(8.0/h, z - dz)
                    f = -(1+z) * (s8p - s8m) / (2 * dz * s8z)
                else:
                    f = -(1+z) * (s8p - s8z) / (dz * s8z)
                fs8 = f * s8z
                results.append((z, fs8, s8z, f))
            except Exception as e:
                results.append((z, float('nan'), float('nan'), float('nan')))

        cosmo.struct_cleanup()
        cosmo.empty()

        print(f"\n{'='*70}")
        print(f"{label}: sigma8(z=0) = {s8_0:.4f}")
        print(f"{'='*70}")
        print(f"  {'z':>5s} {'f*s8':>8s} {'s8(z)':>8s} {'f(z)':>8s}")
        print(f"  {'-'*35}")
        for z, fs8, s8z, f in results:
            print(f"  {z:>5.2f} {fs8:>8.4f} {s8z:>8.4f} {f:>8.4f}")
        sys.stdout.flush()
        return results, s8_0
    except Exception as e:
        print(f"{label}: FAILED analysis - {str(e)[:80]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# 1. Standard LCDM
base = {
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
    'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
    'P_k_max_1/Mpc': 10.0, 'z_max_pk': 3.0,
}
lcdm_res = compute_fsigma8(base, "LCDM")

# SMG base params
smg_base = {
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
    'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
    'P_k_max_1/Mpc': 10.0, 'z_max_pk': 3.0,
    'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
    'expansion_model': 'lcdm', 'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
}

# 2. propto_omega models
for cM in [0.0002, 0.0005, 0.001, 0.0015]:
    p = smg_base.copy()
    cB = -cM / 2.0
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'0.0, {cB}, {cM}, 0.0, 1.0'
    compute_fsigma8(p, f"propto_omega cM={cM}")

# 3. propto_scale models
for cM in [0.0001, 0.0003, 0.0005, 0.001]:
    p = smg_base.copy()
    cB = -cM / 2.0
    p['gravity_model'] = 'propto_scale'
    p['parameters_smg'] = f'0.0, {cB}, {cM}, 0.0, 1.0'
    compute_fsigma8(p, f"propto_scale cM={cM}")

# Observational data
print(f"\n{'='*70}")
print("OBSERVATIONAL f*sigma8 DATA (RSD compilation)")
print(f"{'='*70}")
obs_data = [
    (0.02, 0.428, 0.048, "ALFALFA (2018)"),
    (0.10, 0.370, 0.130, "6dFGS (2012)"),
    (0.15, 0.490, 0.145, "2MTF (2017)"),
    (0.38, 0.497, 0.045, "BOSS LOWZ (2017)"),
    (0.51, 0.458, 0.038, "BOSS CMASS (2017)"),
    (0.61, 0.436, 0.034, "BOSS CMASS (2017)"),
    (0.77, 0.490, 0.180, "VIPERS (2018)"),
    (0.85, 0.450, 0.110, "DESI LRG (2024)"),
    (1.40, 0.482, 0.116, "FastSound (2016)"),
]
print(f"  {'z':>5s} {'f*s8':>8s} {'err':>6s} {'Survey':>20s}")
print(f"  {'-'*45}")
for z, fs8, err, survey in obs_data:
    print(f"  {z:>5.2f} {fs8:>8.3f} {err:>6.3f} {survey:>20s}")

# Chi2 comparison
if lcdm_res:
    lcdm_results, _ = lcdm_res
    print(f"\n{'='*70}")
    print("CHI2 COMPARISON: f*sigma8")
    print(f"{'='*70}")
    # Only use data points where we have model predictions
    for model_label in ["LCDM"]:
        chi2 = 0
        n = 0
        for z_obs, fs8_obs, err, survey in obs_data:
            # Find nearest model z
            best_match = None
            for z_m, fs8_m, _, _ in lcdm_results:
                if abs(z_m - z_obs) < 0.05:
                    best_match = fs8_m
                    break
            if best_match is not None:
                chi2 += ((best_match - fs8_obs) / err) ** 2
                n += 1
        if n > 0:
            print(f"  {model_label}: chi2 = {chi2:.2f} ({n} points, chi2/n = {chi2/n:.2f})")

sys.stdout.flush()

#!/usr/bin/env python3
"""
Joint optimization of (c_M, A_s, n_s) for propto_omega and propto_scale.
Goal: Find parameters where chi2 < LCDM AND sigma8 close to 0.811.
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

def compute(gravity_model, cM, h=0.6732, omch2=0.1200, omb=0.02237,
            As=2.1e-9, ns=0.9649, tau=0.0544):
    cB = -cM / 2.0
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': h, 'T_cmb': 2.7255,
        'omega_b': omb, 'omega_cdm': omch2,
        'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
        'tau_reio': tau, 'A_s': As, 'n_s': ns,
        'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
        'gravity_model': gravity_model,
        'parameters_smg': f'0.0, {cB}, {cM}, 0.0, 1.0',
        'expansion_model': 'lcdm', 'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    }
    cosmo.set(params)
    try:
        cosmo.compute()
    except:
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
        return chi2, s8, th
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# LCDM reference
print("=" * 100)
print("LCDM reference (A_s=2.1e-9, n_s=0.9649)")
cosmo = Class()
cosmo.set({
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
    'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
})
cosmo.compute()
cls_ref = cosmo.lensed_cl(2500)
ell = np.arange(2, 2501)
T = 2.7255e6
Dl_ref = cls_ref['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
s8_ref = cosmo.sigma8()
th_ref = cosmo.theta_s_100() / 100.0
cosmo.struct_cleanup(); cosmo.empty()
res_ref = (np.interp(planck_ell, ell, Dl_ref)[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
chi2_ref = float(np.sum(res_ref**2))
print(f"LCDM: chi2={chi2_ref:.1f}, s8={s8_ref:.4f}, th={th_ref:.6f}")
print("=" * 100)
sys.stdout.flush()

# ============================================================
# SCAN 1: propto_omega, CFM relation, varying (cM, A_s)
# ============================================================
print("\nSCAN 1: propto_omega, alpha_B = -cM/2")
print(f"{'cM':>8s} {'A_s':>10s} | {'chi2':>8s} {'dchi2':>8s} {'chi2/n':>7s} {'s8':>7s} {'ds8%':>7s} {'theta':>10s}")
print("-" * 85)
sys.stdout.flush()

cM_vals = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
As_vals = [1.5e-9, 1.7e-9, 1.8e-9, 1.9e-9, 2.0e-9, 2.1e-9, 2.2e-9]

best_po = None
for cM in cM_vals:
    for As in As_vals:
        r = compute('propto_omega', cM, As=As)
        if r is None:
            print(f"{cM:>8.4f} {As:>10.2e} | {'FAIL':>8s}")
            sys.stdout.flush()
            continue
        chi2, s8, th = r
        dchi2 = chi2 - chi2_ref
        ds8 = (s8 - 0.811) / 0.811 * 100
        marker = ""
        if dchi2 < 0 and abs(ds8) < 5:
            marker = " ***"
        elif dchi2 < 0 and abs(ds8) < 10:
            marker = " **"
        elif dchi2 < 0:
            marker = " *"
        print(f"{cM:>8.4f} {As:>10.2e} | {chi2:>8.1f} {dchi2:>+8.1f} {chi2/npts:>7.3f} {s8:>7.4f} {ds8:>+7.1f} {th:>10.6f}{marker}")
        sys.stdout.flush()
        if best_po is None or chi2 < best_po[2]:
            best_po = (cM, As, chi2, s8, th)

if best_po:
    print(f"\nBest propto_omega: cM={best_po[0]:.4f}, As={best_po[1]:.2e}")
    print(f"  chi2={best_po[2]:.1f} (dchi2={best_po[2]-chi2_ref:+.1f}), s8={best_po[3]:.4f}, th={best_po[4]:.6f}")

# ============================================================
# SCAN 2: propto_scale, CFM relation, varying (cM, A_s)
# ============================================================
print(f"\n{'='*100}")
print("SCAN 2: propto_scale, alpha_B = -cM/2")
print(f"{'cM':>8s} {'A_s':>10s} | {'chi2':>8s} {'dchi2':>8s} {'chi2/n':>7s} {'s8':>7s} {'ds8%':>7s} {'theta':>10s}")
print("-" * 85)
sys.stdout.flush()

cM_vals_ps = [0.0003, 0.0005, 0.001, 0.0015, 0.002]
best_ps = None

for cM in cM_vals_ps:
    for As in As_vals:
        r = compute('propto_scale', cM, As=As)
        if r is None:
            print(f"{cM:>8.4f} {As:>10.2e} | {'FAIL':>8s}")
            sys.stdout.flush()
            continue
        chi2, s8, th = r
        dchi2 = chi2 - chi2_ref
        ds8 = (s8 - 0.811) / 0.811 * 100
        marker = ""
        if dchi2 < 0 and abs(ds8) < 5:
            marker = " ***"
        elif dchi2 < 0 and abs(ds8) < 10:
            marker = " **"
        elif dchi2 < 0:
            marker = " *"
        print(f"{cM:>8.4f} {As:>10.2e} | {chi2:>8.1f} {dchi2:>+8.1f} {chi2/npts:>7.3f} {s8:>7.4f} {ds8:>+7.1f} {th:>10.6f}{marker}")
        sys.stdout.flush()
        if best_ps is None or chi2 < best_ps[2]:
            best_ps = (cM, As, chi2, s8, th)

if best_ps:
    print(f"\nBest propto_scale: cM={best_ps[0]:.4f}, As={best_ps[1]:.2e}")
    print(f"  chi2={best_ps[2]:.1f} (dchi2={best_ps[2]-chi2_ref:+.1f}), s8={best_ps[3]:.4f}, th={best_ps[4]:.6f}")

# ============================================================
# SCAN 3: constant_alphas, alpha_B=-aM (std f(R)), varying (aM, A_s)
# ============================================================
print(f"\n{'='*100}")
print("SCAN 3: constant_alphas, alpha_B = -alpha_M (std f(R))")
print(f"{'aM':>8s} {'A_s':>10s} | {'chi2':>8s} {'dchi2':>8s} {'chi2/n':>7s} {'s8':>7s} {'ds8%':>7s} {'theta':>10s}")
print("-" * 85)
sys.stdout.flush()

aM_vals = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
best_ca = None

for aM in aM_vals:
    for As in As_vals:
        aB = -aM  # Standard f(R) relation!
        cosmo = Class()
        params = {
            'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
            'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
            'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
            'tau_reio': 0.0544, 'A_s': As, 'n_s': 0.9649,
            'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
            'gravity_model': 'constant_alphas',
            'parameters_smg': f'0.0, {aB}, {aM}, 0.0, 1.0',
            'expansion_model': 'lcdm', 'expansion_smg': '0.5',
            'method_qs_smg': 'quasi_static',
            'skip_stability_tests_smg': 'yes',
            'pert_qs_ic_tolerance_test_smg': -1,
        }
        cosmo.set(params)
        try:
            cosmo.compute()
            cls = cosmo.lensed_cl(2500)
            ell = np.arange(2, 2501)
            T = 2.7255e6
            Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
            s8 = cosmo.sigma8()
            th = cosmo.theta_s_100() / 100.0
            cosmo.struct_cleanup(); cosmo.empty()
            model_Dl = np.interp(planck_ell, ell, Dl)
            res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
            chi2 = float(np.sum(res**2))
            dchi2 = chi2 - chi2_ref
            ds8 = (s8 - 0.811) / 0.811 * 100
            marker = ""
            if dchi2 < 0 and abs(ds8) < 5:
                marker = " ***"
            elif dchi2 < 0 and abs(ds8) < 10:
                marker = " **"
            elif dchi2 < 0:
                marker = " *"
            print(f"{aM:>8.4f} {As:>10.2e} | {chi2:>8.1f} {dchi2:>+8.1f} {chi2/npts:>7.3f} {s8:>7.4f} {ds8:>+7.1f} {th:>10.6f}{marker}")
            if best_ca is None or chi2 < best_ca[2]:
                best_ca = (aM, As, chi2, s8, th)
        except Exception as e:
            print(f"{aM:>8.4f} {As:>10.2e} | FAIL: {str(e)[:60]}")
            try: cosmo.struct_cleanup(); cosmo.empty()
            except: pass
        sys.stdout.flush()

if best_ca:
    print(f"\nBest constant_alphas (fR): aM={best_ca[0]:.4f}, As={best_ca[1]:.2e}")
    print(f"  chi2={best_ca[2]:.1f} (dchi2={best_ca[2]-chi2_ref:+.1f}), s8={best_ca[3]:.4f}, th={best_ca[4]:.6f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*100}")
print("FINAL COMPARISON")
print(f"{'='*100}")
print(f"LCDM:            chi2={chi2_ref:.1f}, sigma8={s8_ref:.4f}")
if best_po:
    ds8 = (best_po[3]-0.811)/0.811*100
    print(f"propto_omega:    chi2={best_po[2]:.1f} ({best_po[2]-chi2_ref:+.1f}), sigma8={best_po[3]:.4f} ({ds8:+.1f}%), cM={best_po[0]:.4f}, As={best_po[1]:.2e}")
if best_ps:
    ds8 = (best_ps[3]-0.811)/0.811*100
    print(f"propto_scale:    chi2={best_ps[2]:.1f} ({best_ps[2]-chi2_ref:+.1f}), sigma8={best_ps[3]:.4f} ({ds8:+.1f}%), cM={best_ps[0]:.4f}, As={best_ps[1]:.2e}")
if best_ca:
    ds8 = (best_ca[3]-0.811)/0.811*100
    print(f"const_alphas fR: chi2={best_ca[2]:.1f} ({best_ca[2]-chi2_ref:+.1f}), sigma8={best_ca[3]:.4f} ({ds8:+.1f}%), aM={best_ca[0]:.4f}, As={best_ca[1]:.2e}")
print(f"\nPoints with *** = chi2 < LCDM AND |dsigma8| < 5% (sigma8 ∈ [0.770, 0.852])")
print(f"Points with **  = chi2 < LCDM AND |dsigma8| < 10%")
print(f"Points with *   = chi2 < LCDM")

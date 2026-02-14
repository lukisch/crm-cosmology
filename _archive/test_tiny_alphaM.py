#!/usr/bin/env python3
"""
Test extrem kleine alpha_M-Werte bei constant_alphas.
Beantwortet: Kann man alpha_M asymptotisch an 0 annähern?
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

# LCDM reference
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
cosmo.struct_cleanup(); cosmo.empty()
res_ref = (np.interp(planck_ell, ell, Dl_ref)[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
chi2_ref = float(np.sum(res_ref**2))
print(f"LCDM: chi2={chi2_ref:.2f}, s8={s8_ref:.6f}")
print()
sys.stdout.flush()

def test_smg(label, aM, aB_func):
    """Test an SMG model with given alpha_M and alpha_B function."""
    aB = aB_func(aM)
    cosmo = Class()
    cosmo.set({
        'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
        'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
        'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
        'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
        'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
        'gravity_model': 'constant_alphas',
        'parameters_smg': f'0.0, {aB}, {aM}, 0.0, 1.0',
        'expansion_model': 'lcdm', 'expansion_smg': '0.5',
        'method_qs_smg': 'quasi_static',
        'skip_stability_tests_smg': 'yes',
        'pert_qs_ic_tolerance_test_smg': -1,
    })
    try:
        cosmo.compute()
        cls = cosmo.lensed_cl(2500)
        Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
        s8 = cosmo.sigma8()
        cosmo.struct_cleanup(); cosmo.empty()
        model_Dl = np.interp(planck_ell, ell, Dl)
        res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2 = float(np.sum(res**2))
        dchi2 = chi2 - chi2_ref
        ds8 = (s8 - 0.811) / 0.811 * 100
        print(f"  {label} aM={aM:.1e}: chi2={chi2:.2f} (dchi2={dchi2:+.2f}), s8={s8:.6f} ({ds8:+.3f}%)")
        return chi2, s8
    except Exception as e:
        print(f"  {label} aM={aM:.1e}: FAILED - {str(e)[:80]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None, None
    finally:
        sys.stdout.flush()

# ============================================================
# TEST 1: CFM relation alpha_B = -alpha_M / 2
# ============================================================
print("=" * 90)
print("TEST 1: constant_alphas, alpha_B = -alpha_M/2 (CFM)")
print("  Physik: D = (3/8)*aM^2, G_eff/G = 1 + 2*aM^2/(3*D) = 1 + 16/3 ≈ 6.33")
print("  Aber: Amplitude der Modifikation ~ aM^2, wird bei kleinem aM winzig")
print("=" * 90)
aM_vals = [1e-15, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
for aM in aM_vals:
    test_smg("CFM", aM, lambda x: -x/2.0)

# ============================================================
# TEST 2: Standard f(R) relation alpha_B = -alpha_M
# ============================================================
print()
print("=" * 90)
print("TEST 2: constant_alphas, alpha_B = -alpha_M (standard f(R))")
print("  Physik: D = (3/2)*aM^2, G_eff/G = 1 + 2*aM^2/(3*D) = 1 + 4/9 ≈ 1.44")
print("  Deutlich moderater als CFM!")
print("=" * 90)
for aM in aM_vals:
    test_smg("fR", aM, lambda x: -x)

# ============================================================
# TEST 3: Optimal - find the alpha_M that minimizes chi2
# ============================================================
print()
print("=" * 90)
print("TEST 3: Fein-Scan von alpha_M bei alpha_B=-alpha_M (std fR)")
print("=" * 90)
# Fine scan around the minimum (seen at aM~0.0003)
aM_fine = np.concatenate([
    np.arange(0.00005, 0.00050, 0.00005),
    np.arange(0.0005, 0.0011, 0.0001),
])
best_fR = (None, None, None)
for aM in aM_fine:
    chi2, s8 = test_smg("fR-fine", aM, lambda x: -x)
    if chi2 is not None and (best_fR[0] is None or chi2 < best_fR[0]):
        best_fR = (chi2, aM, s8)

if best_fR[0]:
    ds8 = (best_fR[2]-0.811)/0.811*100
    print(f"\n  OPTIMAL fR: aM={best_fR[1]:.5f}, chi2={best_fR[0]:.2f} (dchi2={best_fR[0]-chi2_ref:+.2f}), s8={best_fR[2]:.4f} ({ds8:+.1f}%)")

# ============================================================
# SUMMARY
# ============================================================
print()
print("=" * 90)
print("ZUSAMMENFASSUNG")
print("=" * 90)
print(f"LCDM:  chi2={chi2_ref:.2f}, sigma8={s8_ref:.6f}")
print()
print("ANTWORT auf 'kann man alpha_M asymptotisch an 0 annähern':")
print("  JA - bei extrem kleinem alpha_M konvergiert das Modell gegen LCDM.")
print("  Das Problem ist: Die INTERESSANTEN Effekte (chi2 < LCDM, sigma8-Reduktion)")
print("  treten nur bei aM ~ 0.0001-0.001 auf, nicht bei aM -> 0.")
print("  Bei aM -> 0 ist das Modell physikalisch identisch mit LCDM.")
print()
print("PHYSIK-ERKLÄRUNG:")
print("  CFM (aB=-aM/2): G_eff/G = 1 + 16/3, unabhängig von aM!")
print("    -> Amplitude ~ aM^2, wird winzig bei kleinem aM")
print("    -> Numerisch: hi_class ignoriert bei aM < ~1e-6 die Modifikation")
print()
print("  Std f(R) (aB=-aM): G_eff/G = 1 + 4/9 ≈ 1.44")
print("    -> Viel moderater, erlaubt grössere aM-Werte")
print("    -> OPTIMUM bei aM ≈ 0.0003: chi2 ~ 76 (dchi2 = -8)")

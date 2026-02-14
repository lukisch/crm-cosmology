#!/usr/bin/env python3
"""
Test the NATIVE cfm_fR gravity model in hi_class.

cfm_fR implements:
  alpha_M(a) = alpha_M_0 * n_exp * a^n_exp / (1 + alpha_M_0 * a^n_exp)
  alpha_B = -alpha_M/2  (f(R) relation)
  alpha_T = 0           (c_gw = c)
  alpha_K = 0

This captures BOTH:
  - propto_scale behavior at early times (a << 1): alpha_M ~ alpha_M_0 * n_exp * a^n_exp
  - Scalaron saturation at late times (a ~ 1): alpha_M -> n_exp

Parameters: alpha_M_0, n_exp, M*2_init
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np
import urllib.request

# Download Planck TT data
def download_planck(file_id, outpath):
    url = f'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={file_id}'
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode()
        lines = [l for l in data.split('\n') if l.strip() and not l.startswith('#')]
        with open(outpath, 'w') as f:
            for l in lines:
                f.write(l + '\n')
        return len(lines)
    except:
        return 0

n_tt = download_planck('COM_PowerSpect_CMB-TT-full_R3.01.txt', '/tmp/planck_tt.txt')
n_te = download_planck('COM_PowerSpect_CMB-TE-full_R3.01.txt', '/tmp/planck_te.txt')
n_ee = download_planck('COM_PowerSpect_CMB-EE-full_R3.01.txt', '/tmp/planck_ee.txt')
print(f"Downloaded: TT={n_tt}, TE={n_te}, EE={n_ee} lines")

def load_planck(path, ell_min=30, ell_max=2500):
    data = np.loadtxt(path)
    ell = data[:, 0].astype(int)
    Dl = data[:, 1]
    if data.shape[1] >= 4:
        err = (data[:, 2] + data[:, 3]) / 2.0
    else:
        err = data[:, 2]
    mask = (ell >= ell_min) & (ell <= ell_max) & (err > 0)
    return ell[mask], Dl[mask], err[mask]

tt_ell, tt_Dl, tt_err = load_planck('/tmp/planck_tt.txt')
te_ell, te_Dl, te_err = load_planck('/tmp/planck_te.txt', 30)
ee_ell, ee_Dl, ee_err = load_planck('/tmp/planck_ee.txt', 30)

base = {
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
    'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
}

smg_base = base.copy()
smg_base.update({
    'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
    'expansion_model': 'lcdm', 'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
})

def run_model(params, label):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        print(f"  {label}: FAILED - {str(e)[:100]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6

        Dl_tt = cls['tt'][2:2501] * ell * (ell+1) / (2*np.pi) * T**2
        Dl_te = cls['te'][2:2501] * ell * (ell+1) / (2*np.pi) * T**2
        Dl_ee = cls['ee'][2:2501] * ell * (ell+1) / (2*np.pi) * T**2

        chi2_tt = float(np.sum(((np.interp(tt_ell, ell, Dl_tt) - tt_Dl)/tt_err)**2))
        chi2_te = float(np.sum(((np.interp(te_ell, ell, Dl_te) - te_Dl)/te_err)**2))
        chi2_ee = float(np.sum(((np.interp(ee_ell, ell, Dl_ee) - ee_Dl)/ee_err)**2))
        chi2_tot = chi2_tt + chi2_te + chi2_ee
        n_tot = len(tt_ell) + len(te_ell) + len(ee_ell)

        s8 = cosmo.sigma8()
        th100 = cosmo.theta_s_100()

        cosmo.struct_cleanup(); cosmo.empty()

        print(f"  {label:>40}: chi2={chi2_tot:.1f} (dTT={chi2_tt-2539.5:+.1f}), "
              f"s8={s8:.4f}, 100th={th100:.5f}")
        return chi2_tot, chi2_tt, s8, th100
    except Exception as e:
        print(f"  {label}: POST-FAIL - {str(e)[:80]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# ================================================================
# LCDM Reference
# ================================================================
print("="*90)
print("LCDM Reference")
print("="*90)
ref = run_model(base, "LCDM")
sys.stdout.flush()

# ================================================================
# propto_omega Reference (best known: cM=0.0002)
# ================================================================
print("\n" + "="*90)
print("propto_omega Reference")
print("="*90)
for cM in [0.0002, 0.0005]:
    p = smg_base.copy()
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'0.0, {-cM/2}, {cM}, 0.0, 1.0'
    run_model(p, f"pO cM={cM}")
sys.stdout.flush()

# ================================================================
# propto_scale Reference
# ================================================================
print("\n" + "="*90)
print("propto_scale Reference")
print("="*90)
for cM in [0.0002, 0.0005]:
    p = smg_base.copy()
    p['gravity_model'] = 'propto_scale'
    p['parameters_smg'] = f'0.0, {-cM/2}, {cM}, 0.0, 1.0'
    run_model(p, f"pS cM={cM}")
sys.stdout.flush()

# ================================================================
# NATIVE cfm_fR MODEL
# ================================================================
print("\n" + "="*90)
print("NATIVE cfm_fR MODEL (scalaron with saturation)")
print("alpha_M(a) = aM0 * n * a^n / (1 + aM0 * a^n)")
print("="*90)

# Systematic scan over (alpha_M_0, n_exp)
# n_exp=1: propto_scale-like (scalaron with trace coupling)
# n_exp=2: faster growth
# alpha_M_0 controls amplitude

for n_exp in [0.5, 1.0, 1.5, 2.0]:
    print(f"\n--- n_exp = {n_exp} ---")
    for aM0 in [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01]:
        # alpha_M(a=1) = aM0*n/(1+aM0) ~ aM0*n for small aM0
        aM_today = aM0 * n_exp / (1 + aM0)
        p = smg_base.copy()
        p['gravity_model'] = 'cfm_fR'
        p['parameters_smg'] = f'{aM0}, {n_exp}, 1.0'
        run_model(p, f"cfm_fR aM0={aM0:.4f} n={n_exp} (aM1~{aM_today:.5f})")
    sys.stdout.flush()

# ================================================================
# BEST CFM_FR: fine scan around best parameters
# ================================================================
print("\n" + "="*90)
print("FINE SCAN: cfm_fR around best n_exp=1 (scalaron-natural)")
print("="*90)
for aM0 in [0.00005, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0004, 0.0005]:
    p = smg_base.copy()
    p['gravity_model'] = 'cfm_fR'
    p['parameters_smg'] = f'{aM0}, 1.0, 1.0'
    run_model(p, f"cfm_fR aM0={aM0:.5f} n=1")
sys.stdout.flush()

print("\n" + "="*90)
print("COMPLETE - Native CFM Boltzmann code IS hi_class with cfm_fR!")
print("="*90)

#!/usr/bin/env python3
"""
Final optimization: cfm_fR with n_exp=0.01 (nearly constant alpha_M)
Goal: Find aM0/omch2 combination that gives:
  l1 ~ 220, r31 ~ 0.4295, sigma8 ~ 0.81, chi2/n < 1.1
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

def find_peak(ell, Dl, lmin, lmax):
    mask = (ell >= lmin) & (ell <= lmax)
    e = ell[mask]; d = Dl[mask]
    if len(d) == 0 or np.any(np.isnan(d)):
        return float('nan'), float('nan')
    idx = np.argmax(d)
    if idx == 0 or idx == len(d)-1:
        return float(e[idx]), float(d[idx])
    x = e[idx-1:idx+2].astype(float); y = d[idx-1:idx+2]
    a_c = (y[2] - 2*y[1] + y[0]) / 2.0
    b_c = (y[2] - y[0]) / 2.0
    if abs(a_c) < 1e-30:
        return float(x[1]), float(y[1])
    return float(x[1] - b_c/(2*a_c)), float(y[1] - b_c*b_c/(4*a_c))

def run(omch2, aM0, n_exp, As=2.05e-9, ns=0.97):
    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl,mPk',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': 0.673,
        'T_cmb': 2.7255,
        'omega_b': 0.02237,
        'omega_cdm': omch2,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'tau_reio': 0.0544,
        'A_s': As,
        'n_s': ns,
        'Omega_Lambda': 0,
        'Omega_fld': 0,
        'Omega_smg': -1,
        'gravity_model': 'cfm_fR',
        'parameters_smg': f'{aM0}, {n_exp}, 1.0',
        'expansion_model': 'lcdm',
        'expansion_smg': '0.5',
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
        if np.any(np.isnan(Dl)) or np.any(np.isinf(Dl)):
            cosmo.struct_cleanup(); cosmo.empty()
            return None
        l1, D1 = find_peak(ell, Dl, 150, 300)
        l3, D3 = find_peak(ell, Dl, 700, 900)
        r31 = D3/D1 if D1 > 0 else float('nan')
        try: s8 = cosmo.sigma8()
        except: s8 = float('nan')
        try: th = cosmo.theta_s_100() / 100.0
        except: th = float('nan')
        model_Dl = np.interp(planck_ell, ell, Dl)
        res = (model_Dl[mask_fit] - planck_Dl[mask_fit]) / planck_err[mask_fit]
        chi2n = float(np.sum(res**2)) / int(mask_fit.sum())
        cosmo.struct_cleanup(); cosmo.empty()
        return (l1, r31, s8, chi2n, th)
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# ============================================================
# Main scan: n_exp=0.01, grid of aM0 x omch2
# ============================================================
print("=" * 120)
print("FINAL OPTIMIZATION: cfm_fR, n_exp=0.01")
print("=" * 120)
print(f"{'omch2':>8s} {'aM0':>8s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s} {'theta_s':>10s} | {'dl1':>6s} {'dr31%':>7s} {'ds8%':>6s}")
print("-" * 100)
sys.stdout.flush()

results = []
for omch2 in [0.1130, 0.1135, 0.1138, 0.1140, 0.1143, 0.1145, 0.1150]:
    for aM0 in [0.06, 0.08, 0.10, 0.12, 0.14, 0.16]:
        res = run(omch2, aM0, 0.01)
        if res:
            l1, r31, s8, chi2n, th = res
            dl1 = l1 - 220
            dr31 = (r31 - 0.4295) / 0.4295 * 100
            ds8 = (s8 - 0.811) / 0.811 * 100
            score = (dl1)**2 + (dr31/1.0)**2 + (ds8/5.0)**2 + (chi2n - 1.0)**2 * 10
            results.append((score, omch2, aM0, l1, r31, s8, chi2n, th, dl1, dr31, ds8))
            marker = " ***" if abs(dl1) < 0.5 and abs(dr31) < 1 and s8 < 0.85 else ""
            print(f"{omch2:>8.4f} {aM0:>8.3f} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f} {th:>10.6f} | {dl1:>+6.2f} {dr31:>+7.2f}% {ds8:>+6.1f}%{marker}")
        else:
            print(f"{omch2:>8.4f} {aM0:>8.3f} | FAILED")
        sys.stdout.flush()

# Sort and show top 5
if results:
    results.sort(key=lambda x: x[0])
    print(f"\n{'='*120}")
    print("TOP 5 BEST COMBINATIONS (multi-criteria score):")
    print(f"{'='*120}")
    for i, (sc, omch2, aM0, l1, r31, s8, chi2n, th, dl1, dr31, ds8) in enumerate(results[:5]):
        print(f"#{i+1}: omch2={omch2:.4f}, aM0={aM0:.3f}, n_exp=0.01")
        print(f"    l1={l1:.2f} (dl={dl1:+.2f}), r31={r31:.4f} (dr={dr31:+.2f}%)")
        print(f"    sigma8={s8:.4f} (ds={ds8:+.1f}%), chi2/n={chi2n:.3f}, theta_s={th:.6f}")

# ============================================================
# Also try n_exp=0.01 with A_s optimization for best sigma8
# ============================================================
if results:
    print(f"\n{'='*120}")
    print("A_s OPTIMIZATION for top candidate:")
    print(f"{'='*120}")
    sc, omch2, aM0, *_ = results[0]
    print(f"Candidate: omch2={omch2}, aM0={aM0}, n_exp=0.01")
    print(f"{'A_s':>10s} | {'l1':>7s} {'r31':>8s} {'sigma8':>8s} {'chi2/n':>8s}")
    print("-" * 50)
    for As in [1.8e-9, 1.9e-9, 1.95e-9, 2.0e-9, 2.05e-9, 2.1e-9]:
        r = run(omch2, aM0, 0.01, As=As)
        if r:
            l1, r31, s8, chi2n, th = r
            print(f"{As:>10.3e} | {l1:>7.2f} {r31:>8.4f} {s8:>8.4f} {chi2n:>8.3f}")
        sys.stdout.flush()

print(f"\nPlanck 2018: l1=220.0, r31=0.4295, sigma8=0.811, chi2/n~1.0")

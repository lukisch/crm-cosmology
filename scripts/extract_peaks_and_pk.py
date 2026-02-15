#!/usr/bin/env python3
"""
Q7: Extract CMB peak positions (l_1, l_2, l_3, P_2/P_1, P_3/P_1)
N8: Extract P(k) at Lyman-alpha scales (k ~ 0.1-10 h/Mpc)

For LCDM and native cfm_fR models.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

# Planck 2018 base parameters
base = {
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'P_k_max_1/Mpc': 20.0,  # need high k for Lyman-alpha
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


def find_peaks(ell, Dl, n_peaks=7):
    """Find local maxima in D_l spectrum."""
    peaks = []
    for i in range(2, len(Dl) - 2):
        if Dl[i] > Dl[i-1] and Dl[i] > Dl[i+1] and Dl[i] > Dl[i-2] and Dl[i] > Dl[i+2]:
            peaks.append((int(ell[i]), float(Dl[i])))
    return peaks[:n_peaks]


def find_troughs(ell, Dl, n_troughs=7):
    """Find local minima in D_l spectrum."""
    troughs = []
    for i in range(2, len(Dl) - 2):
        if Dl[i] < Dl[i-1] and Dl[i] < Dl[i+1] and Dl[i] < Dl[i-2] and Dl[i] < Dl[i+2]:
            troughs.append((int(ell[i]), float(Dl[i])))
    return troughs[:n_troughs]


def run_and_extract(params, label):
    """Run hi_class and extract peaks + P(k)."""
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        print(f"  {label}: FAILED - {str(e)[:120]}")
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except:
            pass
        return None

    try:
        # === CMB PEAKS (Q7) ===
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6  # muK
        Dl_tt = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2

        peaks = find_peaks(ell, Dl_tt)
        troughs = find_troughs(ell, Dl_tt)

        # === P(k) at Lyman-alpha scales (N8) ===
        h = cosmo.h()
        k_values_hMpc = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # h/Mpc
        pk_values = []
        for kh in k_values_hMpc:
            k_1Mpc = kh * h  # convert h/Mpc to 1/Mpc
            try:
                pk = cosmo.pk(k_1Mpc, 0.0)  # P(k, z=0)
                pk_values.append(pk)
            except:
                pk_values.append(np.nan)

        # Also get P(k) at z=2.3 (typical Lyman-alpha redshift)
        pk_lya = []
        for kh in k_values_hMpc:
            k_1Mpc = kh * h
            try:
                pk = cosmo.pk(k_1Mpc, 2.3)  # P(k, z=2.3)
                pk_lya.append(pk)
            except:
                pk_lya.append(np.nan)

        s8 = cosmo.sigma8()
        th100 = cosmo.theta_s_100()

        cosmo.struct_cleanup()
        cosmo.empty()

        return {
            'label': label,
            'peaks': peaks,
            'troughs': troughs,
            'sigma8': s8,
            'theta_s_100': th100,
            'k_hMpc': k_values_hMpc,
            'pk_z0': pk_values,
            'pk_z23': pk_lya,
        }
    except Exception as e:
        print(f"  {label}: POST-FAIL - {str(e)[:120]}")
        try:
            cosmo.struct_cleanup()
            cosmo.empty()
        except:
            pass
        return None


def print_results(res, ref_res=None):
    """Print peak positions and P(k) comparison."""
    print(f"\n{'='*80}")
    print(f"  {res['label']}")
    print(f"  sigma_8 = {res['sigma8']:.4f}, 100*theta_s = {res['theta_s_100']:.5f}")
    print(f"{'='*80}")

    # Peaks
    print(f"\n  CMB TT Peaks (Q7):")
    print(f"  {'Peak':>6} {'ell':>6} {'D_l [muK^2]':>14}", end='')
    if ref_res:
        print(f" {'ref ell':>10} {'delta_ell':>10} {'D_l ratio':>10}", end='')
    print()

    for i, (l, d) in enumerate(res['peaks']):
        print(f"  P_{i+1:>4} {l:>6} {d:>14.1f}", end='')
        if ref_res and i < len(ref_res['peaks']):
            rl, rd = ref_res['peaks'][i]
            print(f" {rl:>10} {l - rl:>+10} {d/rd:>10.4f}", end='')
        print()

    # Peak ratios
    if len(res['peaks']) >= 3:
        p1 = res['peaks'][0][1]
        p2 = res['peaks'][1][1]
        p3 = res['peaks'][2][1]
        print(f"\n  Peak ratios:  P2/P1 = {p2/p1:.4f},  P3/P1 = {p3/p1:.4f}")
        if ref_res and len(ref_res['peaks']) >= 3:
            rp1 = ref_res['peaks'][0][1]
            rp2 = ref_res['peaks'][1][1]
            rp3 = ref_res['peaks'][2][1]
            print(f"  LCDM ratios:  P2/P1 = {rp2/rp1:.4f},  P3/P1 = {rp3/rp1:.4f}")

    # Troughs
    print(f"\n  CMB TT Troughs:")
    for i, (l, d) in enumerate(res['troughs'][:3]):
        print(f"  T_{i+1:>4} {l:>6} {d:>14.1f}")

    # P(k)
    print(f"\n  Matter Power Spectrum P(k) [Mpc^3/h^3] (N8):")
    print(f"  {'k [h/Mpc]':>12} {'P(k,z=0)':>14} {'P(k,z=2.3)':>14}", end='')
    if ref_res:
        print(f" {'ratio z=0':>12} {'ratio z=2.3':>12}", end='')
    print()
    for i, kh in enumerate(res['k_hMpc']):
        h = 0.6732
        pk0 = res['pk_z0'][i] * h**3  # convert from Mpc^3 to (Mpc/h)^3
        pk23 = res['pk_z23'][i] * h**3
        print(f"  {kh:>12.1f} {pk0:>14.2f} {pk23:>14.2f}", end='')
        if ref_res:
            rpk0 = ref_res['pk_z0'][i] * h**3
            rpk23 = ref_res['pk_z23'][i] * h**3
            print(f" {pk0/rpk0:>12.4f} {pk23/rpk23:>12.4f}", end='')
        print()


# ================================================================
# RUN MODELS
# ================================================================
models = []

# LCDM
print("Running LCDM...")
sys.stdout.flush()
lcdm = run_and_extract(base, "LCDM (Planck 2018)")
if lcdm:
    models.append(lcdm)

# cfm_fR: conservative (n=0.5, aM0=0.0003)
print("Running cfm_fR conservative (n=0.5, aM0=0.0003)...")
sys.stdout.flush()
p = smg_base.copy()
p['gravity_model'] = 'cfm_fR'
p['parameters_smg'] = '0.0003, 0.5, 1.0'
res = run_and_extract(p, "cfm_fR n=0.5, aM0=0.0003 (conservative)")
if res:
    models.append(res)

# cfm_fR: MCMC best-fit (n=0.28, aM0=0.0024)
print("Running cfm_fR MCMC best-fit (n=0.28, aM0=0.0024)...")
sys.stdout.flush()
p = smg_base.copy()
p['gravity_model'] = 'cfm_fR'
p['parameters_smg'] = '0.0024, 0.28, 1.0'
res = run_and_extract(p, "cfm_fR n=0.28, aM0=0.0024 (MCMC best)")
if res:
    models.append(res)

# cfm_fR: scalaron-natural (n=1, aM0=0.0002)
print("Running cfm_fR scalaron-natural (n=1, aM0=0.0002)...")
sys.stdout.flush()
p = smg_base.copy()
p['gravity_model'] = 'cfm_fR'
p['parameters_smg'] = '0.0002, 1.0, 1.0'
res = run_and_extract(p, "cfm_fR n=1.0, aM0=0.0002 (scalaron)")
if res:
    models.append(res)

# cfm_fR: aggressive (n=0.5, aM0=0.001)
print("Running cfm_fR aggressive (n=0.5, aM0=0.001)...")
sys.stdout.flush()
p = smg_base.copy()
p['gravity_model'] = 'cfm_fR'
p['parameters_smg'] = '0.001, 0.5, 1.0'
res = run_and_extract(p, "cfm_fR n=0.5, aM0=0.001 (aggressive)")
if res:
    models.append(res)

# ================================================================
# PRINT RESULTS
# ================================================================
print("\n\n" + "#"*80)
print("#  RESULTS SUMMARY")
print("#"*80)

ref = models[0] if models else None
for m in models:
    print_results(m, ref_res=ref if m != ref else None)

# Planck measured values for comparison
print(f"\n\n{'='*80}")
print("  PLANCK 2018 MEASURED VALUES (for comparison)")
print(f"{'='*80}")
print("  l_1 = 220.0 +/- 0.5")
print("  l_2 = 537.5 +/- 0.7")
print("  l_3 = 810.8 +/- 0.7")
print("  P3/P1 ~ 0.430 (from D_l ratio)")
print("  100*theta_s = 1.04110 +/- 0.00031")
print("  sigma_8 = 0.8111 +/- 0.0060")
print()
print("  Lyman-alpha constraints (eBOSS DR14, Chabanier+ 2019):")
print("  P(k) slope and amplitude at k = 0.1-2 h/Mpc, z ~ 2-4")
print("  Main constraint: shape of P(k)/P_LCDM(k) should be < 10-20% deviation")
print("  at k < 2 h/Mpc; larger deviations allowed at k > 5 h/Mpc")
print()

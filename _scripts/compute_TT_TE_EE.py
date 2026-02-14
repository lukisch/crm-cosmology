#!/usr/bin/env python3
"""
Compute CMB TT+TE+EE chi2 for LCDM and CFM models using hi_class.
Uses Planck 2018 binned TT, TE, EE power spectra.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np

# Download Planck TT data
import urllib.request

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
sys.stdout.flush()

def load_planck(path, ell_min=30, ell_max=2500):
    data = np.loadtxt(path)
    ell = data[:, 0].astype(int)
    Dl = data[:, 1]
    # Error: average of asymmetric errors
    if data.shape[1] >= 4:
        err = (data[:, 2] + data[:, 3]) / 2.0
    else:
        err = data[:, 2]
    mask = (ell >= ell_min) & (ell <= ell_max) & (err > 0)
    return ell[mask], Dl[mask], err[mask]

# Load all spectra
tt_ell, tt_Dl, tt_err = load_planck('/tmp/planck_tt.txt')
print(f"TT: {len(tt_ell)} points, ell={tt_ell[0]}-{tt_ell[-1]}")

try:
    te_ell, te_Dl, te_err = load_planck('/tmp/planck_te.txt', ell_min=30)
    print(f"TE: {len(te_ell)} points, ell={te_ell[0]}-{te_ell[-1]}")
    has_te = True
except:
    print("TE: Failed to load")
    has_te = False

try:
    ee_ell, ee_Dl, ee_err = load_planck('/tmp/planck_ee.txt', ell_min=30)
    print(f"EE: {len(ee_ell)} points, ell={ee_ell[0]}-{ee_ell[-1]}")
    has_ee = True
except:
    print("EE: Failed to load")
    has_ee = False

sys.stdout.flush()

def compute_chi2(params, label):
    cosmo = Class()
    cosmo.set(params)
    try:
        cosmo.compute()
    except Exception as e:
        print(f"{label}: FAILED - {str(e)[:80]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6  # T_cmb in microK

        # TT spectrum
        Dl_tt = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
        model_tt = np.interp(tt_ell, ell, Dl_tt)
        chi2_tt = float(np.sum(((model_tt - tt_Dl) / tt_err)**2))

        chi2_te = 0.0
        n_te_pts = 0
        if has_te:
            Dl_te = cls['te'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
            model_te = np.interp(te_ell, ell, Dl_te)
            chi2_te = float(np.sum(((model_te - te_Dl) / te_err)**2))
            n_te_pts = len(te_ell)

        chi2_ee = 0.0
        n_ee_pts = 0
        if has_ee:
            Dl_ee = cls['ee'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
            model_ee = np.interp(ee_ell, ell, Dl_ee)
            chi2_ee = float(np.sum(((model_ee - ee_Dl) / ee_err)**2))
            n_ee_pts = len(ee_ell)

        chi2_total = chi2_tt + chi2_te + chi2_ee
        n_total = len(tt_ell) + n_te_pts + n_ee_pts

        s8 = cosmo.sigma8()
        cosmo.struct_cleanup(); cosmo.empty()

        print(f"\n{label}:")
        print(f"  TT:  chi2={chi2_tt:.1f} ({len(tt_ell)} pts, chi2/n={chi2_tt/len(tt_ell):.3f})")
        if has_te:
            print(f"  TE:  chi2={chi2_te:.1f} ({n_te_pts} pts, chi2/n={chi2_te/n_te_pts:.3f})")
        if has_ee:
            print(f"  EE:  chi2={chi2_ee:.1f} ({n_ee_pts} pts, chi2/n={chi2_ee/n_ee_pts:.3f})")
        print(f"  TOT: chi2={chi2_total:.1f} ({n_total} pts, chi2/n={chi2_total/n_total:.3f})")
        print(f"  sigma8={s8:.4f}")
        sys.stdout.flush()

        return chi2_tt, chi2_te, chi2_ee, chi2_total, s8
    except Exception as e:
        print(f"{label}: FAILED analysis - {str(e)[:80]}")
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return None

# ================================================================
# 1. LCDM Reference
# ================================================================
base = {
    'output': 'tCl,pCl,lCl,mPk', 'l_max_scalars': 2500, 'lensing': 'yes',
    'h': 0.6732, 'T_cmb': 2.7255, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
    'tau_reio': 0.0544, 'A_s': 2.1e-9, 'n_s': 0.9649,
}
lcdm = compute_chi2(base, "LCDM")

# ================================================================
# 2. CFM propto_omega models
# ================================================================
smg_base = base.copy()
smg_base.update({
    'Omega_Lambda': 0, 'Omega_fld': 0, 'Omega_smg': -1,
    'expansion_model': 'lcdm', 'expansion_smg': '0.5',
    'method_qs_smg': 'quasi_static',
    'skip_stability_tests_smg': 'yes',
    'pert_qs_ic_tolerance_test_smg': -1,
})

for cM in [0.0002, 0.0003, 0.0005, 0.001]:
    p = smg_base.copy()
    cB = -cM / 2.0
    p['gravity_model'] = 'propto_omega'
    p['parameters_smg'] = f'0.0, {cB}, {cM}, 0.0, 1.0'
    compute_chi2(p, f"propto_omega cM={cM}")

# ================================================================
# 3. CFM propto_scale models
# ================================================================
for cM in [0.0001, 0.0003, 0.0005]:
    p = smg_base.copy()
    cB = -cM / 2.0
    p['gravity_model'] = 'propto_scale'
    p['parameters_smg'] = f'0.0, {cB}, {cM}, 0.0, 1.0'
    compute_chi2(p, f"propto_scale cM={cM}")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print("ZUSAMMENFASSUNG TT+TE+EE")
print(f"{'='*70}")
if lcdm:
    print(f"LCDM: TT={lcdm[0]:.1f}, TE={lcdm[1]:.1f}, EE={lcdm[2]:.1f}, TOT={lcdm[3]:.1f}, s8={lcdm[4]:.4f}")
print("Alle Modelle verwenden Planck 2018 best-fit Parameter")
print("Chi2 = sum((model-data)/error)^2, keine Kovarianzmatrix")
sys.stdout.flush()

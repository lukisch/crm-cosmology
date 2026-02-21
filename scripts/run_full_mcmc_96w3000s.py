#!/usr/bin/env python3
"""
EXTENDED MCMC RUN: Native cfm_fR model - 96 Walkers × 3000 Steps
5 parameters: alpha_M_0, n_exp, omega_cdm, ln(10^10 A_s), n_s

Uses TT+TE+EE chi2 against Planck 2018 binned data.
emcee EnsembleSampler with 96 walkers (20x ndim for better coverage).

Starting from converged 48W×5000S results as initial ball.

Estimated runtime: ~80-120 hours on Hetzner CCX33 (8 cores)
Total samples: 96 × 3000 = 288,000 samples
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np
import emcee
import time
import os
from multiprocessing import Pool
import resource

# ================================================================
# RESOURCE LIMITS (prevent OOM)
# ================================================================
# Limit each worker to 4.5 GB
soft, hard = 4500 * 1024 * 1024, 4500 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

# ================================================================
# 1. LOAD PLANCK DATA
# ================================================================
import urllib.request

def download_planck(file_id, outpath):
    url = f'https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID={file_id}'
    try:
        data = urllib.request.urlopen(url, timeout=30).read().decode()
        lines = [l for l in data.split('\n') if l.strip() and not l.startswith('#')]
        with open(outpath, 'w') as f:
            for l in lines: f.write(l + '\n')
        return len(lines)
    except:
        return 0

def load_planck(path, ell_min=30, ell_max=2500):
    data = np.loadtxt(path)
    ell = data[:, 0].astype(int)
    Dl = data[:, 1]
    err = (data[:, 2] + data[:, 3]) / 2.0 if data.shape[1] >= 4 else data[:, 2]
    mask = (ell >= ell_min) & (ell <= ell_max) & (err > 0)
    return ell[mask], Dl[mask], err[mask]

print("Downloading Planck 2018 data...")
download_planck('COM_PowerSpect_CMB-TT-full_R3.01.txt', '/tmp/planck_tt.txt')
download_planck('COM_PowerSpect_CMB-TE-full_R3.01.txt', '/tmp/planck_te.txt')
download_planck('COM_PowerSpect_CMB-EE-full_R3.01.txt', '/tmp/planck_ee.txt')

tt_ell, tt_Dl, tt_err = load_planck('/tmp/planck_tt.txt')
te_ell, te_Dl, te_err = load_planck('/tmp/planck_te.txt', 30)
ee_ell, ee_Dl, ee_err = load_planck('/tmp/planck_ee.txt', 30)
n_data = len(tt_ell) + len(te_ell) + len(ee_ell)
print(f"Data: TT={len(tt_ell)}, TE={len(te_ell)}, EE={len(ee_ell)}, total={n_data}")

# ================================================================
# 2. LIKELIHOOD FUNCTION
# ================================================================
N_EVAL = 0
T_START = time.time()

def log_likelihood(theta):
    """
    Compute log-likelihood for cfm_fR model.
    theta = [alpha_M_0, n_exp, omega_cdm, logAs, n_s]
    """
    global N_EVAL
    N_EVAL += 1

    aM0, n_exp, omega_cdm, logAs, n_s = theta
    As = np.exp(logAs) * 1e-10

    cosmo = Class()
    params = {
        'output': 'tCl,pCl,lCl',
        'l_max_scalars': 2500,
        'lensing': 'yes',
        'h': 0.6732,
        'T_cmb': 2.7255,
        'omega_b': 0.02237,
        'omega_cdm': omega_cdm,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'tau_reio': 0.0544,
        'A_s': As,
        'n_s': n_s,
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
        return -1e30

    try:
        cls = cosmo.lensed_cl(2500)
        ell = np.arange(2, 2501)
        T = 2.7255e6

        Dl_tt = cls['tt'][2:2501] * ell * (ell+1) / (2*np.pi) * T**2
        Dl_te = cls['te'][2:2501] * ell * (ell+1) / (2*np.pi) * T**2
        Dl_ee = cls['ee'][2:2501] * ell * (ell+1) / (2*np.pi) * T**2

        chi2_tt = np.sum(((np.interp(tt_ell, ell, Dl_tt) - tt_Dl) / tt_err)**2)
        chi2_te = np.sum(((np.interp(te_ell, ell, Dl_te) - te_Dl) / te_err)**2)
        chi2_ee = np.sum(((np.interp(ee_ell, ell, Dl_ee) - ee_Dl) / ee_err)**2)

        cosmo.struct_cleanup(); cosmo.empty()
        return -0.5 * (chi2_tt + chi2_te + chi2_ee)
    except:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        return -1e30

# ================================================================
# 3. PRIOR
# ================================================================
# Parameters: alpha_M_0, n_exp, omega_cdm, logAs, n_s
PARAM_NAMES = ['alpha_M_0', 'n_exp', 'omega_cdm', 'logAs', 'n_s']
PARAM_LABELS = [r'$\alpha_{M,0}$', r'$n$', r'$\omega_{cdm}$',
                r'$\ln(10^{10}A_s)$', r'$n_s$']

# Flat priors (generous but physical)
PRIOR_LO = np.array([0.0,    0.1,  0.10,  2.5,  0.90])
PRIOR_HI = np.array([0.003,  2.0,  0.14,  3.5,  1.02])

def log_prior(theta):
    if np.all(theta >= PRIOR_LO) and np.all(theta <= PRIOR_HI):
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# ================================================================
# 4. MCMC SETUP - 96 WALKERS
# ================================================================
ndim = 5
nwalkers = 96  # 20x ndim for excellent coverage

# Starting point: FROM CONVERGED 48W×5000S RESULTS
# Mean values from previous MCMC:
# alpha_M_0 = 0.001273, n_exp = 0.655, omega_cdm = 0.120013
# logAs = 3.044365, n_s = 0.965631
p0_center = np.array([
    0.001273,     # alpha_M_0 (from 48W×5000S mean)
    0.655,        # n_exp (from 48W×5000S mean)
    0.120013,     # omega_cdm
    3.044365,     # logAs
    0.965631,     # n_s
])

# Scatter widths: ~1.5x std from previous run for broader exploration
# Previous stds: [0.000725, 0.403, 0.0003, 0.00194, 0.00241]
scatter = np.array([
    0.0011,    # alpha_M_0 (1.5x previous std)
    0.60,      # n_exp (1.5x previous std)
    0.00045,   # omega_cdm
    0.0029,    # logAs
    0.0036,    # n_s
])

# Initial ball around center
p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    while True:
        proposal = p0_center + scatter * np.random.randn(ndim)
        if np.all(proposal >= PRIOR_LO) and np.all(proposal <= PRIOR_HI):
            p0[i] = proposal
            break

# ================================================================
# 5. RUN MCMC WITH MULTIPROCESSING
# ================================================================
print("\n" + "="*80)
print("EXTENDED MCMC: cfm_fR with 96 walkers × 3000 steps")
print(f"  Walkers: {nwalkers}, Burn-in: 150, Production: 3000")
print(f"  Parameters: {PARAM_NAMES}")
print(f"  Starting at (48W×5000S converged values):")
print(f"    alpha_M_0={p0_center[0]:.6f}, n={p0_center[1]:.3f}")
print(f"    omega_cdm={p0_center[2]:.5f}, logAs={p0_center[3]:.6f}, n_s={p0_center[4]:.6f}")
print(f"  Expected samples: {nwalkers * 3000} = 288,000")
print("="*80)
sys.stdout.flush()

# Use multiprocessing pool (6 cores on Hetzner CCX33 to avoid OOM)
with Pool(processes=6, maxtasksperchild=50) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

    # Burn-in
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting burn-in (150 steps)...")
    sys.stdout.flush()
    state = sampler.run_mcmc(p0, 150, progress=False)
    print(f"[{time.strftime('%H:%M:%S')}] Burn-in complete. "
          f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    print(f"  Evaluations so far: {N_EVAL}, "
          f"Time: {(time.time()-T_START)/60:.1f} min")
    sys.stdout.flush()

    # Check burn-in results
    chain_burnin = sampler.get_chain(flat=True)
    print(f"\nBurn-in statistics:")
    for i, name in enumerate(PARAM_NAMES):
        vals = chain_burnin[:, i]
        print(f"  {name:>12s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")
    sys.stdout.flush()

    sampler.reset()

    # Production with checkpoints
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting production (3000 steps)...")
    sys.stdout.flush()

    n_production = 3000
    checkpoint_interval = 250
    chunk_size = 50

    for chunk_start in range(0, n_production, chunk_size):
        state = sampler.run_mcmc(state, chunk_size, progress=False)
        step = chunk_start + chunk_size
        elapsed = (time.time() - T_START) / 60
        accept = np.mean(sampler.acceptance_fraction)
        best_ll = np.max(sampler.get_log_prob(flat=True))
        print(f"  [{time.strftime('%H:%M:%S')}] Step {step}/{n_production}, "
              f"accept={accept:.3f}, best_logL={best_ll:.1f}, "
              f"evals={N_EVAL}, time={elapsed:.1f}min")
        sys.stdout.flush()

        # Checkpoint every 250 steps
        if step % checkpoint_interval == 0:
            chain_so_far = sampler.get_chain(flat=True)
            logp_so_far = sampler.get_log_prob(flat=True)
            best_idx = np.argmax(logp_so_far)
            best_chi2 = -2 * logp_so_far[best_idx]
            checkpoint_path = f'/home/cfm-cosmology/results/checkpoint_96w_{step}.npz'
            np.savez(checkpoint_path,
                     chain=chain_so_far, log_prob=logp_so_far,
                     param_names=PARAM_NAMES,
                     best_params=chain_so_far[best_idx],
                     best_chi2=best_chi2)
            print(f"    -> Checkpoint saved: {checkpoint_path}")
            sys.stdout.flush()

# ================================================================
# 6. RESULTS
# ================================================================
print("\n" + "="*80)
print("MCMC RESULTS")
print("="*80)

chain = sampler.get_chain(flat=True)
log_prob = sampler.get_log_prob(flat=True)

print(f"\nChain shape: {chain.shape}")
print(f"Total evaluations: {N_EVAL}")
print(f"Total time: {(time.time()-T_START)/60:.1f} min ({(time.time()-T_START)/3600:.1f} hours)")
print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

# Best-fit
best_idx = np.argmax(log_prob)
best_params = chain[best_idx]
best_chi2 = -2 * log_prob[best_idx]

print(f"\nBest-fit (lowest chi2 = {best_chi2:.1f}, dchi2 = {best_chi2 - 6628.8:.1f}):")
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name:>12s} = {best_params[i]:.6f}")

# Marginalized constraints
print(f"\nMarginalized 1D constraints (mean +/- std):")
print(f"{'Parameter':>12s} {'Mean':>12s} {'Std':>10s} {'Median':>12s} {'16%':>10s} {'84%':>10s}")
print("-"*70)
for i, name in enumerate(PARAM_NAMES):
    vals = chain[:, i]
    q16, q50, q84 = np.percentile(vals, [16, 50, 84])
    print(f"  {name:>12s} {np.mean(vals):>12.6f} {np.std(vals):>10.6f} "
          f"{q50:>12.6f} {q16:>10.6f} {q84:>10.6f}")

# Derived parameters
print(f"\nDerived parameters at best-fit:")
As_best = np.exp(best_params[3]) * 1e-10
aM_today = best_params[0] * best_params[1] / (1 + best_params[0])
h = 0.6732
Omega_m = (0.02237 + best_params[2]) / h**2
print(f"  A_s = {As_best:.4e}")
print(f"  alpha_M(a=1) = {aM_today:.6f}")
print(f"  Omega_m = {Omega_m:.4f}")

# Significance of alpha_M_0 > 0
aM0_chain = chain[:, 0]
frac_above_zero = np.mean(aM0_chain > 1e-6)
print(f"\n  P(alpha_M_0 > 0) = {frac_above_zero:.4f}")
if frac_above_zero > 0.5:
    mean_aM0 = np.mean(aM0_chain)
    std_aM0 = np.std(aM0_chain)
    if std_aM0 > 0:
        sigma_detection = mean_aM0 / std_aM0
        print(f"  alpha_M_0 detection significance: {sigma_detection:.2f} sigma")

# Correlation matrix
print(f"\nCorrelation matrix:")
corr = np.corrcoef(chain.T)
print(f"{'':>12s}", end='')
for name in PARAM_NAMES:
    print(f" {name:>10s}", end='')
print()
for i, name in enumerate(PARAM_NAMES):
    print(f"  {name:>12s}", end='')
    for j in range(ndim):
        print(f" {corr[i,j]:>10.3f}", end='')
    print()

# Save chain
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
_save_args = dict(chain=chain, log_prob=log_prob,
                  param_names=PARAM_NAMES,
                  best_params=best_params, best_chi2=best_chi2)
_persistent_path = os.path.join(_project_dir, 'results', 'cfm_fR_mcmc_96w3000s_chain.npz')
np.savez(_persistent_path, **_save_args)
np.savez('/tmp/cfm_fR_mcmc_results.npz', **_save_args)
print(f"\nChain saved to: {_persistent_path}")
print(f"Chain also saved to: /tmp/cfm_fR_mcmc_results.npz")

print(f"\n{'='*80}")
print("MCMC COMPLETE")
print(f"{'='*80}")

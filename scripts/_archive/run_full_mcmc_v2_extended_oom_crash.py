#!/usr/bin/env python3
"""
FULL MCMC RUN: Native cfm_fR model (EXTENDED)
5 parameters: alpha_M_0, n_exp, omega_cdm, ln(10^10 A_s), n_s

Uses TT+TE+EE chi2 against Planck 2018 binned data.
emcee EnsembleSampler with 48 walkers, 5000 production steps.
Parallelized with multiprocessing (12 cores).

Target: 240,000 samples, R-hat < 1.01, ESS > 1000
Estimated runtime: ~16-24 hours with 12 cores
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np
import emcee
import time
import os
import multiprocessing

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
# 4. MCMC SETUP
# ================================================================
ndim = 5
nwalkers = 48  # ~10x ndim (reviewer requirement)

# Starting point: around known best-fit
# Best cfm_fR: aM0=0.0005, n=0.5-1.0, rest = Planck best-fit
p0_center = np.array([
    0.0005,                          # alpha_M_0
    0.75,                            # n_exp (between 0.5 and 1.0)
    0.1200,                          # omega_cdm
    np.log(2.1e-9 * 1e10),          # logAs = 3.044
    0.9649,                          # n_s
])

# Initial ball around center
p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    while True:
        proposal = p0_center + np.array([
            np.random.normal(0, 0.0002),   # aM0
            np.random.normal(0, 0.2),      # n_exp
            np.random.normal(0, 0.005),    # omega_cdm
            np.random.normal(0, 0.05),     # logAs
            np.random.normal(0, 0.005),    # n_s
        ])
        if np.all(proposal >= PRIOR_LO) and np.all(proposal <= PRIOR_HI):
            p0[i] = proposal
            break

# ================================================================
# 5. RUN MCMC
# ================================================================
# Persistent save path (OneDrive-synced)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
_results_dir = os.path.join(_project_dir, 'results')
os.makedirs(_results_dir, exist_ok=True)
_persistent_path = os.path.join(_results_dir, 'cfm_fR_mcmc_chain.npz')

n_production = 5000
chunk_size = 100
checkpoint_interval = 500
n_burn = 200
N_CORES = min(8, multiprocessing.cpu_count())

print("\n" + "="*80)
print("FULL MCMC: cfm_fR with 5 parameters (EXTENDED RUN)")
print(f"  Walkers: {nwalkers}, Burn-in: {n_burn}, Production: {n_production}")
print(f"  Cores: {N_CORES}, Checkpoint every {checkpoint_interval} steps")
print(f"  Parameters: {PARAM_NAMES}")
print(f"  Starting at: aM0={p0_center[0]:.4f}, n={p0_center[1]:.2f}, "
      f"ocdm={p0_center[2]:.4f}, logAs={p0_center[3]:.3f}, ns={p0_center[4]:.4f}")
print(f"  Save path: {_persistent_path}")
print("="*80)
sys.stdout.flush()

pool = multiprocessing.Pool(processes=N_CORES)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

# Burn-in
print(f"\n[{time.strftime('%H:%M:%S')}] Starting burn-in ({n_burn} steps)...")
sys.stdout.flush()
state = sampler.run_mcmc(p0, n_burn, progress=False)
print(f"[{time.strftime('%H:%M:%S')}] Burn-in complete. "
      f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
print(f"  Time: {(time.time()-T_START)/60:.1f} min")
sys.stdout.flush()

# Check burn-in results
chain_burnin = sampler.get_chain(flat=True)
print(f"\nBurn-in statistics:")
for i, name in enumerate(PARAM_NAMES):
    vals = chain_burnin[:, i]
    print(f"  {name:>12s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")
sys.stdout.flush()

sampler.reset()

# Production
print(f"\n[{time.strftime('%H:%M:%S')}] Starting production ({n_production} steps)...")
sys.stdout.flush()

for chunk in range(n_production // chunk_size):
    state = sampler.run_mcmc(state, chunk_size, progress=False)
    step = (chunk + 1) * chunk_size
    elapsed = (time.time() - T_START) / 60
    accept = np.mean(sampler.acceptance_fraction)
    best_ll = np.max(sampler.get_log_prob(flat=True))
    print(f"  [{time.strftime('%H:%M:%S')}] Step {step}/{n_production}, "
          f"accept={accept:.3f}, best_logL={best_ll:.1f}, "
          f"time={elapsed:.1f}min")
    sys.stdout.flush()

    # Checkpoint
    if step % checkpoint_interval == 0:
        _ckpt = dict(
            chain=sampler.get_chain(flat=True),
            log_prob=sampler.get_log_prob(flat=True),
            param_names=PARAM_NAMES,
            step=step,
            acceptance=np.mean(sampler.acceptance_fraction),
        )
        np.savez(_persistent_path.replace('.npz', f'_checkpoint_{step}.npz'), **_ckpt)
        np.savez(_persistent_path, **_ckpt)
        np.savez('/tmp/cfm_fR_mcmc_results.npz', **_ckpt)
        print(f"    -> Checkpoint saved ({step} steps, {sampler.get_chain(flat=True).shape[0]} samples)")
        sys.stdout.flush()

pool.close()
pool.join()

# ================================================================
# 5b. CONVERGENCE DIAGNOSTICS
# ================================================================
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

# Autocorrelation time (emcee built-in)
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\nAutocorrelation times:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:>12s}: tau = {tau[i]:.1f}")
    print(f"\nMean autocorrelation time: {np.mean(tau):.1f}")
    print(f"Chain length / tau: {n_production / np.mean(tau):.1f} (should be >> 50)")
except Exception as e:
    tau = None
    print(f"WARNING: Autocorrelation time estimation failed: {e}")

# Effective Sample Size
chain_3d = sampler.get_chain()  # shape: (n_steps, n_walkers, n_dim)
n_steps_actual = chain_3d.shape[0]
if tau is not None:
    ess = n_steps_actual * nwalkers / np.mean(tau)
    print(f"Effective Sample Size (ESS): {ess:.0f}")
else:
    print("ESS: could not compute (tau unavailable)")

# Gelman-Rubin R-hat (split each walker chain in half)
print(f"\nGelman-Rubin R-hat (split-chain):")
for i, name in enumerate(PARAM_NAMES):
    half = n_steps_actual // 2
    chains_split = []
    for w in range(nwalkers):
        chains_split.append(chain_3d[:half, w, i])
        chains_split.append(chain_3d[half:, w, i])
    chains_split = np.array(chains_split)

    M = len(chains_split)
    N = len(chains_split[0])
    chain_means = np.mean(chains_split, axis=1)
    grand_mean = np.mean(chain_means)
    B = N / (M - 1) * np.sum((chain_means - grand_mean)**2)
    W = np.mean(np.var(chains_split, axis=1, ddof=1))
    var_hat = (N - 1) / N * W + B / N
    R_hat = np.sqrt(var_hat / W) if W > 0 else float('inf')
    status = "OK" if R_hat < 1.01 else "WARNING"
    print(f"  {name:>12s}: R-hat = {R_hat:.4f}  [{status}]")

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
print(f"Total time: {(time.time()-T_START)/60:.1f} min")
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
    # Compute how many sigma alpha_M_0 is above 0
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

# Save final chain
_save_args = dict(chain=chain, log_prob=log_prob,
                  param_names=PARAM_NAMES,
                  best_params=best_params, best_chi2=best_chi2,
                  nwalkers=nwalkers, n_production=n_production,
                  acceptance=np.mean(sampler.acceptance_fraction))
np.savez(_persistent_path, **_save_args)
np.savez('/tmp/cfm_fR_mcmc_results.npz', **_save_args)
print(f"\nChain saved to: {_persistent_path}")
print(f"Chain also saved to: /tmp/cfm_fR_mcmc_results.npz (backward compat)")

# Also save human-readable summary
with open('/tmp/cfm_fR_mcmc_summary.txt', 'w') as f:
    f.write("CFM_FR FULL MCMC RESULTS\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Walkers: {nwalkers}, Steps: {n_production}\n")
    f.write(f"Total samples: {chain.shape[0]}\n")
    f.write(f"Total evaluations: {N_EVAL}\n")
    f.write(f"Runtime: {(time.time()-T_START)/60:.1f} min\n")
    f.write(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}\n\n")
    f.write(f"Best chi2: {best_chi2:.1f} (dchi2 = {best_chi2-6628.8:.1f})\n")
    for i, name in enumerate(PARAM_NAMES):
        vals = chain[:, i]
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        f.write(f"{name}: {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}\n")

print(f"\n{'='*80}")
print("MCMC COMPLETE")
print(f"{'='*80}")

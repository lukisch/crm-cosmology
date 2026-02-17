#!/usr/bin/env python3
"""
MCMC RESUME from checkpoint.

Fixes vs. original run_full_mcmc_extended.py:
  1. maxtasksperchild=50  -- Worker-Prozesse werden nach 50 Evaluationen recycelt
     -> Verhindert Memory Leak durch hi_class C-Bibliothek
  2. 6 statt 8 Kerne      -- Laesst ~6 GB RAM frei fuer OS + Puffer
  3. Checkpoint alle 250   -- Haeufigere Sicherung
  4. Laedt Checkpoint und setzt ab dort fort
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
from classy import Class
import numpy as np
import emcee
import time
import os
import multiprocessing
import resource
import gc

# ================================================================
# 0. MEMORY LIMIT PER WORKER (Safety Net)
# ================================================================
def limit_memory():
    """Setze Soft-Limit auf 4.5 GB pro Worker-Prozess."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (4_500_000_000, hard))

def worker_init():
    """Initializer fuer jeden neuen Worker-Prozess."""
    limit_memory()

# ================================================================
# 1. LOAD PLANCK DATA
# ================================================================
import urllib.request

def download_planck(file_id, outpath):
    if os.path.exists(outpath) and os.path.getsize(outpath) > 100:
        return sum(1 for _ in open(outpath))
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

print("Loading Planck 2018 data...")
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
    except Exception:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        del cosmo
        gc.collect()
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

        cosmo.struct_cleanup()
        cosmo.empty()
        del cosmo
        gc.collect()
        return -0.5 * (chi2_tt + chi2_te + chi2_ee)
    except Exception:
        try: cosmo.struct_cleanup(); cosmo.empty()
        except: pass
        del cosmo
        gc.collect()
        return -1e30

# ================================================================
# 3. PRIOR
# ================================================================
PARAM_NAMES = ['alpha_M_0', 'n_exp', 'omega_cdm', 'logAs', 'n_s']
PARAM_LABELS = [r'$\alpha_{M,0}$', r'$n$', r'$\omega_{cdm}$',
                r'$\ln(10^{10}A_s)$', r'$n_s$']

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
# 4. LOAD CHECKPOINT
# ================================================================
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
_results_dir = os.path.join(_project_dir, 'results')
os.makedirs(_results_dir, exist_ok=True)
_persistent_path = os.path.join(_results_dir, 'cfm_fR_mcmc_chain.npz')

# Suche den neuesten Checkpoint
checkpoint_files = sorted([
    f for f in os.listdir(_results_dir)
    if f.startswith('cfm_fR_mcmc_chain_checkpoint_') and f.endswith('.npz')
])

if not checkpoint_files:
    print("FEHLER: Kein Checkpoint gefunden!")
    sys.exit(1)

latest_ckpt = os.path.join(_results_dir, checkpoint_files[-1])
print(f"Loading checkpoint: {latest_ckpt}")

ckpt = np.load(latest_ckpt, allow_pickle=True)
chain_prev = ckpt['chain']       # flat chain: (nwalkers*steps, ndim)
log_prob_prev = ckpt['log_prob'] # flat log_prob: (nwalkers*steps,)
start_step = int(ckpt['step'])
prev_acceptance = float(ckpt['acceptance'])

ndim = 5
nwalkers = 48

print(f"Checkpoint: Step {start_step}, {len(chain_prev)} samples, acceptance={prev_acceptance:.3f}")

# Letzte Walker-Positionen extrahieren (letzte 48 Eintraege der flat chain)
last_positions = chain_prev[-nwalkers:]  # shape: (48, 5)
last_log_probs = log_prob_prev[-nwalkers:]

print(f"Resuming from {nwalkers} walker positions:")
for i, name in enumerate(PARAM_NAMES):
    vals = last_positions[:, i]
    print(f"  {name:>12s}: {np.mean(vals):.6f} +/- {np.std(vals):.6f}")

# ================================================================
# 5. RESUME MCMC
# ================================================================
n_production_total = 5000
n_remaining = n_production_total - start_step
chunk_size = 100
checkpoint_interval = 250  # Haeufiger als vorher (war 500)
N_CORES = 6  # Reduziert von 8 -> laesst ~6 GB RAM frei

print(f"\n{'='*80}")
print(f"MCMC RESUME: cfm_fR (continuing from step {start_step})")
print(f"  Remaining steps: {n_remaining}")
print(f"  Walkers: {nwalkers}, Cores: {N_CORES}")
print(f"  maxtasksperchild: 50 (Worker-Recycling gegen Memory Leak)")
print(f"  Checkpoint every {checkpoint_interval} steps")
print(f"  Save path: {_persistent_path}")
print(f"{'='*80}")
sys.stdout.flush()

# FIX 1: maxtasksperchild recycelt Worker nach 50 Evaluationen
#         -> verhindert unbegrenztes RAM-Wachstum durch hi_class C-Leak
# FIX 2: worker_init setzt Memory-Limit pro Prozess (4.5 GB)
pool = multiprocessing.Pool(
    processes=N_CORES,
    maxtasksperchild=50,
    initializer=worker_init,
)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)

# Initiale State aus Checkpoint-Positionen
from emcee.state import State
state = State(last_positions, log_prob=last_log_probs)

print(f"\n[{time.strftime('%H:%M:%S')}] Starting production (steps {start_step+1}-{n_production_total})...")
sys.stdout.flush()

# Sammle vorherige Chain fuer den finalen Output
all_chain_parts = [chain_prev]
all_logprob_parts = [log_prob_prev]

for chunk in range(n_remaining // chunk_size):
    state = sampler.run_mcmc(state, chunk_size, progress=False)
    step = start_step + (chunk + 1) * chunk_size
    elapsed = (time.time() - T_START) / 60
    accept = np.mean(sampler.acceptance_fraction)
    best_ll_new = np.max(sampler.get_log_prob(flat=True))
    best_ll_prev = np.max(log_prob_prev)
    best_ll = max(best_ll_new, best_ll_prev)

    # Memory-Status
    import psutil
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / (1024**3)
    mem_avail_gb = mem.available / (1024**3)

    print(f"  [{time.strftime('%H:%M:%S')}] Step {step}/{n_production_total}, "
          f"accept={accept:.3f}, best_logL={best_ll:.1f}, "
          f"time={elapsed:.1f}min, "
          f"RAM={mem_used_gb:.1f}/{mem.total/(1024**3):.1f}GB (free={mem_avail_gb:.1f}GB)")
    sys.stdout.flush()

    # Checkpoint
    if (chunk + 1) * chunk_size % checkpoint_interval == 0:
        new_chain = sampler.get_chain(flat=True)
        new_logprob = sampler.get_log_prob(flat=True)

        # Kombiniere alte + neue Chain
        combined_chain = np.vstack([chain_prev, new_chain])
        combined_logprob = np.concatenate([log_prob_prev, new_logprob])

        _ckpt = dict(
            chain=combined_chain,
            log_prob=combined_logprob,
            param_names=PARAM_NAMES,
            step=step,
            acceptance=accept,
        )
        ckpt_file = _persistent_path.replace('.npz', f'_checkpoint_{step}.npz')
        np.savez(ckpt_file, **_ckpt)
        np.savez(_persistent_path, **_ckpt)
        np.savez('/tmp/cfm_fR_mcmc_results.npz', **_ckpt)
        print(f"    -> Checkpoint saved (step {step}, {len(combined_chain)} total samples)")
        sys.stdout.flush()

pool.close()
pool.join()

# ================================================================
# 5b. CONVERGENCE DIAGNOSTICS
# ================================================================
print(f"\n{'='*80}")
print("CONVERGENCE DIAGNOSTICS")
print(f"{'='*80}")

# Neue Chain aus Resume-Lauf
new_chain_3d = sampler.get_chain()  # (n_steps, nwalkers, ndim)
n_steps_new = new_chain_3d.shape[0]

try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\nAutocorrelation times (resume segment only, {n_steps_new} steps):")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:>12s}: tau = {tau[i]:.1f}")
    print(f"\nMean autocorrelation time: {np.mean(tau):.1f}")
    print(f"Chain length / tau: {n_steps_new / np.mean(tau):.1f} (should be >> 50)")
except Exception as e:
    tau = None
    print(f"WARNING: Autocorrelation time estimation failed: {e}")

if tau is not None:
    ess = n_steps_new * nwalkers / np.mean(tau)
    print(f"Effective Sample Size (ESS): {ess:.0f}")

# Gelman-Rubin R-hat
print(f"\nGelman-Rubin R-hat (split-chain, resume segment):")
for i, name in enumerate(PARAM_NAMES):
    half = n_steps_new // 2
    chains_split = []
    for w in range(nwalkers):
        chains_split.append(new_chain_3d[:half, w, i])
        chains_split.append(new_chain_3d[half:, w, i])
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
# 6. FINAL RESULTS
# ================================================================
print(f"\n{'='*80}")
print("MCMC RESULTS (combined: checkpoint + resume)")
print(f"{'='*80}")

new_chain_flat = sampler.get_chain(flat=True)
new_logprob_flat = sampler.get_log_prob(flat=True)

# Kombiniere alles
chain = np.vstack([chain_prev, new_chain_flat])
log_prob_all = np.concatenate([log_prob_prev, new_logprob_flat])

print(f"\nCombined chain shape: {chain.shape}")
print(f"  Previous (checkpoint): {chain_prev.shape[0]} samples")
print(f"  New (resume): {new_chain_flat.shape[0]} samples")
print(f"Total evaluations (this run): {N_EVAL}")
print(f"Total time (this run): {(time.time()-T_START)/60:.1f} min")
print(f"Acceptance fraction (resume): {np.mean(sampler.acceptance_fraction):.3f}")

# Best-fit
best_idx = np.argmax(log_prob_all)
best_params = chain[best_idx]
best_chi2 = -2 * log_prob_all[best_idx]

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

# Derived
print(f"\nDerived parameters at best-fit:")
As_best = np.exp(best_params[3]) * 1e-10
aM_today = best_params[0] * best_params[1] / (1 + best_params[0])
h = 0.6732
Omega_m = (0.02237 + best_params[2]) / h**2
print(f"  A_s = {As_best:.4e}")
print(f"  alpha_M(a=1) = {aM_today:.6f}")
print(f"  Omega_m = {Omega_m:.4f}")

# Significance
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

# Save
_save_args = dict(chain=chain, log_prob=log_prob_all,
                  param_names=PARAM_NAMES,
                  best_params=best_params, best_chi2=best_chi2,
                  nwalkers=nwalkers, n_production=n_production_total,
                  acceptance=np.mean(sampler.acceptance_fraction),
                  resumed_from_step=start_step)
np.savez(_persistent_path, **_save_args)
np.savez('/tmp/cfm_fR_mcmc_results.npz', **_save_args)
print(f"\nChain saved to: {_persistent_path}")
print(f"Chain also saved to: /tmp/cfm_fR_mcmc_results.npz")

with open('/tmp/cfm_fR_mcmc_summary.txt', 'w') as f:
    f.write("CFM_FR FULL MCMC RESULTS (RESUMED)\n")
    f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"Walkers: {nwalkers}, Total Steps: {n_production_total}\n")
    f.write(f"Resumed from step: {start_step}\n")
    f.write(f"Total samples: {chain.shape[0]}\n")
    f.write(f"Total evaluations (resume): {N_EVAL}\n")
    f.write(f"Runtime (resume): {(time.time()-T_START)/60:.1f} min\n")
    f.write(f"Acceptance: {np.mean(sampler.acceptance_fraction):.3f}\n\n")
    f.write(f"Best chi2: {best_chi2:.1f} (dchi2 = {best_chi2-6628.8:.1f})\n")
    for i, name in enumerate(PARAM_NAMES):
        vals = chain[:, i]
        q16, q50, q84 = np.percentile(vals, [16, 50, 84])
        f.write(f"{name}: {q50:.6f} +{q84-q50:.6f} -{q50-q16:.6f}\n")

print(f"\n{'='*80}")
print("MCMC COMPLETE")
print(f"{'='*80}")

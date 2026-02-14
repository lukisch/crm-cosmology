#!/usr/bin/env python3
"""
Setup and test cobaya MCMC for CFM+constant_alphas model.
Uses hi_class/classy as the theory code with custom likelihood.
This is a simplified chi2 likelihood against Planck TT binned data.
"""
import sys
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
import numpy as np
from cobaya.run import run
from cobaya.log import LoggedError

# ============================================================
# Step 1: Test that cobaya + classy integration works
# ============================================================
print("=" * 80)
print("COBAYA + CLASSY INTEGRATION TEST")
print("=" * 80)

# Load Planck TT data
planck_data = np.loadtxt('/tmp/planck_tt.txt')
planck_ell = planck_data[:, 0].astype(int)
planck_Dl = planck_data[:, 1]
planck_err = (planck_data[:, 2] + planck_data[:, 3]) / 2.0

# Save as global for the likelihood
np.savez('/tmp/planck_tt_data.npz',
         ell=planck_ell, Dl=planck_Dl, err=planck_err)

# ============================================================
# Step 2: Define custom likelihood class
# ============================================================
# We need a simple chi2 likelihood that calls classy directly
# Since cobaya's built-in classy provider may not support hi_class SMG,
# we use a custom likelihood that calls classy internally

from classy import Class

class CFMChi2Likelihood:
    """Custom chi2 likelihood for CFM models against Planck TT."""
    def __init__(self):
        data = np.load('/tmp/planck_tt_data.npz')
        self.planck_ell = data['ell'].astype(int)
        self.planck_Dl = data['Dl']
        self.planck_err = data['err']
        self.mask = (self.planck_ell >= 30) & (self.planck_ell <= 2500) & (self.planck_err > 0)
        self.npts = self.mask.sum()

    def compute_loglike(self, omega_cdm, logAs, n_s, alpha_M):
        """Compute log-likelihood for given parameters."""
        As = np.exp(logAs) * 1e-10  # ln(10^10 A_s)
        alpha_B = -alpha_M / 2.0  # f(R) relation

        cosmo = Class()
        params = {
            'output': 'tCl,pCl,lCl',
            'l_max_scalars': 2500,
            'lensing': 'yes',
            'h': 0.673,
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
            'gravity_model': 'constant_alphas',
            'parameters_smg': f'0.0, {alpha_B}, {alpha_M}, 0.0, 1.0',
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
            return -1e30  # Very bad likelihood if computation fails

        try:
            cls = cosmo.lensed_cl(2500)
            ell = np.arange(2, 2501)
            T = 2.7255e6
            Dl = cls['tt'][2:2501] * ell * (ell + 1) / (2 * np.pi) * T**2
            cosmo.struct_cleanup()
            cosmo.empty()
        except:
            try: cosmo.struct_cleanup(); cosmo.empty()
            except: pass
            return -1e30

        model_Dl = np.interp(self.planck_ell, ell, Dl)
        residual = (model_Dl[self.mask] - self.planck_Dl[self.mask]) / self.planck_err[self.mask]
        chi2 = np.sum(residual**2)
        return -chi2 / 2.0

# ============================================================
# Step 3: Test the likelihood at known good parameters
# ============================================================
print("\nTesting likelihood at known good parameters...")
lik = CFMChi2Likelihood()

# Test at our best-fit point
loglike = lik.compute_loglike(
    omega_cdm=0.1143,
    logAs=np.log(2.05e-9 * 1e10),  # ln(10^10 A_s)
    n_s=0.97,
    alpha_M=0.0005
)
chi2 = -2 * loglike
print(f"  Best CFM point: chi2={chi2:.1f}, chi2/n={chi2/lik.npts:.3f}, loglike={loglike:.1f}")

# Test at LCDM-like point (alpha_M=0)
loglike_lcdm = lik.compute_loglike(
    omega_cdm=0.1200,
    logAs=np.log(2.1e-9 * 1e10),
    n_s=0.9649,
    alpha_M=0.0
)
chi2_lcdm = -2 * loglike_lcdm
print(f"  LCDM-like:      chi2={chi2_lcdm:.1f}, chi2/n={chi2_lcdm/lik.npts:.3f}, loglike={loglike_lcdm:.1f}")

# ============================================================
# Step 4: Quick MCMC with emcee (simpler than cobaya for testing)
# ============================================================
print("\n" + "=" * 80)
print("QUICK MCMC with emcee (4 parameters, 16 walkers, 200 steps)")
print("Parameters: omega_cdm, ln(10^10 A_s), n_s, alpha_M")
print("=" * 80)

import emcee

# Parameter names and priors
param_names = ['omega_cdm', 'logAs', 'n_s', 'alpha_M']
param_labels = [r'$\omega_{cdm}$', r'$\ln(10^{10}A_s)$', r'$n_s$', r'$\alpha_M$']

# Flat priors
prior_lo = np.array([0.10, 2.5, 0.90, 0.0])
prior_hi = np.array([0.13, 3.5, 1.00, 0.005])

def log_prior(theta):
    if np.all(theta >= prior_lo) and np.all(theta <= prior_hi):
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = lik.compute_loglike(*theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

ndim = 4
nwalkers = 16

# Initial positions centered on best-fit
p0_center = np.array([0.1143, np.log(2.05e-9 * 1e10), 0.97, 0.0005])
p0 = p0_center + 1e-4 * np.random.randn(nwalkers, ndim)

# Clip to priors
p0 = np.clip(p0, prior_lo + 1e-6, prior_hi - 1e-6)

print(f"Starting {nwalkers} walkers from omega_cdm={p0_center[0]:.4f}, "
      f"logAs={p0_center[1]:.4f}, n_s={p0_center[2]:.4f}, alpha_M={p0_center[3]:.4f}")
sys.stdout.flush()

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

# Run burn-in
print("Running burn-in (50 steps)...")
sys.stdout.flush()
state = sampler.run_mcmc(p0, 50, progress=False)
sampler.reset()

# Production run
print("Running production (200 steps)...")
sys.stdout.flush()
sampler.run_mcmc(state, 200, progress=False)

# Results
print(f"\nAcceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
chain = sampler.get_chain(flat=True)
print(f"Chain shape: {chain.shape}")

print(f"\n{'Parameter':>15s} {'Mean':>10s} {'Std':>10s} {'Median':>10s}")
print("-" * 50)
for i, (name, label) in enumerate(zip(param_names, param_labels)):
    vals = chain[:, i]
    print(f"{name:>15s} {np.mean(vals):>10.5f} {np.std(vals):>10.5f} {np.median(vals):>10.5f}")

# Convert logAs back to A_s for interpretation
As_chain = np.exp(chain[:, 1]) * 1e-10
print(f"\nDerived: A_s = {np.mean(As_chain):.3e} +/- {np.std(As_chain):.3e}")
print(f"         alpha_M = {np.mean(chain[:, 3]):.5f} +/- {np.std(chain[:, 3]):.5f}")
print(f"         alpha_B = {-np.mean(chain[:, 3])/2:.5f} (f(R) relation)")

# Save chain
np.save('/tmp/cfm_mcmc_chain.npy', chain)
print("\nChain saved to /tmp/cfm_mcmc_chain.npy")

# Best-fit from chain
best_idx = np.argmax(sampler.get_log_prob(flat=True))
best_params = chain[best_idx]
print(f"\nBest-fit from MCMC:")
print(f"  omega_cdm = {best_params[0]:.5f}")
print(f"  A_s       = {np.exp(best_params[1]) * 1e-10:.4e}")
print(f"  n_s       = {best_params[2]:.5f}")
print(f"  alpha_M   = {best_params[3]:.6f}")
print(f"  loglike   = {np.max(sampler.get_log_prob(flat=True)):.1f}")

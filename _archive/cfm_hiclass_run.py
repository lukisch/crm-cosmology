#!/usr/bin/env python3
"""CFM+MOND hi_class perturbation analysis"""
from classy import Class
import numpy as np
import time

t0 = time.time()
lines = []
def out(s=""):
    print(s)
    lines.append(s)

MU_PI = np.sqrt(np.pi)
H0_CFM = 67.3
h_cfm = H0_CFM / 100.0
Ob_cfm = 0.0495
alpha_cfm = 0.695
beta_early = 2.82
a_t = 0.0984
mu_late = MU_PI
mu_early = 1.00
a_mu = 2.55e-4
z_star = 1089.92
a_star = 1.0/(1+z_star)

def mu_of_a(a):
    return mu_late + (mu_early - mu_late)/(1.0+(a/a_mu)**4)
def beta_of_a(a):
    return 2.02 + (beta_early - 2.02)/(1.0+(a/a_t)**4)

mu_zs = mu_of_a(a_star)
Om_geom = alpha_cfm * a_star**(3 - beta_of_a(a_star))
Om_eff = mu_zs * Ob_cfm + Om_geom
ombh2_cfm = Ob_cfm * h_cfm**2
omch2_eff = (Om_eff - Ob_cfm) * h_cfm**2
As_cfm = 3.039e-9
ns_cfm = 0.9638
tau_cfm = 0.074

def find_peaks(ell, Dl):
    pks = {}
    for name, lo, hi in [('1',150,350),('2',400,650),('3',700,1000)]:
        mask = (ell>=lo)&(ell<=hi)
        if np.any(mask) and np.max(Dl[mask])>10:
            idx = np.argmax(Dl[mask])
            pks['l'+name] = int(ell[mask][idx])
            pks['Dl'+name] = Dl[mask][idx]
        else:
            pks['l'+name] = 0; pks['Dl'+name] = 0
    pks['r31'] = pks['Dl3']/pks['Dl1'] if pks['Dl1']>0 else 0
    pks['r21'] = pks['Dl2']/pks['Dl1'] if pks['Dl1']>0 else 0
    return pks

def chi2_cls(Dl1, Dl2, lmin=30, lmax=2000):
    n = min(len(Dl1), len(Dl2), lmax+1)
    d1 = Dl1[lmin:n]; d2 = Dl2[lmin:n]
    mask = (d1>0)&(d2>0)
    if np.sum(mask)<50: return 1e8
    A = np.sum(d2[mask]*d1[mask])/np.sum(d1[mask]**2)
    res = d2[mask]-A*d1[mask]
    ll = np.arange(lmin,n)[mask]
    sigma = np.maximum(np.sqrt(2./(2*ll+1))*d2[mask], 1.)
    return np.sum((res/sigma)**2)

def run_standard(ombh2, omch2, H0, ns, As, tau):
    cosmo = Class()
    cosmo.set({'output':'tCl,pCl,lCl','lensing':'yes','l_max_scalars':2500,
               'omega_b':ombh2,'omega_cdm':omch2,'H0':H0,
               'n_s':ns,'A_s':As,'tau_reio':tau,'T_cmb':2.7255})
    cosmo.compute()
    cls = cosmo.lensed_cl(2500)
    ell = np.arange(len(cls['tt']))
    Dl = cls['tt']*ell*(ell+1)/(2*np.pi)*(2.7255e6)**2
    cosmo.struct_cleanup(); cosmo.empty()
    return ell, Dl

def run_mg(ombh2, omch2, H0, ns, As, tau, aM=0., aB=0., aK=1., aT=0.):
    cosmo = Class()
    cosmo.set({'output':'tCl,pCl,lCl','lensing':'yes','l_max_scalars':2500,
               'omega_b':ombh2,'omega_cdm':omch2,'H0':H0,
               'n_s':ns,'A_s':As,'tau_reio':tau,'T_cmb':2.7255,
               'Omega_Lambda':0,'Omega_fld':0,'Omega_smg':-1,
               'gravity_model':'propto_omega',
               'parameters_smg': '%g, %g, %g, %g, 1.' % (aK, aB, aM, aT),
               'expansion_model':'lcdm','expansion_smg':'0.5',
               'skip_stability_tests_smg':'yes'})
    cosmo.compute()
    cls = cosmo.lensed_cl(2500)
    ell = np.arange(len(cls['tt']))
    Dl = cls['tt']*ell*(ell+1)/(2*np.pi)*(2.7255e6)**2
    cosmo.struct_cleanup(); cosmo.empty()
    return ell, Dl

out("="*70)
out("  CFM+MOND: hi_class PERTURBATIONSANALYSE")
out("="*70)
out("  Om_eff=%.4f, mu(z*)=%.4f" % (Om_eff, mu_zs))
out("  ombh2=%.5f, omch2_eff=%.5f" % (ombh2_cfm, omch2_eff))
out("")

# 1. LCDM Reference
out("--- 1. LCDM REFERENZ ---")
ell_ref, Dl_ref = run_standard(0.02236, 0.1202, 67.36, 0.9649, 2.1e-9, 0.054)
pk_ref = find_peaks(ell_ref, Dl_ref)
out("  l1=%d, l3=%d, Pk3/Pk1=%.4f" % (pk_ref['l1'], pk_ref['l3'], pk_ref['r31']))
out("")

# 2. CFM Effective CDM
out("--- 2. CFM EFFECTIVE CDM (Standard CLASS) ---")
ell_cfm, Dl_cfm = run_standard(ombh2_cfm, omch2_eff, H0_CFM, ns_cfm, As_cfm, tau_cfm)
pk_cfm = find_peaks(ell_cfm, Dl_cfm)
c2_cfm = chi2_cls(Dl_cfm, Dl_ref)
out("  l1=%d, l3=%d, Pk3/Pk1=%.4f" % (pk_cfm['l1'], pk_cfm['l3'], pk_cfm['r31']))
out("  chi2 vs LCDM: %.1f" % c2_cfm)
out("")

# 3. alpha_M scan
out("--- 3. alpha_M SCAN (propto_omega) ---")
out("  %6s  %5s  %5s  %7s  %10s" % ('aM','l1','l3','Pk3/1','chi2'))
out("  "+"-"*40)

best_c2 = 1e30
best_aM = 0
for aM in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    try:
        e, D = run_mg(ombh2_cfm, omch2_eff, H0_CFM, ns_cfm, As_cfm, tau_cfm, aM=aM)
        pk = find_peaks(e, D)
        c2 = chi2_cls(D, Dl_ref)
        tag = " <--best" if c2 < best_c2 else ""
        if c2 < best_c2: best_c2=c2; best_aM=aM
        out("  %6.2f  %5d  %5d  %7.4f  %10.1f%s" % (aM, pk['l1'], pk['l3'], pk['r31'], c2, tag))
    except Exception as ex:
        out("  %6.2f  ERROR: %s" % (aM, str(ex)[:50]))
out("")
out("  Bester alpha_M: %.2f (chi2=%.1f)" % (best_aM, best_c2))
out("")

# 4. alpha_B scan
out("--- 4. alpha_B SCAN ---")
out("  %6s  %5s  %7s  %10s" % ('aB','l1','Pk3/1','chi2'))
out("  "+"-"*35)
best_c2B = 1e30; best_aB = 0
for aB in [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]:
    try:
        e, D = run_mg(ombh2_cfm, omch2_eff, H0_CFM, ns_cfm, As_cfm, tau_cfm, aB=aB)
        pk = find_peaks(e, D)
        c2 = chi2_cls(D, Dl_ref)
        tag = " <--best" if c2 < best_c2B else ""
        if c2 < best_c2B: best_c2B=c2; best_aB=aB
        out("  %6.2f  %5d  %7.4f  %10.1f%s" % (aB, pk['l1'], pk['r31'], c2, tag))
    except Exception as ex:
        out("  %6.2f  ERROR: %s" % (aB, str(ex)[:50]))
out("")
out("  Bester alpha_B: %.2f (chi2=%.1f)" % (best_aB, best_c2B))
out("")

# 5. Combined scan
out("--- 5. KOMBINIERTER alpha_M + alpha_B SCAN ---")
out("  %5s  %5s  %5s  %7s  %10s" % ('aM','aB','l1','Pk3/1','chi2'))
out("  "+"-"*42)
best_c2C = 1e30; best_pars = (0,0)
for aM in [0., 0.1, 0.2, 0.3, 0.5, 1.0]:
    for aB in [-0.2, -0.1, 0., 0.1, 0.2]:
        try:
            e, D = run_mg(ombh2_cfm, omch2_eff, H0_CFM, ns_cfm, As_cfm, tau_cfm, aM=aM, aB=aB)
            pk = find_peaks(e, D)
            c2 = chi2_cls(D, Dl_ref)
            tag = " <--best" if c2 < best_c2C else ""
            if c2 < best_c2C: best_c2C=c2; best_pars=(aM,aB)
            out("  %5.2f  %5.2f  %5d  %7.4f  %10.1f%s" % (aM, aB, pk['l1'], pk['r31'], c2, tag))
        except Exception as ex:
            out("  %5.2f  %5.2f  ERR: %s" % (aM, aB, str(ex)[:40]))
out("")
out("  Bestes Paar: aM=%.2f, aB=%.2f (chi2=%.1f)" % (best_pars[0], best_pars[1], best_c2C))
out("")

# 6. G_eff diagnostics
out("--- 6. G_eff DIAGNOSTIK ---")
aM_b, aB_b = best_pars
cosmo_d = Class()
cosmo_d.set({'output':'tCl,pCl,lCl','lensing':'yes','l_max_scalars':2500,
             'omega_b':ombh2_cfm,'omega_cdm':omch2_eff,'H0':H0_CFM,
             'n_s':ns_cfm,'A_s':As_cfm,'tau_reio':tau_cfm,'T_cmb':2.7255,
             'Omega_Lambda':0,'Omega_fld':0,'Omega_smg':-1,
             'gravity_model':'propto_omega',
             'parameters_smg':'1., %g, %g, 0., 1.' % (aB_b, aM_b),
             'expansion_model':'lcdm','expansion_smg':'0.5',
             'skip_stability_tests_smg':'yes',
             'output_background_smg':'2'})
cosmo_d.compute()
out("  aM=%.2f, aB=%.2f" % (aM_b, aB_b))
out("  %8s  %10s  %10s  %10s" % ('z','G_eff','G_light','slip'))
out("  "+"-"*45)
for z in [0, 0.5, 1, 2, 5, 10, 100, 500, 1090]:
    try:
        ge = cosmo_d.G_eff_at_z_smg(z)
        gl = cosmo_d.G_light_at_z_smg(z)
        sl = cosmo_d.slip_eff_at_z_smg(z)
        out("  %8.0f  %10.6f  %10.6f  %10.6f" % (z, ge, gl, sl))
    except:
        out("  %8.0f  -- not available --" % z)
cosmo_d.struct_cleanup(); cosmo_d.empty()
out("")

# Summary
out("="*70)
out("  ZUSAMMENFASSUNG")
out("="*70)
out("  %30s  %4s  %6s  %8s" % ('Modell','l1','P3/P1','chi2'))
out("  "+"-"*55)
out("  %30s  %4d  %6.4f  %8s" % ('Planck 2018', 220, 0.4295, '---'))
out("  %30s  %4d  %6.4f  %8s" % ('LCDM (CLASS)', pk_ref['l1'], pk_ref['r31'], 'ref'))
out("  %30s  %4d  %6.4f  %8.1f" % ('CFM eff.CDM', pk_cfm['l1'], pk_cfm['r31'], c2_cfm))
e_b, D_b = run_mg(ombh2_cfm, omch2_eff, H0_CFM, ns_cfm, As_cfm, tau_cfm, aM=best_pars[0], aB=best_pars[1])
pk_b = find_peaks(e_b, D_b)
c2_b = chi2_cls(D_b, Dl_ref)
out("  %30s  %4d  %6.4f  %8.1f" % ('CFM + MG (hi_class)', pk_b['l1'], pk_b['r31'], c2_b))
out("")
r31_improvement = (pk_b['r31'] - pk_cfm['r31']) / (0.4295 - pk_cfm['r31']) * 100
out("  Verbesserung Pk3/Pk1: %.4f -> %.4f (%.1f%% der Luecke geschlossen)" % (pk_cfm['r31'], pk_b['r31'], r31_improvement))
out("  Planck Zielwert:      0.4295")
out("  Laufzeit: %.1fs" % (time.time()-t0))

with open('/home/cfm_hiclass_results.txt','w') as f:
    f.write('\n'.join(lines))
print("\n  -> Gespeichert: /home/cfm_hiclass_results.txt")

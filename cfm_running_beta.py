"""
CFM mit laufendem Beta: Kruemmungs-Rueckgabepotential mit Phasenuebergang
=========================================================================
beta_eff(a) = beta_late + (beta_early - beta_late) / (1 + (a/a_t)^n)

Hochkruemmung (z > z_t): beta_eff -> beta_early (materiaeaehnlich)
Tiefkruemmung (z < z_t): beta_eff -> beta_late = 2.02 (geometrisch)

Joint Fit: SN (Pantheon+) + CMB (l_A, R) + BAO (BOSS DR12 + 6dFGS + Ly-alpha)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time, os

# ===============================================================
# Konstanten
# ===============================================================
c_light = 299792.458  # km/s
z_star = 1089.80
z_drag = 1059.94
Omega_r = 9.15e-5
omega_b_BBN = 0.02237

# MCMC-fixierte CFM-Parameter
beta_late_fix = 2.02
k_fsat = 9.81
a_trans_fsat = 0.971

# CMB-Zielwerte (Planck 2018)
lA_planck = 301.471
lA_err = 0.14
R_planck = 1.7502
R_err = 0.0046

# BAO-Daten
BAO_DATA = [
    (0.15, 'DV/rd', 4.466, 0.168),
    (0.38, 'DM/rd', 10.27, 0.15),
    (0.38, 'DH/rd', 25.00, 0.76),
    (0.51, 'DM/rd', 13.38, 0.18),
    (0.51, 'DH/rd', 22.33, 0.58),
    (0.61, 'DM/rd', 15.45, 0.20),
    (0.61, 'DH/rd', 20.75, 0.46),
    (2.334, 'DM/rd', 37.6, 1.1),
    (2.334, 'DH/rd', 8.86, 0.29),
]

# ===============================================================
# SN-Daten laden
# ===============================================================
def load_pantheon():
    path = os.path.join(os.path.dirname(__file__), '_data', 'Pantheon+SH0ES.dat')
    z_sn, mb_sn, mberr_sn = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                z = float(parts[2])       # zHD
                mb = float(parts[8])      # m_b_corr
                mbe = float(parts[9])     # m_b_corr_err_DIAG
            except (ValueError, IndexError):
                continue
            if z > 0.01:
                z_sn.append(z)
                mb_sn.append(mb)
                mberr_sn.append(mbe)
    return np.array(z_sn), np.array(mb_sn), np.array(mberr_sn)

# ===============================================================
# Hilfsfunktionen
# ===============================================================
def f_sat(a):
    x = k_fsat * (a - a_trans_fsat)
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

# ===============================================================
# Kosmologie-Modell
# ===============================================================
class CosmoModel:
    """Basis fuer CFM und LCDM"""
    def __init__(self, H0):
        self.H0 = H0
        h = H0 / 100.0
        self.Omega_b = omega_b_BBN / h**2
        self.Omega_gamma = 2.469e-5 / h**2
        self._dL_interp = None

    def E2(self, a):
        raise NotImplementedError

    def Hz(self, z):
        a = 1.0 / (1.0 + z)
        e2 = self.E2(a)
        if e2 <= 0: return 1e10
        return self.H0 * np.sqrt(e2)

    def d_C(self, z):
        res, _ = quad(lambda zz: 1.0/self.Hz(zz), 0, z, limit=2000)
        return c_light * res

    def r_s(self, z_target):
        Rb_fac = 3.0 * self.Omega_b / (4.0 * self.Omega_gamma)
        def integ(a):
            Rb = Rb_fac * a
            cs = 1.0 / np.sqrt(3.0 * (1.0 + Rb))
            e2 = self.E2(a)
            if e2 <= 0: return 0.0
            return cs / (a**2 * self.H0 * np.sqrt(e2))
        a_t = 1.0 / (1.0 + z_target)
        res, _ = quad(integ, 1e-12, a_t, limit=2000)
        return c_light * res

    def cmb_observables(self):
        """l_A, R, r_s(z*), r_d, d_C(z*)"""
        dc = self.d_C(z_star)
        rs = self.r_s(z_star)
        rd = self.r_s(z_drag)
        la = np.pi * dc / rs if rs > 0 else 1e10
        # R: Omega_m_eff aus E2 bei z=500
        a500 = 1.0 / 501.0
        e2_500 = self.E2(a500)
        Om_eff = max((e2_500 - Omega_r * a500**(-4)) * a500**3, 0.001)
        R = np.sqrt(Om_eff) * dc * self.H0 / c_light
        return la, R, rs, rd, dc

    def _build_dL_interp(self, z_max=2.5, N=500):
        zg = np.linspace(0, z_max, N+1)
        inv_E = np.array([1.0/np.sqrt(max(self.E2(1.0/(1.0+z)), 1e-30)) for z in zg])
        dz = zg[1] - zg[0]
        dC = np.cumsum(np.concatenate(([0], 0.5*(inv_E[:-1]+inv_E[1:])*dz))) * c_light / self.H0
        dL = (1.0 + zg) * dC
        self._dL_interp = interp1d(zg, dL, kind='cubic', fill_value='extrapolate')

    def chi2_SN(self, z_sn, mb_sn, mberr_sn):
        if self._dL_interp is None:
            self._build_dL_interp(z_max=max(z_sn) + 0.1)
        dL = np.maximum(self._dL_interp(z_sn), 1e-10)
        mu_th = 5.0 * np.log10(dL) + 25.0
        delta = mb_sn - mu_th
        w = 1.0 / mberr_sn**2
        return np.sum(delta**2 * w) - (np.sum(delta * w))**2 / np.sum(w)

    def chi2_CMB(self):
        la, R, _, _, _ = self.cmb_observables()
        return ((la - lA_planck)/lA_err)**2 + ((R - R_planck)/R_err)**2

    def chi2_BAO(self):
        _, _, _, rd, _ = self.cmb_observables()
        if rd <= 0: return 1e10
        chi2 = 0.0
        for z_b, otype, oval, oerr in BAO_DATA:
            if otype == 'DM/rd':
                th = self.d_C(z_b) / rd
            elif otype == 'DH/rd':
                th = c_light / (self.Hz(z_b) * rd)
            elif otype == 'DV/rd':
                dC = self.d_C(z_b)
                dH = c_light / self.Hz(z_b)
                th = (z_b * dC**2 * dH)**(1.0/3.0) / rd
            else:
                continue
            chi2 += ((th - oval)/oerr)**2
        return chi2

    def full_chi2(self, z_sn, mb_sn, mberr_sn):
        self._dL_interp = None
        return self.chi2_SN(z_sn, mb_sn, mberr_sn) + self.chi2_CMB() + self.chi2_BAO()


class CFMRunningBeta(CosmoModel):
    def __init__(self, H0, alpha, beta_early, a_t, n_trans=4):
        super().__init__(H0)
        self.alpha = alpha
        self.beta_early = beta_early
        self.a_t = a_t
        self.n_trans = n_trans
        # Closure E2(1)=1
        self.Phi0 = (1.0 - self.Omega_b - Omega_r - alpha) / f_sat(1.0)

    def beta_eff(self, a):
        return beta_late_fix + (self.beta_early - beta_late_fix) / (1.0 + (a/self.a_t)**self.n_trans)

    def E2(self, a):
        if a <= 0: return 1e30
        b = self.beta_eff(a)
        return (self.Omega_b * a**(-3) + Omega_r * a**(-4)
                + self.Phi0 * f_sat(a) + self.alpha * a**(-b))


class LCDMModel(CosmoModel):
    def __init__(self, H0=67.36, Omega_m=0.3153):
        super().__init__(H0)
        self.Omega_m = Omega_m
        self.Omega_L = 1.0 - Omega_m - Omega_r

    def E2(self, a):
        return self.Omega_m * a**(-3) + Omega_r * a**(-4) + self.Omega_L

    def cmb_observables(self):
        dc = self.d_C(z_star)
        rs = self.r_s(z_star)
        rd = self.r_s(z_drag)
        la = np.pi * dc / rs
        R = np.sqrt(self.Omega_m) * dc * self.H0 / c_light
        return la, R, rs, rd, dc


# ===============================================================
# HAUPTPROGRAMM
# ===============================================================
def main():
    t0 = time.time()

    out_dir = os.path.join(os.path.dirname(__file__), '_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Running_Beta_Fit.txt')

    lines = []
    def log(s=''):
        lines.append(s)
        print(s)

    log("  CFM MIT LAUFENDEM BETA: KRUEMMUNGS-RUECKGABEPOTENTIAL")
    log("  " + "="*55)
    log(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log()

    z_sn, mb_sn, mberr_sn = load_pantheon()
    log(f"  Pantheon+ geladen: {len(z_sn)} SNe (z > 0.01)")
    log()

    # ==========================================================
    # SCHRITT 0: LCDM-Referenz
    # ==========================================================
    log("="*70)
    log("  SCHRITT 0: LCDM-REFERENZ")
    log("="*70)

    lcdm = LCDMModel()
    la_L, R_L, rs_L, rd_L, dc_L = lcdm.cmb_observables()
    c2sn_L = lcdm.chi2_SN(z_sn, mb_sn, mberr_sn)
    c2cmb_L = lcdm.chi2_CMB()
    c2bao_L = lcdm.chi2_BAO()
    c2tot_L = c2sn_L + c2cmb_L + c2bao_L

    log(f"  l_A = {la_L:.3f}  R = {R_L:.4f}  r_d = {rd_L:.2f} Mpc")
    log(f"  chi2_SN={c2sn_L:.1f}  chi2_CMB={c2cmb_L:.1f}  chi2_BAO={c2bao_L:.1f}  TOTAL={c2tot_L:.1f}")
    log()

    # ==========================================================
    # SCHRITT 1: Original-CFM Referenz (beta=const=2.02)
    # ==========================================================
    log("="*70)
    log("  SCHRITT 1: ORIGINAL-CFM (beta=const=2.02)")
    log("="*70)

    cfm0 = CFMRunningBeta(67.36, 0.68, 2.02, 0.01)  # beta_early=beta_late => kein Uebergang
    la0, R0, rs0, rd0, dc0 = cfm0.cmb_observables()
    log(f"  l_A = {la0:.3f}  R = {R0:.4f}  r_d = {rd0:.2f}  r_s = {rs0:.2f} Mpc")
    log()

    # ==========================================================
    # SCHRITT 2: GRID-SCAN
    # ==========================================================
    log("="*70)
    log("  SCHRITT 2: GRID-SCAN")
    log("="*70)
    log()

    n_fix = 4
    be_grid = [2.5, 2.7, 2.8, 2.85, 2.9, 3.0, 3.2]
    at_grid = [5e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    al_grid = [0.50, 0.60, 0.68, 0.78, 0.88]
    h0_grid = [60, 63, 65, 67.4, 70, 73, 76]
    N_tot = len(be_grid)*len(at_grid)*len(al_grid)*len(h0_grid)
    log(f"  {len(be_grid)}x{len(at_grid)}x{len(al_grid)}x{len(h0_grid)} = {N_tot} Punkte, n={n_fix}")
    log()

    best_c2 = 1e30
    best_p = None
    results_scan = []
    cnt = 0
    t_s = time.time()

    for be in be_grid:
        for at in at_grid:
            for al in al_grid:
                for h0 in h0_grid:
                    cnt += 1
                    if cnt % 40 == 0:
                        el = time.time() - t_s
                        print(f"  ... {cnt}/{N_tot} ({el:.0f}s)", end='\r')

                    h = h0/100.0
                    Ob = omega_b_BBN / h**2
                    Phi0 = (1.0 - Ob - Omega_r - al) / f_sat(1.0)
                    if Phi0 < 0:
                        continue

                    try:
                        m = CFMRunningBeta(h0, al, be, at, n_fix)
                        # Schnellcheck E2 > 0
                        ok = all(m.E2(a) > 0 for a in [1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1.0])
                        if not ok:
                            continue

                        c2s = m.chi2_SN(z_sn, mb_sn, mberr_sn)
                        m._dL_interp = None
                        la_m, R_m, rs_m, rd_m, dc_m = m.cmb_observables()

                        c2c = ((la_m - lA_planck)/lA_err)**2 + ((R_m - R_planck)/R_err)**2

                        # BAO
                        c2b = 0.0
                        for z_b, ot, ov, oe in BAO_DATA:
                            if ot == 'DM/rd':
                                th = m.d_C(z_b) / rd_m
                            elif ot == 'DH/rd':
                                th = c_light / (m.Hz(z_b) * rd_m)
                            elif ot == 'DV/rd':
                                dC_b = m.d_C(z_b)
                                dH_b = c_light / m.Hz(z_b)
                                th = (z_b * dC_b**2 * dH_b)**(1./3.) / rd_m
                            else:
                                continue
                            c2b += ((th - ov)/oe)**2

                        c2t = c2s + c2c + c2b
                        results_scan.append((be, at, al, h0, c2s, c2c, c2b, c2t, la_m, R_m, rd_m))

                        if c2t < best_c2:
                            best_c2 = c2t
                            best_p = (be, at, al, h0)
                    except Exception:
                        continue

    print(" "*60, end='\r')
    log(f"  Scan fertig: {len(results_scan)} gueltige Punkte in {time.time()-t_s:.1f}s")
    log()

    # Sortieren
    results_scan.sort(key=lambda x: x[7])

    log("  TOP-15 Ergebnisse:")
    log(f"  {'be':>5} {'a_t':>8} {'al':>5} {'H0':>5}  {'X2sn':>8} {'X2cmb':>8} {'X2bao':>8} {'X2tot':>8}  {'l_A':>7} {'R':>7} {'r_d':>7}")
    log("  " + "-"*95)
    for r in results_scan[:15]:
        log(f"  {r[0]:5.2f} {r[1]:8.1e} {r[2]:5.2f} {r[3]:5.1f}"
            f"  {r[4]:8.1f} {r[5]:8.1f} {r[6]:8.1f} {r[7]:8.1f}"
            f"  {r[8]:7.2f} {r[9]:7.4f} {r[10]:7.2f}")
    log()

    # ==========================================================
    # SCHRITT 3: FEINOPTIMIERUNG
    # ==========================================================
    log("="*70)
    log("  SCHRITT 3: FEINOPTIMIERUNG")
    log("="*70)
    log()

    if best_p:
        be0, at0, al0, h00 = best_p
        log(f"  Start: be={be0}, at={at0:.1e}, al={al0}, H0={h00}")
        log(f"  Start chi2 = {best_c2:.1f}")

        def objective(p):
            be, lat, al, h0 = p[0], 10**p[1], p[2], p[3]
            if be < 2.0 or be > 4.0: return 1e15
            if al < 0.1 or al > 0.95: return 1e15
            if h0 < 60 or h0 > 85: return 1e15
            Phi0 = (1.0 - omega_b_BBN/(h0/100)**2 - Omega_r - al) / f_sat(1.0)
            if Phi0 < 0: return 1e15
            try:
                m = CFMRunningBeta(h0, al, be, lat, n_fix)
                if any(m.E2(a) <= 0 for a in [1e-6, 1e-4, 0.01, 0.5, 1.0]):
                    return 1e15
                return m.full_chi2(z_sn, mb_sn, mberr_sn)
            except:
                return 1e15

        x0 = [be0, np.log10(at0), al0, h00]
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 400, 'xatol': 0.005, 'fatol': 0.5,
                               'adaptive': True})

        be_o, at_o, al_o, h0_o = res.x[0], 10**res.x[1], res.x[2], res.x[3]
        log(f"  Optimierung: {res.nfev} Evaluationen, chi2 = {res.fun:.1f}")
        log()

        # Detailauswertung
        m_opt = CFMRunningBeta(h0_o, al_o, be_o, at_o, n_fix)
        la_o, R_o, rs_o, rd_o, dc_o = m_opt.cmb_observables()
        c2s_o = m_opt.chi2_SN(z_sn, mb_sn, mberr_sn)
        m_opt._dL_interp = None
        c2c_o = ((la_o - lA_planck)/lA_err)**2 + ((R_o - R_planck)/R_err)**2
        c2b_o = 0.0
        for z_b, ot, ov, oe in BAO_DATA:
            if ot == 'DM/rd':
                th = m_opt.d_C(z_b) / rd_o
            elif ot == 'DH/rd':
                th = c_light / (m_opt.Hz(z_b) * rd_o)
            elif ot == 'DV/rd':
                dC_b = m_opt.d_C(z_b)
                dH_b = c_light / m_opt.Hz(z_b)
                th = (z_b * dC_b**2 * dH_b)**(1./3.) / rd_o
            else:
                continue
            c2b_o += ((th - ov)/oe)**2
        c2t_o = c2s_o + c2c_o + c2b_o

        log("  OPTIMIERTES ERGEBNIS:")
        log("  " + "="*50)
        log(f"  beta_early = {be_o:.4f}")
        log(f"  beta_late  = {beta_late_fix}")
        log(f"  a_t        = {at_o:.6f}  (z_t = {1/at_o - 1:.1f})")
        log(f"  n_trans    = {n_fix}")
        log(f"  alpha      = {al_o:.4f}")
        log(f"  H0         = {h0_o:.2f} km/s/Mpc")
        log(f"  Omega_b    = {m_opt.Omega_b:.5f}")
        log(f"  Phi0       = {m_opt.Phi0:.4f}")
        log()
        log(f"  {'':20s} {'CFM(b_run)':>12s} {'LCDM':>12s} {'Planck':>12s}")
        log(f"  {'l_A':20s} {la_o:12.3f} {la_L:12.3f} {lA_planck:12.3f}")
        log(f"  {'R':20s} {R_o:12.4f} {R_L:12.4f} {R_planck:12.4f}")
        log(f"  {'r_s(z*) [Mpc]':20s} {rs_o:12.2f} {rs_L:12.2f}")
        log(f"  {'r_d [Mpc]':20s} {rd_o:12.2f} {rd_L:12.2f}")
        log(f"  {'d_C(z*) [Mpc]':20s} {dc_o:12.2f} {dc_L:12.2f}")
        log()
        log(f"  {'':20s} {'CFM':>12s} {'LCDM':>12s}")
        log(f"  {'chi2_SN':20s} {c2s_o:12.1f} {c2sn_L:12.1f}")
        log(f"  {'chi2_CMB':20s} {c2c_o:12.1f} {c2cmb_L:12.1f}")
        log(f"  {'chi2_BAO':20s} {c2b_o:12.1f} {c2bao_L:12.1f}")
        log(f"  {'chi2_TOTAL':20s} {c2t_o:12.1f} {c2tot_L:12.1f}")
        log(f"  {'Delta_chi2':20s} {c2t_o - c2tot_L:12.1f}")
        log()

        # beta_eff Profil
        log("  beta_eff-Profil:")
        log(f"  {'z':>8s}  {'beta_eff':>8s}  {'H_CFM/H_LCDM':>13s}")
        log("  " + "-"*35)
        for z in [0, 0.5, 1, 2, 10, 50, 100, 500, 1090, 5000]:
            a = 1.0/(1.0+z)
            beff = m_opt.beta_eff(a)
            e2c = m_opt.E2(a)
            e2l = lcdm.E2(a)
            rat = np.sqrt(e2c/e2l) if e2l > 0 and e2c > 0 else 0
            log(f"  {z:8.1f}  {beff:8.3f}  {rat:13.4f}")
        log()

        # Omega_m_eff Vergleich
        log("  Effektive Materiedichte:")
        log(f"  {'z':>6s}  {'Om_eff(CFM)':>12s}  {'Om(LCDM)':>10s}")
        log("  " + "-"*35)
        for z in [0, 1, 10, 100, 500, 1090]:
            a = 1.0/(1.0+z)
            e2 = m_opt.E2(a)
            Om_eff = (e2 - Omega_r * a**(-4)) * a**3
            log(f"  {z:6d}  {Om_eff:12.4f}  {0.3153:10.4f}")
        log()

    # ==========================================================
    # SCHRITT 4: n_trans Variation
    # ==========================================================
    log("="*70)
    log("  SCHRITT 4: n_trans VARIATION")
    log("="*70)
    log()

    if best_p:
        log(f"  Bei: be={be_o:.3f}, at={at_o:.4f}, al={al_o:.3f}, H0={h0_o:.1f}")
        log(f"  {'n':>4s}  {'X2sn':>8s} {'X2cmb':>8s} {'X2bao':>8s} {'X2tot':>8s}  {'l_A':>7s} {'R':>7s} {'r_d':>7s}")
        log("  " + "-"*65)
        for nt in [1, 2, 3, 4, 6, 8, 12]:
            try:
                mn = CFMRunningBeta(h0_o, al_o, be_o, at_o, nt)
                la_n, R_n, _, rd_n, _ = mn.cmb_observables()
                c2s_n = mn.chi2_SN(z_sn, mb_sn, mberr_sn)
                mn._dL_interp = None
                c2c_n = ((la_n - lA_planck)/lA_err)**2 + ((R_n - R_planck)/R_err)**2
                c2b_n = 0.0
                for z_b, ot, ov, oe in BAO_DATA:
                    if ot == 'DM/rd':
                        th = mn.d_C(z_b)/rd_n
                    elif ot == 'DH/rd':
                        th = c_light/(mn.Hz(z_b)*rd_n)
                    elif ot == 'DV/rd':
                        dC_b = mn.d_C(z_b)
                        dH_b = c_light / mn.Hz(z_b)
                        th = (z_b * dC_b**2 * dH_b)**(1./3.)/rd_n
                    else: continue
                    c2b_n += ((th-ov)/oe)**2
                log(f"  {nt:4d}  {c2s_n:8.1f} {c2c_n:8.1f} {c2b_n:8.1f} {c2s_n+c2c_n+c2b_n:8.1f}  {la_n:7.2f} {R_n:7.4f} {rd_n:7.2f}")
            except:
                log(f"  {nt:4d}  FEHLER")
        log()

    # ==========================================================
    # ZUSAMMENFASSUNG
    # ==========================================================
    log("="*70)
    log("  ZUSAMMENFASSUNG")
    log("="*70)
    log()

    if best_p:
        z_t = 1.0/at_o - 1 if at_o > 0 else 0
        dc2 = c2t_o - c2tot_L
        log(f"  CFM mit laufendem beta vs LCDM:  Delta_chi2 = {dc2:+.1f}")
        log()
        log(f"  Uebergangs-Rotverschiebung: z_t = {z_t:.0f}")
        log(f"  beta_early = {be_o:.3f}  (Hochkruemmung)")
        log(f"  beta_late  = {beta_late_fix}  (Tiefkruemmung, SN)")
        log()
        if abs(be_o - 3.0) < 0.3:
            log("  *** beta_early ~ 3: Rueckgabepotential mimikt CDM bei z > z_t ***")
        log()
        if dc2 < 10:
            log("  >>> CFM IST KOMPETITIV MIT LCDM! <<<")
        elif dc2 < 100:
            log("  CFM ist moderat schlechter als LCDM")
        else:
            log(f"  CFM bleibt {dc2:.0f} chi2-Einheiten schlechter als LCDM")

    log()
    log(f"  Laufzeit: {time.time()-t0:.1f} Sekunden")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    log(f"  Gespeichert: {out_path}")


if __name__ == '__main__':
    main()

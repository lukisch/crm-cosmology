"""
CFM+MOND Verfeinerter Fit: mu_eff im Bereich 1.3 - 2.2
========================================================
Aufbauend auf cfm_mond_background.py:
- Erster Lauf fand: mu_eff(frei) = 1.96, H0 = 80 (zu hoch)
- mu_eff = 4/3 -> H0 = 57.8 (zu niedrig)
- Sweet Spot vermutlich bei mu ~ 1.5 - 1.7

Dieser Lauf:
1. Feines mu_eff-Profil: Optimierung bei JEDEM festen mu in [1.2, 2.2]
2. Freie Multistart-Optimierung aus verschiedenen Startpunkten
3. Physikalische Analyse: mu_eff(a) als skalenabhaengig?
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import time, os

# ===============================================================
# Konstanten
# ===============================================================
c_light = 299792.458      # km/s
c_SI = 2.998e8            # m/s
z_star = 1089.80
z_drag = 1059.94
Omega_r = 9.15e-5
omega_b_BBN = 0.02237
beta_late_fix = 2.02
k_fsat = 9.81
a_trans_fsat = 0.971

lA_pl = 301.471;  lA_err = 0.14
R_pl  = 1.7502;   R_err  = 0.0046

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

def load_pantheon():
    path = os.path.join(os.path.dirname(__file__), '_data', 'Pantheon+SH0ES.dat')
    z_sn, mb_sn, mberr_sn = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10: continue
            try:
                z = float(parts[2])
                mb = float(parts[8])
                mbe = float(parts[9])
            except: continue
            if z > 0.01:
                z_sn.append(z); mb_sn.append(mb); mberr_sn.append(mbe)
    return np.array(z_sn), np.array(mb_sn), np.array(mberr_sn)

def f_sat(a):
    x = k_fsat * (a - a_trans_fsat)
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

# ===============================================================
# Modelle
# ===============================================================
class CosmoBase:
    def __init__(self, H0, mu_eff=1.0):
        self.H0 = H0
        h = H0/100.0
        self.Omega_b = omega_b_BBN / h**2
        self.Omega_gamma = 2.469e-5 / h**2
        self.mu_eff = mu_eff
        self._dL_interp = None

    def E2(self, a): raise NotImplementedError

    def Hz(self, z):
        a = 1.0/(1.0+z)
        e2 = self.E2(a)
        return self.H0 * np.sqrt(max(e2, 1e-30))

    def d_C(self, z):
        r, _ = quad(lambda zz: 1.0/self.Hz(zz), 0, z, limit=2000)
        return c_light * r

    def r_s(self, z_t):
        Rb_f = 3.0 * self.mu_eff * self.Omega_b / (4.0 * self.Omega_gamma)
        def integ(a):
            Rb = Rb_f * a
            cs = 1.0/np.sqrt(3.0*(1.0+Rb))
            e2 = self.E2(a)
            if e2 <= 0: return 0.0
            return cs/(a**2 * self.H0 * np.sqrt(e2))
        r, _ = quad(integ, 1e-12, 1.0/(1.0+z_t), limit=2000)
        return c_light * r

    def cmb_obs(self):
        dc = self.d_C(z_star)
        rs = self.r_s(z_star)
        rd = self.r_s(z_drag)
        la = np.pi*dc/rs if rs > 0 else 1e10
        a500 = 1.0/501.0
        Om_eff = max((self.E2(a500) - Omega_r*a500**(-4))*a500**3, 0.001)
        R = np.sqrt(Om_eff)*dc*self.H0/c_light
        return la, R, rs, rd, dc

    def _build_dL(self, z_max=2.5, N=500):
        zg = np.linspace(0, z_max, N+1)
        inv_E = np.array([1.0/np.sqrt(max(self.E2(1.0/(1.0+z)),1e-30)) for z in zg])
        dz = zg[1]-zg[0]
        dC = np.cumsum(np.r_[0, 0.5*(inv_E[:-1]+inv_E[1:])*dz]) * c_light/self.H0
        self._dL_interp = interp1d(zg, (1.0+zg)*dC, kind='cubic', fill_value='extrapolate')

    def chi2_SN(self, z_sn, mb, mbe):
        if self._dL_interp is None: self._build_dL(max(z_sn)+0.1)
        dL = np.maximum(self._dL_interp(z_sn), 1e-10)
        mu = 5.0*np.log10(dL)+25.0
        d = mb - mu; w = 1.0/mbe**2
        return np.sum(d**2*w) - (np.sum(d*w))**2/np.sum(w)

    def chi2_BAO(self, rd):
        c2 = 0.0
        for z_b, ot, ov, oe in BAO_DATA:
            if ot == 'DM/rd': th = self.d_C(z_b)/rd
            elif ot == 'DH/rd': th = c_light/(self.Hz(z_b)*rd)
            elif ot == 'DV/rd':
                dC = self.d_C(z_b); dH = c_light/self.Hz(z_b)
                th = (z_b*dC**2*dH)**(1./3.)/rd
            else: continue
            c2 += ((th-ov)/oe)**2
        return c2

    def full_chi2(self, z_sn, mb, mbe):
        self._dL_interp = None
        c2s = self.chi2_SN(z_sn, mb, mbe)
        la, R, _, rd, _ = self.cmb_obs()
        c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
        c2b = self.chi2_BAO(rd)
        return c2s + c2c + c2b

    def detailed_chi2(self, z_sn, mb, mbe):
        self._dL_interp = None
        c2s = self.chi2_SN(z_sn, mb, mbe)
        la, R, rs, rd, dc = self.cmb_obs()
        c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
        c2b = self.chi2_BAO(rd)
        return c2s, c2c, c2b, la, R, rs, rd, dc


class LCDM(CosmoBase):
    def __init__(self, H0=67.36, Om=0.3153):
        super().__init__(H0, mu_eff=1.0)
        self.Om = Om; self.OL = 1.0-Om-Omega_r
    def E2(self, a):
        return self.Om*a**(-3) + Omega_r*a**(-4) + self.OL
    def cmb_obs(self):
        dc = self.d_C(z_star); rs = self.r_s(z_star); rd = self.r_s(z_drag)
        la = np.pi*dc/rs
        R = np.sqrt(self.Om)*dc*self.H0/c_light
        return la, R, rs, rd, dc


class CFM_MOND(CosmoBase):
    def __init__(self, H0, alpha, beta_early, a_t, mu_eff=4.0/3.0,
                 n_trans=4, f_ede=0, a_ede=1e-3, p_ede=6):
        super().__init__(H0, mu_eff)
        self.alpha = alpha
        self.beta_early = beta_early
        self.a_t = a_t
        self.n_trans = n_trans
        self.f_ede = f_ede
        self.a_ede = a_ede
        self.p_ede = p_ede
        self.ede_at_1 = f_ede/(1.0+(1.0/a_ede)**p_ede) if f_ede > 0 and a_ede > 0 else 0
        self.Phi0 = (1.0 - self.mu_eff*self.Omega_b - Omega_r - alpha) / f_sat(1.0)

    def beta_eff(self, a):
        return beta_late_fix + (self.beta_early - beta_late_fix)/(1.0+(a/self.a_t)**self.n_trans)

    def ede(self, a):
        if self.f_ede <= 0: return 0.0
        return self.f_ede/(1.0+(a/self.a_ede)**self.p_ede) - self.ede_at_1

    def E2(self, a):
        if a <= 0: return 1e30
        b = self.beta_eff(a)
        return (self.mu_eff * self.Omega_b * a**(-3)
                + Omega_r * a**(-4)
                + self.Phi0 * f_sat(a)
                + self.alpha * a**(-b)
                + self.ede(a))


# ===============================================================
# HAUPTPROGRAMM
# ===============================================================
def main():
    t0 = time.time()
    out_dir = os.path.join(os.path.dirname(__file__), '_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'CFM_MOND_Refined.txt')

    L = []
    def log(s=''):
        L.append(s); print(s)

    log("  CFM+MOND VERFEINERTER FIT")
    log("  " + "="*55)
    log(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log()

    z_sn, mb_sn, mbe_sn = load_pantheon()
    log(f"  {len(z_sn)} SNe geladen")
    log()

    # LCDM Referenz
    lcdm = LCDM()
    la_L, R_L, rs_L, rd_L, dc_L = lcdm.cmb_obs()
    c2sn_L = lcdm.chi2_SN(z_sn, mb_sn, mbe_sn)
    c2cmb_L = ((la_L-lA_pl)/lA_err)**2 + ((R_L-R_pl)/R_err)**2
    c2bao_L = lcdm.chi2_BAO(rd_L)
    c2tot_L = c2sn_L + c2cmb_L + c2bao_L
    log(f"  LCDM: l_A={la_L:.3f} R={R_L:.4f} r_d={rd_L:.2f} chi2={c2tot_L:.1f}")
    log()

    n_fix = 4
    p_ede_fix = 6

    # ===========================================================
    # SCHRITT 1: PROFIL - Optimierung bei JEDEM festen mu_eff
    # ===========================================================
    log("="*70)
    log("  SCHRITT 1: mu_eff-PROFIL (6D-Optimierung bei festem mu)")
    log("="*70)
    log()
    log("  Fuer jedes mu_eff: Optimiere (be, at, al, H0, fe, ae)")
    log("  mit Nelder-Mead aus mehreren Startpunkten")
    log()

    mu_profile = np.arange(1.30, 2.15, 0.10)
    profile_results = []

    # Startpunkte fuer Multistart (reduziert fuer Geschwindigkeit)
    starts = [
        # (be, at, al, H0, fe, log10(ae))
        [2.80, 0.10, 0.72, 70.0, 1e8, -3.0],   # bester aus erstem Lauf
        [2.74, 0.14, 0.72, 80.0, 3e8, -2.9],    # Optimum aus erstem Lauf
        [2.90, 0.08, 0.55, 67.4, 5e7, -3.2],
    ]

    log(f"  {'mu_eff':>6} {'H0':>6} {'be':>5} {'at':>5} {'al':>5} {'fe':>8} {'ae':>7}"
        f"  {'rd':>6} {'lA':>7} {'R':>6} {'X2sn':>6} {'X2cmb':>5} {'X2bao':>5} {'X2tot':>7} {'dX2':>6}")
    log("  "+"-"*110)

    for mu_val in mu_profile:
        def obj_fixed(p, mu=mu_val):
            be, at, al, h0 = p[0], p[1], p[2], p[3]
            fe, ae = max(p[4], 0), 10**p[5]
            if be<2.0 or be>4.0: return 1e15
            if at<0.005 or at>0.3: return 1e15
            if al<0.1 or al>0.95: return 1e15
            if h0<50 or h0>85: return 1e15
            if ae<1e-5 or ae>0.1: return 1e15
            h = h0/100.0
            Ob = omega_b_BBN/h**2
            Phi0 = (1.0 - mu*Ob - Omega_r - al) / f_sat(1.0)
            if Phi0 < 0: return 1e15
            try:
                m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                             f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
                if any(m.E2(a)<=0 for a in [1e-6, 1e-4, 0.01, 1.0]):
                    return 1e15
                return m.full_chi2(z_sn, mb_sn, mbe_sn)
            except:
                return 1e15

        best_this = 1e30
        best_x = None

        for s in starts:
            try:
                res = minimize(obj_fixed, s, method='Nelder-Mead',
                             options={'maxiter':600, 'xatol':0.01, 'fatol':1.0, 'adaptive':True})
                if res.fun < best_this:
                    best_this = res.fun
                    best_x = res.x.copy()
            except:
                continue

        # Zweiter Lauf vom besten Punkt
        if best_x is not None and best_this < 1e10:
            try:
                res2 = minimize(obj_fixed, best_x, method='Nelder-Mead',
                               options={'maxiter':400, 'xatol':0.005, 'fatol':0.5, 'adaptive':True})
                if res2.fun < best_this:
                    best_this = res2.fun
                    best_x = res2.x.copy()
            except:
                pass

        if best_x is not None and best_this < 1e10:
            be, at, al, h0 = best_x[0], best_x[1], best_x[2], best_x[3]
            fe, ae = max(best_x[4], 0), 10**best_x[5]
            m = CFM_MOND(h0, al, be, at, mu_eff=mu_val, n_trans=n_fix,
                         f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
            c2s, c2c, c2b, la, R, rs, rd, dc = m.detailed_chi2(z_sn, mb_sn, mbe_sn)
            c2t = c2s + c2c + c2b
            profile_results.append((mu_val, h0, be, at, al, fe, ae, rd, la, R, c2s, c2c, c2b, c2t))
            log(f"  {mu_val:6.3f} {h0:6.1f} {be:5.2f} {at:5.3f} {al:5.3f} {fe:8.1e} {ae:7.1e}"
                f"  {rd:6.1f} {la:7.3f} {R:6.4f} {c2s:6.1f} {c2c:5.1f} {c2b:5.1f} {c2t:7.1f} {c2t-c2tot_L:+6.1f}")
        else:
            log(f"  {mu_val:6.3f}  -- keine gueltige Loesung --")

    log()

    # Bestes aus Profil
    if profile_results:
        profile_results.sort(key=lambda x: x[13])
        best_pr = profile_results[0]
        log(f"  BESTES mu_eff-Profil: mu={best_pr[0]:.3f}, H0={best_pr[1]:.1f}, "
            f"rd={best_pr[7]:.1f}, chi2={best_pr[13]:.1f} (dX2={best_pr[13]-c2tot_L:+.1f})")
        log()

        # H0 vs mu_eff Trend
        log("  mu_eff vs H0 Trend:")
        log(f"  {'mu_eff':>6} {'H0':>6} {'rd':>6} {'dX2':>7}")
        log("  "+"-"*30)
        for pr in sorted(profile_results, key=lambda x: x[0]):
            log(f"  {pr[0]:6.3f} {pr[1]:6.1f} {pr[7]:6.1f} {pr[13]-c2tot_L:+7.1f}")
        log()

    # ===========================================================
    # SCHRITT 2: VOLLE 7D-OPTIMIERUNG (Multistart)
    # ===========================================================
    log("="*70)
    log("  SCHRITT 2: VOLLE 7D-OPTIMIERUNG (mu_eff FREI, Multistart)")
    log("="*70)
    log()

    def objective_full(p):
        be, at, al, h0, mu, fe, log_ae = p
        ae = 10**log_ae
        if be<2.0 or be>4.0: return 1e15
        if at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95: return 1e15
        if h0<50 or h0>85: return 1e15
        if mu<0.8 or mu>3.5: return 1e15
        if ae<1e-5 or ae>0.1: return 1e15
        h = h0/100.0
        Ob = omega_b_BBN/h**2
        Phi0 = (1.0 - mu*Ob - Omega_r - al) / f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                         f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
            if any(m.E2(a)<=0 for a in [1e-6, 1e-4, 0.01, 1.0]):
                return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)
        except:
            return 1e15

    # Startpunkte: beste aus Profil + Ergebnisse des ersten Laufs
    starts_7d = [
        [2.74, 0.14, 0.72, 80.0, 1.96, 3.1e8, -2.93],  # Optimum aus Lauf 1
    ]

    # Beste 5 aus Profil als Startpunkte
    for pr in profile_results[:5]:
        starts_7d.append([pr[2], pr[3], pr[4], pr[1], pr[0], pr[5], np.log10(max(pr[6], 1e-5))])

    # Zusaetzliche diverse Starts
    starts_7d += [
        [2.80, 0.10, 0.65, 67.4, 1.50, 1e8, -3.0],
        [2.90, 0.08, 0.55, 70.0, 1.70, 2e8, -3.0],
        [2.80, 0.12, 0.70, 73.0, 1.80, 5e8, -3.0],
    ]

    best_7d = 1e30
    best_7d_x = None
    best_7d_nfev = 0

    log(f"  {len(starts_7d)} Startpunkte ...")
    log()

    for i, s7 in enumerate(starts_7d):
        try:
            res = minimize(objective_full, s7, method='Nelder-Mead',
                         options={'maxiter':1200, 'xatol':0.005, 'fatol':0.5, 'adaptive':True})
            if res.fun < best_7d:
                best_7d = res.fun
                best_7d_x = res.x.copy()
                best_7d_nfev += res.nfev
                log(f"  Start {i+1}: chi2={res.fun:.1f} mu={res.x[4]:.3f} H0={res.x[3]:.1f} "
                    f"be={res.x[0]:.2f} al={res.x[2]:.3f}")
        except:
            continue

    # Nachpolieren vom Besten
    if best_7d_x is not None:
        for _ in range(2):
            try:
                res = minimize(objective_full, best_7d_x, method='Nelder-Mead',
                             options={'maxiter':800, 'xatol':0.002, 'fatol':0.2, 'adaptive':True})
                if res.fun < best_7d:
                    best_7d = res.fun
                    best_7d_x = res.x.copy()
                best_7d_nfev += res.nfev
            except:
                break

    log()

    if best_7d_x is not None and best_7d < 1e10:
        be_o, at_o, al_o, h0_o, mu_o, fe_o, log_ae_o = best_7d_x
        ae_o = 10**log_ae_o
        fe_o = max(fe_o, 0)

        m_o = CFM_MOND(h0_o, al_o, be_o, at_o, mu_eff=mu_o, n_trans=n_fix,
                       f_ede=fe_o, a_ede=ae_o, p_ede=p_ede_fix)
        c2s_o, c2c_o, c2b_o, la_o, R_o, rs_o, rd_o, dc_o = m_o.detailed_chi2(z_sn, mb_sn, mbe_sn)
        c2t_o = c2s_o + c2c_o + c2b_o

        log("  BESTES 7D-ERGEBNIS (mu_eff FREI):")
        log("  " + "="*55)
        log(f"  MOND-Kopplung:")
        log(f"    mu_eff     = {mu_o:.4f}  (4/3 = {4/3:.4f})")
        log(f"    Ob_phys    = {m_o.Omega_b:.5f}")
        log(f"    Ob_eff     = {mu_o*m_o.Omega_b:.5f}  (= mu_eff * Ob)")
        log(f"  Running Beta:")
        log(f"    beta_early = {be_o:.4f}")
        log(f"    beta_late  = {beta_late_fix}")
        log(f"    a_t        = {at_o:.5f}  (z_t = {1/at_o-1:.1f})")
        log(f"    alpha      = {al_o:.4f}")
        log(f"  Early Dark Energy:")
        log(f"    f_ede      = {fe_o:.2e}")
        log(f"    a_ede      = {ae_o:.2e}  (z_ede = {1/ae_o-1:.0f})")
        if fe_o > 0:
            E2_star = m_o.E2(1.0/(1.0+z_star))
            ede_star = m_o.ede(1.0/(1.0+z_star))
            f_frac = ede_star / E2_star if E2_star > 0 else 0
            log(f"    f_EDE(z*)  = {f_frac:.4f}  ({f_frac*100:.2f}%)")
        log(f"  Hintergrund:")
        log(f"    H0         = {h0_o:.2f} km/s/Mpc")
        log(f"    Phi0       = {m_o.Phi0:.4f}")
        log()
        log(f"  {'':24s} {'CFM+MOND':>12s} {'LCDM':>12s} {'Planck':>12s}")
        log(f"  {'l_A':24s} {la_o:12.3f} {la_L:12.3f} {lA_pl:12.3f}")
        log(f"  {'R':24s} {R_o:12.4f} {R_L:12.4f} {R_pl:12.4f}")
        log(f"  {'r_s(z*) [Mpc]':24s} {rs_o:12.2f} {rs_L:12.2f}")
        log(f"  {'r_d [Mpc]':24s} {rd_o:12.2f} {rd_L:12.2f}")
        log(f"  {'d_C(z*) [Mpc]':24s} {dc_o:12.2f} {dc_L:12.2f}")
        log()
        log(f"  {'':24s} {'CFM+MOND':>12s} {'LCDM':>12s}")
        log(f"  {'chi2_SN':24s} {c2s_o:12.1f} {c2sn_L:12.1f}")
        log(f"  {'chi2_CMB (lA+R)':24s} {c2c_o:12.1f} {c2cmb_L:12.1f}")
        log(f"  {'chi2_BAO':24s} {c2b_o:12.1f} {c2bao_L:12.1f}")
        log(f"  {'chi2_TOTAL':24s} {c2t_o:12.1f} {c2tot_L:12.1f}")
        log(f"  {'Delta_chi2':24s} {c2t_o-c2tot_L:12.1f}")
        log()

    # ===========================================================
    # SCHRITT 3: SPEZIALFALL - Optimierung bei H0 ~ 67-68
    # ===========================================================
    log("="*70)
    log("  SCHRITT 3: FIT MIT H0-CONSTRAINT (66 < H0 < 69)")
    log("="*70)
    log()
    log("  Suche: Welches mu_eff brauchen wir fuer H0 ~ 67.4?")
    log()

    def obj_h0_constrained(p):
        be, at, al, h0, mu, fe, log_ae = p
        ae = 10**log_ae
        if be<2.0 or be>4.0: return 1e15
        if at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95: return 1e15
        if h0<66 or h0>69: return 1e15  # H0 CONSTRAINT!
        if mu<0.8 or mu>3.5: return 1e15
        if ae<1e-5 or ae>0.1: return 1e15
        h = h0/100.0
        Ob = omega_b_BBN/h**2
        Phi0 = (1.0 - mu*Ob - Omega_r - al) / f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                         f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
            if any(m.E2(a)<=0 for a in [1e-6, 1e-4, 0.01, 1.0]):
                return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)
        except:
            return 1e15

    starts_h0 = [
        [2.80, 0.10, 0.72, 67.4, 1.50, 1e8, -3.0],
        [2.80, 0.10, 0.65, 67.4, 1.80, 2e8, -3.0],
        [2.90, 0.08, 0.60, 67.4, 2.00, 1e8, -3.0],
        [2.80, 0.12, 0.70, 68.0, 1.60, 5e8, -3.0],
        [2.80, 0.10, 0.50, 67.4, 2.50, 1e8, -3.0],
    ]

    best_h0c = 1e30
    best_h0c_x = None

    for i, s in enumerate(starts_h0):
        try:
            res = minimize(obj_h0_constrained, s, method='Nelder-Mead',
                         options={'maxiter':1500, 'xatol':0.002, 'fatol':0.3, 'adaptive':True})
            if res.fun < best_h0c:
                best_h0c = res.fun
                best_h0c_x = res.x.copy()
                log(f"  Start {i+1}: chi2={res.fun:.1f} mu={res.x[4]:.3f} H0={res.x[3]:.1f}")
        except:
            continue

    # Nachpolieren
    if best_h0c_x is not None:
        try:
            res = minimize(obj_h0_constrained, best_h0c_x, method='Nelder-Mead',
                         options={'maxiter':1000, 'xatol':0.001, 'fatol':0.1, 'adaptive':True})
            if res.fun < best_h0c:
                best_h0c = res.fun
                best_h0c_x = res.x.copy()
        except:
            pass

    if best_h0c_x is not None and best_h0c < 1e10:
        be_h, at_h, al_h, h0_h, mu_h, fe_h, log_ae_h = best_h0c_x
        ae_h = 10**log_ae_h
        fe_h = max(fe_h, 0)
        m_h = CFM_MOND(h0_h, al_h, be_h, at_h, mu_eff=mu_h, n_trans=n_fix,
                       f_ede=fe_h, a_ede=ae_h, p_ede=p_ede_fix)
        c2s_h, c2c_h, c2b_h, la_h, R_h, rs_h, rd_h, dc_h = m_h.detailed_chi2(z_sn, mb_sn, mbe_sn)
        c2t_h = c2s_h + c2c_h + c2b_h

        log()
        log("  ERGEBNIS (H0 ~ 67.4):")
        log("  " + "="*55)
        log(f"    mu_eff     = {mu_h:.4f}")
        log(f"    H0         = {h0_h:.2f} km/s/Mpc")
        log(f"    beta_early = {be_h:.4f}")
        log(f"    a_t        = {at_h:.5f}  (z_t = {1/at_h-1:.1f})")
        log(f"    alpha      = {al_h:.4f}")
        log(f"    f_ede      = {fe_h:.2e}")
        log(f"    a_ede      = {ae_h:.2e}")
        log(f"    Phi0       = {m_h.Phi0:.4f}")
        log(f"    r_d        = {rd_h:.2f} Mpc")
        log(f"    l_A        = {la_h:.3f}")
        log(f"    R          = {R_h:.4f}")
        if fe_h > 0:
            E2_st = m_h.E2(1.0/(1.0+z_star))
            ede_st = m_h.ede(1.0/(1.0+z_star))
            f_fr = ede_st / E2_st if E2_st > 0 else 0
            log(f"    f_EDE(z*)  = {f_fr:.4f}  ({f_fr*100:.2f}%)")
        log()
        log(f"  chi2: SN={c2s_h:.1f} CMB={c2c_h:.1f} BAO={c2b_h:.1f} TOT={c2t_h:.1f} (dX2={c2t_h-c2tot_L:+.1f})")
        log()

    # ===========================================================
    # SCHRITT 4: SPEZIALFALL - Optimierung bei H0 ~ 73 (SH0ES)
    # ===========================================================
    log("="*70)
    log("  SCHRITT 4: FIT MIT H0-CONSTRAINT (72 < H0 < 75)")
    log("="*70)
    log()
    log("  Suche: Kann CFM+MOND auch den lokalen H0-Wert (SH0ES) treffen?")
    log()

    def obj_h0_shoes(p):
        be, at, al, h0, mu, fe, log_ae = p
        ae = 10**log_ae
        if be<2.0 or be>4.0: return 1e15
        if at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95: return 1e15
        if h0<72 or h0>75: return 1e15  # SH0ES CONSTRAINT!
        if mu<0.8 or mu>3.5: return 1e15
        if ae<1e-5 or ae>0.1: return 1e15
        h = h0/100.0
        Ob = omega_b_BBN/h**2
        Phi0 = (1.0 - mu*Ob - Omega_r - al) / f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                         f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
            if any(m.E2(a)<=0 for a in [1e-6, 1e-4, 0.01, 1.0]):
                return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)
        except:
            return 1e15

    starts_shoes = [
        [2.80, 0.10, 0.72, 73.0, 1.60, 1e8, -3.0],
        [2.74, 0.14, 0.72, 73.0, 1.80, 3e8, -2.9],
        [2.80, 0.10, 0.60, 73.0, 2.00, 2e8, -3.0],
    ]

    best_sh = 1e30
    best_sh_x = None

    for i, s in enumerate(starts_shoes):
        try:
            res = minimize(obj_h0_shoes, s, method='Nelder-Mead',
                         options={'maxiter':1500, 'xatol':0.002, 'fatol':0.3, 'adaptive':True})
            if res.fun < best_sh:
                best_sh = res.fun
                best_sh_x = res.x.copy()
                log(f"  Start {i+1}: chi2={res.fun:.1f} mu={res.x[4]:.3f} H0={res.x[3]:.1f}")
        except:
            continue

    if best_sh_x is not None:
        try:
            res = minimize(obj_h0_shoes, best_sh_x, method='Nelder-Mead',
                         options={'maxiter':1000, 'xatol':0.001, 'fatol':0.1, 'adaptive':True})
            if res.fun < best_sh:
                best_sh = res.fun
                best_sh_x = res.x.copy()
        except:
            pass

    if best_sh_x is not None and best_sh < 1e10:
        be_s, at_s, al_s, h0_s, mu_s, fe_s, log_ae_s = best_sh_x
        ae_s = 10**log_ae_s
        fe_s = max(fe_s, 0)
        m_s = CFM_MOND(h0_s, al_s, be_s, at_s, mu_eff=mu_s, n_trans=n_fix,
                       f_ede=fe_s, a_ede=ae_s, p_ede=p_ede_fix)
        c2s_s, c2c_s, c2b_s, la_s, R_s, rs_s, rd_s, dc_s = m_s.detailed_chi2(z_sn, mb_sn, mbe_sn)
        c2t_s = c2s_s + c2c_s + c2b_s

        log()
        log("  ERGEBNIS (H0 ~ 73, SH0ES):")
        log("  " + "="*55)
        log(f"    mu_eff     = {mu_s:.4f}")
        log(f"    H0         = {h0_s:.2f} km/s/Mpc")
        log(f"    beta_early = {be_s:.4f}")
        log(f"    a_t        = {at_s:.5f}  (z_t = {1/at_s-1:.1f})")
        log(f"    alpha      = {al_s:.4f}")
        log(f"    f_ede      = {fe_s:.2e}")
        log(f"    a_ede      = {ae_s:.2e}")
        log(f"    Phi0       = {m_s.Phi0:.4f}")
        log(f"    r_d        = {rd_s:.2f} Mpc")
        log(f"    l_A        = {la_s:.3f}")
        log(f"    R          = {R_s:.4f}")
        if fe_s > 0:
            E2_st = m_s.E2(1.0/(1.0+z_star))
            ede_st = m_s.ede(1.0/(1.0+z_star))
            f_fr = ede_st / E2_st if E2_st > 0 else 0
            log(f"    f_EDE(z*)  = {f_fr:.4f}  ({f_fr*100:.2f}%)")
        log()
        log(f"  chi2: SN={c2s_s:.1f} CMB={c2c_s:.1f} BAO={c2b_s:.1f} TOT={c2t_s:.1f} (dX2={c2t_s-c2tot_L:+.1f})")
        log()

    # ===========================================================
    # ZUSAMMENFASSUNG
    # ===========================================================
    log("="*70)
    log("  GESAMTZUSAMMENFASSUNG")
    log("="*70)
    log()
    log(f"  {'Modell':30s} {'H0':>6} {'mu':>5} {'rd':>7} {'lA':>7} {'R':>6} {'f_EDE%':>6} {'X2tot':>7} {'dX2':>6}")
    log("  "+"-"*90)
    log(f"  {'LCDM':30s} {67.4:6.1f} {1.0:5.2f} {rd_L:7.2f} {la_L:7.3f} {R_L:6.4f} {'0.0':>6} {c2tot_L:7.1f} {'+0.0':>6}")
    log(f"  {'CFM ohne MOND (frueher)':30s} {'60.0':>6} {'1.00':>5} {'165.0':>7} {'301.48':>7} {'1.7502':>6} {'52':>6} {'705.2':>7} {'-5.1':>6}")

    if best_7d_x is not None and best_7d < 1e10:
        E2_st = m_o.E2(1.0/(1.0+z_star))
        ede_st = m_o.ede(1.0/(1.0+z_star))
        f_fr_o = ede_st / E2_st * 100 if E2_st > 0 else 0
        log(f"  {'CFM+MOND (mu frei)':30s} {h0_o:6.1f} {mu_o:5.2f} {rd_o:7.2f} {la_o:7.3f} {R_o:6.4f} {f_fr_o:6.1f} {c2t_o:7.1f} {c2t_o-c2tot_L:+6.1f}")

    if best_h0c_x is not None and best_h0c < 1e10:
        E2_st = m_h.E2(1.0/(1.0+z_star))
        ede_st = m_h.ede(1.0/(1.0+z_star))
        f_fr_h = ede_st / E2_st * 100 if E2_st > 0 else 0
        log(f"  {'CFM+MOND (H0~67)':30s} {h0_h:6.1f} {mu_h:5.2f} {rd_h:7.2f} {la_h:7.3f} {R_h:6.4f} {f_fr_h:6.1f} {c2t_h:7.1f} {c2t_h-c2tot_L:+6.1f}")

    if best_sh_x is not None and best_sh < 1e10:
        E2_st = m_s.E2(1.0/(1.0+z_star))
        ede_st = m_s.ede(1.0/(1.0+z_star))
        f_fr_s = ede_st / E2_st * 100 if E2_st > 0 else 0
        log(f"  {'CFM+MOND (H0~73, SH0ES)':30s} {h0_s:6.1f} {mu_s:5.2f} {rd_s:7.2f} {la_s:7.3f} {R_s:6.4f} {f_fr_s:6.1f} {c2t_s:7.1f} {c2t_s-c2tot_L:+6.1f}")

    log()

    # Physikalische Interpretation
    log("="*70)
    log("  PHYSIKALISCHE INTERPRETATION")
    log("="*70)
    log()
    if best_7d_x is not None and best_7d < 1e10:
        log(f"  Optimales mu_eff = {mu_o:.3f}")
        log(f"  Verhaeltnis zu 4/3: mu_opt / (4/3) = {mu_o/(4/3):.3f}")
        log(f"  Verhaeltnis zu 2:   mu_opt / 2     = {mu_o/2:.3f}")
        log()
        log("  MOND-Interpolation mu(x) = x/sqrt(1+x^2):")
        log("  Im deep-MOND-Regime (g << a0): mu -> g/a0 -> mu_eff -> a0/g")
        if best_h0c_x is not None and best_h0c < 1e10:
            log(f"  Fuer H0~67: mu_eff = {mu_h:.3f} -> g/a0 = {1/mu_h:.3f} -> g = {1/mu_h*1.2e-10:.2e} m/s^2")
        log()
        log("  Effektive Omega_b fuer verschiedene Szenarien:")
        if best_h0c_x is not None and best_h0c < 1e10:
            log(f"    H0~67: Ob_eff = {mu_h:.3f} * {omega_b_BBN/(h0_h/100)**2:.4f} = {mu_h*omega_b_BBN/(h0_h/100)**2:.4f}")
        if best_sh_x is not None and best_sh < 1e10:
            log(f"    H0~73: Ob_eff = {mu_s:.3f} * {omega_b_BBN/(h0_s/100)**2:.4f} = {mu_s*omega_b_BBN/(h0_s/100)**2:.4f}")
    log()

    log(f"  Laufzeit: {time.time()-t0:.1f} Sekunden")

    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    print(f"\n  -> Ergebnisse in {out_path}")


if __name__ == '__main__':
    main()

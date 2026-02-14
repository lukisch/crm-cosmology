"""
CFM+MOND Joint Fit: MOND auf Hintergrund-Ebene
================================================
Schluesselaenderung: mu_eff = 4/3 wird auf den baryonischen Term
in der Friedmann-Gleichung angewendet:

  H^2(a) = H0^2 [mu_eff * Ob * a^-3 + Or * a^-4 + Phi0*f_sat(a) + alpha*a^{-beta_eff(a)} + EDE(a)]

Physik:
- MOND verstaerkt die effektive Gravitationskopplung: G_eff = mu_eff * G_N
- Dies erhoet H(z) bei z > 1000 -> reduziert Schallhorizont r_d
- Zusaetzlich aendert sich die Schallgeschwindigkeit c_s wegen R_b = 3*rho_b_eff/(4*rho_gamma)
- Erwartung: r_d sinkt, H0 kann steigen

Test: mu_eff als freien Parameter fitten (nicht nur 4/3)
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time, os

# ===============================================================
# Konstanten
# ===============================================================
c_light = 299792.458      # km/s
c_SI = 2.998e8            # m/s
a0_MOND = 1.2e-10         # m/s^2
z_star = 1089.80
z_drag = 1059.94
Omega_r = 9.15e-5
omega_b_BBN = 0.02237
beta_late_fix = 2.02
k_fsat = 9.81
a_trans_fsat = 0.971

# Planck 2018 compressed
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
        self.mu_eff = mu_eff  # MOND-Verstaerkung
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
        """Sound horizon mit MOND-modifizierter Baryonendichte"""
        # R_b = 3 * rho_b_eff / (4 * rho_gamma)
        # Bei MOND: rho_b_eff = mu_eff * rho_b -> R_b wird groesser
        # -> c_s wird kleiner -> r_s wird kleiner (genau was wir wollen!)
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
    """CFM + MOND auf Hintergrund-Ebene: mu_eff * Omega_b"""
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
        # EDE-Korrektur bei a=1
        self.ede_at_1 = f_ede/(1.0+(1.0/a_ede)**p_ede) if f_ede > 0 and a_ede > 0 else 0
        # Phi0 aus Closure: E2(1) = 1
        # mu_eff * Ob + Or + Phi0*f_sat(1) + alpha + ede(1)=0 -> Phi0 = (1 - mu*Ob - Or - alpha) / f_sat(1)
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
    out_path = os.path.join(out_dir, 'CFM_MOND_Background.txt')

    L = []
    def log(s=''):
        L.append(s); print(s)

    log("  CFM+MOND HINTERGRUND-FIT: mu_eff IN FRIEDMANN-GLEICHUNG")
    log("  " + "="*55)
    log(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log()
    log("  Schluesselaenderung gegenueber bisherigem Fit:")
    log("  H^2 = H0^2 [mu_eff*Ob*a^-3 + Or*a^-4 + Phi0*f_sat + alpha*a^-beta + EDE]")
    log("  mu_eff = 4/3 (MOND-Verstaerkung auf Hintergrund-Ebene)")
    log("  -> Erhoet H(z) bei z>1000 -> reduziert r_d -> erlaubt hoeheres H0")
    log()

    z_sn, mb_sn, mbe_sn = load_pantheon()
    log(f"  {len(z_sn)} SNe geladen")
    log()

    # ===========================================================
    # LCDM Referenz
    # ===========================================================
    log("="*70)
    log("  LCDM-REFERENZ")
    log("="*70)
    lcdm = LCDM()
    la_L, R_L, rs_L, rd_L, dc_L = lcdm.cmb_obs()
    c2sn_L = lcdm.chi2_SN(z_sn, mb_sn, mbe_sn)
    c2cmb_L = ((la_L-lA_pl)/lA_err)**2 + ((R_L-R_pl)/R_err)**2
    c2bao_L = lcdm.chi2_BAO(rd_L)
    c2tot_L = c2sn_L + c2cmb_L + c2bao_L
    log(f"  l_A={la_L:.3f} R={R_L:.4f} r_d={rd_L:.2f} rs={rs_L:.2f}")
    log(f"  chi2: SN={c2sn_L:.1f} CMB={c2cmb_L:.1f} BAO={c2bao_L:.1f} TOT={c2tot_L:.1f}")
    log()

    # ===========================================================
    # SCHRITT 1: VERGLEICH mu_eff Werte
    # ===========================================================
    log("="*70)
    log("  SCHRITT 1: EFFEKT VON mu_eff AUF r_d")
    log("="*70)
    log()
    log("  Feste Parameter: be=2.82, at=0.092, al=0.63, H0=60, kein EDE")
    log()
    log(f"  {'mu_eff':>7} {'Ob_eff':>7} {'Phi0':>7} {'r_s':>8} {'r_d':>8} {'l_A':>8} {'R':>8}")
    log("  " + "-"*65)
    for mu in [1.0, 1.1, 4.0/3.0, 1.5, 1.7, 2.0, 2.5, 3.0]:
        try:
            m = CFM_MOND(60.0, 0.63, 2.82, 0.092, mu_eff=mu, f_ede=0)
            if m.Phi0 < 0:
                log(f"  {mu:7.3f}  -> Phi0 < 0, nicht physikalisch")
                continue
            if any(m.E2(a) <= 0 for a in [1e-6, 1e-4, 0.01, 1.0]):
                log(f"  {mu:7.3f}  -> E2 < 0 irgendwo")
                continue
            la, R, rs, rd, dc = m.cmb_obs()
            log(f"  {mu:7.3f} {mu*m.Omega_b:7.4f} {m.Phi0:7.4f} {rs:8.2f} {rd:8.2f} {la:8.2f} {R:8.4f}")
        except Exception as e:
            log(f"  {mu:7.3f}  -> Fehler: {e}")
    log()

    # ===========================================================
    # SCHRITT 2: mu_eff=4/3 bei verschiedenen H0
    # ===========================================================
    log("="*70)
    log("  SCHRITT 2: mu_eff=4/3 BEI VERSCHIEDENEN H0")
    log("="*70)
    log()
    log(f"  {'H0':>5} {'Ob':>7} {'mu*Ob':>7} {'Phi0':>7} {'r_d':>7} {'l_A':>8} {'R':>8} {'X2sn':>8} {'X2cmb':>8} {'X2bao':>8} {'X2tot':>8}")
    log("  " + "-"*100)
    for h0 in [55, 58, 60, 63, 65, 67.4, 70, 73]:
        try:
            m = CFM_MOND(h0, 0.63, 2.82, 0.092, mu_eff=4.0/3.0, f_ede=0)
            if m.Phi0 < 0: continue
            if any(m.E2(a) <= 0 for a in [1e-6, 1e-4, 0.01, 1.0]): continue
            la, R, rs, rd, dc = m.cmb_obs()
            c2s = m.chi2_SN(z_sn, mb_sn, mbe_sn)
            m._dL_interp = None
            c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
            c2b = m.chi2_BAO(rd)
            log(f"  {h0:5.1f} {m.Omega_b:7.4f} {4/3*m.Omega_b:7.4f} {m.Phi0:7.4f} {rd:7.2f} {la:8.2f} {R:8.4f} {c2s:8.1f} {c2c:8.1f} {c2b:8.1f} {c2s+c2c+c2b:8.1f}")
        except:
            pass
    log()

    # ===========================================================
    # SCHRITT 3: GRID-SCAN mit mu_eff
    # ===========================================================
    log("="*70)
    log("  SCHRITT 3: GRID-SCAN (7D: be, at, al, H0, fe, ae, mu)")
    log("="*70)
    log()

    n_fix = 4
    p_ede_fix = 6

    be_grid  = [2.7, 2.8, 2.9, 3.0, 3.2]
    at_grid  = [0.03, 0.06, 0.1]
    al_grid  = [0.45, 0.55, 0.63, 0.72]
    h0_grid  = [60, 63, 65, 67.4, 70, 73]
    fe_grid  = [0, 1e8, 5e8]
    ae_grid  = [5e-4, 1e-3]
    mu_grid  = [1.0, 4.0/3.0, 1.5, 2.0]

    N_tot = len(be_grid)*len(at_grid)*len(al_grid)*len(h0_grid)*len(fe_grid)*len(ae_grid)*len(mu_grid)
    log(f"  Grid: {len(be_grid)}x{len(at_grid)}x{len(al_grid)}x{len(h0_grid)}x{len(fe_grid)}x{len(ae_grid)}x{len(mu_grid)} = {N_tot}")
    log()

    best_c2 = 1e30
    best_p = None
    results = []
    cnt = 0
    t_s = time.time()

    for be in be_grid:
      for at in at_grid:
        for al in al_grid:
          for h0 in h0_grid:
            for mu in mu_grid:
              for fe in fe_grid:
                for ae in ae_grid:
                    cnt += 1
                    if cnt % 200 == 0:
                        el = time.time()-t_s
                        print(f"  ... {cnt}/{N_tot} ({el:.0f}s)", end='\r')

                    if fe == 0 and ae != ae_grid[0]:
                        continue

                    h = h0/100.0
                    Ob = omega_b_BBN/h**2
                    Phi0 = (1.0-mu*Ob-Omega_r-al)/f_sat(1.0)
                    if Phi0 < 0: continue

                    try:
                        m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                                     f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
                        if any(m.E2(a) <= 0 for a in [1e-6,1e-4,1e-3,0.01,0.1,1.0]):
                            continue

                        c2s = m.chi2_SN(z_sn, mb_sn, mbe_sn)
                        m._dL_interp = None
                        la, R, rs, rd, dc = m.cmb_obs()
                        c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
                        c2b = m.chi2_BAO(rd)
                        c2t = c2s+c2c+c2b

                        results.append((be,at,al,h0,mu,fe,ae, c2s,c2c,c2b,c2t, la,R,rd))

                        if c2t < best_c2:
                            best_c2 = c2t
                            best_p = (be,at,al,h0,mu,fe,ae)
                    except:
                        continue

    print(" "*60, end='\r')
    log(f"  Scan fertig: {len(results)} Punkte in {time.time()-t_s:.1f}s")
    log()

    results.sort(key=lambda x: x[10])

    log("  TOP-20:")
    log(f"  {'be':>5} {'at':>5} {'al':>5} {'H0':>5} {'mu':>5} {'f_ede':>8} {'a_ede':>7}"
        f"  {'X2sn':>7} {'X2cmb':>7} {'X2bao':>7} {'X2tot':>7}  {'lA':>7} {'R':>7} {'rd':>6}")
    log("  "+"-"*115)
    for r in results[:20]:
        log(f"  {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:5.1f} {r[4]:5.2f} {r[5]:8.0e} {r[6]:7.1e}"
            f"  {r[7]:7.1f} {r[8]:7.1f} {r[9]:7.1f} {r[10]:7.1f}"
            f"  {r[11]:7.2f} {r[12]:7.4f} {r[13]:6.1f}")
    log()

    # Vergleich mu_eff Werte in Top-Ergebnissen
    for mu_v in [1.0, 4.0/3.0, 1.5, 2.0]:
        sub = [r for r in results if abs(r[4]-mu_v) < 0.01]
        if sub:
            log(f"  Bester mu_eff={mu_v:.2f}: chi2={sub[0][10]:.1f} H0={sub[0][3]:.1f} rd={sub[0][13]:.1f}")
    log()

    # ===========================================================
    # SCHRITT 4: FEINOPTIMIERUNG
    # ===========================================================
    log("="*70)
    log("  SCHRITT 4: FEINOPTIMIERUNG (7D Nelder-Mead)")
    log("="*70)
    log()

    if best_p:
        be0,at0,al0,h00,mu0,fe0,ae0 = best_p
        log(f"  Start: be={be0} at={at0} al={al0} H0={h00} mu={mu0:.2f} fe={fe0:.0e} ae={ae0:.0e}")
        log(f"  Start chi2 = {best_c2:.1f}")
        log()

        def objective(p):
            be = p[0]; at = p[1]; al = p[2]; h0 = p[3]
            mu = p[4]; fe = max(p[5], 0); ae = 10**p[6]
            if be<2.0 or be>4.0: return 1e15
            if at<0.005 or at>0.3: return 1e15
            if al<0.1 or al>0.95: return 1e15
            if h0<55 or h0>80: return 1e15
            if mu<0.8 or mu>3.5: return 1e15
            if ae<1e-5 or ae>0.1: return 1e15
            Phi0 = (1.0-mu*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
            if Phi0 < 0: return 1e15
            try:
                m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                             f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
                if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1.0]):
                    return 1e15
                return m.full_chi2(z_sn, mb_sn, mbe_sn)
            except:
                return 1e15

        x0 = [be0, at0, al0, h00, mu0, fe0, np.log10(max(ae0,1e-5))]
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol':0.003, 'fatol':0.3, 'adaptive':True})

        be_o = res.x[0]; at_o = res.x[1]; al_o = res.x[2]; h0_o = res.x[3]
        mu_o = res.x[4]; fe_o = max(res.x[5],0); ae_o = 10**res.x[6]

        log(f"  Optimierung: {res.nfev} Eval, chi2 = {res.fun:.1f}")
        log()

        m_o = CFM_MOND(h0_o, al_o, be_o, at_o, mu_eff=mu_o, n_trans=n_fix,
                       f_ede=fe_o, a_ede=ae_o, p_ede=p_ede_fix)
        la_o, R_o, rs_o, rd_o, dc_o = m_o.cmb_obs()
        c2s_o = m_o.chi2_SN(z_sn, mb_sn, mbe_sn)
        m_o._dL_interp = None
        c2c_o = ((la_o-lA_pl)/lA_err)**2 + ((R_o-R_pl)/R_err)**2
        c2b_o = m_o.chi2_BAO(rd_o)
        c2t_o = c2s_o + c2c_o + c2b_o

        log("  OPTIMIERTES ERGEBNIS:")
        log("  " + "="*50)
        log(f"  MOND-Kopplung:")
        log(f"    mu_eff     = {mu_o:.4f}  (4/3 = {4/3:.4f})")
        log(f"    Ob_phys    = {m_o.Omega_b:.5f}")
        log(f"    Ob_eff     = {mu_o*m_o.Omega_b:.5f}  (= mu_eff * Ob)")
        log(f"  Running Beta:")
        log(f"    beta_early = {be_o:.4f}")
        log(f"    beta_late  = {beta_late_fix}")
        log(f"    a_t        = {at_o:.5f}  (z_t = {1/at_o-1:.1f})")
        log(f"    n_trans    = {n_fix}")
        log(f"    alpha      = {al_o:.4f}")
        log(f"  Early Dark Energy:")
        log(f"    f_ede      = {fe_o:.2e}")
        log(f"    a_ede      = {ae_o:.2e}  (z_ede = {1/ae_o-1:.0f})")
        log(f"    p_ede      = {p_ede_fix}")
        if fe_o > 0:
            E2_star = m_o.E2(1.0/(1.0+z_star))
            ede_star = m_o.ede(1.0/(1.0+z_star))
            f_frac = ede_star / E2_star if E2_star > 0 else 0
            log(f"    f_EDE(z*)  = {f_frac:.4f}  ({f_frac*100:.2f}%)")
        log(f"  Hintergrund:")
        log(f"    H0         = {h0_o:.2f} km/s/Mpc")
        log(f"    Omega_b    = {m_o.Omega_b:.5f}")
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

        # Profil
        log("  Profil:")
        log(f"  {'z':>7} {'beta_eff':>8} {'H/H_L':>7} {'f_EDE%':>7} {'Om_eff':>8}")
        log("  "+"-"*45)
        for z in [0, 0.5, 1, 2, 10, 50, 100, 500, 1090, 5000]:
            a = 1.0/(1.0+z)
            beff = m_o.beta_eff(a)
            e2 = m_o.E2(a); e2l = lcdm.E2(a)
            hr = np.sqrt(e2/e2l) if e2l>0 and e2>0 else 0
            ede_frac = m_o.ede(a)/e2*100 if e2 > 0 else 0
            Om = (e2-Omega_r*a**(-4))*a**3
            log(f"  {z:7.0f} {beff:8.3f} {hr:7.4f} {ede_frac:7.2f} {Om:8.4f}")
        log()

    # ===========================================================
    # SCHRITT 5: mu_eff FIXIERT AUF 4/3 (reiner MOND-Test)
    # ===========================================================
    log("="*70)
    log("  SCHRITT 5: FIT MIT FIXIERTEM mu_eff = 4/3")
    log("="*70)
    log()

    def objective_fixed_mu(p):
        be = p[0]; at = p[1]; al = p[2]; h0 = p[3]
        fe = max(p[4], 0); ae = 10**p[5]
        mu = 4.0/3.0  # FIXIERT!
        if be<2.0 or be>4.0: return 1e15
        if at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95: return 1e15
        if h0<55 or h0>80: return 1e15
        if ae<1e-5 or ae>0.1: return 1e15
        Phi0 = (1.0-mu*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_MOND(h0, al, be, at, mu_eff=mu, n_trans=n_fix,
                         f_ede=fe, a_ede=ae, p_ede=p_ede_fix)
            if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1.0]):
                return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)
        except:
            return 1e15

    # Start aus Grid-Ergebnis fuer mu=4/3
    mu43_results = [r for r in results if abs(r[4]-4.0/3.0)<0.01]
    if mu43_results:
        b43 = mu43_results[0]
        x0_43 = [b43[0], b43[1], b43[2], b43[3], b43[5], np.log10(max(b43[6],1e-5))]
    else:
        x0_43 = [2.82, 0.092, 0.55, 65.0, 1e8, np.log10(1e-3)]

    log(f"  Start: {x0_43[:4]}")

    res43 = minimize(objective_fixed_mu, x0_43, method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol':0.003, 'fatol':0.3, 'adaptive':True})

    be43 = res43.x[0]; at43 = res43.x[1]; al43 = res43.x[2]; h043 = res43.x[3]
    fe43 = max(res43.x[4],0); ae43 = 10**res43.x[5]

    log(f"  Optimierung: {res43.nfev} Eval, chi2 = {res43.fun:.1f}")
    log()

    m43 = CFM_MOND(h043, al43, be43, at43, mu_eff=4.0/3.0, n_trans=n_fix,
                   f_ede=fe43, a_ede=ae43, p_ede=p_ede_fix)
    la43, R43, rs43, rd43, dc43 = m43.cmb_obs()
    c2s43 = m43.chi2_SN(z_sn, mb_sn, mbe_sn)
    m43._dL_interp = None
    c2c43 = ((la43-lA_pl)/lA_err)**2 + ((R43-R_pl)/R_err)**2
    c2b43 = m43.chi2_BAO(rd43)
    c2t43 = c2s43 + c2c43 + c2b43

    log("  ERGEBNIS (mu_eff = 4/3 fixiert):")
    log("  " + "="*50)
    log(f"    beta_early = {be43:.4f}")
    log(f"    a_t        = {at43:.5f}  (z_t = {1/at43-1:.1f})")
    log(f"    alpha      = {al43:.4f}")
    log(f"    H0         = {h043:.2f} km/s/Mpc")
    log(f"    f_ede      = {fe43:.2e}")
    log(f"    a_ede      = {ae43:.2e}")
    log(f"    Phi0       = {m43.Phi0:.4f}")
    log(f"    r_d        = {rd43:.2f} Mpc")
    log(f"    l_A        = {la43:.3f}")
    log(f"    R          = {R43:.4f}")
    log()
    log(f"  chi2_SN  = {c2s43:.1f}")
    log(f"  chi2_CMB = {c2c43:.1f}")
    log(f"  chi2_BAO = {c2b43:.1f}")
    log(f"  chi2_TOT = {c2t43:.1f}  (LCDM: {c2tot_L:.1f})")
    log(f"  Delta_chi2 = {c2t43-c2tot_L:+.1f}")
    log()

    # ===========================================================
    # ZUSAMMENFASSUNG
    # ===========================================================
    log("="*70)
    log("  ZUSAMMENFASSUNG")
    log("="*70)
    log()
    log(f"  {'Modell':25s} {'H0':>6} {'mu':>5} {'rd':>7} {'lA':>8} {'R':>7} {'X2tot':>7} {'dX2':>7}")
    log("  "+"-"*80)
    log(f"  {'LCDM':25s} {67.4:6.1f} {1.0:5.2f} {rd_L:7.2f} {la_L:8.3f} {R_L:7.4f} {c2tot_L:7.1f} {0.0:+7.1f}")

    if best_p:
        log(f"  {'CFM+MOND (mu frei)':25s} {h0_o:6.1f} {mu_o:5.2f} {rd_o:7.2f} {la_o:8.3f} {R_o:7.4f} {c2t_o:7.1f} {c2t_o-c2tot_L:+7.1f}")

    log(f"  {'CFM+MOND (mu=4/3)':25s} {h043:6.1f} {4/3:5.2f} {rd43:7.2f} {la43:8.3f} {R43:7.4f} {c2t43:7.1f} {c2t43-c2tot_L:+7.1f}")

    # Vergleich mit altem Fit (mu=1)
    log(f"  {'CFM ohne MOND (alt)':25s} {'60.0':>6} {'1.00':>5} {'165.0':>7} {'301.477':>8} {'1.7502':>7} {'705.2':>7} {'-5.1':>7}")
    log()

    if best_p:
        log(f"  mu_eff optimal: {mu_o:.3f}  (4/3 = {4/3:.3f})")
        if h0_o > 63:
            log(f"  -> H0 VERBESSERT! {h0_o:.1f} vs. 60.0 (alter Fit)")
        if rd_o < 160:
            log(f"  -> r_d VERBESSERT! {rd_o:.1f} vs. 165.0 (alter Fit)")

    log()
    log(f"  Laufzeit: {time.time()-t0:.1f} Sekunden")

    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    print(f"\n  -> Ergebnisse in {out_path}")


if __name__ == '__main__':
    main()

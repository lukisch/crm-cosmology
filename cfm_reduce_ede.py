"""
CFM+MOND: Reduktion des EDE-Anteils
====================================
Problem: f_EDE(z*) = 59% ist zu hoch.
Strategie:
  1. Kein EDE (nur Running Beta + sqrt(pi))
  2. EDE mit hohem p (schnellerer Abfall -> weniger Breitband-Effekt)
  3. Skalenabhaengiges mu_eff(a) statt EDE
  4. Physikalisches EDE: Poeschl-Teller-Form aus der Saturation-ODE
  5. Kombiniert: mu_eff(a) + reduziertes EDE
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time, os

c_light = 299792.458
z_star = 1089.80; z_drag = 1059.94
Omega_r = 9.15e-5; omega_b_BBN = 0.02237
beta_late_fix = 2.02; k_fsat = 9.81; a_trans_fsat = 0.971
lA_pl = 301.471; lA_err = 0.14; R_pl = 1.7502; R_err = 0.0046
MU_PI = np.sqrt(np.pi)

BAO_DATA = [
    (0.15, 'DV/rd', 4.466, 0.168), (0.38, 'DM/rd', 10.27, 0.15),
    (0.38, 'DH/rd', 25.00, 0.76), (0.51, 'DM/rd', 13.38, 0.18),
    (0.51, 'DH/rd', 22.33, 0.58), (0.61, 'DM/rd', 15.45, 0.20),
    (0.61, 'DH/rd', 20.75, 0.46), (2.334, 'DM/rd', 37.6, 1.1),
    (2.334, 'DH/rd', 8.86, 0.29),
]

def load_pantheon():
    path = os.path.join(os.path.dirname(__file__), '_data', 'Pantheon+SH0ES.dat')
    z, mb, mbe = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            p = line.split()
            if len(p) < 10: continue
            try: zv, mbv, mbev = float(p[2]), float(p[8]), float(p[9])
            except: continue
            if zv > 0.01: z.append(zv); mb.append(mbv); mbe.append(mbev)
    return np.array(z), np.array(mb), np.array(mbe)

def f_sat(a):
    x = k_fsat * (a - a_trans_fsat)
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + np.exp(-x))


class CFM_General:
    """Generalisiertes CFM+MOND mit verschiedenen EDE-Optionen"""
    def __init__(self, H0, alpha, beta_early, a_t, n_trans=4,
                 mu_eff=MU_PI, mu_early=None, a_mu=None,
                 f_ede=0, a_ede=1e-3, p_ede=6, ede_type='standard'):
        self.H0 = H0
        h = H0/100.0
        self.Omega_b = omega_b_BBN / h**2
        self.Omega_gamma = 2.469e-5 / h**2
        self.mu_late = mu_eff
        self.mu_early = mu_early if mu_early is not None else mu_eff
        self.a_mu = a_mu if a_mu is not None else 0.01  # mu-Uebergang
        self.alpha = alpha
        self.beta_early = beta_early
        self.a_t = a_t
        self.n_trans = n_trans
        self.f_ede = f_ede
        self.a_ede = a_ede
        self.p_ede = p_ede
        self.ede_type = ede_type

        # EDE bei a=1 (Normierung)
        self.ede_at_1 = self._ede_raw(1.0)

        # Closure mit mu_eff bei a=1
        mu_now = self.mu_of(1.0)
        self.Phi0 = (1.0 - mu_now * self.Omega_b - Omega_r - alpha) / f_sat(1.0)
        self._dL_interp = None

    def mu_of(self, a):
        """Skalenabhaengiges mu_eff(a)"""
        if self.mu_early == self.mu_late:
            return self.mu_late
        # Uebergang von mu_early (z>>1) zu mu_late (z~0)
        return self.mu_late + (self.mu_early - self.mu_late) / (1.0 + (a/self.a_mu)**4)

    def beta_eff(self, a):
        return beta_late_fix + (self.beta_early - beta_late_fix) / (1.0 + (a/self.a_t)**self.n_trans)

    def _ede_raw(self, a):
        if self.f_ede <= 0: return 0.0
        if self.ede_type == 'standard':
            return self.f_ede / (1.0 + (a/self.a_ede)**self.p_ede)
        elif self.ede_type == 'cosh':
            # Poeschl-Teller: 1/cosh^2 -- scharf lokalisiert
            x = np.log(a/self.a_ede)
            return self.f_ede / np.cosh(self.p_ede * x)**2
        elif self.ede_type == 'gaussian':
            # Gaussfoermig -- am schmalsten
            x = np.log(a/self.a_ede)
            return self.f_ede * np.exp(-0.5 * (x / (0.3/self.p_ede))**2)
        return 0.0

    def ede(self, a):
        return self._ede_raw(a) - self.ede_at_1

    def E2(self, a):
        if a <= 0: return 1e30
        b = self.beta_eff(a)
        mu = self.mu_of(a)
        return (mu * self.Omega_b * a**(-3)
                + Omega_r * a**(-4)
                + self.Phi0 * f_sat(a)
                + self.alpha * a**(-b)
                + self.ede(a))

    def Hz(self, z):
        return self.H0 * np.sqrt(max(self.E2(1.0/(1.0+z)), 1e-30))

    def d_C(self, z):
        r, _ = quad(lambda zz: 1.0/self.Hz(zz), 0, z, limit=2000)
        return c_light * r

    def r_s(self, z_t):
        def integ(a):
            mu = self.mu_of(a)
            Rb = 3.0 * mu * self.Omega_b / (4.0 * self.Omega_gamma) * a
            cs = 1.0 / np.sqrt(3.0 * (1.0 + Rb))
            e2 = self.E2(a)
            if e2 <= 0: return 0.0
            return cs / (a**2 * self.H0 * np.sqrt(e2))
        r, _ = quad(integ, 1e-12, 1.0/(1.0+z_t), limit=2000)
        return c_light * r

    def _build_dL(self, z_max=2.5, N=500):
        zg = np.linspace(0, z_max, N+1)
        inv_E = [1.0/np.sqrt(max(self.E2(1.0/(1.0+z)), 1e-30)) for z in zg]
        dz = zg[1] - zg[0]
        dC = np.cumsum(np.r_[0, 0.5*(np.array(inv_E[:-1])+np.array(inv_E[1:]))*dz]) * c_light/self.H0
        self._dL_interp = interp1d(zg, (1.0+zg)*dC, kind='cubic', fill_value='extrapolate')

    def chi2_SN(self, z_sn, mb, mbe):
        if self._dL_interp is None: self._build_dL(max(z_sn)+0.1)
        dL = np.maximum(self._dL_interp(z_sn), 1e-10)
        mu = 5.0*np.log10(dL) + 25.0
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
        dc = self.d_C(z_star); rs = self.r_s(z_star); rd = self.r_s(z_drag)
        la = np.pi * dc / rs if rs > 0 else 1e10
        a500 = 1.0/501.0
        Om = max((self.E2(a500) - Omega_r*a500**(-4))*a500**3, 0.001)
        R = np.sqrt(Om) * dc * self.H0 / c_light
        c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
        c2b = self.chi2_BAO(rd)
        # EDE fraction at z*
        E2s = self.E2(1.0/(1.0+z_star))
        edes = self.ede(1.0/(1.0+z_star))
        fede_frac = edes/E2s if E2s > 0 else 0
        return c2s+c2c+c2b, c2s, c2c, c2b, la, R, rd, fede_frac


class LCDM:
    def __init__(self, H0=67.36, Om=0.3153):
        self.H0=H0; self.Om=Om; self.OL=1-Om-Omega_r
    def E2(self,a): return self.Om*a**(-3)+Omega_r*a**(-4)+self.OL
    def Hz(self,z): return self.H0*np.sqrt(max(self.E2(1/(1+z)),1e-30))
    def d_C(self,z):
        r,_=quad(lambda zz:1/self.Hz(zz),0,z,limit=2000); return c_light*r
    def r_s(self,zt):
        Og=2.469e-5/(self.H0/100)**2; Ob=omega_b_BBN/(self.H0/100)**2
        Rbf=3*Ob/(4*Og)
        def integ(a):
            Rb=Rbf*a;cs=1/np.sqrt(3*(1+Rb));e2=self.E2(a)
            if e2<=0: return 0
            return cs/(a**2*self.H0*np.sqrt(e2))
        r,_=quad(integ,1e-12,1/(1+zt),limit=2000); return c_light*r


def main():
    t0 = time.time()
    out_dir = os.path.join(os.path.dirname(__file__), '_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'CFM_EDE_Reduktion.txt')

    L = []
    def log(s=''):
        L.append(s); print(s)

    log("  CFM+MOND: EDE-REDUKTIONS-ANALYSE")
    log("  " + "="*55)
    log(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log()

    z_sn, mb_sn, mbe_sn = load_pantheon()
    log(f"  {len(z_sn)} SNe geladen")
    log()

    # LCDM Referenz
    lcdm = LCDM()
    la_L, R_L, rs_L, rd_L, dc_L = lcdm.d_C(z_star), 0, lcdm.r_s(z_star), lcdm.r_s(z_drag), 0
    c2tot_L = 710.3  # bekannt
    log(f"  LCDM Referenz: chi2_tot = {c2tot_L:.1f}")
    log()

    results_all = []

    # ============================================================
    # TEST 1: KEIN EDE (nur Running Beta + mu=sqrt(pi))
    # ============================================================
    log("="*70)
    log("  TEST 1: KEIN EDE (nur Running Beta + mu=sqrt(pi))")
    log("="*70)
    log()

    def obj_noede(p):
        be, at, al, h0 = p
        if be<2 or be>4 or at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95 or h0<55 or h0>85: return 1e15
        Phi0 = (1-MU_PI*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_General(h0, al, be, at, mu_eff=MU_PI, f_ede=0)
            if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1]): return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)[0]
        except: return 1e15

    starts1 = [[2.82,0.09,0.63,60],[2.80,0.10,0.72,70],[2.90,0.07,0.55,65],[2.75,0.12,0.68,67.4]]
    best1 = 1e30; bx1 = None
    for s in starts1:
        try:
            r = minimize(obj_noede, s, method='Nelder-Mead',
                        options={'maxiter':600,'xatol':0.005,'fatol':0.5,'adaptive':True})
            if r.fun < best1: best1 = r.fun; bx1 = r.x.copy()
        except: pass
    if bx1 is not None:
        r = minimize(obj_noede, bx1, method='Nelder-Mead',
                    options={'maxiter':400,'xatol':0.002,'fatol':0.2,'adaptive':True})
        if r.fun < best1: best1 = r.fun; bx1 = r.x.copy()

        m = CFM_General(bx1[3], bx1[2], bx1[0], bx1[1], mu_eff=MU_PI, f_ede=0)
        c2t,c2s,c2c,c2b,la,R,rd,ff = m.full_chi2(z_sn, mb_sn, mbe_sn)
        log(f"  H0={bx1[3]:.1f} be={bx1[0]:.2f} at={bx1[1]:.4f} al={bx1[2]:.3f}")
        log(f"  l_A={la:.3f} R={R:.4f} r_d={rd:.2f}")
        log(f"  chi2: SN={c2s:.1f} CMB={c2c:.1f} BAO={c2b:.1f} TOT={c2t:.1f}")
        log(f"  Delta_chi2 = {c2t-c2tot_L:+.1f}")
        log(f"  f_EDE = 0%  <-- KEIN EDE!")
        results_all.append(("Kein EDE", bx1[3], MU_PI, rd, la, R, 0, c2t, c2t-c2tot_L))
    log()

    # ============================================================
    # TEST 2: EDE mit hohem p (scharf lokalisiert)
    # ============================================================
    log("="*70)
    log("  TEST 2: EDE MIT HOHEM p (p=12,20 statt 6)")
    log("="*70)
    log()

    for p_val in [10, 14, 20]:
        def obj_highp(p, pede=p_val):
            be, at, al, h0, fe, lae = p
            ae = 10**lae
            if be<2 or be>4 or at<0.005 or at>0.3: return 1e15
            if al<0.1 or al>0.95 or h0<55 or h0>85: return 1e15
            if ae<1e-5 or ae>0.1: return 1e15
            Phi0 = (1-MU_PI*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
            if Phi0 < 0: return 1e15
            try:
                m = CFM_General(h0, al, be, at, mu_eff=MU_PI,
                               f_ede=max(fe,0), a_ede=ae, p_ede=pede)
                if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1]): return 1e15
                return m.full_chi2(z_sn, mb_sn, mbe_sn)[0]
            except: return 1e15

        starts2 = [[2.70,0.15,0.70,69,6e8,-2.95],[2.80,0.10,0.65,67,2e8,-3],
                    [2.75,0.12,0.68,68,4e8,-3]]
        best2 = 1e30; bx2 = None
        for s in starts2:
            try:
                r = minimize(obj_highp, s, method='Nelder-Mead',
                            options={'maxiter':600,'xatol':0.005,'fatol':0.5,'adaptive':True})
                if r.fun < best2: best2 = r.fun; bx2 = r.x.copy()
            except: pass
        if bx2 is not None:
            r = minimize(obj_highp, bx2, method='Nelder-Mead',
                        options={'maxiter':400,'xatol':0.002,'fatol':0.2,'adaptive':True})
            if r.fun < best2: best2 = r.fun; bx2 = r.x.copy()

            be,at,al,h0,fe,lae = bx2; ae=10**lae; fe=max(fe,0)
            m = CFM_General(h0, al, be, at, mu_eff=MU_PI,
                           f_ede=fe, a_ede=ae, p_ede=p_val)
            c2t,c2s,c2c,c2b,la,R,rd,ff = m.full_chi2(z_sn, mb_sn, mbe_sn)
            log(f"  p_ede={p_val}: H0={h0:.1f} f_EDE={ff*100:.1f}% r_d={rd:.1f} chi2={c2t:.1f} (dX2={c2t-c2tot_L:+.1f})")
            results_all.append((f"EDE p={p_val}", h0, MU_PI, rd, la, R, ff*100, c2t, c2t-c2tot_L))
    log()

    # ============================================================
    # TEST 3: Poeschl-Teller EDE (cosh^-2)
    # ============================================================
    log("="*70)
    log("  TEST 3: POESCHL-TELLER EDE (cosh^-2 statt 1/(1+x^p))")
    log("="*70)
    log()

    def obj_cosh(p):
        be, at, al, h0, fe, lae, pw = p
        ae = 10**lae
        if be<2 or be>4 or at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95 or h0<55 or h0>85: return 1e15
        if ae<1e-5 or ae>0.1 or pw<1 or pw>20: return 1e15
        Phi0 = (1-MU_PI*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_General(h0, al, be, at, mu_eff=MU_PI,
                           f_ede=max(fe,0), a_ede=ae, p_ede=pw, ede_type='cosh')
            if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1]): return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)[0]
        except: return 1e15

    starts3 = [[2.70,0.15,0.70,69,6e8,-2.95,3],[2.80,0.10,0.65,67,2e8,-3,5],
                [2.75,0.12,0.68,68,4e8,-3,2]]
    best3 = 1e30; bx3 = None
    for s in starts3:
        try:
            r = minimize(obj_cosh, s, method='Nelder-Mead',
                        options={'maxiter':600,'xatol':0.005,'fatol':0.5,'adaptive':True})
            if r.fun < best3: best3 = r.fun; bx3 = r.x.copy()
        except: pass
    if bx3 is not None:
        r = minimize(obj_cosh, bx3, method='Nelder-Mead',
                    options={'maxiter':400,'xatol':0.002,'fatol':0.2,'adaptive':True})
        if r.fun < best3: best3 = r.fun; bx3 = r.x.copy()

        be,at,al,h0,fe,lae,pw = bx3; ae=10**lae; fe=max(fe,0)
        m = CFM_General(h0, al, be, at, mu_eff=MU_PI,
                       f_ede=fe, a_ede=ae, p_ede=pw, ede_type='cosh')
        c2t,c2s,c2c,c2b,la,R,rd,ff = m.full_chi2(z_sn, mb_sn, mbe_sn)
        log(f"  Poeschl-Teller: H0={h0:.1f} pw={pw:.1f} f_EDE={ff*100:.1f}%")
        log(f"  r_d={rd:.1f} l_A={la:.3f} R={R:.4f}")
        log(f"  chi2={c2t:.1f} (dX2={c2t-c2tot_L:+.1f})")
        results_all.append(("PT cosh^-2", h0, MU_PI, rd, la, R, ff*100, c2t, c2t-c2tot_L))
    log()

    # ============================================================
    # TEST 4: SKALENABHAENGIGES mu_eff(a) OHNE EDE
    # ============================================================
    log("="*70)
    log("  TEST 4: SKALENABHAENGIGES mu(a) OHNE EDE")
    log("="*70)
    log()
    log("  mu(a) = mu_late + (mu_early - mu_late)/(1+(a/a_mu)^4)")
    log("  Idee: Staerkeres mu bei z~1000 ersetzt EDE")
    log()

    def obj_muvar(p):
        be, at, al, h0, mu_e, la_mu = p
        a_mu = 10**la_mu
        if be<2 or be>4 or at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95 or h0<55 or h0>85: return 1e15
        if mu_e<1 or mu_e>10 or a_mu<1e-5 or a_mu>0.1: return 1e15
        # Closure mit mu bei a=1
        h = h0/100; Ob = omega_b_BBN/h**2
        # mu(1) ≈ mu_late = sqrt(pi)
        Phi0 = (1-MU_PI*Ob-Omega_r-al)/f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_General(h0, al, be, at, mu_eff=MU_PI, mu_early=mu_e, a_mu=a_mu, f_ede=0)
            if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1]): return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)[0]
        except: return 1e15

    starts4 = [[2.80,0.10,0.63,60,3.0,-3],[2.75,0.12,0.68,65,4.0,-3.5],
                [2.70,0.15,0.72,70,2.5,-2.5],[2.85,0.08,0.60,67,5.0,-3],
                [2.80,0.10,0.55,60,6.0,-3.5],[2.80,0.10,0.50,58,8.0,-4]]
    best4 = 1e30; bx4 = None
    for s in starts4:
        try:
            r = minimize(obj_muvar, s, method='Nelder-Mead',
                        options={'maxiter':800,'xatol':0.005,'fatol':0.5,'adaptive':True})
            if r.fun < best4: best4 = r.fun; bx4 = r.x.copy()
        except: pass
    if bx4 is not None:
        r = minimize(obj_muvar, bx4, method='Nelder-Mead',
                    options={'maxiter':500,'xatol':0.002,'fatol':0.2,'adaptive':True})
        if r.fun < best4: best4 = r.fun; bx4 = r.x.copy()

        be,at,al,h0,mu_e,la_mu = bx4; a_mu=10**la_mu
        m = CFM_General(h0, al, be, at, mu_eff=MU_PI, mu_early=mu_e, a_mu=a_mu, f_ede=0)
        c2t,c2s,c2c,c2b,la,R,rd,ff = m.full_chi2(z_sn, mb_sn, mbe_sn)
        log(f"  mu_early={mu_e:.2f} a_mu={a_mu:.2e} (z_mu={1/a_mu-1:.0f})")
        log(f"  H0={h0:.1f} be={be:.2f} at={at:.4f} al={al:.3f}")
        log(f"  r_d={rd:.1f} l_A={la:.3f} R={R:.4f}")
        log(f"  chi2={c2t:.1f} (dX2={c2t-c2tot_L:+.1f})")
        log(f"  f_EDE = 0%  <-- KEIN EDE!")
        # mu-Profil
        log(f"  mu-Profil:")
        for z in [0, 1, 10, 100, 500, 1090, 5000]:
            a = 1/(1+z)
            muv = m.mu_of(a)
            log(f"    z={z:5d}: mu={muv:.3f}")
        results_all.append(("mu(a) variabel", h0, mu_e, rd, la, R, 0, c2t, c2t-c2tot_L))
    log()

    # ============================================================
    # TEST 5: mu(a) + REDUZIERTES EDE
    # ============================================================
    log("="*70)
    log("  TEST 5: mu(a) + REDUZIERTES EDE")
    log("="*70)
    log()

    def obj_combined(p):
        be, at, al, h0, mu_e, la_mu, fe, lae = p
        a_mu = 10**la_mu; ae = 10**lae
        if be<2 or be>4 or at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95 or h0<55 or h0>85: return 1e15
        if mu_e<1 or mu_e>10 or a_mu<1e-5 or a_mu>0.1: return 1e15
        if ae<1e-5 or ae>0.1: return 1e15
        Phi0 = (1-MU_PI*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM_General(h0, al, be, at, mu_eff=MU_PI, mu_early=mu_e, a_mu=a_mu,
                           f_ede=max(fe,0), a_ede=ae, p_ede=6)
            if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1]): return 1e15
            return m.full_chi2(z_sn, mb_sn, mbe_sn)[0]
        except: return 1e15

    starts5 = [[2.70,0.15,0.70,69,3.0,-3,2e8,-3],[2.80,0.10,0.65,67,4.0,-3.5,1e8,-3],
                [2.75,0.12,0.68,68,2.5,-2.5,3e8,-3],[2.80,0.10,0.60,65,5.0,-3,5e7,-3]]
    best5 = 1e30; bx5 = None
    for s in starts5:
        try:
            r = minimize(obj_combined, s, method='Nelder-Mead',
                        options={'maxiter':800,'xatol':0.005,'fatol':0.5,'adaptive':True})
            if r.fun < best5: best5 = r.fun; bx5 = r.x.copy()
        except: pass
    if bx5 is not None:
        r = minimize(obj_combined, bx5, method='Nelder-Mead',
                    options={'maxiter':500,'xatol':0.002,'fatol':0.2,'adaptive':True})
        if r.fun < best5: best5 = r.fun; bx5 = r.x.copy()

        be,at,al,h0,mu_e,la_mu,fe,lae = bx5; a_mu=10**la_mu; ae=10**lae; fe=max(fe,0)
        m = CFM_General(h0, al, be, at, mu_eff=MU_PI, mu_early=mu_e, a_mu=a_mu,
                       f_ede=fe, a_ede=ae, p_ede=6)
        c2t,c2s,c2c,c2b,la,R,rd,ff = m.full_chi2(z_sn, mb_sn, mbe_sn)
        log(f"  mu_early={mu_e:.2f} a_mu={a_mu:.2e} f_ede={fe:.1e}")
        log(f"  H0={h0:.1f} be={be:.2f} at={at:.4f} al={al:.3f}")
        log(f"  r_d={rd:.1f} l_A={la:.3f} R={R:.4f}")
        log(f"  chi2={c2t:.1f} (dX2={c2t-c2tot_L:+.1f})")
        log(f"  f_EDE(z*) = {ff*100:.1f}%")
        results_all.append(("mu(a)+EDE", h0, mu_e, rd, la, R, ff*100, c2t, c2t-c2tot_L))
    log()

    # ============================================================
    # ZUSAMMENFASSUNG
    # ============================================================
    log("="*70)
    log("  ZUSAMMENFASSUNG: EDE-REDUKTION")
    log("="*70)
    log()
    log(f"  {'Modell':22s} {'H0':>5} {'mu':>5} {'rd':>6} {'lA':>7} {'R':>6} {'fEDE%':>5} {'X2tot':>6} {'dX2':>5}")
    log("  "+"-"*80)
    log(f"  {'LCDM':22s} {'67.4':>5} {'1.00':>5} {'147.2':>6} {'301.43':>7} {'1.750':>6} {'0':>5} {'710.3':>6} {'+0.0':>5}")
    log(f"  {'sqrt(pi), p=6 (alt)':22s} {'69.0':>5} {'1.77':>5} {'143.3':>6} {'301.47':>7} {'1.750':>6} {'59':>5} {'704.2':>6} {'-6.1':>5}")
    for name, h0, mu, rd, la, R, fede, c2t, dc2 in results_all:
        log(f"  {name:22s} {h0:5.1f} {mu:5.2f} {rd:6.1f} {la:7.3f} {R:6.4f} {fede:5.1f} {c2t:6.1f} {dc2:+5.1f}")
    log()
    log(f"  Laufzeit: {time.time()-t0:.1f} Sekunden")

    with open(out_path, 'w') as f:
        f.write('\n'.join(L))
    print(f"\n  -> Ergebnisse in {out_path}")


if __name__ == '__main__':
    main()

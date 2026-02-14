"""Quick test: CFM+MOND with mu_eff = sqrt(pi) = 1.7725 (fixed)"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import os, time

c_light = 299792.458
z_star = 1089.80; z_drag = 1059.94
Omega_r = 9.15e-5; omega_b_BBN = 0.02237
beta_late_fix = 2.02; k_fsat = 9.81; a_trans_fsat = 0.971
lA_pl = 301.471; lA_err = 0.14; R_pl = 1.7502; R_err = 0.0046
n_fix = 4; p_ede_fix = 6
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
            parts = line.split()
            if len(parts) < 10: continue
            try: zv = float(parts[2]); mbv = float(parts[8]); mbev = float(parts[9])
            except: continue
            if zv > 0.01: z.append(zv); mb.append(mbv); mbe.append(mbev)
    return np.array(z), np.array(mb), np.array(mbe)

def f_sat(a):
    x = k_fsat * (a - a_trans_fsat)
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0 / (1.0 + np.exp(-x))

class CFM:
    def __init__(self, H0, al, be, at, mu, fe=0, ae=1e-3):
        self.H0 = H0; h = H0/100.0
        self.Ob = omega_b_BBN/h**2; self.Og = 2.469e-5/h**2
        self.mu = mu; self.al = al; self.be = be; self.at = at
        self.fe = fe; self.ae = ae
        self.ede1 = fe/(1.0+(1.0/ae)**p_ede_fix) if fe > 0 else 0
        self.Phi0 = (1.0 - mu*self.Ob - Omega_r - al) / f_sat(1.0)
        self._dL = None

    def beta(self, a):
        return beta_late_fix + (self.be - beta_late_fix)/(1.0+(a/self.at)**n_fix)

    def ede(self, a):
        if self.fe <= 0: return 0.0
        return self.fe/(1.0+(a/self.ae)**p_ede_fix) - self.ede1

    def E2(self, a):
        if a <= 0: return 1e30
        return (self.mu*self.Ob*a**(-3) + Omega_r*a**(-4)
                + self.Phi0*f_sat(a) + self.al*a**(-self.beta(a)) + self.ede(a))

    def Hz(self, z): return self.H0*np.sqrt(max(self.E2(1/(1+z)), 1e-30))

    def dC(self, z):
        r, _ = quad(lambda zz: 1.0/self.Hz(zz), 0, z, limit=2000)
        return c_light * r

    def rs(self, zt):
        Rbf = 3.0*self.mu*self.Ob/(4.0*self.Og)
        def integ(a):
            Rb = Rbf*a; cs = 1.0/np.sqrt(3.0*(1.0+Rb))
            e2 = self.E2(a)
            if e2 <= 0: return 0.0
            return cs/(a**2*self.H0*np.sqrt(e2))
        r, _ = quad(integ, 1e-12, 1.0/(1.0+zt), limit=2000)
        return c_light * r

    def build_dL(self, zmax=2.5, N=500):
        zg = np.linspace(0, zmax, N+1)
        iE = [1.0/np.sqrt(max(self.E2(1/(1+z)),1e-30)) for z in zg]
        dz = zg[1]-zg[0]
        dC = np.cumsum(np.r_[0, 0.5*(np.array(iE[:-1])+np.array(iE[1:]))*dz])*c_light/self.H0
        self._dL = interp1d(zg, (1+zg)*dC, kind='cubic', fill_value='extrapolate')

    def chi2_SN(self, zs, mb, mbe):
        if self._dL is None: self.build_dL(max(zs)+0.1)
        dL = np.maximum(self._dL(zs), 1e-10)
        mu = 5*np.log10(dL)+25; d = mb-mu; w = 1.0/mbe**2
        return np.sum(d**2*w) - (np.sum(d*w))**2/np.sum(w)

    def chi2_BAO(self, rd):
        c2 = 0.0
        for zb, ot, ov, oe in BAO_DATA:
            if ot == 'DM/rd': th = self.dC(zb)/rd
            elif ot == 'DH/rd': th = c_light/(self.Hz(zb)*rd)
            elif ot == 'DV/rd':
                dc = self.dC(zb); dh = c_light/self.Hz(zb)
                th = (zb*dc**2*dh)**(1./3.)/rd
            else: continue
            c2 += ((th-ov)/oe)**2
        return c2

    def full(self, zs, mb, mbe):
        self._dL = None
        c2s = self.chi2_SN(zs, mb, mbe)
        dc = self.dC(z_star); rss = self.rs(z_star); rd = self.rs(z_drag)
        la = np.pi*dc/rss
        a500 = 1.0/501.0
        Om = max((self.E2(a500)-Omega_r*a500**(-4))*a500**3, 0.001)
        R = np.sqrt(Om)*dc*self.H0/c_light
        c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
        c2b = self.chi2_BAO(rd)
        return c2s+c2c+c2b, c2s, c2c, c2b, la, R, rd

def main():
    t0 = time.time()
    zs, mb, mbe = load_pantheon()
    print(f"CFM+MOND mit mu_eff = sqrt(pi) = {MU_PI:.5f}")
    print("="*60)
    print(f"{len(zs)} SNe geladen\n")

    def obj(p):
        be, at, al, h0, fe, lae = p
        ae = 10**lae
        if be<2.0 or be>4.0: return 1e15
        if at<0.005 or at>0.3: return 1e15
        if al<0.1 or al>0.95: return 1e15
        if h0<50 or h0>85: return 1e15
        if ae<1e-5 or ae>0.1: return 1e15
        Phi0 = (1.0-MU_PI*omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
        if Phi0 < 0: return 1e15
        try:
            m = CFM(h0, al, be, at, MU_PI, fe, ae)
            if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1.0]): return 1e15
            return m.full(zs, mb, mbe)[0]
        except: return 1e15

    starts = [
        [2.70, 0.15, 0.69, 66.0, 4.8e8, np.log10(1.13e-3)],
        [2.80, 0.10, 0.72, 70.0, 1e8, -3.0],
        [2.74, 0.14, 0.72, 65.0, 3e8, -2.9],
        [2.90, 0.08, 0.60, 67.4, 2e8, -3.0],
        [2.80, 0.12, 0.65, 68.0, 5e8, -3.0],
    ]

    best = 1e30; bx = None
    for i, s in enumerate(starts):
        try:
            r = minimize(obj, s, method='Nelder-Mead',
                        options={'maxiter':800, 'xatol':0.005, 'fatol':0.5, 'adaptive':True})
            if r.fun < best: best = r.fun; bx = r.x.copy()
            print(f"  Start {i+1}: chi2={r.fun:.1f} H0={r.x[3]:.1f} be={r.x[0]:.2f}")
        except: pass

    # Polish
    for _ in range(3):
        try:
            r = minimize(obj, bx, method='Nelder-Mead',
                        options={'maxiter':600, 'xatol':0.002, 'fatol':0.2, 'adaptive':True})
            if r.fun < best: best = r.fun; bx = r.x.copy()
        except: break

    be, at, al, h0, fe, lae = bx
    ae = 10**lae; fe = max(fe, 0)
    m = CFM(h0, al, be, at, MU_PI, fe, ae)
    c2t, c2s, c2c, c2b, la, R, rd = m.full(zs, mb, mbe)

    print(f"\n{'='*60}")
    print(f"  ERGEBNIS: mu_eff = sqrt(pi) = {MU_PI:.5f}")
    print(f"{'='*60}")
    print(f"  H0         = {h0:.2f} km/s/Mpc")
    print(f"  beta_early = {be:.4f}")
    print(f"  a_t        = {at:.5f}  (z_t = {1/at-1:.1f})")
    print(f"  alpha      = {al:.4f}")
    print(f"  f_ede      = {fe:.2e}")
    print(f"  a_ede      = {ae:.2e}")
    print(f"  Phi0       = {m.Phi0:.4f}")
    print(f"  Ob_phys    = {m.Ob:.5f}")
    print(f"  Ob_eff     = {MU_PI*m.Ob:.5f}")
    print(f"  3*sqrt(pi)*Ob = {3*MU_PI*m.Ob:.4f}  (LCDM: Om_m=0.315)")
    print()
    print(f"  l_A  = {la:.3f}  (Planck: 301.471)")
    print(f"  R    = {R:.4f}  (Planck: 1.7502)")
    print(f"  r_d  = {rd:.2f} Mpc  (LCDM: 147.18)")
    print()
    print(f"  chi2_SN  = {c2s:.1f}  (LCDM: 700.9)")
    print(f"  chi2_CMB = {c2c:.1f}  (LCDM: 0.1)")
    print(f"  chi2_BAO = {c2b:.1f}  (LCDM: 9.3)")
    print(f"  chi2_TOT = {c2t:.1f}  (LCDM: 710.3)")
    print(f"  Delta_chi2 = {c2t-710.3:+.1f}")

    if fe > 0:
        E2s = m.E2(1/(1+z_star))
        es = m.ede(1/(1+z_star))
        ff = es/E2s if E2s > 0 else 0
        print(f"\n  f_EDE(z*) = {ff:.4f} ({ff*100:.1f}%)")

    print(f"\n  Laufzeit: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()

"""
CFM Combined Fit: Running Beta + Early Dark Energy + MOND-Analyse
=================================================================
1. Running beta: beta_eff(a) Uebergang von ~3 (CDM) zu ~2 (MOND)
2. Parametrisches EDE: Zusatzenergie bei z > z_ede zur r_d-Korrektur
3. MOND-Analyse: Beschleunigungsschwellen und Interaktionseffekte

Frage: Was passiert wenn keine Kruemmung mehr zurueckgegeben wird?
-> Uebergang zum MOND-Regime auf Galaxienskalen
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
H0_SI = 2.183e-18         # s^-1 (fuer H0=67.36)
a0_MOND = 1.2e-10         # m/s^2
z_star = 1089.80
z_drag = 1059.94
Omega_r = 9.15e-5
omega_b_BBN = 0.02237
beta_late_fix = 2.02
k_fsat = 9.81
a_trans_fsat = 0.971

# Planck 2018
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
    def __init__(self, H0):
        self.H0 = H0
        h = H0/100.0
        self.Omega_b = omega_b_BBN / h**2
        self.Omega_gamma = 2.469e-5 / h**2
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
        Rb_f = 3.0*self.Omega_b/(4.0*self.Omega_gamma)
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
        super().__init__(H0)
        self.Om = Om; self.OL = 1.0-Om-Omega_r
    def E2(self, a):
        return self.Om*a**(-3) + Omega_r*a**(-4) + self.OL
    def cmb_obs(self):
        dc = self.d_C(z_star); rs = self.r_s(z_star); rd = self.r_s(z_drag)
        la = np.pi*dc/rs
        R = np.sqrt(self.Om)*dc*self.H0/c_light
        return la, R, rs, rd, dc


class CFMCombined(CosmoBase):
    """Running Beta + parametrisches EDE"""
    def __init__(self, H0, alpha, beta_early, a_t, n_trans=4,
                 f_ede=0, a_ede=1e-3, p_ede=6):
        super().__init__(H0)
        self.alpha = alpha
        self.beta_early = beta_early
        self.a_t = a_t
        self.n_trans = n_trans
        self.f_ede = f_ede
        self.a_ede = a_ede
        self.p_ede = p_ede
        # EDE-Korrektur bei a=1 (Closure)
        self.ede_at_1 = f_ede/(1.0+(1.0/a_ede)**p_ede) if f_ede > 0 and a_ede > 0 else 0
        # Phi0 (DE)
        self.Phi0 = (1.0 - self.Omega_b - Omega_r - alpha) / f_sat(1.0)

    def beta_eff(self, a):
        return beta_late_fix + (self.beta_early - beta_late_fix)/(1.0+(a/self.a_t)**self.n_trans)

    def ede(self, a):
        if self.f_ede <= 0: return 0.0
        return self.f_ede/(1.0+(a/self.a_ede)**self.p_ede) - self.ede_at_1

    def E2(self, a):
        if a <= 0: return 1e30
        b = self.beta_eff(a)
        return (self.Omega_b*a**(-3) + Omega_r*a**(-4)
                + self.Phi0*f_sat(a) + self.alpha*a**(-b) + self.ede(a))


# ===============================================================
# HAUPTPROGRAMM
# ===============================================================
def main():
    t0 = time.time()
    out_dir = os.path.join(os.path.dirname(__file__), '_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Combined_Fit.txt')

    L = []
    def log(s=''):
        L.append(s); print(s)

    log("  CFM COMBINED: RUNNING BETA + EDE + MOND-ANALYSE")
    log("  " + "="*55)
    log(f"  Datum: {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
    # TEIL A: MOND-ANALYSE
    # ===========================================================
    log("="*70)
    log("  TEIL A: MOND-ANALYSE")
    log("="*70)
    log()
    log("  Ist MOND bereits beruecksichtigt?")
    log("  ----------------------------------")
    log("  MOND wirkt auf PERTURBATIONS-Ebene (Galaxienskalen),")
    log("  NICHT auf Hintergrund-Ebene (Friedmann-Gleichung).")
    log("  -> Im aktuellen Fit: NEIN, MOND ist nicht enthalten.")
    log("  -> MOND aendert mu_eff=4/3 fuer Strukturwachstum,")
    log("     nicht H(z) oder r_s oder l_A.")
    log()
    log("  Kosmologische vs. Galaxien-Beschleunigung:")
    log(f"  a_0(MOND) = {a0_MOND:.1e} m/s^2")
    log()
    log(f"  {'z':>6}  {'H(z) [km/s/Mpc]':>16}  {'a_H [m/s^2]':>12}  {'a_H/a_0':>8}  {'Regime':>12}")
    log("  " + "-"*62)

    # Berechne mit dem Running-Beta Modell (vorlaeufig)
    cfm_ref = CFMCombined(60.0, 0.628, 2.82, 0.092)
    for z in [0, 0.5, 1, 2, 5, 10, 50, 100, 500, 1090]:
        Hz = cfm_ref.Hz(z)
        H_SI = Hz * 1e3 / 3.086e22  # km/s/Mpc -> s^-1
        a_H = H_SI * c_SI  # Hubble-Beschleunigung
        ratio = a_H / a0_MOND
        regime = "MOND" if ratio < 1 else ("Uebergang" if ratio < 10 else "Newton")
        log(f"  {z:6.0f}  {Hz:16.2f}  {a_H:12.2e}  {ratio:8.1f}  {regime:>12}")
    log()

    log("  Was passiert wenn keine Kruemmung mehr zurueckgegeben wird?")
    log("  -----------------------------------------------------------")
    log("  z > z_t (~10): Beta_eff ~ 3 -> CDM-artige Rueckgabe aktiv")
    log("    -> Starke Gravitationspotentiale, Newton-Regime")
    log("    -> Strukturen wachsen wie in LCDM")
    log()
    log("  z < z_t (~10): Beta_eff -> 2 -> Rueckgabe klingt ab")
    log("    -> Gravitationspotentiale werden flacher")
    log("    -> Auf Galaxienskalen: Beschleunigung sinkt unter a_0")
    log("    -> MOND-Regime aktiviert sich NATUERLICH")
    log("    -> Flache Rotationskurven entstehen OHNE CDM-Halos")
    log()
    log("  INTERAKTIONSEFFEKT:")
    log("  Der Uebergang beta=3 -> beta=2 IST der MOND-Uebergang!")
    log("  - Kosmologisch: geometrische DM statt Teilchen-CDM")
    log("  - Galaxien: MOND-Enhancement (mu_eff=4/3) kompensiert")
    log("    den Verlust der CDM-artigen Potentiale")
    log("  - Selbstkonsistenz: Dieselbe Kruemmungsschwelle steuert")
    log("    sowohl den Hintergrund-Uebergang als auch MOND")
    log()

    # ===========================================================
    # TEIL B: GRID-SCAN Running Beta + EDE
    # ===========================================================
    log("="*70)
    log("  TEIL B: GRID-SCAN (Running Beta + EDE)")
    log("="*70)
    log()

    n_fix = 4
    p_ede_fix = 6

    # Parameter-Gitter
    be_grid  = [2.7, 2.8, 2.85, 2.9, 3.0]
    at_grid  = [0.03, 0.06, 0.1]
    al_grid  = [0.55, 0.63, 0.72]
    h0_grid  = [60, 63, 65, 67.4, 70]
    fe_grid  = [0, 5e7, 1e8, 2e8, 4e8]
    ae_grid  = [1e-4, 5e-4, 1e-3]

    N_tot = len(be_grid)*len(at_grid)*len(al_grid)*len(h0_grid)*len(fe_grid)*len(ae_grid)
    log(f"  Grid: {len(be_grid)}x{len(at_grid)}x{len(al_grid)}x{len(h0_grid)}x{len(fe_grid)}x{len(ae_grid)} = {N_tot}")
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
            for fe in fe_grid:
              for ae in ae_grid:
                cnt += 1
                if cnt % 100 == 0:
                    el = time.time()-t_s
                    print(f"  ... {cnt}/{N_tot} ({el:.0f}s)", end='\r')

                # Skip: wenn kein EDE, nur eine ae-Iteration
                if fe == 0 and ae != ae_grid[0]:
                    continue

                h = h0/100.0
                Ob = omega_b_BBN/h**2
                Phi0 = (1.0-Ob-Omega_r-al)/f_sat(1.0)
                if Phi0 < 0: continue

                try:
                    m = CFMCombined(h0, al, be, at, n_fix, fe, ae, p_ede_fix)
                    if any(m.E2(a) <= 0 for a in [1e-6,1e-4,1e-3,0.01,0.1,1.0]):
                        continue

                    c2s = m.chi2_SN(z_sn, mb_sn, mbe_sn)
                    m._dL_interp = None
                    la, R, rs, rd, dc = m.cmb_obs()
                    c2c = ((la-lA_pl)/lA_err)**2 + ((R-R_pl)/R_err)**2
                    c2b = m.chi2_BAO(rd)
                    c2t = c2s+c2c+c2b

                    results.append((be,at,al,h0,fe,ae, c2s,c2c,c2b,c2t, la,R,rd,rs))

                    if c2t < best_c2:
                        best_c2 = c2t
                        best_p = (be,at,al,h0,fe,ae)
                except:
                    continue

    print(" "*60, end='\r')
    log(f"  Scan fertig: {len(results)} Punkte in {time.time()-t_s:.1f}s")
    log()

    results.sort(key=lambda x: x[9])

    log("  TOP-15:")
    log(f"  {'be':>5} {'at':>5} {'al':>5} {'H0':>5} {'f_ede':>8} {'a_ede':>7}  {'X2sn':>7} {'X2cmb':>7} {'X2bao':>7} {'X2tot':>7}  {'lA':>7} {'R':>7} {'rd':>6}")
    log("  "+"-"*105)
    for r in results[:15]:
        log(f"  {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:5.1f} {r[4]:8.0e} {r[5]:7.1e}"
            f"  {r[6]:7.1f} {r[7]:7.1f} {r[8]:7.1f} {r[9]:7.1f}"
            f"  {r[10]:7.2f} {r[11]:7.4f} {r[12]:6.1f}")
    log()

    # Vergleich: mit vs ohne EDE in Top-Ergebnissen
    no_ede = [r for r in results if r[4] == 0]
    with_ede = [r for r in results if r[4] > 0]
    if no_ede:
        log(f"  Bester OHNE EDE: chi2={no_ede[0][9]:.1f} (lA={no_ede[0][10]:.2f}, R={no_ede[0][11]:.4f}, rd={no_ede[0][12]:.1f})")
    if with_ede:
        log(f"  Bester MIT  EDE: chi2={with_ede[0][9]:.1f} (lA={with_ede[0][10]:.2f}, R={with_ede[0][11]:.4f}, rd={with_ede[0][12]:.1f})")
    log()

    # ===========================================================
    # TEIL C: FEINOPTIMIERUNG
    # ===========================================================
    log("="*70)
    log("  TEIL C: FEINOPTIMIERUNG (6D Nelder-Mead)")
    log("="*70)
    log()

    if best_p:
        be0,at0,al0,h00,fe0,ae0 = best_p
        log(f"  Start: be={be0} at={at0} al={al0} H0={h00} fe={fe0:.0e} ae={ae0:.0e}")
        log(f"  Start chi2 = {best_c2:.1f}")
        log()

        def objective(p):
            be = p[0]; at = p[1]; al = p[2]; h0 = p[3]
            fe = max(p[4], 0); ae = 10**p[5]
            if be<2.0 or be>4.0: return 1e15
            if at<0.005 or at>0.3: return 1e15
            if al<0.2 or al>0.95: return 1e15
            if h0<60 or h0>80: return 1e15
            if ae<1e-5 or ae>0.1: return 1e15
            Phi0 = (1.0-omega_b_BBN/(h0/100)**2-Omega_r-al)/f_sat(1.0)
            if Phi0 < 0: return 1e15
            try:
                m = CFMCombined(h0, al, be, at, n_fix, fe, ae, p_ede_fix)
                if any(m.E2(a)<=0 for a in [1e-6,1e-4,0.01,1.0]):
                    return 1e15
                return m.full_chi2(z_sn, mb_sn, mbe_sn)
            except:
                return 1e15

        x0 = [be0, at0, al0, h00, fe0, np.log10(max(ae0,1e-5))]
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 800, 'xatol':0.005, 'fatol':0.5, 'adaptive':True})

        be_o = res.x[0]; at_o = res.x[1]; al_o = res.x[2]; h0_o = res.x[3]
        fe_o = max(res.x[4],0); ae_o = 10**res.x[5]

        log(f"  Optimierung: {res.nfev} Eval, chi2={res.fun:.1f}")
        log()

        m_o = CFMCombined(h0_o, al_o, be_o, at_o, n_fix, fe_o, ae_o, p_ede_fix)
        la_o, R_o, rs_o, rd_o, dc_o = m_o.cmb_obs()
        c2s_o = m_o.chi2_SN(z_sn, mb_sn, mbe_sn)
        m_o._dL_interp = None
        c2c_o = ((la_o-lA_pl)/lA_err)**2 + ((R_o-R_pl)/R_err)**2
        c2b_o = m_o.chi2_BAO(rd_o)
        c2t_o = c2s_o + c2c_o + c2b_o

        log("  OPTIMIERTES ERGEBNIS:")
        log("  " + "="*50)
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
            # Fraktion bei z*
            E2_star = m_o.E2(1.0/(1.0+z_star))
            ede_star = m_o.ede(1.0/(1.0+z_star))
            f_frac = ede_star / E2_star if E2_star > 0 else 0
            log(f"    f_EDE(z*)  = {f_frac:.4f}  ({f_frac*100:.2f}%)")
        log(f"  Hintergrund:")
        log(f"    H0         = {h0_o:.2f} km/s/Mpc")
        log(f"    Omega_b    = {m_o.Omega_b:.5f}")
        log(f"    Phi0       = {m_o.Phi0:.4f}")
        log()
        log(f"  {'':22s} {'CFM+EDE':>12s} {'LCDM':>12s} {'Planck':>12s}")
        log(f"  {'l_A':22s} {la_o:12.3f} {la_L:12.3f} {lA_pl:12.3f}")
        log(f"  {'R':22s} {R_o:12.4f} {R_L:12.4f} {R_pl:12.4f}")
        log(f"  {'r_s(z*) [Mpc]':22s} {rs_o:12.2f} {rs_L:12.2f}")
        log(f"  {'r_d [Mpc]':22s} {rd_o:12.2f} {rd_L:12.2f}")
        log(f"  {'d_C(z*) [Mpc]':22s} {dc_o:12.2f} {dc_L:12.2f}")
        log()
        log(f"  {'':22s} {'CFM+EDE':>12s} {'LCDM':>12s}")
        log(f"  {'chi2_SN':22s} {c2s_o:12.1f} {c2sn_L:12.1f}")
        log(f"  {'chi2_CMB (lA+R)':22s} {c2c_o:12.1f} {c2cmb_L:12.1f}")
        log(f"  {'chi2_BAO':22s} {c2b_o:12.1f} {c2bao_L:12.1f}")
        log(f"  {'chi2_TOTAL':22s} {c2t_o:12.1f} {c2tot_L:12.1f}")
        log(f"  {'Delta_chi2':22s} {c2t_o-c2tot_L:12.1f}")
        log()

        # beta_eff + EDE Profil
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
    # TEIL D: MOND-INTERAKTIONSEFFEKTE
    # ===========================================================
    log("="*70)
    log("  TEIL D: MOND-INTERAKTIONSEFFEKTE IM DETAIL")
    log("="*70)
    log()

    if best_p:
        z_t = 1.0/at_o - 1
        log(f"  Uebergangs-Rotverschiebung z_t = {z_t:.1f}")
        log()
        log("  PHASEN DES CFM+MOND-UNIVERSUMS:")
        log()
        log("  Phase 1: z > z_t (Hochkruemmungs-Regime)")
        log(f"    beta_eff ~ {be_o:.2f} (nahe CDM-Skalierung a^-3)")
        log("    Starke Kruemmungsrueckgabe -> tiefe Potentialtoepfe")
        log("    Galaxien-Beschleunigung a >> a_0 -> Newton-Regime")
        log("    MOND-Effekte: NICHT aktiv (zu hohe Beschleunigungen)")
        log("    -> Universum VERHAELT SICH wie LCDM")
        log()
        log("  Phase 2: z ~ z_t (Uebergangsphase)")
        log("    beta_eff wandert von ~3 zu ~2")
        log("    Kruemmungsrueckgabe endet, Potentiale werden flacher")
        log("    Galaxien-Beschleunigung sinkt Richtung a_0")
        log("    -> CDM-artiges Verhalten schwaecher")
        log("    -> MOND beginnt sich zu manifestieren")
        log()
        log("  Phase 3: z < z_t (Tiefkruemmungs-Regime)")
        log(f"    beta_eff ~ {beta_late_fix} (geometrisch, ~Kruemmung)")
        log("    Keine aktive Rueckgabe mehr -> rein geometrischer Term")
        log("    Auf Galaxienskalen: a ~ a_0 -> MOND aktiv")
        log("    mu_eff = 4/3 -> flache Rotationskurven OHNE CDM-Halos")
        log("    -> MOND ERSETZT CDM auf kleinen Skalen")
        log()
        log("  DER SCHLUESSEL-MECHANISMUS:")
        log("  Wenn die Kruemmungsrueckgabe endet (z < z_t), fehlt den")
        log("  Galaxien die 'CDM-artige' gravitationsunterstuetzung.")
        log("  Aber genau DANN aktiviert sich MOND und kompensiert:")
        log("    CDM-Beschleunigung:  a_CDM = G*M/r^2 * (Om_CDM/Om_b)")
        log(f"    MOND-Enhancement:    a_MOND = a_Newton * mu_eff = a_N * 4/3")
        log("    Noetig: mu_eff * Om_b ~ Om_total(LCDM)")
        log(f"    Check: 4/3 * {m_o.Omega_b:.3f} = {4/3*m_o.Omega_b:.3f}")
        log(f"           vs Om_m(LCDM) = 0.315")
        log()
        if 4/3 * m_o.Omega_b < 0.15:
            log("  -> MOND allein reicht NICHT aus!")
            log(f"    4/3 * Ob = {4/3*m_o.Omega_b:.3f} << 0.315")
            log("    Der geometrische Term (alpha*a^-2) traegt den Rest bei.")
            log(f"    Bei z=0: alpha = {al_o:.3f} -> effektive 'Masse' = {al_o:.3f}")
            log(f"    Total: Ob + alpha = {m_o.Omega_b + al_o:.3f}")
        log()

    # ===========================================================
    # ZUSAMMENFASSUNG
    # ===========================================================
    log("="*70)
    log("  ZUSAMMENFASSUNG")
    log("="*70)
    log()

    if best_p:
        dc2 = c2t_o - c2tot_L
        log(f"  Delta_chi2 (CFM+EDE vs LCDM) = {dc2:+.1f}")
        log()
        log(f"  Parameter:")
        log(f"    beta_early={be_o:.3f}, a_t={at_o:.4f} (z_t={1/at_o-1:.0f})")
        log(f"    alpha={al_o:.3f}, H0={h0_o:.1f}")
        log(f"    f_ede={fe_o:.1e}, a_ede={ae_o:.1e}")
        log()

        # Diagnose
        if rd_o > 160:
            log(f"  r_d = {rd_o:.1f} Mpc (LCDM: {rd_L:.1f}) -> r_d-Problem bleibt")
            log("  Moegliche Loesungen:")
            log("    1. Staerkeres EDE (mehr Energie bei z > 1000)")
            log("    2. beta_early naeher an 3.0")
            log("    3. Zusaetzliche Physik (Neutrino-Masse, etc.)")
        elif abs(rd_o - rd_L) < 10:
            log(f"  r_d = {rd_o:.1f} Mpc -> r_d-Problem GELOEST!")

        if dc2 < 20:
            log("  >>> CFM IST KOMPETITIV MIT LCDM! <<<")
        elif dc2 < 100:
            log(f"  CFM ist moderat schlechter (Delta={dc2:.0f})")
        else:
            log(f"  CFM bleibt {dc2:.0f} chi2-Einheiten schlechter")

    log()
    log(f"  Laufzeit: {time.time()-t0:.1f} Sekunden")

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log(f"  Gespeichert: {out_path}")


if __name__ == '__main__':
    main()

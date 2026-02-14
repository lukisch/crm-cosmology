#!/usr/bin/env python3
"""
===============================================================================
BETA + ALPHA RE-FIT MIT SKALARFELD
===============================================================================
Letzte Chance fuer CFM: Kann eine Erhoehung von beta (naeher an CDM a^{-3})
zusammen mit alpha-Anpassung und Skalarfeld ALLE drei CMB/BAO-Observablen
gleichzeitig retten?

Probleme bei beta=2.02:
  - r_d = 200 Mpc (soll 148) -> BAO kaputt
  - R = 0.44 (soll 1.75) -> CMB shift kaputt
  - l_A = 317 (soll 301) -> Skalarfeld loest das

Hoffnung: beta -> 3 macht geom. DM ~ CDM:
  - r_d sinkt (mehr Materie bei z > z*)
  - R steigt (Omega_m_eff groesser)
  - alpha muss re-fitted werden
  - Skalarfeld feintuned l_A

Scan: beta x alpha x H0 x omega_V x psi0
===============================================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
import os, time, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_results")
DATA_DIR = os.path.join(SCRIPT_DIR, "_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_lines = []
def out(text=""):
    print(text)
    output_lines.append(text)

def save_output():
    path = os.path.join(OUTPUT_DIR, "Beta_Alpha_Refit.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    out(f"\n  [Gespeichert: {path}]")

# ================================================================
# KONSTANTEN
# ================================================================
c_light = 299792.458
Omega_gamma = 5.38e-5
Omega_nu = 3.65e-5
Omega_r = Omega_gamma + Omega_nu
z_star = 1089.92
a_star = 1.0 / (1 + z_star)
z_drag = 1059.62

PLANCK_lA = 301.471;  PLANCK_lA_err = 0.090
PLANCK_R  = 1.7502;   PLANCK_R_err  = 0.0046
PLANCK_wb = 0.02236;  PLANCK_wb_err = 0.00015

# BAO (BOSS DR12 + SDSS MGS + eBOSS Lya)
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

# CFM feste Parameter
K_CFM = 9.81
A_TRANS = 0.971


# ================================================================
# DATEN LADEN
# ================================================================
def load_pantheon():
    path = os.path.join(DATA_DIR, "Pantheon+SH0ES.dat")
    z_arr, mb_arr, mb_err_arr = [], [], []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        header = True
        for line in f:
            if header:
                if line.startswith('#') or line.strip() == '':
                    continue
                header = False
                continue
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                z = float(parts[2])
                mb = float(parts[8])
                mb_err = float(parts[9])
                if z > 0.01 and mb_err > 0 and mb_err < 5.0:
                    z_arr.append(z)
                    mb_arr.append(mb)
                    mb_err_arr.append(mb_err)
            except (ValueError, IndexError):
                continue
    return np.array(z_arr), np.array(mb_arr), np.array(mb_err_arr)


# ================================================================
# CFM E^2 MIT ALLEN PARAMETERN
# ================================================================
def cfm_E2(a, Ob, alpha, beta):
    """CFM E^2(a) = H^2/H0^2 mit variablem alpha, beta"""
    a_eq = Omega_r / max(Ob, 1e-10)
    S = 1.0 / (1.0 + a_eq / np.maximum(a, 1e-20))
    s0 = np.tanh(K_CFM * A_TRANS)
    f_sat = (np.tanh(K_CFM * (a - A_TRANS)) + s0) / (1.0 + s0)
    # Phi0 aus Closure E^2(1) = 1
    S1 = 1.0 / (1.0 + a_eq)
    f1 = (np.tanh(K_CFM * (1.0 - A_TRANS)) + s0) / (1.0 + s0)
    Phi0 = (1.0 - Ob - Omega_r - alpha * S1) / max(f1, 1e-15)
    return Ob * a**(-3) + Omega_r * a**(-4) + Phi0 * f_sat + alpha * a**(-beta) * S


# ================================================================
# SCHNELLES MODELL (ohne Skalarfeld-ODE fuer Vorscan)
# ================================================================
class CFMFastModel:
    """CFM ohne Skalarfeld, schnell fuer Grid-Scan"""

    def __init__(self, H0, Ob, alpha, beta):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Ob = Ob
        self.alpha = alpha
        self.beta = beta
        self._dC_interp = None

    def E2(self, a):
        return cfm_E2(a, self.Ob, self.alpha, self.beta)

    def Hz(self, z):
        a = 1.0 / (1.0 + z)
        return self.H0 * np.sqrt(max(self.E2(a), 1e-30))

    def _build_interp(self, z_max=2.5, n_z=400):
        z_grid = np.linspace(0, z_max, n_z + 1)
        dc_grid = np.zeros(n_z + 1)
        for i in range(1, n_z + 1):
            z_mid = 0.5 * (z_grid[i-1] + z_grid[i])
            dz = z_grid[i] - z_grid[i-1]
            dc_grid[i] = dc_grid[i-1] + dz / self.Hz(z_mid)
        dc_grid *= c_light
        self._dC_interp = interp1d(z_grid, dc_grid, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
        self._z_max = z_max

    def d_C(self, z):
        """Komitbewegende Distanz: Interpolator fuer z<3, quad fuer z>3"""
        if z < 3.0:
            # Schnell via Interpolator (genug Aufloesung bei z<3)
            if self._dC_interp is None:
                self._build_interp(z_max=3.5)
            return float(self._dC_interp(min(z, 3.49)))
        else:
            # Genau via quad (fuer z* etc.)
            def integrand(zz):
                return 1.0 / self.Hz(zz)
            result, _ = quad(integrand, 0, z, limit=2000)
            return c_light * result

    def d_A(self, z):
        return self.d_C(z) / (1 + z)

    def DV(self, z):
        dA = self.d_A(z)
        Hz_val = self.Hz(z)
        return (dA**2 * c_light * z / Hz_val) ** (1.0/3.0)

    def mu_dist_array(self, z_arr):
        """Schnelle SN-Distanzen via Interpolator (nur fuer z < z_max_SN)"""
        if self._dC_interp is None:
            self._build_interp(z_max=max(z_arr) * 1.1)
        dC = self._dC_interp(z_arr)
        dL = (1 + z_arr) * dC
        dL = np.maximum(dL, 1e-10)
        return 5 * np.log10(dL) + 25

    def r_s(self, z_end):
        a_end = 1.0 / (1 + z_end)
        def integrand(a):
            if a < 1e-15:
                return 0.0
            E2 = max(self.E2(a), 1e-30)
            R_b = 3.0 * self.Ob * a / (4.0 * Omega_gamma)
            c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
            return c_s / (a**2 * np.sqrt(E2))
        result, _ = quad(integrand, 1e-10, a_end, limit=1500)
        return result * c_light / self.H0

    def r_d(self):
        return self.r_s(z_drag)

    def lA(self):
        rs = self.r_s(z_star)
        dc = self.d_C(z_star)
        return np.pi * dc / rs if rs > 0 else 0

    def R_shift(self):
        # Omega_m_eff aus E^2-Verhalten bei z~500 (nach matter-radiation Gleichgewicht)
        # E^2(a) ~ Omega_m_eff * a^{-3} + Omega_r * a^{-4}
        # => Omega_m_eff = (E^2(a) - Omega_r * a^{-4}) * a^3
        a_test = 1.0 / 501.0  # z=500
        E2_test = self.E2(a_test)
        Omega_m_eff = (E2_test - Omega_r * a_test**(-4)) * a_test**3
        Omega_m_eff = max(Omega_m_eff, 0.001)
        dc = self.d_C(z_star)
        return np.sqrt(Omega_m_eff) * dc * self.H0 / c_light


# ================================================================
# CFM MIT SKALARFELD (langsamer, fuer Feinsuche)
# ================================================================
class CFMScalarModel(CFMFastModel):
    """CFM + Poeschl-Teller Skalarfeld"""

    def __init__(self, H0, Ob, alpha, beta, omega_V, psi0, N_points=1000):
        super().__init__(H0, Ob, alpha, beta)
        self.omega_V = omega_V
        self.psi0 = max(psi0, 1e-6)

        if omega_V > 0:
            self._compute_scalar_background(N_points)

    def _compute_scalar_background(self, N_points):
        N_min = np.log(1e-8)
        N_max = 0.0
        N_grid = np.linspace(N_min, N_max, N_points)
        a_grid = np.exp(N_grid)

        E2_base = np.array([cfm_E2(a, self.Ob, self.alpha, self.beta) for a in a_grid])
        E2_base = np.maximum(E2_base, 1e-30)
        E2_base_interp = interp1d(N_grid, E2_base, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')

        # Klein-Gordon loesen
        omega_V = self.omega_V
        psi0 = self.psi0

        def eps_H(N):
            dN_h = 0.005
            Np = min(N + dN_h, N_max)
            Nm = max(N - dN_h, N_min)
            E2c = E2_base_interp(N)
            if E2c < 1e-30:
                return -2.0
            return (E2_base_interp(Np) - E2_base_interp(Nm)) / (2 * dN_h * 2 * E2c)

        def rhs(N, y):
            psi, dpsi = y
            E2 = max(float(E2_base_interp(N)), 1e-30)
            eH = eps_H(N)
            x = np.clip(psi / psi0, -50, 50)
            dVdpsi = -2.0 * omega_V / psi0 * np.tanh(x) / np.cosh(x)**2
            ddpsi = -(3.0 + eH) * dpsi - dVdpsi / E2
            return [dpsi, ddpsi]

        sol = solve_ivp(rhs, [N_min, N_max], [0.01, 0.0],
                        t_eval=N_grid, method='RK45',
                        rtol=1e-6, atol=1e-8, max_step=0.3)

        if sol.success:
            psi_arr, dpsi_arr = sol.y[0], sol.y[1]
        else:
            psi_arr = np.full(N_points, 0.01)
            dpsi_arr = np.zeros(N_points)

        x_arr = np.clip(psi_arr / psi0, -50, 50)
        Omega_phi = E2_base * dpsi_arr**2 / 6.0 + omega_V / np.cosh(x_arr)**2

        # Closure: E^2(1) = 1
        Ophi_0 = Omega_phi[-1]
        E2_full = E2_base + Omega_phi - Ophi_0
        E2_full = np.maximum(E2_full, 1e-30)

        self._E2_full_interp = interp1d(N_grid, E2_full, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
        self._N_min = N_min
        self._Omega_phi = Omega_phi
        self._E2_full = E2_full
        self._a_grid = a_grid

    def E2(self, a):
        if self.omega_V <= 0:
            return cfm_E2(a, self.Ob, self.alpha, self.beta)
        if a < 1e-8:
            return self.Ob * a**(-3) + Omega_r * a**(-4)
        N = np.log(a)
        if N < self._N_min:
            return self.Ob * a**(-3) + Omega_r * a**(-4)
        return max(float(self._E2_full_interp(N)), 1e-30)


# ================================================================
# LCDM REFERENZ
# ================================================================
class LCDMModel(CFMFastModel):
    def __init__(self, H0=67.36, Om=0.315):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Om = Om
        self.Ob = PLANCK_wb / self.h**2
        self.OL = 1.0 - Om - Omega_r
        self._dC_interp = None

    def E2(self, a):
        return self.Om * a**(-3) + Omega_r * a**(-4) + self.OL

    def R_shift(self):
        dc = self.d_C(z_star)
        return np.sqrt(self.Om) * dc * self.H0 / c_light


# ================================================================
# CHI^2 FUNKTIONEN
# ================================================================
def chi2_SN(model, z_data, mu_data, mu_err):
    mu_th = model.mu_dist_array(z_data)
    delta = mu_data - mu_th
    w = 1.0 / mu_err**2
    A = np.sum(w * delta**2)
    B = np.sum(w * delta)
    C = np.sum(w)
    return A - B**2 / C

def chi2_CMB(model):
    lA_th = model.lA()
    R_th = model.R_shift()
    wb_th = model.Ob * model.h**2
    chi2  = ((lA_th - PLANCK_lA) / PLANCK_lA_err)**2
    chi2 += ((R_th - PLANCK_R) / PLANCK_R_err)**2
    chi2 += ((wb_th - PLANCK_wb) / PLANCK_wb_err)**2
    return chi2, lA_th, R_th, wb_th

def chi2_BAO(model):
    rd = model.r_d()
    chi2 = 0.0
    details = []
    for z, obs_type, obs_val, obs_err in BAO_DATA:
        if obs_type == 'DV/rd':
            th = model.DV(z) / rd
        elif obs_type == 'DM/rd':
            th = model.d_C(z) / rd
        elif obs_type == 'DH/rd':
            th = c_light / (model.Hz(z) * rd)
        chi2_i = ((th - obs_val) / obs_err)**2
        chi2 += chi2_i
        details.append((z, obs_type, obs_val, th, chi2_i))
    return chi2, details

def chi2_total(model, z_SN, mu_SN, mu_err_SN):
    c2_sn = chi2_SN(model, z_SN, mu_SN, mu_err_SN)
    c2_cmb, lA, R, wb = chi2_CMB(model)
    c2_bao, _ = chi2_BAO(model)
    return c2_sn + c2_cmb + c2_bao, c2_sn, c2_cmb, c2_bao, lA, R


# ================================================================
# HAUPTANALYSE
# ================================================================
if __name__ == "__main__":
    t0 = time.time()
    out("=" * 78)
    out("  BETA + ALPHA RE-FIT MIT SKALARFELD")
    out("=" * 78)
    out("  Datum: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    out()

    z_SN, mu_SN, mu_err_SN = load_pantheon()
    out("  Pantheon+: {} Supernovae".format(len(z_SN)))
    out()

    # LCDM Referenz
    lcdm = LCDMModel()
    c2_tot_lcdm, c2_sn_l, c2_cmb_l, c2_bao_l, lA_l, R_l = chi2_total(
        lcdm, z_SN, mu_SN, mu_err_SN)
    out("  LCDM: chi2_tot={:.1f} (SN={:.1f}, CMB={:.1f}, BAO={:.1f})".format(
        c2_tot_lcdm, c2_sn_l, c2_cmb_l, c2_bao_l))
    out("        l_A={:.3f}, R={:.4f}, r_d={:.1f}".format(lA_l, R_l, lcdm.r_d()))
    out()

    # ================================================================
    # SCHRITT 1: Grober beta-alpha Scan OHNE Skalarfeld
    # ================================================================
    out("=" * 78)
    out("  SCHRITT 1: beta x alpha SCAN (ohne Skalarfeld, H0=67.36)")
    out("=" * 78)
    out()
    out("  Frage: Welches (beta, alpha) bringt r_d und R naeher an Planck?")
    out()

    beta_vals = np.arange(2.0, 3.21, 0.2)
    # alpha muss Closure erhalten: bei a=1 ist E^2=1 automatisch durch Phi0
    # Aber alpha bestimmt wieviel 'geometrischer DM' es gibt
    alpha_vals = np.arange(0.1, 1.01, 0.1)

    out("  {:>6s}  {:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>10s}  {:>10s}".format(
        "beta", "alpha", "l_A", "R", "r_d", "Phi0", "chi2_SN", "chi2_CMB", "chi2_BAO"))
    out("  " + "-" * 90)

    scan_results = []

    for beta in beta_vals:
        for alpha in alpha_vals:
            try:
                m = CFMFastModel(67.36, 0.05, alpha, beta)
                # Pruefe ob Phi0 > 0 (physikalisch)
                a_eq = Omega_r / 0.05
                S1 = 1.0 / (1.0 + a_eq)
                s0 = np.tanh(K_CFM * A_TRANS)
                f1 = (np.tanh(K_CFM * (1.0 - A_TRANS)) + s0) / (1.0 + s0)
                Phi0 = (1.0 - 0.05 - Omega_r - alpha * S1) / max(f1, 1e-15)
                if Phi0 < -0.5:
                    continue  # unphysikalisch

                c2_tot, c2_sn, c2_cmb, c2_bao, lA, R = chi2_total(
                    m, z_SN, mu_SN, mu_err_SN)
                rd = m.r_d()

                scan_results.append({
                    'beta': beta, 'alpha': alpha, 'H0': 67.36,
                    'lA': lA, 'R': R, 'rd': rd, 'Phi0': Phi0,
                    'c2_sn': c2_sn, 'c2_cmb': c2_cmb, 'c2_bao': c2_bao,
                    'c2_tot': c2_tot
                })

                marker = ""
                if abs(R - PLANCK_R) < 0.3:
                    marker = " <R>"
                if abs(lA - PLANCK_lA) < 10:
                    marker += " <lA>"
                if rd < 160:
                    marker += " <rd>"

                out("  {:6.2f}  {:6.2f}  {:8.1f}  {:8.4f}  {:8.1f}  {:8.3f}  {:10.1f}  {:10.0f}  {:10.1f}{}".format(
                    beta, alpha, lA, R, rd, Phi0,
                    c2_sn, c2_cmb, c2_bao, marker))
            except Exception as e:
                pass
        out()  # Leerzeile zwischen beta-Bloecken

    # Finde bestes (beta, alpha) nach chi2_tot
    if scan_results:
        scan_results.sort(key=lambda x: x['c2_tot'])
        best = scan_results[0]
        out("  BESTES (beta,alpha) ohne Skalarfeld:")
        out("  beta={:.2f}, alpha={:.2f}, chi2_tot={:.1f}".format(
            best['beta'], best['alpha'], best['c2_tot']))
        out("  l_A={:.1f}, R={:.4f}, r_d={:.1f}".format(
            best['lA'], best['R'], best['rd']))
        out()

    # ================================================================
    # SCHRITT 2: H0-Variation bei bestem (beta, alpha)
    # ================================================================
    out("=" * 78)
    out("  SCHRITT 2: H0-SCAN bei besten (beta, alpha)-Kandidaten")
    out("=" * 78)
    out()

    # Top 5 (beta,alpha) nach chi2_tot
    top5 = scan_results[:5]
    H0_vals = [60, 63, 67.36, 70, 75, 80]

    best_overall = None
    best_overall_chi2 = 1e30

    for cand in top5:
        out("  --- beta={:.2f}, alpha={:.2f} ---".format(cand['beta'], cand['alpha']))
        for H0 in H0_vals:
            hh = H0 / 100.0
            Ob = PLANCK_wb / hh**2
            try:
                m = CFMFastModel(H0, Ob, cand['alpha'], cand['beta'])
                c2_tot, c2_sn, c2_cmb, c2_bao, lA, R = chi2_total(
                    m, z_SN, mu_SN, mu_err_SN)
                rd = m.r_d()
                out("    H0={:.1f}: l_A={:.1f}, R={:.4f}, r_d={:.1f}, chi2={:.0f} (SN={:.0f}, CMB={:.0f}, BAO={:.0f})".format(
                    H0, lA, R, rd, c2_tot, c2_sn, c2_cmb, c2_bao))
                if c2_tot < best_overall_chi2:
                    best_overall_chi2 = c2_tot
                    best_overall = {
                        'beta': cand['beta'], 'alpha': cand['alpha'],
                        'H0': H0, 'Ob': Ob, 'lA': lA, 'R': R, 'rd': rd,
                        'c2_sn': c2_sn, 'c2_cmb': c2_cmb, 'c2_bao': c2_bao,
                        'c2_tot': c2_tot
                    }
            except:
                pass
        out()

    if best_overall:
        out("  BESTES ERGEBNIS (ohne Skalarfeld):")
        out("  " + "=" * 55)
        b = best_overall
        out("  beta={:.2f}, alpha={:.2f}, H0={:.1f}, Ob={:.4f}".format(
            b['beta'], b['alpha'], b['H0'], b['Ob']))
        out("  l_A={:.1f}, R={:.4f}, r_d={:.1f}".format(b['lA'], b['R'], b['rd']))
        out("  chi2_tot={:.0f} (SN={:.0f}, CMB={:.0f}, BAO={:.0f})".format(
            b['c2_tot'], b['c2_sn'], b['c2_cmb'], b['c2_bao']))
        out("  Delta_chi2 vs LCDM = {:.0f}".format(b['c2_tot'] - c2_tot_lcdm))
        out()

    # ================================================================
    # SCHRITT 3: Skalarfeld hinzufuegen
    # ================================================================
    out("=" * 78)
    out("  SCHRITT 3: SKALARFELD bei bestem (beta, alpha, H0)")
    out("=" * 78)
    out()

    if best_overall:
        b = best_overall
        out("  Basis: beta={:.2f}, alpha={:.2f}, H0={:.1f}".format(
            b['beta'], b['alpha'], b['H0']))
        out()

        oV_vals = [0, 1e5, 5e5, 1e6, 3e6, 5e6, 1e7, 3e7, 5e7]
        psi0_vals = [0.15, 0.2, 0.3, 0.4, 0.5]

        best_scalar = None
        best_scalar_chi2 = best_overall['c2_tot']

        out("  {:>10s}  {:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>8s}  {:>8s}".format(
            "omega_V", "psi0", "l_A", "R", "r_d", "chi2_tot", "chi2_CMB", "chi2_BAO"))
        out("  " + "-" * 78)

        for oV in oV_vals:
            for p0 in psi0_vals:
                try:
                    m = CFMScalarModel(b['H0'], b['Ob'], b['alpha'], b['beta'],
                                        oV, p0, N_points=800)
                    c2_tot, c2_sn, c2_cmb, c2_bao, lA, R = chi2_total(
                        m, z_SN, mu_SN, mu_err_SN)
                    rd = m.r_d()
                    marker = " <<<" if c2_tot < best_scalar_chi2 else ""
                    out("  {:10.1e}  {:6.2f}  {:8.1f}  {:8.4f}  {:8.1f}  {:10.0f}  {:8.0f}  {:8.0f}{}".format(
                        oV, p0, lA, R, rd, c2_tot, c2_cmb, c2_bao, marker))
                    if c2_tot < best_scalar_chi2:
                        best_scalar_chi2 = c2_tot
                        best_scalar = {
                            'beta': b['beta'], 'alpha': b['alpha'],
                            'H0': b['H0'], 'Ob': b['Ob'],
                            'omega_V': oV, 'psi0': p0,
                            'lA': lA, 'R': R, 'rd': rd,
                            'c2_sn': c2_sn, 'c2_cmb': c2_cmb, 'c2_bao': c2_bao,
                            'c2_tot': c2_tot
                        }
                except:
                    pass
            if oV > 0:
                out()

        out()

    # ================================================================
    # SCHRITT 4: Breiter kombinierter Scan
    # ================================================================
    out("=" * 78)
    out("  SCHRITT 4: BREITER KOMBINIERTER SCAN")
    out("=" * 78)
    out()
    out("  Scan: beta x alpha x H0 x omega_V x psi0")
    out("  Ziel: Minimiere chi2_total = chi2_SN + chi2_CMB + chi2_BAO")
    out()

    # Fokussiertes Grid basierend auf Schritt 1-3 Erkenntnissen
    beta_scan = [2.4, 2.6, 2.8, 3.0]
    alpha_scan = [0.2, 0.3, 0.5, 0.7]
    H0_scan = [63, 67.36, 73]
    oV_scan = [0, 1e6, 5e6, 1e7]
    psi0_scan = [0.2, 0.3, 0.5]

    total = len(beta_scan) * len(alpha_scan) * len(H0_scan) * len(oV_scan) * len(psi0_scan)
    out("  Grid: {} x {} x {} x {} x {} = {} Evaluierungen".format(
        len(beta_scan), len(alpha_scan), len(H0_scan), len(oV_scan), len(psi0_scan), total))
    out()

    best_combined = None
    best_combined_chi2 = 1e30
    n_eval = 0
    t_scan = time.time()

    for beta in beta_scan:
        for alpha in alpha_scan:
            # Quick check: Phi0 physikalisch?
            a_eq = Omega_r / 0.05
            S1 = 1.0 / (1.0 + a_eq)
            s0 = np.tanh(K_CFM * A_TRANS)
            f1 = (np.tanh(K_CFM * (1.0 - A_TRANS)) + s0) / (1.0 + s0)
            Phi0 = (1.0 - 0.05 - Omega_r - alpha * S1) / max(f1, 1e-15)
            if Phi0 < -0.5:
                n_eval += len(H0_scan) * len(oV_scan) * len(psi0_scan)
                continue

            for H0 in H0_scan:
                hh = H0 / 100.0
                Ob = PLANCK_wb / hh**2
                for oV in oV_scan:
                    for p0 in psi0_scan:
                        n_eval += 1
                        try:
                            if oV > 0:
                                m = CFMScalarModel(H0, Ob, alpha, beta,
                                                    oV, p0, N_points=600)
                            else:
                                m = CFMFastModel(H0, Ob, alpha, beta)
                            c2_tot, c2_sn, c2_cmb, c2_bao, lA, R = chi2_total(
                                m, z_SN, mu_SN, mu_err_SN)

                            if c2_tot < best_combined_chi2:
                                best_combined_chi2 = c2_tot
                                rd = m.r_d()
                                best_combined = {
                                    'beta': beta, 'alpha': alpha,
                                    'H0': H0, 'Ob': Ob,
                                    'omega_V': oV, 'psi0': p0,
                                    'lA': lA, 'R': R, 'rd': rd,
                                    'c2_sn': c2_sn, 'c2_cmb': c2_cmb,
                                    'c2_bao': c2_bao, 'c2_tot': c2_tot
                                }
                        except:
                            pass

                        if n_eval % 100 == 0:
                            elapsed = time.time() - t_scan
                            print("  [{}/{}] {:.0f}s, beste chi2={:.0f}".format(
                                n_eval, total, elapsed, best_combined_chi2))

    dt_scan = time.time() - t_scan
    out("  Scan: {:.0f}s ({:.2f}s/Eval)".format(dt_scan, dt_scan/max(n_eval,1)))
    out()

    if best_combined:
        bc = best_combined
        out("  BESTES KOMBINIERTES ERGEBNIS:")
        out("  " + "=" * 60)
        out("  beta    = {:.2f}".format(bc['beta']))
        out("  alpha   = {:.2f}".format(bc['alpha']))
        out("  H0      = {:.2f} km/s/Mpc".format(bc['H0']))
        out("  Ob      = {:.5f}".format(bc['Ob']))
        out("  omega_V = {:.2e}".format(bc['omega_V']))
        out("  psi0    = {:.3f}".format(bc['psi0']))
        out()
        out("  l_A     = {:.3f}  (Planck: {:.3f})".format(bc['lA'], PLANCK_lA))
        out("  R       = {:.4f}  (Planck: {:.4f})".format(bc['R'], PLANCK_R))
        out("  r_d     = {:.2f} Mpc (LCDM: {:.2f})".format(bc['rd'], lcdm.r_d()))
        out()
        out("  chi2_SN  = {:.1f}  (LCDM: {:.1f})".format(bc['c2_sn'], c2_sn_l))
        out("  chi2_CMB = {:.1f}  (LCDM: {:.1f})".format(bc['c2_cmb'], c2_cmb_l))
        out("  chi2_BAO = {:.1f}  (LCDM: {:.1f})".format(bc['c2_bao'], c2_bao_l))
        out("  chi2_TOT = {:.1f}  (LCDM: {:.1f})".format(bc['c2_tot'], c2_tot_lcdm))
        out("  Delta_chi2 = {:.1f}".format(bc['c2_tot'] - c2_tot_lcdm))
        out()

        # ================================================================
        # SCHRITT 5: Feinsuche um bestes Ergebnis
        # ================================================================
        out("=" * 78)
        out("  SCHRITT 5: FEINSUCHE")
        out("=" * 78)
        out()

        beta_fine = np.linspace(max(bc['beta']-0.15, 2.0), min(bc['beta']+0.15, 3.2), 5)
        alpha_fine = np.linspace(max(bc['alpha']-0.1, 0.05), min(bc['alpha']+0.1, 1.0), 5)
        H0_fine = np.linspace(max(bc['H0']-3, 55), bc['H0']+3, 5)

        if bc['omega_V'] > 0:
            oV_fine = np.logspace(np.log10(bc['omega_V'])-0.3,
                                   np.log10(bc['omega_V'])+0.3, 5)
            psi0_fine = np.linspace(max(bc['psi0']-0.08, 0.05), bc['psi0']+0.08, 5)
        else:
            oV_fine = [0, 1e5, 5e5, 1e6]
            psi0_fine = [0.2, 0.3]

        total_fine = len(beta_fine)*len(alpha_fine)*len(H0_fine)*len(oV_fine)*len(psi0_fine)
        out("  Fein-Grid: {} Evaluierungen".format(total_fine))

        n_fine = 0
        for beta in beta_fine:
            for alpha in alpha_fine:
                for H0 in H0_fine:
                    hh = H0 / 100.0
                    Ob = PLANCK_wb / hh**2
                    for oV in oV_fine:
                        for p0 in psi0_fine:
                            n_fine += 1
                            try:
                                if oV > 0:
                                    m = CFMScalarModel(H0, Ob, alpha, beta,
                                                        oV, p0, N_points=600)
                                else:
                                    m = CFMFastModel(H0, Ob, alpha, beta)
                                c2_tot, c2_sn, c2_cmb, c2_bao, lA, R = chi2_total(
                                    m, z_SN, mu_SN, mu_err_SN)

                                if c2_tot < best_combined_chi2:
                                    best_combined_chi2 = c2_tot
                                    rd = m.r_d()
                                    best_combined = {
                                        'beta': beta, 'alpha': alpha,
                                        'H0': H0, 'Ob': Ob,
                                        'omega_V': oV, 'psi0': p0,
                                        'lA': lA, 'R': R, 'rd': rd,
                                        'c2_sn': c2_sn, 'c2_cmb': c2_cmb,
                                        'c2_bao': c2_bao, 'c2_tot': c2_tot
                                    }
                            except:
                                pass

        bc = best_combined
        out()
        out("  FINALES ERGEBNIS NACH FEINSUCHE:")
        out("  " + "=" * 60)
        out("  beta    = {:.3f}".format(bc['beta']))
        out("  alpha   = {:.3f}".format(bc['alpha']))
        out("  H0      = {:.2f} km/s/Mpc".format(bc['H0']))
        out("  Ob      = {:.5f}".format(bc['Ob']))
        out("  omega_V = {:.2e}".format(bc['omega_V']))
        out("  psi0    = {:.3f}".format(bc['psi0']))
        out()
        out("  l_A     = {:.3f}  (Planck: {:.3f},  Abw.: {:.1f} sigma)".format(
            bc['lA'], PLANCK_lA, (bc['lA']-PLANCK_lA)/PLANCK_lA_err))
        out("  R       = {:.4f}  (Planck: {:.4f},  Abw.: {:.1f} sigma)".format(
            bc['R'], PLANCK_R, (bc['R']-PLANCK_R)/PLANCK_R_err))
        out("  r_d     = {:.2f} Mpc  (LCDM: {:.2f})".format(bc['rd'], lcdm.r_d()))
        out()
        out("  chi2_SN  = {:.1f}  (LCDM: {:.1f},  Delta={:+.1f})".format(
            bc['c2_sn'], c2_sn_l, bc['c2_sn']-c2_sn_l))
        out("  chi2_CMB = {:.1f}  (LCDM: {:.1f},  Delta={:+.1f})".format(
            bc['c2_cmb'], c2_cmb_l, bc['c2_cmb']-c2_cmb_l))
        out("  chi2_BAO = {:.1f}  (LCDM: {:.1f},  Delta={:+.1f})".format(
            bc['c2_bao'], c2_bao_l, bc['c2_bao']-c2_bao_l))
        out("  chi2_TOT = {:.1f}  (LCDM: {:.1f},  Delta={:+.1f})".format(
            bc['c2_tot'], c2_tot_lcdm, bc['c2_tot']-c2_tot_lcdm))
        out()

        # BAO Details
        if bc['omega_V'] > 0:
            m_final = CFMScalarModel(bc['H0'], bc['Ob'], bc['alpha'], bc['beta'],
                                       bc['omega_V'], bc['psi0'], N_points=1000)
        else:
            m_final = CFMFastModel(bc['H0'], bc['Ob'], bc['alpha'], bc['beta'])

        _, bao_details = chi2_BAO(m_final)
        out("  BAO Details (Finales Modell):")
        out("  {:>6s}  {:>8s}  {:>10s}  {:>10s}  {:>8s}".format(
            "z", "Typ", "Beob.", "Theorie", "chi2_i"))
        out("  " + "-" * 48)
        for z, typ, val, th, c2i in bao_details:
            out("  {:6.3f}  {:>8s}  {:10.3f}  {:10.3f}  {:8.2f}".format(
                z, typ, val, th, c2i))
        out()

    # ================================================================
    # ZUSAMMENFASSUNG
    # ================================================================
    out("=" * 78)
    out("  ZUSAMMENFASSUNG: BETA+ALPHA RE-FIT")
    out("=" * 78)
    out()

    if best_combined:
        bc = best_combined
        delta_chi2 = bc['c2_tot'] - c2_tot_lcdm

        out("  LCDM:           chi2 = {:.1f}  (2 Parameter)".format(c2_tot_lcdm))
        out("  CFM beta=2.02:  chi2 = {:.1f}  (Delta = +{:.0f})".format(
            scan_results[-1]['c2_tot'] if scan_results else 0,
            (scan_results[-1]['c2_tot'] if scan_results else 0) - c2_tot_lcdm))
        out("  CFM optimiert:  chi2 = {:.1f}  (Delta = {:+.0f})".format(
            bc['c2_tot'], delta_chi2))
        out()

        if delta_chi2 < 10:
            out("  *** CFM IST KOMPETITIV MIT LCDM! ***")
            out("  Delta_chi2 = {:.1f} bei {} zusaetzlichen Parametern".format(
                delta_chi2, 8 - 2))
        elif delta_chi2 < 100:
            out("  CFM ist MARGINAL schlechter als LCDM.")
            out("  Delta_chi2 = {:.1f} -- koennte durch erweiterten Scan verbessert werden.".format(
                delta_chi2))
        elif delta_chi2 < 1000:
            out("  CFM hat SIGNIFIKANTE Abweichungen.")
            out("  Delta_chi2 = {:.1f} -- moeglicherweise rettbar mit feinerem Tuning.".format(
                delta_chi2))
        else:
            out("  CFM bleibt DEUTLICH schlechter als LCDM.")
            out("  Delta_chi2 = {:.0f}".format(delta_chi2))
            out()
            out("  FUNDAMENTALE URSACHE:")
            out("  Ohne echte kalte dunkle Materie (CDM) kann keine")
            out("  Parameterkombination GLEICHZEITIG alle drei erfuellen:")
            out("    1. l_A = 301.5 (akustische Skala)")
            out("    2. R = 1.75 (Shift-Parameter)")
            out("    3. r_d = 148 Mpc (Sound Horizon)")
            out()
            out("  Das CFM-Modell reproduziert SN-Daten gut,")
            out("  scheitert aber am fruehen Universum (z > 100).")
    else:
        out("  Kein gueltiges Ergebnis gefunden.")

    out()
    elapsed = time.time() - t0
    out("  Gesamtlaufzeit: {:.0f} Sekunden ({:.1f} Minuten)".format(elapsed, elapsed/60))

    save_output()

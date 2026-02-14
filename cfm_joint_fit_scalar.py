#!/usr/bin/env python3
"""
===============================================================================
JOINT FIT: CFM + POESCHL-TELLER-SKALARFELD
===============================================================================
Kombinierter Fit an:
  - Supernovae (Pantheon+ 1701 SNe)
  - CMB (Planck 2018: l_A, R, omega_b)
  - BAO (SDSS, BOSS, eBOSS)

CFM-Parameter:  alpha, beta, k, a_trans  (aus vorherigem MCMC)
Skalarfeld:     omega_V, psi0           (neu: loest l_A-Problem)
Kosmologie:     H0, Omega_b

Vergleich mit LCDM.
===============================================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
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
    path = os.path.join(OUTPUT_DIR, "Joint_Fit_Scalar.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    out(f"\n  [Gespeichert: {path}]")

# ================================================================
# KONSTANTEN
# ================================================================
c_light = 299792.458   # km/s
T_CMB = 2.7255
Omega_gamma = 5.38e-5
Omega_nu = 3.65e-5
Omega_r = Omega_gamma + Omega_nu
z_star = 1089.92
a_star = 1.0 / (1 + z_star)
z_drag = 1059.62

# Planck 2018 CMB-Observablen
PLANCK_lA = 301.471
PLANCK_lA_err = 0.090
PLANCK_R = 1.7502
PLANCK_R_err = 0.0046
PLANCK_wb = 0.02236
PLANCK_wb_err = 0.00015

# BAO-Daten (DV/rd oder DA/rd, H*rd)
# Format: z, observable, value, error, type
BAO_DATA = [
    # SDSS MGS (Ross et al. 2015)
    (0.15, 'DV/rd', 4.466, 0.168),
    # BOSS DR12 (Alam et al. 2017): DM/rd und DH/rd = c/(H(z)*rd)
    (0.38, 'DM/rd', 10.27, 0.15),
    (0.38, 'DH/rd', 25.00, 0.76),
    (0.51, 'DM/rd', 13.38, 0.18),
    (0.51, 'DH/rd', 22.33, 0.58),
    (0.61, 'DM/rd', 15.45, 0.20),
    (0.61, 'DH/rd', 20.75, 0.46),
    # eBOSS Lyman-alpha (du Mas des Bourboux et al. 2020)
    (2.334, 'DM/rd', 37.6, 1.1),
    (2.334, 'DH/rd', 8.86, 0.29),
]


# ================================================================
# CFM-MODELL MIT SKALARFELD
# ================================================================

def cfm_E2_base(a, Ob, alpha, beta, k, a_trans):
    """Basis-CFM E^2(a) ohne Skalarfeld"""
    a_eq = Omega_r / max(Ob, 1e-10)
    S = 1.0 / (1.0 + a_eq / np.maximum(a, 1e-20))
    s0 = np.tanh(k * a_trans)
    f_sat = (np.tanh(k * (a - a_trans)) + s0) / (1.0 + s0)
    S1 = 1.0 / (1.0 + a_eq)
    f1 = (np.tanh(k * (1.0 - a_trans)) + s0) / (1.0 + s0)
    Phi0 = (1.0 - Ob - Omega_r - alpha * S1) / max(f1, 1e-15)
    return Ob * a**(-3) + Omega_r * a**(-4) + Phi0 * f_sat + alpha * a**(-beta) * S


class CFMScalarModel:
    """CFM-Modell mit Poeschl-Teller-Skalarfeld"""

    def __init__(self, H0, Ob, alpha, beta, k, a_trans, omega_V, psi0,
                 N_points=1200):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Ob = Ob
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.a_trans = a_trans
        self.omega_V = omega_V
        self.psi0 = max(psi0, 1e-6)
        self.N_points = N_points

        # Hintergrund berechnen
        self._compute_background()

    def _compute_background(self):
        N_min = np.log(1e-8)
        N_max = 0.0
        self.N_grid = np.linspace(N_min, N_max, self.N_points)
        self.a_grid = np.exp(self.N_grid)

        # Basis-CFM
        E2_base = np.array([cfm_E2_base(a, self.Ob, self.alpha, self.beta,
                                          self.k, self.a_trans)
                             for a in self.a_grid])
        E2_base = np.maximum(E2_base, 1e-30)

        if self.omega_V <= 0:
            self.E2_full = E2_base
            self.Omega_phi = np.zeros(self.N_points)
        else:
            # Skalarfeld loesen
            E2_interp = interp1d(self.N_grid, E2_base, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')

            psi_arr, dpsi_arr, Omega_phi = self._solve_scalar(E2_interp)

            # Closure: subtrahiere Omega_phi(0) damit E^2(1)=1
            Ophi_0 = Omega_phi[-1]
            self.E2_full = E2_base + Omega_phi - Ophi_0
            self.E2_full = np.maximum(self.E2_full, 1e-30)
            self.Omega_phi = Omega_phi

        self.E2_interp = interp1d(self.N_grid, self.E2_full, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')

    def _solve_scalar(self, E2_interp):
        N_min, N_max = self.N_grid[0], self.N_grid[-1]
        omega_V = self.omega_V
        psi0 = self.psi0

        def eps_H(N):
            dN = 0.005
            Np = min(N + dN, N_max)
            Nm = max(N - dN, N_min)
            E2c = E2_interp(N)
            if E2c < 1e-30:
                return -2.0
            return (E2_interp(Np) - E2_interp(Nm)) / (2 * dN * 2 * E2c)

        def rhs(N, y):
            psi, dpsi = y
            E2 = max(float(E2_interp(N)), 1e-30)
            eH = eps_H(N)
            x = np.clip(psi / psi0, -50, 50)
            dVdpsi = -2.0 * omega_V / psi0 * np.tanh(x) / np.cosh(x)**2
            ddpsi = -(3.0 + eH) * dpsi - dVdpsi / E2
            return [dpsi, ddpsi]

        sol = solve_ivp(rhs, [N_min, N_max], [0.01, 0.0],
                        t_eval=self.N_grid, method='RK45',
                        rtol=1e-6, atol=1e-8, max_step=0.3)

        if sol.success:
            psi_arr = sol.y[0]
            dpsi_arr = sol.y[1]
        else:
            psi_arr = np.full(self.N_points, 0.01)
            dpsi_arr = np.zeros(self.N_points)

        E2_arr = np.array([E2_interp(N) for N in self.N_grid])
        x_arr = np.clip(psi_arr / psi0, -50, 50)
        Omega_phi = E2_arr * dpsi_arr**2 / 6.0 + omega_V / np.cosh(x_arr)**2

        return psi_arr, dpsi_arr, Omega_phi

    def E2(self, a):
        """E^2(a) = H^2(a)/H0^2"""
        if a < 1e-8:
            return self.Ob * a**(-3) + Omega_r * a**(-4)
        N = np.log(a)
        if N < self.N_grid[0]:
            return self.Ob * a**(-3) + Omega_r * a**(-4)
        return max(float(self.E2_interp(N)), 1e-30)

    def E(self, a):
        return np.sqrt(self.E2(a))

    def Hz(self, z):
        """H(z) in km/s/Mpc"""
        return self.H0 * self.E(1.0 / (1 + z))

    def _build_dL_interp(self, z_max=2.5, n_z=300):
        """Baue dL-Interpolator auf z-Grid (SCHNELL fuer viele SNe)"""
        z_grid = np.linspace(0, z_max, n_z + 1)
        dL_grid = np.zeros(n_z + 1)
        # Kumulative Trapezregel fuer int_0^z dz'/H(z')
        for i in range(1, n_z + 1):
            z_mid = 0.5 * (z_grid[i-1] + z_grid[i])
            dz = z_grid[i] - z_grid[i-1]
            dL_grid[i] = dL_grid[i-1] + dz / self.Hz(z_mid)
        # dL = c*(1+z)*int
        dC_grid = c_light * dL_grid
        self._dC_interp = interp1d(z_grid, dC_grid, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
        self._z_max_interp = z_max

    def dL(self, z):
        """Leuchtkraft-Distanz in Mpc"""
        if not hasattr(self, '_dC_interp'):
            self._build_dL_interp()
        return (1 + z) * float(self._dC_interp(z))

    def mu_dist(self, z):
        """Distanzmodul"""
        dl = self.dL(z)
        if dl <= 0:
            return 50.0
        return 5 * np.log10(dl) + 25

    def mu_dist_array(self, z_arr):
        """Distanzmodule fuer Array von z (SCHNELL)"""
        if not hasattr(self, '_dC_interp'):
            self._build_dL_interp(z_max=max(z_arr) * 1.1)
        dC = self._dC_interp(z_arr)
        dL = (1 + z_arr) * dC
        dL = np.maximum(dL, 1e-10)
        return 5 * np.log10(dL) + 25

    def r_s(self, z_end):
        """Kosmoverschmelzungshorizont r_s(z) in Mpc"""
        a_end = 1.0 / (1 + z_end)
        def integrand(a):
            if a < 1e-15:
                return 0.0
            E2 = self.E2(a)
            R_b = 3.0 * self.Ob * a / (4.0 * Omega_gamma)
            c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
            return c_s / (a**2 * np.sqrt(E2))
        result, _ = quad(integrand, 1e-10, a_end, limit=2000)
        return result * c_light / self.H0

    def d_C(self, z):
        """Komitbewegende Distanz in Mpc"""
        a_low = 1.0 / (1 + z)
        def integrand(a):
            E2 = self.E2(a)
            return 1.0 / (a**2 * np.sqrt(E2))
        result, _ = quad(integrand, a_low, 1.0, limit=2000)
        return result * c_light / self.H0

    def d_A(self, z):
        """Winkeldurchmesser-Distanz"""
        return self.d_C(z) / (1 + z)

    def DV(self, z):
        """BAO Volume-averaged distance"""
        dA = self.d_A(z)
        Hz_val = self.Hz(z)
        return (dA**2 * c_light * z / Hz_val) ** (1.0/3.0)

    def lA(self):
        """Akustische Skala l_A"""
        rs = self.r_s(z_star)
        dc = self.d_C(z_star)
        return np.pi * dc / rs if rs > 0 else 0

    def R_shift(self):
        """CMB Shift-Parameter R = sqrt(Omega_m) * d_C(z*) * H0/c"""
        # Omega_m_eff: gesamte nicht-relativistische Materie bei z=0
        # Fuer CFM: nur Baryonen + geometrischer DM-Effekt
        # R = sqrt(Ob*h^2) * d_C(z*) / (c/H0) ... nein
        # R = sqrt(Omega_m) * H0 * d_C / c
        # Wir verwenden effektives Omega_m aus E^2-Verhalten bei kleinem z
        # E^2(a) ~ Omega_m * a^{-3} + ... => Omega_m_eff = dE^2/d(a^{-3})|_{a=1}
        # Einfacher: E^2(0.99) ~ 1 + 3*Omega_m * 0.01 + ...
        a1 = 0.995
        a2 = 1.005
        E2_1 = self.E2(a1)
        E2_2 = self.E2(a2)
        # dE^2/da bei a=1 ~ (E2_1 - E2_2)/(a1 - a2)
        dE2_da = (E2_1 - E2_2) / (a1 - a2)
        # E^2 ~ 1 + Omega_m*(-3)*da + ... => dE2/da|_1 = -3*Omega_m
        Omega_m_eff = -dE2_da / 3.0
        Omega_m_eff = max(Omega_m_eff, 0.01)

        dc = self.d_C(z_star)
        return np.sqrt(Omega_m_eff) * dc * self.H0 / c_light

    def r_d(self):
        """Sound horizon at drag epoch"""
        return self.r_s(z_drag)


# ================================================================
# LCDM-MODELL
# ================================================================

class LCDMModel:
    def __init__(self, H0=67.36, Om=0.315):
        self.H0 = H0
        self.h = H0 / 100.0
        self.Om = Om
        self.Ob = PLANCK_wb / self.h**2
        self.OL = 1.0 - Om - Omega_r

    def E2(self, a):
        return self.Om * a**(-3) + Omega_r * a**(-4) + self.OL

    def E(self, a):
        return np.sqrt(self.E2(a))

    def Hz(self, z):
        return self.H0 * self.E(1.0 / (1 + z))

    def _build_dL_interp(self, z_max=2.5, n_z=300):
        z_grid = np.linspace(0, z_max, n_z + 1)
        dL_grid = np.zeros(n_z + 1)
        for i in range(1, n_z + 1):
            z_mid = 0.5 * (z_grid[i-1] + z_grid[i])
            dz = z_grid[i] - z_grid[i-1]
            dL_grid[i] = dL_grid[i-1] + dz / self.Hz(z_mid)
        dC_grid = c_light * dL_grid
        self._dC_interp = interp1d(z_grid, dC_grid, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')

    def dL(self, z):
        if not hasattr(self, '_dC_interp'):
            self._build_dL_interp()
        return (1 + z) * float(self._dC_interp(z))

    def mu_dist(self, z):
        dl = self.dL(z)
        if dl <= 0:
            return 50.0
        return 5 * np.log10(dl) + 25

    def mu_dist_array(self, z_arr):
        if not hasattr(self, '_dC_interp'):
            self._build_dL_interp(z_max=max(z_arr) * 1.1)
        dC = self._dC_interp(z_arr)
        dL = (1 + z_arr) * dC
        dL = np.maximum(dL, 1e-10)
        return 5 * np.log10(dL) + 25

    def r_s(self, z_end):
        a_end = 1.0 / (1 + z_end)
        def integrand(a):
            if a < 1e-15:
                return 0.0
            E2 = self.E2(a)
            R_b = 3.0 * self.Ob * a / (4.0 * Omega_gamma)
            c_s = 1.0 / np.sqrt(3.0 * (1.0 + R_b))
            return c_s / (a**2 * np.sqrt(E2))
        result, _ = quad(integrand, 1e-10, a_end, limit=2000)
        return result * c_light / self.H0

    def d_C(self, z):
        a_low = 1.0 / (1 + z)
        def integrand(a):
            E2 = self.E2(a)
            return 1.0 / (a**2 * np.sqrt(E2))
        result, _ = quad(integrand, a_low, 1.0, limit=2000)
        return result * c_light / self.H0

    def d_A(self, z):
        return self.d_C(z) / (1 + z)

    def DV(self, z):
        dA = self.d_A(z)
        Hz_val = self.Hz(z)
        return (dA**2 * c_light * z / Hz_val) ** (1.0/3.0)

    def lA(self):
        rs = self.r_s(z_star)
        dc = self.d_C(z_star)
        return np.pi * dc / rs

    def R_shift(self):
        dc = self.d_C(z_star)
        return np.sqrt(self.Om) * dc * self.H0 / c_light

    def r_d(self):
        return self.r_s(z_drag)


# ================================================================
# LIKELIHOODS
# ================================================================

def load_pantheon():
    """Lade Pantheon+ Daten (m_b_corr + err, zHD > 0.01)"""
    path = os.path.join(DATA_DIR, "Pantheon+SH0ES.dat")
    if not os.path.exists(path):
        out("  [WARNUNG: Pantheon+ nicht gefunden, generiere Mock-Daten]")
        np.random.seed(42)
        z = np.sort(np.random.uniform(0.01, 2.3, 1000))
        lcdm = LCDMModel()
        mu = np.array([lcdm.mu_dist(zi) for zi in z])
        mu_err = np.random.uniform(0.1, 0.3, len(z))
        return z, mu, mu_err

    # Spalten: CID(0) IDSURVEY(1) zHD(2) zHDERR(3) zCMB(4) zCMBERR(5)
    #          zHEL(6) zHELERR(7) m_b_corr(8) m_b_corr_err_DIAG(9)
    #          MU_SH0ES(10) MU_SH0ES_ERR_DIAG(11) ...
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
                z = float(parts[2])       # zHD
                mb = float(parts[8])      # m_b_corr (scheinbare Helligkeit)
                mb_err = float(parts[9])  # m_b_corr_err_DIAG
                if z > 0.01 and mb_err > 0 and mb_err < 5.0:
                    z_arr.append(z)
                    mb_arr.append(mb)
                    mb_err_arr.append(mb_err)
            except (ValueError, IndexError):
                continue

    return np.array(z_arr), np.array(mb_arr), np.array(mb_err_arr)


def chi2_SN(model, z_data, mu_data, mu_err):
    """SN chi^2 mit analytischer Marginalisierung ueber M (vektorisiert)"""
    if hasattr(model, 'mu_dist_array'):
        mu_th = model.mu_dist_array(z_data)
    else:
        mu_th = np.array([model.mu_dist(z) for z in z_data])
    delta = mu_data - mu_th
    w = 1.0 / mu_err**2

    A = np.sum(w * delta**2)
    B = np.sum(w * delta)
    C = np.sum(w)
    return A - B**2 / C


def chi2_CMB(model):
    """CMB chi^2 aus l_A und R (Planck compressed likelihood)"""
    lA_th = model.lA()
    R_th = model.R_shift()
    wb_th = model.Ob * model.h**2 if hasattr(model, 'Ob') else PLANCK_wb

    chi2 = ((lA_th - PLANCK_lA) / PLANCK_lA_err)**2
    chi2 += ((R_th - PLANCK_R) / PLANCK_R_err)**2
    chi2 += ((wb_th - PLANCK_wb) / PLANCK_wb_err)**2

    return chi2, lA_th, R_th, wb_th


def chi2_BAO(model):
    """BAO chi^2"""
    rd = model.r_d()
    chi2 = 0.0
    details = []

    for entry in BAO_DATA:
        z, obs_type, obs_val, obs_err = entry

        if obs_type == 'DV/rd':
            th = model.DV(z) / rd
        elif obs_type == 'DM/rd':
            th = model.d_C(z) / rd  # DM = comoving distance
        elif obs_type == 'DH/rd':
            th = c_light / (model.Hz(z) * rd)  # DH = c/H(z)

        chi2_i = ((th - obs_val) / obs_err)**2
        chi2 += chi2_i
        details.append((z, obs_type, obs_val, th, chi2_i))

    return chi2, details


# ================================================================
# HAUPTANALYSE
# ================================================================

if __name__ == "__main__":
    t0 = time.time()
    out("=" * 78)
    out("  JOINT FIT: CFM + POESCHL-TELLER-SKALARFELD")
    out("=" * 78)
    out("  Datum: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
    out()

    # Daten laden
    z_SN, mu_SN, mu_err_SN = load_pantheon()
    out("  Pantheon+: {} Supernovae geladen".format(len(z_SN)))
    out("  BAO:       {} Datenpunkte".format(len(BAO_DATA)))
    out()

    # ================================================================
    # 1. LCDM Referenz
    # ================================================================
    out("=" * 78)
    out("  1. LCDM REFERENZ")
    out("=" * 78)
    out()

    lcdm = LCDMModel(H0=67.36, Om=0.315)
    chi2_sn_lcdm = chi2_SN(lcdm, z_SN, mu_SN, mu_err_SN)
    chi2_cmb_lcdm, lA_lcdm, R_lcdm, wb_lcdm = chi2_CMB(lcdm)
    chi2_bao_lcdm, bao_det_lcdm = chi2_BAO(lcdm)
    chi2_tot_lcdm = chi2_sn_lcdm + chi2_cmb_lcdm + chi2_bao_lcdm

    out("  H0 = {:.2f}, Omega_m = {:.3f}".format(lcdm.H0, lcdm.Om))
    out("  l_A = {:.3f}  (Planck: {:.3f})".format(lA_lcdm, PLANCK_lA))
    out("  R   = {:.4f}  (Planck: {:.4f})".format(R_lcdm, PLANCK_R))
    out("  r_d = {:.2f} Mpc".format(lcdm.r_d()))
    out()
    out("  chi2_SN  = {:.1f}  ({} SNe)".format(chi2_sn_lcdm, len(z_SN)))
    out("  chi2_CMB = {:.1f}  (l_A + R + wb)".format(chi2_cmb_lcdm))
    out("  chi2_BAO = {:.1f}  ({} Punkte)".format(chi2_bao_lcdm, len(BAO_DATA)))
    out("  chi2_TOT = {:.1f}".format(chi2_tot_lcdm))
    out()

    # BAO Details
    out("  BAO Details:")
    out("  {:>6s}  {:>8s}  {:>10s}  {:>10s}  {:>8s}".format(
        "z", "Typ", "Beob.", "Theorie", "chi2_i"))
    out("  " + "-" * 48)
    for z, typ, val, th, c2i in bao_det_lcdm:
        out("  {:6.3f}  {:>8s}  {:10.3f}  {:10.3f}  {:8.2f}".format(
            z, typ, val, th, c2i))
    out()

    # ================================================================
    # 2. CFM OHNE SKALARFELD (Basis)
    # ================================================================
    out("=" * 78)
    out("  2. CFM OHNE SKALARFELD (Basis)")
    out("=" * 78)
    out()

    cfm_base = CFMScalarModel(H0=67.36, Ob=0.05,
                                alpha=0.68, beta=2.02,
                                k=9.81, a_trans=0.971,
                                omega_V=0, psi0=1.0)

    chi2_sn_base = chi2_SN(cfm_base, z_SN, mu_SN, mu_err_SN)
    chi2_cmb_base, lA_base, R_base, wb_base = chi2_CMB(cfm_base)
    chi2_bao_base, bao_det_base = chi2_BAO(cfm_base)
    chi2_tot_base = chi2_sn_base + chi2_cmb_base + chi2_bao_base

    out("  H0 = 67.36, Ob = 0.05, alpha = 0.68, beta = 2.02")
    out("  l_A = {:.3f}  (Planck: {:.3f})".format(lA_base, PLANCK_lA))
    out("  R   = {:.4f}  (Planck: {:.4f})".format(R_base, PLANCK_R))
    out("  r_d = {:.2f} Mpc".format(cfm_base.r_d()))
    out()
    out("  chi2_SN  = {:.1f}".format(chi2_sn_base))
    out("  chi2_CMB = {:.1f}  ***KATASTROPHAL***".format(chi2_cmb_base))
    out("  chi2_BAO = {:.1f}".format(chi2_bao_base))
    out("  chi2_TOT = {:.1f}".format(chi2_tot_base))
    out()

    # ================================================================
    # 3. CFM MIT OPTIMALEM SKALARFELD (aus Feinsuche)
    # ================================================================
    out("=" * 78)
    out("  3. CFM + SKALARFELD (omega_V = 3.16e6, psi0 = 0.317)")
    out("=" * 78)
    out()

    cfm_scalar = CFMScalarModel(H0=67.36, Ob=0.05,
                                  alpha=0.68, beta=2.02,
                                  k=9.81, a_trans=0.971,
                                  omega_V=3.16e6, psi0=0.317)

    chi2_sn_sc = chi2_SN(cfm_scalar, z_SN, mu_SN, mu_err_SN)
    chi2_cmb_sc, lA_sc, R_sc, wb_sc = chi2_CMB(cfm_scalar)
    chi2_bao_sc, bao_det_sc = chi2_BAO(cfm_scalar)
    chi2_tot_sc = chi2_sn_sc + chi2_cmb_sc + chi2_bao_sc

    out("  omega_V = 3.16e6, psi0 = 0.317")
    out("  l_A = {:.3f}  (Planck: {:.3f})".format(lA_sc, PLANCK_lA))
    out("  R   = {:.4f}  (Planck: {:.4f})".format(R_sc, PLANCK_R))
    out("  r_d = {:.2f} Mpc".format(cfm_scalar.r_d()))
    out()
    out("  chi2_SN  = {:.1f}".format(chi2_sn_sc))
    out("  chi2_CMB = {:.1f}".format(chi2_cmb_sc))
    out("  chi2_BAO = {:.1f}".format(chi2_bao_sc))
    out("  chi2_TOT = {:.1f}".format(chi2_tot_sc))
    out()

    # BAO Details
    out("  BAO Details:")
    out("  {:>6s}  {:>8s}  {:>10s}  {:>10s}  {:>8s}".format(
        "z", "Typ", "Beob.", "Theorie", "chi2_i"))
    out("  " + "-" * 48)
    for z, typ, val, th, c2i in bao_det_sc:
        out("  {:6.3f}  {:>8s}  {:10.3f}  {:10.3f}  {:8.2f}".format(
            z, typ, val, th, c2i))
    out()

    # ================================================================
    # 4. GRID-OPTIMIERUNG: omega_V, psi0, H0
    # ================================================================
    out("=" * 78)
    out("  4. GRID-OPTIMIERUNG: omega_V x psi0 x H0")
    out("=" * 78)
    out()

    out("  Minimiere chi2_total = chi2_SN + chi2_CMB + chi2_BAO")
    out("  CFM-Parameter (alpha, beta, k, a_trans) FEST aus MCMC")
    out()

    # Grobe Suche
    best_chi2 = 1e30
    best_params = None

    H0_grid = [63, 67.36, 73]
    oV_grid = [0, 5e5, 1e6, 3e6, 5e6, 1e7]
    psi0_grid = [0.2, 0.3, 0.4, 0.5]

    total_evals = len(H0_grid) * len(oV_grid) * len(psi0_grid)
    out("  Grobe Suche: {} x {} x {} = {} Evaluierungen".format(
        len(H0_grid), len(oV_grid), len(psi0_grid), total_evals))

    n_eval = 0
    t_grid = time.time()

    for H0_try in H0_grid:
        hh = H0_try / 100.0
        Ob_try = PLANCK_wb / hh**2  # BBN-konsistent
        for oV in oV_grid:
            for p0 in psi0_grid:
                n_eval += 1
                try:
                    m = CFMScalarModel(H0_try, Ob_try, 0.68, 2.02, 9.81, 0.971,
                                        oV, p0, N_points=800)
                    c2_sn = chi2_SN(m, z_SN, mu_SN, mu_err_SN)
                    c2_cmb, _, _, _ = chi2_CMB(m)
                    c2_bao, _ = chi2_BAO(m)
                    c2_tot = c2_sn + c2_cmb + c2_bao

                    if c2_tot < best_chi2:
                        best_chi2 = c2_tot
                        best_params = {
                            'H0': H0_try, 'Ob': Ob_try,
                            'omega_V': oV, 'psi0': p0,
                            'chi2_SN': c2_sn, 'chi2_CMB': c2_cmb,
                            'chi2_BAO': c2_bao, 'chi2_tot': c2_tot,
                            'model': m
                        }
                except:
                    pass

                if n_eval % 50 == 0:
                    print("  [{}/{}] beste chi2 = {:.1f}".format(
                        n_eval, total_evals, best_chi2))

    dt_grid = time.time() - t_grid
    out("  Grid-Suche: {:.1f}s ({:.2f}s/Eval)".format(dt_grid, dt_grid/max(n_eval,1)))
    out()

    if best_params:
        bp = best_params
        m_best = bp['model']
        lA_best = m_best.lA()
        R_best = m_best.R_shift()
        rd_best = m_best.r_d()

        out("  BESTES ERGEBNIS (grobe Suche):")
        out("  " + "=" * 55)
        out("  H0      = {:.2f} km/s/Mpc".format(bp['H0']))
        out("  Omega_b = {:.5f}".format(bp['Ob']))
        out("  omega_V = {:.2e}".format(bp['omega_V']))
        out("  psi0    = {:.3f}".format(bp['psi0']))
        out()
        out("  l_A     = {:.3f}  (Planck: {:.3f})".format(lA_best, PLANCK_lA))
        out("  R       = {:.4f}  (Planck: {:.4f})".format(R_best, PLANCK_R))
        out("  r_d     = {:.2f} Mpc".format(rd_best))
        out()
        out("  chi2_SN  = {:.1f}".format(bp['chi2_SN']))
        out("  chi2_CMB = {:.1f}".format(bp['chi2_CMB']))
        out("  chi2_BAO = {:.1f}".format(bp['chi2_BAO']))
        out("  chi2_TOT = {:.1f}".format(bp['chi2_tot']))
        out()

        # Feinsuche um bestes Ergebnis
        out("  Feinsuche um optimale Parameter...")
        out()

        H0_fine = np.linspace(max(bp['H0']-2, 60), bp['H0']+2, 5)
        if bp['omega_V'] > 0:
            oV_fine = np.logspace(np.log10(bp['omega_V'])-0.3,
                                   np.log10(bp['omega_V'])+0.3, 5)
        else:
            oV_fine = [0, 5e5, 1e6, 3e6, 5e6]
        psi0_fine = np.linspace(max(bp['psi0']-0.06, 0.05), bp['psi0']+0.06, 5)

        for H0_try in H0_fine:
            hh = H0_try / 100.0
            Ob_try = PLANCK_wb / hh**2
            for oV in oV_fine:
                for p0 in psi0_fine:
                    try:
                        m = CFMScalarModel(H0_try, Ob_try, 0.68, 2.02, 9.81, 0.971,
                                            oV, p0, N_points=800)
                        c2_sn = chi2_SN(m, z_SN, mu_SN, mu_err_SN)
                        c2_cmb, _, _, _ = chi2_CMB(m)
                        c2_bao, _ = chi2_BAO(m)
                        c2_tot = c2_sn + c2_cmb + c2_bao

                        if c2_tot < best_chi2:
                            best_chi2 = c2_tot
                            best_params = {
                                'H0': H0_try, 'Ob': Ob_try,
                                'omega_V': oV, 'psi0': p0,
                                'chi2_SN': c2_sn, 'chi2_CMB': c2_cmb,
                                'chi2_BAO': c2_bao, 'chi2_tot': c2_tot,
                                'model': m
                            }
                    except:
                        pass

        bp = best_params
        m_best = bp['model']
        lA_best = m_best.lA()
        R_best = m_best.R_shift()
        rd_best = m_best.r_d()

        out("  BESTES ERGEBNIS (nach Feinsuche):")
        out("  " + "=" * 55)
        out("  H0      = {:.2f} km/s/Mpc".format(bp['H0']))
        out("  Omega_b = {:.5f}".format(bp['Ob']))
        out("  omega_V = {:.2e}".format(bp['omega_V']))
        out("  psi0    = {:.3f}".format(bp['psi0']))
        out()
        out("  l_A     = {:.3f}  (Planck: {:.3f})".format(lA_best, PLANCK_lA))
        out("  R       = {:.4f}  (Planck: {:.4f})".format(R_best, PLANCK_R))
        out("  r_d     = {:.2f} Mpc".format(rd_best))
        out()
        out("  chi2_SN  = {:.1f}".format(bp['chi2_SN']))
        out("  chi2_CMB = {:.1f}".format(bp['chi2_CMB']))
        out("  chi2_BAO = {:.1f}".format(bp['chi2_BAO']))
        out("  chi2_TOT = {:.1f}".format(bp['chi2_tot']))
        out()

        # BAO Details fuer bestes Modell
        _, bao_det_best = chi2_BAO(m_best)
        out("  BAO Details:")
        out("  {:>6s}  {:>8s}  {:>10s}  {:>10s}  {:>8s}".format(
            "z", "Typ", "Beob.", "Theorie", "chi2_i"))
        out("  " + "-" * 48)
        for z, typ, val, th, c2i in bao_det_best:
            out("  {:6.3f}  {:>8s}  {:10.3f}  {:10.3f}  {:8.2f}".format(
                z, typ, val, th, c2i))
        out()

    # ================================================================
    # 5. VERGLEICHSTABELLE
    # ================================================================
    out("=" * 78)
    out("  5. VERGLEICHSTABELLE")
    out("=" * 78)
    out()

    out("  {:25s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
        "", "LCDM", "CFM-Basis", "CFM+Skalar", "CFM-Optimal"))
    out("  " + "-" * 78)

    # Werte sammeln
    lA_opt = m_best.lA() if best_params else 0
    R_opt = m_best.R_shift() if best_params else 0
    rd_opt = m_best.r_d() if best_params else 0

    out("  {:25s}  {:12.3f}  {:12.3f}  {:12.3f}  {:12.3f}".format(
        "l_A", lA_lcdm, lA_base, lA_sc, lA_opt))
    out("  {:25s}  {:12.4f}  {:12.4f}  {:12.4f}  {:12.4f}".format(
        "R (shift)", R_lcdm, R_base, R_sc, R_opt))
    out("  {:25s}  {:12.2f}  {:12.2f}  {:12.2f}  {:12.2f}".format(
        "r_d [Mpc]", lcdm.r_d(), cfm_base.r_d(), cfm_scalar.r_d(), rd_opt))
    out()
    out("  {:25s}  {:12.1f}  {:12.1f}  {:12.1f}  {:12.1f}".format(
        "chi2_SN", chi2_sn_lcdm, chi2_sn_base, chi2_sn_sc, bp['chi2_SN']))
    out("  {:25s}  {:12.1f}  {:12.1f}  {:12.1f}  {:12.1f}".format(
        "chi2_CMB", chi2_cmb_lcdm, chi2_cmb_base, chi2_cmb_sc, bp['chi2_CMB']))
    out("  {:25s}  {:12.1f}  {:12.1f}  {:12.1f}  {:12.1f}".format(
        "chi2_BAO", chi2_bao_lcdm, chi2_bao_base, chi2_bao_sc, bp['chi2_BAO']))
    out("  {:25s}  {:12.1f}  {:12.1f}  {:12.1f}  {:12.1f}".format(
        "chi2_TOTAL", chi2_tot_lcdm, chi2_tot_base, chi2_tot_sc, bp['chi2_tot']))
    out()

    out("  Delta_chi2 (vs LCDM):")
    out("  {:25s}  {:>12s}  {:12.1f}  {:12.1f}  {:12.1f}".format(
        "", "---", chi2_tot_base - chi2_tot_lcdm,
        chi2_tot_sc - chi2_tot_lcdm, bp['chi2_tot'] - chi2_tot_lcdm))
    out()

    # N_param Vergleich
    out("  Modell-Parameter:")
    out("  LCDM:        2 (H0, Omega_m)")
    out("  CFM-Basis:   6 (H0, Ob, alpha, beta, k, a_trans)")
    out("  CFM+Skalar:  8 (+ omega_V, psi0)")
    out()

    # AIC/BIC
    N_data = len(z_SN) + 3 + len(BAO_DATA)  # SN + CMB(3) + BAO
    for name, chi2_val, n_params in [
        ("LCDM", chi2_tot_lcdm, 2),
        ("CFM-Basis", chi2_tot_base, 6),
        ("CFM+Skalar", chi2_tot_sc, 8),
        ("CFM-Optimal", bp['chi2_tot'], 8)
    ]:
        AIC = chi2_val + 2 * n_params
        BIC = chi2_val + n_params * np.log(N_data)
        out("  {:15s}  AIC = {:10.1f}  BIC = {:10.1f}".format(name, AIC, BIC))
    out()

    # ================================================================
    # 6. PHYSIKALISCHE INTERPRETATION
    # ================================================================
    out("=" * 78)
    out("  6. PHYSIKALISCHE INTERPRETATION")
    out("=" * 78)
    out()

    if best_params:
        bp = best_params
        out("  Das Poeschl-Teller-Skalarfeld mit V(phi) = V0/cosh^2(phi/phi0)")
        out("  wirkt als natuerliche 'Early Dark Energy' (EDE) Komponente.")
        out()
        out("  Optimale Parameter:")
        out("    V0/rho_crit = {:.2e}  (Potentialhoehe)".format(bp['omega_V']))
        out("    phi0/M_Pl   = {:.3f}   (Potentialbreite)".format(bp['psi0']))
        out("    H0          = {:.2f}   km/s/Mpc".format(bp['H0']))
        out()

        # Energieskala
        # rho_crit = 3H0^2/(8piG) = 8.5e-10 J/m^3
        # V0 = omega_V * rho_crit
        rho_crit_eV4 = 3.7e-11  # (eV)^4
        V0_eV4 = bp['omega_V'] * rho_crit_eV4
        V0_eV = V0_eV4 ** 0.25
        out("    V0 = {:.2e} (eV)^4 = ({:.2f} eV)^4".format(V0_eV4, V0_eV))
        out("    Vergleich: Lambda_QCD ~ (200 MeV)^4, m_nu ~ (0.1 eV)")
        out("    V0^(1/4) / M_Pl = {:.2e}".format(V0_eV / 1.22e28))
        out()

        # EDE Anteil
        m_opt = bp['model']
        idx_star = np.argmin(np.abs(m_opt.a_grid - a_star))
        Ophi_star = m_opt.Omega_phi[idx_star]
        E2_star = m_opt.E2_full[idx_star]
        f_star = Ophi_star / E2_star if E2_star > 0 else 0

        out("    Skalarfeld-Anteil bei z* = {:.4f} ({:.2f}%)".format(f_star, f_star*100))
        out("    Zum Vergleich: EDE-Modelle benoetigen ~5-10% bei z*")
        out()

    out("  FAZIT:")
    out("  " + "=" * 55)
    if best_params and bp['chi2_tot'] < chi2_tot_lcdm + 50:
        out("  CFM + Skalarfeld ist KOMPETITIV mit LCDM!")
        out("  Delta_chi2 = {:.1f} (positiv = schlechter als LCDM)".format(
            bp['chi2_tot'] - chi2_tot_lcdm))
    elif best_params:
        out("  CFM + Skalarfeld verbessert sich DRAMATISCH:")
        out("  Ohne Skalarfeld: Delta_chi2 = {:.0f} (katastrophal)".format(
            chi2_tot_base - chi2_tot_lcdm))
        out("  Mit Skalarfeld:  Delta_chi2 = {:.0f}".format(
            bp['chi2_tot'] - chi2_tot_lcdm))
    out()

    elapsed = time.time() - t0
    out("  Gesamtlaufzeit: {:.1f} Sekunden".format(elapsed))

    save_output()

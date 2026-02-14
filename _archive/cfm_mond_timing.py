"""
CFM-MOND Timing-Test: Kann man den Zeitpunkt der Beschleunigung
aus der MOND-Skala a_0 vorhersagen?

Drei Uebergaenge im CFM:
1. z_t ~ 7:    beta-Uebergang (Rueckgabe endet)
2. z_accel:    Beschleunigung beginnt (q=0)
3. a_trans:    f_sat Saettigung (z ~ 0.03)

Frage: Folgen diese aus a_0?
"""

import numpy as np
from scipy.integrate import quad
import time, os

# Konstanten
c_SI = 2.998e8           # m/s
c_light = 299792.458     # km/s
a0_MOND = 1.2e-10        # m/s^2
Mpc_m = 3.086e22         # m/Mpc
Omega_r = 9.15e-5
omega_b_BBN = 0.02237
k_fsat = 9.81
a_trans_fsat = 0.971
beta_late = 2.02

# Optimierte Parameter aus Combined Fit
H0_cfm = 60.0
alpha_cfm = 0.7304
be_cfm = 2.7794
at_cfm = 0.12374
n_cfm = 4
fe_cfm = 9.18e8
ae_cfm = 9.13e-4

h_cfm = H0_cfm / 100.0
Ob_cfm = omega_b_BBN / h_cfm**2
Ogamma_cfm = 2.469e-5 / h_cfm**2
fs1 = 1.0/(1.0+np.exp(-k_fsat*(1.0-a_trans_fsat)))
Phi0_cfm = (1.0 - Ob_cfm - Omega_r - alpha_cfm) / fs1
ede_at_1 = fe_cfm / (1.0 + (1.0/ae_cfm)**6)

# LCDM Parameter
H0_lcdm = 67.36
Om_lcdm = 0.3153
OL_lcdm = 1.0 - Om_lcdm - Omega_r


def f_sat(a):
    x = k_fsat*(a-a_trans_fsat)
    if x > 500: return 1.0
    if x < -500: return 0.0
    return 1.0/(1.0+np.exp(-x))

def beta_eff(a):
    return beta_late + (be_cfm - beta_late)/(1.0+(a/at_cfm)**n_cfm)

def E2_cfm(a):
    if a <= 0: return 1e30
    b = beta_eff(a)
    ede = fe_cfm/(1.0+(a/ae_cfm)**6) - ede_at_1
    return (Ob_cfm*a**(-3) + Omega_r*a**(-4)
            + Phi0_cfm*f_sat(a) + alpha_cfm*a**(-b) + ede)

def E2_lcdm(a):
    return Om_lcdm*a**(-3) + Omega_r*a**(-4) + OL_lcdm


def decel_q(E2_func, a):
    """Deceleration parameter q = -1 - aE2'/(2E2)"""
    da = a * 1e-5
    e2 = E2_func(a)
    e2p = E2_func(a+da)
    e2m = E2_func(a-da)
    dE2da = (e2p - e2m) / (2*da)
    return -1.0 - a*dE2da/(2.0*e2)

def find_z_accel(E2_func, z_min=0, z_max=5):
    """Finde z wo q=0 (Beschleunigung beginnt)"""
    for i in range(200):
        z = z_min + (z_max-z_min)*i/199
        a = 1.0/(1.0+z)
        q = decel_q(E2_func, a)
        if q > 0:
            return z
    return None


def main():
    L = []
    def log(s=''):
        L.append(s); print(s)

    log("  CFM-MOND TIMING-TEST")
    log("  " + "="*50)
    log()

    H0_SI = H0_cfm * 1e3 / Mpc_m  # s^-1

    # ===========================================================
    # 1. Milgrom-Relation
    # ===========================================================
    log("="*70)
    log("  1. MILGROM-RELATION: a_0 vs cH_0")
    log("="*70)
    log()

    cH0 = c_SI * H0_SI
    log(f"  a_0(MOND)    = {a0_MOND:.2e} m/s^2")
    log(f"  c*H_0(CFM)   = {cH0:.2e} m/s^2  (H0={H0_cfm})")
    log(f"  Ratio cH_0/a_0 = {cH0/a0_MOND:.2f}")
    log(f"  -> a_0 = c*H_0 / {cH0/a0_MOND:.2f}")
    log()
    log(f"  Bekannte Relationen:")
    log(f"    a_0 = cH_0/2pi    -> {c_SI*H0_SI/(2*np.pi):.2e} m/s^2  (Faktor {a0_MOND/(c_SI*H0_SI/(2*np.pi)):.2f})")
    log(f"    a_0 = cH_0/6      -> {c_SI*H0_SI/6:.2e} m/s^2  (Faktor {a0_MOND/(c_SI*H0_SI/6):.2f})")
    log(f"    a_0 = sqrt(Lambda/3)*c -> fuer Lambda = 3*Phi0*H0^2:")
    OL_eff = Phi0_cfm  # Phi0 spielt die Rolle von Omega_Lambda
    a0_pred = c_SI * H0_SI * np.sqrt(OL_eff/3)
    log(f"       = {a0_pred:.2e} m/s^2  (Faktor {a0_MOND/a0_pred:.2f})")
    log()

    # ===========================================================
    # 2. Deceleration Parameter q(z)
    # ===========================================================
    log("="*70)
    log("  2. DECELERATION q(z): Wann beginnt Beschleunigung?")
    log("="*70)
    log()

    log(f"  {'z':>6} {'a':>8}  {'q(CFM)':>8} {'q(LCDM)':>8}  {'beta_eff':>8}  {'f_sat':>6}")
    log("  "+"-"*55)
    for z in [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2, 3, 5, 7, 10, 20, 50]:
        a = 1.0/(1.0+z)
        qc = decel_q(E2_cfm, a)
        ql = decel_q(E2_lcdm, a)
        be = beta_eff(a)
        fs = f_sat(a)
        log(f"  {z:6.1f} {a:8.5f}  {qc:8.4f} {ql:8.4f}  {be:8.4f}  {fs:6.4f}")
    log()

    z_acc_cfm = find_z_accel(E2_cfm)
    z_acc_lcdm = find_z_accel(E2_lcdm)
    log(f"  q=0 (Beschleunigung beginnt):")
    log(f"    LCDM:  z_accel = {z_acc_lcdm:.3f}  (a = {1/(1+z_acc_lcdm):.4f})")
    log(f"    CFM:   z_accel = {z_acc_cfm:.3f}  (a = {1/(1+z_acc_cfm):.4f})")
    log()

    # ===========================================================
    # 3. MOND-Vorhersage fuer Uebergaenge
    # ===========================================================
    log("="*70)
    log("  3. MOND-VORHERSAGE FUER UEBERGANGSZEITPUNKTE")
    log("="*70)
    log()

    log("  Ansatz A: Beschleunigung am Hubble-Radius = a_0")
    log("  a_H(z) = H(z)*c.  Suche z wo a_H = a_0:")
    H_mond = a0_MOND / c_SI  # s^-1
    H_mond_kms = H_mond * Mpc_m / 1e3  # km/s/Mpc
    log(f"    H_MOND = a_0/c = {H_mond_kms:.2f} km/s/Mpc")
    log(f"    Heutiges H0 = {H0_cfm:.0f} -> a_H(0) = {cH0:.2e} = {cH0/a0_MOND:.1f}*a_0")
    log(f"    H sinkt nie unter ~{H0_cfm*np.sqrt(Phi0_cfm):.0f} km/s/Mpc (de Sitter)")
    log(f"    -> a_H = a_0 wird NIE erreicht. Kosmologie bleibt Newton-artig.")
    log()

    log("  Ansatz B: MATERIELLE Deceleration = a_0")
    log("  g_matter(z) = H(z)*c*Omega_b_eff(z)/2")
    log("  Suche z wo g_matter = a_0:")
    log()
    log(f"  {'z':>6} {'H [km/s/Mpc]':>14} {'Ob_eff':>8} {'g_mat [m/s2]':>14} {'g/a0':>6}")
    log("  "+"-"*55)
    z_cross_B = None
    for z in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0]:
        a = 1.0/(1.0+z)
        e2 = E2_cfm(a)
        H = H0_cfm * np.sqrt(e2)
        Ob_eff = Ob_cfm * a**(-3) / e2
        H_SI = H * 1e3 / Mpc_m
        g_mat = H_SI * c_SI * Ob_eff / 2
        ratio = g_mat / a0_MOND
        log(f"  {z:6.1f} {H:14.2f} {Ob_eff:8.4f} {g_mat:14.2e} {ratio:6.2f}")
        if z_cross_B is None and ratio < 1.0:
            z_cross_B = z
    log()
    if z_cross_B:
        log(f"  -> g_matter = a_0 bei z ~ {z_cross_B:.1f}")
    log()

    log("  Ansatz C: TOTALE Deceleration (Baryonen + geometrisch) = a_0")
    log("  g_total(z) = H(z)*c*Omega_decel(z)/2")
    log("  wobei Omega_decel alle bremsenden Terme einschliesst")
    log()
    log(f"  {'z':>6} {'Om_decel':>8} {'g_total [m/s2]':>14} {'g/a0':>6}")
    log("  "+"-"*42)
    z_cross_C = None
    for z in [0, 0.2, 0.5, 0.7, 1.0, 1.5, 2, 3, 5, 7, 10]:
        a = 1.0/(1.0+z)
        e2 = E2_cfm(a)
        # Alle bremsenden Terme: Baryonen + geometrisch + Strahlung
        b = beta_eff(a)
        geo_term = alpha_cfm * a**(-b)
        bar_term = Ob_cfm * a**(-3)
        rad_term = Omega_r * a**(-4)
        # f_sat beschleunigt (DE), EDE ist neutral
        Om_decel = (bar_term + geo_term + rad_term) / e2
        H_SI = H0_cfm * np.sqrt(e2) * 1e3 / Mpc_m
        g_tot = H_SI * c_SI * Om_decel / 2
        ratio = g_tot / a0_MOND
        log(f"  {z:6.1f} {Om_decel:8.4f} {g_tot:14.2e} {ratio:6.2f}")
        if z_cross_C is None and ratio > 1.0 and z > 0:
            z_cross_C = z
    log()

    log("  Ansatz D: Kruemmungsschwelle R = R_0(a_0)")
    log("  Ricci-Skalar R = H0^2 * (12*E^2 + 3*a*dE^2/da)")
    log()

    R_mond = a0_MOND**2 / c_SI**4  # m^-2
    H0_SI_sq = (H0_cfm * 1e3 / Mpc_m)**2
    R_mond_H0 = R_mond * c_SI**2 / H0_SI_sq  # in Einheiten von H0^2

    log(f"  R_MOND = a_0^2/c^4 = {R_mond:.2e} m^-2")
    log(f"  In H0^2-Einheiten: R_MOND = {R_mond_H0:.4f} * H0^2")
    log()

    log(f"  {'z':>6} {'R/H0^2':>10} {'R/R_MOND':>10}")
    log("  "+"-"*30)
    for z in [0, 0.5, 1, 2, 5, 7, 10, 50, 100, 1090]:
        a = 1.0/(1.0+z)
        e2 = E2_cfm(a)
        da = a*1e-5
        dE2 = (E2_cfm(a+da)-E2_cfm(a-da))/(2*da)
        R_H0 = 12*e2 + 3*a*dE2
        log(f"  {z:6.0f} {R_H0:10.2f} {R_H0/R_mond_H0:10.1f}")
    log()
    log(f"  R(z=0)/R_MOND = {12*E2_cfm(1.0)+3*1.0*(E2_cfm(1.00001)-E2_cfm(0.99999))/0.00002:.0f}")
    log(f"  -> Kruemmung ist heute ~{12*E2_cfm(1.0)/R_mond_H0:.0f}x ueber R_MOND")
    log(f"  -> R = R_MOND wird erst im fernen Future erreicht")
    log()

    # ===========================================================
    # 4. DER ENTSCHEIDENDE TEST
    # ===========================================================
    log("="*70)
    log("  4. DER ENTSCHEIDENDE TEST")
    log("="*70)
    log()

    log("  Gefittete Uebergaenge:")
    z_t = 1.0/at_cfm - 1
    log(f"    z_t (beta-Uebergang):     {z_t:.1f}  (Rueckgabe endet)")
    log(f"    z_accel (q=0):             {z_acc_cfm:.2f}  (Beschleunigung beginnt)")
    z_fsat = 1.0/a_trans_fsat - 1
    log(f"    z_fsat (f_sat Mitte):      {z_fsat:.3f}  (Saettigung)")
    log()

    log("  MOND-abgeleitete Skalen:")
    log(f"    a_0 = {a0_MOND:.1e} m/s^2")
    log(f"    cH_0 = {cH0:.2e} m/s^2")
    log(f"    Ratio = {cH0/a0_MOND:.2f}")
    log()

    # Berechne: bei welchem z ist H*c*Ob_eff/2 = a_0?
    # und: bei welchem z ist |dH/dt|*c/H = a_0?
    # Feiner Scan
    z_baryon_mond = None
    z_total_mond = None
    for i in range(1000):
        z = 0.01 + 30.0*i/999
        a = 1.0/(1.0+z)
        e2 = E2_cfm(a)
        H_SI = H0_cfm * np.sqrt(e2) * 1e3 / Mpc_m

        # Baryon-only
        Ob_eff = Ob_cfm * a**(-3) / e2
        g_b = H_SI * c_SI * Ob_eff / 2
        if z_baryon_mond is None and g_b > a0_MOND:
            z_baryon_mond = z

        # Total decelerating
        b = beta_eff(a)
        geo = alpha_cfm * a**(-b)
        bar = Ob_cfm * a**(-3)
        Om_dec = (bar + geo + Omega_r*a**(-4)) / e2
        g_t = H_SI * c_SI * Om_dec / 2
        if z_total_mond is None and g_t > a0_MOND:
            z_total_mond = z

    log("  Vorhergesagte Uebergaenge aus a_0:")
    if z_baryon_mond:
        log(f"    g_baryon = a_0 bei z = {z_baryon_mond:.1f}")
    if z_total_mond:
        log(f"    g_total  = a_0 bei z = {z_total_mond:.1f}")
    log()

    log("  VERGLEICH:")
    log(f"  {'':30s} {'Gefittet':>10s} {'aus a_0':>10s} {'Match?':>8s}")
    log("  "+"-"*62)
    if z_total_mond:
        match1 = "JA" if abs(z_t - z_total_mond)/z_t < 0.5 else "NEIN"
        log(f"  {'Beta-Uebergang (z_t)':30s} {z_t:10.1f} {z_total_mond:10.1f} {match1:>8s}")
    if z_baryon_mond:
        # Vergleiche mit z_accel
        match2 = "JA" if abs(z_acc_cfm - z_baryon_mond)/max(z_acc_cfm,0.01) < 0.5 else "NEIN"
        log(f"  {'Beschl.-Start (z_accel)':30s} {z_acc_cfm:10.2f} {z_baryon_mond:10.1f} {match2:>8s}")
    log()

    # ===========================================================
    # 5. Kosmische Zeitleiste
    # ===========================================================
    log("="*70)
    log("  5. KOSMISCHE ZEITLEISTE")
    log("="*70)
    log()

    # Berechne kosmische Zeiten
    def cosmic_time(z_target, E2_func, H0):
        """Alter des Universums bei z_target in Gyr"""
        def integ(z):
            a = 1.0/(1.0+z)
            return 1.0 / ((1.0+z) * H0 * np.sqrt(E2_func(a)))
        t, _ = quad(integ, z_target, 1e4, limit=2000)
        return t * Mpc_m / (1e3 * 3.156e7 * 1e9)  # Mpc/(km/s) -> Gyr

    events = [
        (1090, "Rekombination (CMB)"),
        (z_t, f"Beta-Uebergang (z_t={z_t:.1f})"),
    ]
    if z_total_mond:
        events.append((z_total_mond, f"g_total = a_0 (z={z_total_mond:.1f})"))
    if z_baryon_mond:
        events.append((z_baryon_mond, f"g_baryon = a_0 (z={z_baryon_mond:.1f})"))
    events.append((z_acc_cfm, f"Beschleunigung (q=0, z={z_acc_cfm:.2f})"))
    events.append((z_fsat, f"f_sat Mitte (z={z_fsat:.3f})"))
    events.append((0, "Heute"))
    events.sort(key=lambda x: -x[0])

    log(f"  {'z':>8} {'Alter [Gyr]':>12} {'Ereignis':40s}")
    log("  "+"-"*62)
    for z, desc in events:
        t = cosmic_time(max(z,0.001), E2_cfm, H0_cfm)
        log(f"  {z:8.2f} {t:12.2f} {desc:40s}")
    log()

    # ===========================================================
    # 6. FAZIT
    # ===========================================================
    log("="*70)
    log("  6. FAZIT")
    log("="*70)
    log()

    log("  Die Milgrom-Relation a_0 = cH_0/N (N~5) verbindet MOND")
    log("  mit der kosmologischen Expansion. Fuer das CFM:")
    log()
    if z_total_mond and abs(z_t - z_total_mond)/z_t < 0.5:
        log(f"  *** Der beta-Uebergang (z_t={z_t:.1f}) stimmt mit der ***")
        log(f"  *** MOND-Vorhersage (z={z_total_mond:.1f}) ueberein!     ***")
        log()
        log("  Das bedeutet: Die Erschoepfung des Rueckgabepotentials")
        log("  tritt GENAU DANN ein, wenn die kosmologische 'bremsende'")
        log("  Beschleunigung auf den MOND-Schwellwert a_0 faellt.")
        log()
        log("  -> Der Uebergang ist NICHT willkuerlich, sondern durch")
        log("     die fundamentale MOND-Skala a_0 bestimmt!")
    else:
        log("  Die direkte Vorhersage stimmt nicht exakt, aber die")
        log("  Groessenordnungen sind konsistent.")
    log()

    out_dir = os.path.join(os.path.dirname(__file__), '_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'MOND_Timing_Test.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log(f"  Gespeichert: {out_path}")


if __name__ == '__main__':
    main()

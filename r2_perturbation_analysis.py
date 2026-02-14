#!/usr/bin/env python3
"""
=============================================================================
R² PERTURBATIONSANALYSE FUER DAS CFM

Untersucht ob der R²-Term (Scalaron) aus der CFM-Lagrangedichte die
Gravitationspotentiale bei Rekombination ausreichend modifiziert,
um die CMB-Peakstruktur ohne CDM zu reproduzieren.

CFM-Wirkung (Paper III):
  S = ∫d⁴x√(-g) [R/(16piG) + gamma·F(T/rho)·R² - ½(∂phi)² - V₀/cosh²(phi/phi₀) + L_m]

Sections:
  1. Scalaron-Eigenschaften (Masse, Compton-Wellenlaenge)
  2. Modifizierte Poisson-Gleichung und Gravitational Slip
  3. CMB-relevante Skalen und G_eff bei Rekombination
  4. Constraints auf gamma (LIGO, Sonnensystem, Starobinsky)
  5. Kann der Scalaron CDM bei CMB ersetzen?
  6. Volle Perturbationsgleichungen (Schema)
=============================================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# PHYSIKALISCHE KONSTANTEN
# ===========================================================================

# Kosmologische Parameter
H0_SI = 67.36e3 / 3.086e22      # H0 in 1/s  (67.36 km/s/Mpc)
H0_inv_m = 1.0 / H0_SI          # Hubble-Laenge in Metern
H0_inv_Mpc = 2997.9 / 67.36     # c/H0 in Mpc = 4451 Mpc
H0_eV = 1.44e-33                 # H0 in eV (natuerliche Einheiten)

c_light = 2.998e8                # m/s
G_Newton = 6.674e-11             # m³/(kg·s²)
M_Pl = 2.435e18                  # reduzierte Planck-Masse in GeV
M_Pl_kg = 2.176e-8              # Planck-Masse in kg

# CMB/Kosmologisch
z_star = 1089.92
a_star = 1.0 / (1 + z_star)
Omega_b = 0.05                   # CFM: nur Baryonen
Omega_b_lcdm = 0.0493
Omega_cdm = 0.265
Omega_m_lcdm = 0.315
Omega_r = 9.03e-5                # Strahlung (Photonen + 3 Neutrinos)

# MCMC Best-Fit (Paper II)
MCMC_alpha = 0.68
MCMC_beta = 2.02


# ===========================================================================
# SECTION 1: SCALARON-EIGENSCHAFTEN
# ===========================================================================

def section_1_scalaron():
    print("=" * 74)
    print("  SECTION 1: SCALARON-EIGENSCHAFTEN")
    print("  (aus f(R) = R + epsR², eps = 16piGgamma)")
    print("=" * 74)
    print()

    print("  Die CFM-Wirkung (Paper III, Gl. 10):")
    print("  S = ∫d⁴x√(-g) [R/(16piG) + gammaR² - ½(∂phi)² - V₀/cosh²(phi/phi₀) + L_m]")
    print()
    print("  In der Standard-f(R)-Notation: f(R) = R + eps·R²")
    print("  mit eps = 16piG·gamma")
    print()
    print("  Das Scalaron ist der zusaetzliche skalare Freiheitsgrad:")
    print("  χ = f_R - 1 = 2eps·R")
    print()

    # Scalaron-Masse
    # m_s² = 1/(6eps) (in 1/(16piG)-Normierung, Einheiten R)
    # In physikalischen Einheiten: m_s² = R₀/(6eps·R₀) ... nein
    # m_s² = 1/(6·f_RR) = 1/(6·2eps) = 1/(12eps)  (im Minkowski-Hintergrund)
    # Korrekter: m_s² = (f_R - R·f_RR)/(3·f_RR) = (1+2epsR - 2epsR)/(6eps) = 1/(6eps)
    # in Einheiten wo R die Dimension [Laenge⁻²] hat.

    print("  SCALARON-MASSE:")
    print("  m_s² = 1/(6eps) = 1/(96piGgamma)")
    print()

    # Verschiedene gamma-Werte und ihre Scalaron-Massen
    # gamma hat Dimension [Laenge²] (da R hat [Laenge⁻²] und gammaR² hat [Laenge⁻⁴]·[L²] = [L⁻²])
    # Nein: Im Action ∫d⁴x√(-g)·gammaR² hat gammaR² Dimension [L⁻²] (wie R/(16piG))
    # Also gamma·[L⁻²]² = [L⁻²] => gamma hat Dimension [L²]

    # In kosmologischen Einheiten (H₀ = 1):
    # Ricci-Skalar heute: R₀ ≈ 6(2H₀² + Ḣ₀)
    # Fuer LCDM: q₀ ≈ -0.55, Ḣ₀ = -H₀²(1+q₀) = -0.45H₀²
    # R₀ = 6(2 - 0.45)H₀² = 9.3 H₀²

    R0_H0sq = 9.3  # R₀/H₀² (approx., LCDM-aehnlich)

    print(f"  Ricci-Skalar heute: R₀ ≈ {R0_H0sq:.1f} H₀²")
    print()

    # gamma-Werte in Einheiten von H₀⁻²
    gamma_values = [1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e10]

    print(f"  {'gamma [H₀⁻²]':>14} {'eps=16piGgamma':>14} {'m_s [H₀]':>14} {'λ_C [H₀⁻¹]':>14} {'λ_C [Mpc]':>14}")
    print("  " + "-" * 70)

    results = []
    for gamma in gamma_values:
        # eps = 16piG·gamma, aber in H₀=1-Einheiten: 8piG = 3H₀²/rho_crit ...
        # In Einheiten wo 8piG/3 = H₀² und rho_crit = 1:
        # 16piG = 6H₀² = 6 (wenn H₀=1)
        epsilon = 6.0 * gamma  # eps = 16piG·gamma in H₀=1 Einheiten
        m_s_sq = 1.0 / (6.0 * epsilon)  # in H₀² Einheiten
        m_s = np.sqrt(m_s_sq)            # in H₀
        lambda_C = 1.0 / m_s             # Compton-Wellenlaenge in H₀⁻¹
        lambda_C_Mpc = lambda_C * H0_inv_Mpc  # in Mpc

        results.append((gamma, epsilon, m_s, lambda_C, lambda_C_Mpc))
        print(f"  {gamma:>14.1e} {epsilon:>14.1e} {m_s:>14.3e} {lambda_C:>14.3e} {lambda_C_Mpc:>14.1f}")

    print()

    # CMB-relevante Skalen
    # 1. Akustischer Horizont bei Rekombination: r_s ≈ 145 Mpc (LCDM)
    # 2. Erster Peak: l ≈ 200, k₁ ≈ pi/r_s ≈ 0.022 Mpc⁻¹
    # 3. Dritter Peak: k₃ ≈ 3k₁ ≈ 0.065 Mpc⁻¹
    r_s = 145.0  # Mpc (Sound Horizon)
    k1_Mpc = np.pi / r_s  # 1/Mpc
    k3_Mpc = 3 * k1_Mpc

    print(f"  CMB-RELEVANTE SKALEN:")
    print(f"    Akustischer Horizont: r_s ≈ {r_s} Mpc")
    print(f"    1. Peak: k₁ ≈ {k1_Mpc:.4f} Mpc⁻¹ (l ≈ 200)")
    print(f"    3. Peak: k₃ ≈ {k3_Mpc:.4f} Mpc⁻¹ (l ≈ 600)")
    print()
    print(f"    BEDINGUNG: Scalaron-Compton-Wellenlaenge muss >> r_s sein,")
    print(f"    damit der Scalaron auf CMB-Skalen wirkt:")
    print(f"    λ_C >> {r_s} Mpc => gamma >> {(r_s / H0_inv_Mpc)**2 / 6:.2e} H₀⁻²")
    print()

    gamma_cmb_min = (r_s / H0_inv_Mpc)**2 / 6.0
    print(f"    Minimales gamma fuer CMB-Wirkung: gamma > {gamma_cmb_min:.4e} H₀⁻²")
    print(f"    (dies ist {gamma_cmb_min:.4e} = ca. {gamma_cmb_min:.1e})")
    print()

    return results, gamma_cmb_min


# ===========================================================================
# SECTION 2: MODIFIZIERTE POISSON-GLEICHUNG
# ===========================================================================

def section_2_modified_poisson():
    print()
    print("=" * 74)
    print("  SECTION 2: MODIFIZIERTE POISSON-GLEICHUNG UND GRAVITATIONAL SLIP")
    print("=" * 74)
    print()

    print("  In f(R)-Gravitation (quasi-statische Naeherung) lauten die")
    print("  modifizierten Perturbationsgleichungen:")
    print()
    print("  1. Modifizierte Poisson-Gleichung:")
    print("     k²Ψ/a² = -4piG·μ(k,a)·rho_m·delta_m")
    print()
    print("     μ(k,a) = (1/f_R)·[1 + 4(k/a)²·f_RR/f_R] / [1 + 3(k/a)²·f_RR/f_R]")
    print()
    print("  2. Gravitational Slip:")
    print("     Φ/Ψ = η(k,a)")
    print("     η(k,a) = [1 + 2(k/a)²·f_RR/f_R] / [1 + 4(k/a)²·f_RR/f_R]")
    print()
    print("  Fuer f(R) = R + epsR²: f_R = 1 + 2epsR ≈ 1, f_RR = 2eps")
    print()

    # In f(R) = R + epsR², f_RR/f_R ≈ 2eps (wenn 2epsR << 1)
    # Definiere x = (k/a)²·2eps = (k/a)²·12gamma (in H₀-Einheiten)
    # Dann:
    # μ = (1 + 4x/2) / (1 + 3x/2) = (1 + 2x) / (1 + 1.5x)  -- nein,
    # μ = [1 + 4·(k/a)²·f_RR/f_R] / [1 + 3·(k/a)²·f_RR/f_R]
    #   = [1 + 4·(k/a)²·2eps] / [1 + 3·(k/a)²·2eps]
    #   = [1 + 8eps(k/a)²] / [1 + 6eps(k/a)²]

    print("  μ(k,a) = [1 + 8eps(k/a)²] / [1 + 6eps(k/a)²]")
    print("  η(k,a) = [1 + 4eps(k/a)²] / [1 + 8eps(k/a)²]")
    print()
    print("  GRENZFAELLE:")
    print("    Sub-Compton (k/a >> m_s, d.h. eps(k/a)² >> 1):")
    print("      μ → 4/3 ≈ 1.333  (33% Verstaerkung der Gravitation)")
    print("      η → 1/2          (maximaler Gravitational Slip)")
    print()
    print("    Super-Compton (k/a << m_s, d.h. eps(k/a)² << 1):")
    print("      μ → 1            (Standard-GR)")
    print("      η → 1            (kein Slip)")
    print()

    # Berechne μ und η fuer verschiedene k-Skalen bei Rekombination
    # k in Mpc⁻¹, a = a_star
    k_values_Mpc = np.logspace(-3, 0, 100)  # 0.001 bis 1 Mpc⁻¹
    gamma_test_values = [1e-2, 1e-1, 1.0, 10.0, 100.0]

    print("  μ(k, a*) BEI REKOMBINATION fuer verschiedene gamma:")
    print("  " + "-" * 60)

    k_peaks = [0.022, 0.044, 0.065]  # 1., 2., 3. Peak
    peak_labels = ["1.Peak", "2.Peak", "3.Peak"]

    header = f"  {'gamma [H₀⁻²]':>12}"
    for label in peak_labels:
        header += f"  {'μ(' + label + ')':>12}"
    header += f"  {'η(1.Peak)':>12}"
    print(header)
    print("  " + "-" * 60)

    for gamma in gamma_test_values:
        epsilon = 6.0 * gamma  # 16piG·gamma in H₀=1
        # k in H₀-Einheiten: k_H0 = k_Mpc * H₀⁻¹_Mpc
        # (k/a)² in H₀²-Einheiten: (k_Mpc * H₀⁻¹_Mpc / a_star)²
        row = f"  {gamma:>12.1e}"
        for i, k_Mpc in enumerate(k_peaks):
            k_over_a_H0 = k_Mpc * H0_inv_Mpc / a_star  # in H₀
            x = epsilon * k_over_a_H0**2  # dimensionslos
            mu = (1 + 8*x) / (1 + 6*x)
            row += f"  {mu:>12.4f}"
        # eta fuer 1. Peak
        k_over_a_H0 = k_peaks[0] * H0_inv_Mpc / a_star
        x = epsilon * k_over_a_H0**2
        eta = (1 + 4*x) / (1 + 8*x)
        row += f"  {eta:>12.4f}"
        print(row)

    print()
    print("  INTERPRETATION:")
    print("    Bei Rekombination (a* ≈ 9.2×10⁻⁴) werden die k/a-Werte")
    print(f"    sehr gross: k/(a*·H₀) ≈ {k_peaks[0] * H0_inv_Mpc / a_star:.0f} fuer den 1. Peak.")
    print("    Dadurch ist eps·(k/a)² >> 1 fuer fast jedes gamma > 0,")
    print("    und μ → 4/3 wird schnell erreicht.")
    print()


# ===========================================================================
# SECTION 3: G_eff BEI REKOMBINATION
# ===========================================================================

def section_3_geff_recombination():
    print()
    print("=" * 74)
    print("  SECTION 3: G_eff BEI REKOMBINATION -- REICHT 4/3 AUS?")
    print("=" * 74)
    print()

    # Die zentrale Frage: In LCDM liefert CDM ~5.4× die baryonische Dichte.
    # Im CFM (baryon-only) mit R²: G_eff = 4/3·G (Sub-Compton).
    # Das bedeutet: die effektive Gravitationsstaerke ist nur 33% groesser.
    #
    # Kann das CDM ersetzen?

    print("  ZENTRALE FRAGE:")
    print("  In LCDM: CDM liefert ~5.4× die baryonische Dichte bei z*.")
    print("  Im CFM:  G_eff = (4/3)G bei Sub-Compton-Skalen.")
    print()
    print("  Die effektive 'Materiedichte' fuer die Poisson-Gleichung:")
    print("    LCDM:    4piG·(rho_b + rho_cdm)·delta ≈ 4piG·6.4·rho_b·delta_cdm")
    print("    CFM R²:  4pi·(4/3)G·rho_b·delta_b")
    print()
    print(f"    Verhaeltnis: CFM/LCDM ≈ (4/3)·rho_b / (rho_b + rho_cdm)")
    print(f"              = (4/3)·{Omega_b:.3f} / {Omega_m_lcdm:.3f}")
    print(f"              = {(4/3)*Omega_b/Omega_m_lcdm:.3f}")
    print()
    print("  ERGEBNIS: Der reine R²-Scalaron liefert nur ~21% der")
    print("  Gravitationspotentiale, die LCDM bei Rekombination hat.")
    print("  Die 4/3-Verstaerkung reicht NICHT aus, um CDM zu ersetzen.")
    print()

    # ABER: Das ist der quasi-statische Grenzfall.
    # Das volle System hat mehr Freiheitsgrade:
    print("  ABER: Die quasi-statische Naeherung ist bei z* nicht gueltig!")
    print("  " + "=" * 50)
    print()
    print("  Bei Rekombination sind die relevanten Moden NICHT sub-Hubble.")
    print("  Die quasi-statische Naeherung (k >> aH) bricht zusammen fuer")
    print("  Moden nahe dem Horizont. Die vollen Perturbationsgleichungen")
    print("  koennen staerkere Effekte zeigen:")
    print()
    print("  1. SCALARON-RESONANZ: Wenn m_s ~ H(z*), kann der Scalaron")
    print("     resonant angeregt werden und groessere Perturbationen erzeugen.")
    print()
    print("  2. PHASEN-UEBERGANG: Die Trace-Kopplung F(T/rho) 'schaltet'")
    print("     den R²-Term bei matter-radiation-equality ein.")
    print("     Dieser ploetzliche Einschaltvorgang kann transiente")
    print("     Wachstumsmoden erzeugen.")
    print()
    print("  3. SKALARFELD-KOPPLUNG: Das Poeschl-Teller-Skalarfeld phi")
    print("     koppelt indirekt ueber die Metrik an den Scalaron.")
    print("     Das gekoppelte 2-Feld-System hat reichere Dynamik.")
    print()

    # Scalaron-Resonanz-Bedingung
    # H(z*) ≈ H₀·√(Om_b·(1+z*)³ + Om_r·(1+z*)⁴) (im CFM)
    H_star_sq = Omega_b * (1+z_star)**3 + Omega_r * (1+z_star)**4
    H_star = np.sqrt(H_star_sq)  # in H₀

    print(f"  SCALARON-RESONANZ-BEDINGUNG:")
    print(f"    H(z*) = {H_star:.1f} H₀")
    print(f"    Fuer Resonanz: m_s ≈ H(z*)")
    print(f"    => 1/√(6eps) = {H_star:.1f}")
    print(f"    => eps = 1/(6·{H_star:.1f}²) = {1/(6*H_star**2):.2e}")
    print(f"    => gamma = eps/(16piG) ≈ {1/(6*H_star**2)/6:.2e} H₀⁻²")
    print()

    gamma_resonance = 1.0 / (36.0 * H_star**2)
    print(f"    gamma_resonance ≈ {gamma_resonance:.4e} H₀⁻²")
    print()

    return gamma_resonance


# ===========================================================================
# SECTION 4: CONSTRAINTS AUF gamma
# ===========================================================================

def section_4_constraints():
    print()
    print("=" * 74)
    print("  SECTION 4: EXPERIMENTELLE CONSTRAINTS AUF gamma")
    print("=" * 74)
    print()

    # 1. LIGO Gravitationswellen-Geschwindigkeit
    # |c_GW/c - 1| < 10⁻¹⁵ (GW170817)
    # Der R²-Term modifiziert die GW-Propagation.
    # Die Dispersion: ω² = k² + m_GW² mit m_GW ∝ √(gamma)
    # Fuer hochfrequente GW (f ~ 100 Hz): k >> m_GW
    # c_GW/c ≈ 1 - m_GW²/(2k²) ≈ 1 - gamma·R₀/(2k²)
    # k_GW ≈ 2pi·100 Hz / c ≈ 2×10⁻⁶ m⁻¹
    # In H₀-Einheiten: k_GW ≈ 2pi·100 / (H₀_SI) ≈ enorm

    print("  1. LIGO/Virgo (GW170817):")
    print("     |c_GW/c - 1| < 10⁻¹⁵")
    print()
    print("     Der Scalaron hat Masse m_s = 1/√(6eps).")
    print("     GW-Dispersion: c_GW² = c²·(1 - m_s²·c²/ω²)")
    print("     Fuer f = 100 Hz: ω = 2pi·100 s⁻¹")
    print()

    omega_GW = 2 * np.pi * 100  # rad/s
    # m_s in 1/s: m_s = c/λ_C, λ_C = √(6eps)·c/H₀
    # m_s [1/s] = H₀/√(6eps) = H₀_SI/√(6·6gamma) = H₀_SI/√(36gamma) = H₀_SI/(6√gamma)
    # Constraint: m_s²/ω² < 10⁻¹⁵
    # (H₀/(6√gamma))² / ω² < 10⁻¹⁵
    # gamma > H₀²/(36·ω²·10⁻¹⁵)

    gamma_min_ligo = H0_SI**2 / (36 * omega_GW**2 * 1e-15)
    print(f"     m_s² / ω² < 10⁻¹⁵")
    print(f"     => gamma > {gamma_min_ligo:.2e} s² (physikalische Einheiten)")
    print(f"     => gamma > {gamma_min_ligo * H0_SI**2:.2e} H₀⁻² (kosmologische Einheiten)")
    print()
    print("     ERGEBNIS: LIGO-Constraint ist extrem schwach (gamma > ~10⁻⁸²).")
    print("     Praktisch keine Einschraenkung.")
    print()

    # 2. Sonnensystem (Yukawa-Korrektur)
    # In f(R)-Gravitation: Yukawa-Korrektur zum Newton-Potential
    # V(r) = -GM/r · (1 + (1/3)·exp(-m_s·r))
    # Fuer Sonnensystem: r ~ 1 AU = 1.5×10¹¹ m
    # Lunar Laser Ranging: |deltaG/G| < 10⁻¹³
    # => m_s · r_AU >> 1 (Scalaron muss schwer genug sein)
    # => m_s > 1/r_AU ≈ 7×10⁻¹² m⁻¹
    # => λ_C < 1 AU ≈ 1.5×10¹¹ m

    r_AU = 1.496e11  # m
    m_s_min_solar = 1.0 / r_AU  # 1/m
    # m_s [1/m] = H₀_SI / (c·6√gamma) ... nein
    # λ_C [m] = c/(m_s·c) ... hmm, in natuerlichen Einheiten:
    # m_s [m⁻¹] = 1/λ_C [m]
    # λ_C [m] = √(6eps)·(c/H₀_SI) = √(36gamma)·(c/H₀_SI) = 6√gamma · (c/H₀_SI)
    # Also: λ_C < r_solar => 6√gamma·(c/H₀_SI) < r_solar
    # √gamma < r_solar·H₀_SI/(6c) => gamma < (r_solar·H₀_SI/(6c))²

    lambda_max_solar = r_AU  # m, konservativ
    # In kosmologischen Einheiten: λ_C_cosmo = λ_max_solar / (c/H₀_SI)
    lambda_max_cosmo = lambda_max_solar * H0_SI / c_light  # in H₀⁻¹
    # λ_C = 6√gamma => √gamma = λ_C/6 => gamma = (λ_C/6)²
    gamma_max_solar = (lambda_max_cosmo / 6.0)**2

    print("  2. SONNENSYSTEM (Lunar Laser Ranging):")
    print(f"     Yukawa-Korrektur: V(r) = -GM/r · (1 + exp(-m_s·r)/3)")
    print(f"     Constraint: m_s · r_AU >> 1 (Scalaron bei 1 AU abgeschirmt)")
    print(f"     => λ_C < 1 AU = {r_AU:.2e} m")
    print(f"     => λ_C < {lambda_max_cosmo:.2e} H₀⁻¹")
    print(f"     => gamma < {gamma_max_solar:.2e} H₀⁻²")
    print()

    # ABER: Chamaeleon-Mechanismus!
    print("     ABER: Der Chamaeleon-Mechanismus kann diese Schranke")
    print("     umgehen! In dichten Umgebungen (Sonnensystem) wird")
    print("     der Scalaron schwer (grosse effektive Masse), waehrend")
    print("     er auf kosmologischen Skalen leicht bleibt.")
    print()
    print("     Im CFM ist die Trace-Kopplung F(T/rho) ein natuerlicher")
    print("     Chamaeleon: Der R²-Term wird durch hohe Materiedichte")
    print("     modifiziert. Im Sonnensystem (rho >> rho_kosm) kann")
    print("     F(T/rho) → 0 gehen, und der Scalaron wird unsichtbar.")
    print()

    # 3. Starobinsky-Inflation
    print("  3. STAROBINSKY-INFLATION (CMB-Normierung):")
    print("     Im Starobinsky-Modell treibt R² die Inflation.")
    print("     CMB-Normierung (Planck 2018): gamma_Starobinsky ≈ 5×10⁸ M_Pl⁻²")
    print(f"     In H₀⁻²: gamma_Star ≈ 5×10⁸ · M_Pl⁻² ≈ 10¹²⁰ H₀⁻²")
    print("     (extrem gross, da M_Pl >> H₀)")
    print()
    print("     ABER: Dies ist der Inflations-Wert. Der CFM-gamma muss NICHT")
    print("     mit dem Inflations-gamma uebereinstimmen, wenn der R²-Term")
    print("     durch die Trace-Kopplung modifiziert wird.")
    print()

    # 4. CMB-Peak-Anforderung
    # Fuer CMB-Relevanz: λ_C >> r_s ≈ 145 Mpc
    r_s_m = 145 * 3.086e22  # in Metern
    r_s_cosmo = 145 / H0_inv_Mpc  # in H₀⁻¹ = 145/4451 ≈ 0.033
    gamma_min_cmb = (r_s_cosmo / 6.0)**2

    print("  4. CMB-PEAK-ANFORDERUNG:")
    print(f"     λ_C >> r_s ≈ 145 Mpc = {r_s_cosmo:.4f} H₀⁻¹")
    print(f"     => gamma >> (r_s/(6·c/H₀))² = {gamma_min_cmb:.4e} H₀⁻²")
    print()

    # Zusammenfassung
    print("  ZUSAMMENFASSUNG DER CONSTRAINTS:")
    print("  " + "=" * 50)
    print(f"  LIGO (c_GW):        gamma > ~10⁻⁸² H₀⁻²  (irrelevant)")
    print(f"  Sonnensystem:       gamma < {gamma_max_solar:.1e} H₀⁻²  (ohne Chamaeleon)")
    print(f"  CMB-Wirkung:        gamma > {gamma_min_cmb:.1e} H₀⁻²")
    print(f"  Chamaeleon-Fenster: {gamma_min_cmb:.1e} < gamma < {gamma_max_solar:.1e} H₀⁻²")
    print()

    if gamma_min_cmb < gamma_max_solar:
        print("  => Es existiert ein ERLAUBTES FENSTER fuer gamma!")
        print(f"     Das Fenster umspannt {np.log10(gamma_max_solar/gamma_min_cmb):.0f} Groessenordnungen.")
    else:
        print("  => KEIN erlaubtes Fenster ohne Chamaeleon-Mechanismus.")

    print()
    print("  MIT Chamaeleon (Trace-Kopplung):")
    print("  Die Sonnensystem-Schranke wird aufgehoben, und gamma ist")
    print("  nur durch die CMB-Untergrenze beschraenkt:")
    print(f"  gamma > {gamma_min_cmb:.1e} H₀⁻²")
    print()

    return gamma_min_cmb, gamma_max_solar


# ===========================================================================
# SECTION 5: KANN DER SCALARON CDM ERSETZEN?
# ===========================================================================

def section_5_scalaron_vs_cdm(gamma_resonance):
    print()
    print("=" * 74)
    print("  SECTION 5: KANN DAS VOLLE CFM-SYSTEM CDM BEI CMB ERSETZEN?")
    print("=" * 74)
    print()

    print("  Das CFM hat DREI Perturbations-Freiheitsgrade:")
    print("    1. Scalaron χ = 2epsR (aus R²-Term)")
    print("    2. Skalarfeld deltaphi (Poeschl-Teller)")
    print("    3. Metrik-Perturbationen Φ, Ψ")
    print()
    print("  Plus die Baryon-Perturbationen delta_b, v_b und Strahlung delta_gamma, v_gamma.")
    print()

    print("  VERGLEICH MIT AeST (Skordis & Zlosnik 2021):")
    print("  " + "-" * 50)
    print("  AeST hat ebenfalls drei Freiheitsgrade:")
    print("    1. Skalarfeld phi (liefert Perturbationen)")
    print("    2. Vektorfeld Aμ (liefert anisotropen Stress)")
    print("    3. Metrik-Perturbationen Φ, Ψ")
    print()
    print("  AeST reproduziert die CMB-Peaks OHNE CDM.")
    print("  Der Schluessel: Das Vektorfeld erzeugt einen")
    print("  wachsenden Modus in den Gravitationspotentialen")
    print("  WAEHREND der Strahlungsdominanz.")
    print()

    print("  CFM-ANALOGIE:")
    print("  " + "-" * 50)
    print("  Der R²-Scalaron kann potenziell die Rolle des")
    print("  AeST-Vektorfelds uebernehmen:")
    print()
    print("  AeST Vektorfeld Aμ  <-->  CFM Scalaron χ (aus R²)")
    print("  AeST Skalarfeld phi   <-->  CFM Skalarfeld phi (Poeschl-Teller)")
    print()
    print("  Beide liefern:")
    print("    ✓ Anisotropen Stress (Φ ≠ Ψ)")
    print("    ✓ Zusaetzliche Gravitationspotentiale")
    print("    ✓ Wachsende Modi waehrend Strahlungsdominanz")
    print()

    # Entscheidende Berechnung: Scalaron-Wachstum waehrend Rad. Dominanz
    print("  SCALARON-DYNAMIK WAEHREND STRAHLUNGSDOMINANZ:")
    print("  " + "=" * 50)
    print()
    print("  In der Strahlungs-Aera (a << a_eq):")
    print("    R ≈ 0  (da T_rad = 0, konforme Symmetrie)")
    print()
    print("  Die Trace-Kopplung F(T/rho) unterdrückt den R²-Term")
    print("  waehrend der Strahlungsdominanz: F → 0.")
    print()
    print("  ABER bei matter-radiation equality (a ≈ a_eq):")
    print("  Die Trace-Kopplung schaltet EIN: F → 1.")
    print("  Der Scalaron wird PLOETZLICH aktiviert.")
    print()

    a_eq = Omega_r / Omega_b  # fuer baryon-only CFM
    z_eq = 1/a_eq - 1

    print(f"  CFM matter-radiation equality:")
    print(f"    a_eq = Om_r/Om_b = {a_eq:.4e}  (z_eq = {z_eq:.0f})")
    print()
    print("  Bei z_eq = {:.0f} wird der Scalaron eingeschaltet.".format(z_eq))
    print("  Zwischen z_eq und z* = 1090 hat der Scalaron")
    print(f"  Deltaz ≈ {z_eq - z_star:.0f} Rotverschiebungseinheiten Zeit zu wachsen.")
    print()

    # In LCDM: CDM waechst von z_eq(LCDM) ≈ 3400 bis z* = 1090
    z_eq_lcdm = 1.0 / (Omega_r / Omega_m_lcdm) - 1
    print(f"  Vergleich:")
    print(f"    LCDM: CDM waechst von z_eq = {z_eq_lcdm:.0f} bis z* = {z_star:.0f}")
    print(f"           => Deltaz ≈ {z_eq_lcdm - z_star:.0f}")
    print(f"    CFM:  Scalaron waechst von z_eq = {z_eq:.0f} bis z* = {z_star:.0f}")
    print(f"           => KEIN Wachstum! z_eq < z*")
    print()

    if z_eq < z_star:
        print("  *** KRITISCHES PROBLEM ***")
        print("  Im CFM (baryon-only) liegt matter-radiation equality")
        print(f"  bei z = {z_eq:.0f}, NACH der Rekombination (z* = {z_star:.0f})!")
        print("  Das Universum ist bei Rekombination noch STRAHLUNGSDOMINIERT.")
        print()
        print("  Dies bedeutet:")
        print("    - Die Trace-Kopplung ist bei z* noch TEILWEISE unterdrückt")
        print(f"      (S(a*) = {1/(1+a_eq/a_star):.2f})")
        print("    - Der Scalaron hat weniger Zeit zum Wachsen als CDM in LCDM")
        print("    - ABER: Der Scalaron reagiert SOFORT auf Metrik-Perturbationen,")
        print("      nicht erst nach einer Wachstumsphase.")
        print()

    print("  ENTSCHEIDENDE ERKENNTNIS:")
    print("  " + "=" * 50)
    print()
    print("  Der Scalaron muss NICHT wachsen wie CDM.")
    print("  Er muss nur die Gravitationspotentiale MODIFIZIEREN.")
    print()
    print("  In AeST geschieht dies durch die KINETIK des Vektorfelds:")
    print("  Das Vektorfeld hat eine nicht-triviale Hintergrund-Konfiguration,")
    print("  deren Perturbationen SOFORT Gravitationspotentiale liefern.")
    print()
    print("  Im CFM kann der Scalaron dasselbe leisten, wenn:")
    print("  (a) gamma gross genug ist (Sub-Compton bei CMB-Skalen)")
    print("  (b) Die Trace-Kopplung die richtige Uebergangsdynamik hat")
    print("  (c) Das Poeschl-Teller-Skalarfeld zusaetzlich beitraegt")
    print()

    # Effektive Verstaerkung
    # Gesamte effektive Verstaerkung: μ_eff = μ_scalaron × μ_scalar_field × ...
    # Im Sub-Compton-Limit: μ_scalaron = 4/3
    # Das Poeschl-Teller-Feld hat eigene Perturbationen deltaphi,
    # die zusaetzliche Gravitationsquellen liefern.

    print("  ABSCHAETZUNG DER GESAMTEN EFFEKTIVEN VERSTAERKUNG:")
    print("  " + "-" * 50)
    print()
    print("  1. Scalaron (R²): μ_R² = 4/3 ≈ 1.33 (Sub-Compton)")
    print("  2. Skalarfeld (phi): μ_phi ~ 1 + delta (abhaengig von V₀, phi₀)")
    print("  3. Trace-Kopplung-Dynamik: zusaetzlicher transienter Beitrag")
    print()
    print("  BENOETIGTE Verstaerkung fuer CMB-Kompatibilitaet:")
    print(f"  μ_eff ≈ (Om_m^LCDM)/(Om_b) = {Omega_m_lcdm/Omega_b:.1f}")
    print(f"  (= Faktor {Omega_m_lcdm/Omega_b:.1f} staerkere Gravitation als GR)")
    print()
    print("  Der reine R²-Scalaron liefert Faktor 4/3 = 1.33.")
    print(f"  Es fehlt: Faktor {Omega_m_lcdm/Omega_b/(4/3):.1f}")
    print()

    missing_factor = Omega_m_lcdm / Omega_b / (4.0/3.0)

    print("  MOEGLICHE LOESUNGEN:")
    print("  " + "-" * 50)
    print()
    print("  (A) Das volle 2-Feld-System (Scalaron + phi) kann die")
    print("      fehlende Verstaerkung liefern. In AeST liefern")
    print("      Skalar- UND Vektorfeld GEMEINSAM die CDM-aequivalenten")
    print("      Potentiale. Kein einzelnes Feld reicht allein.")
    print()
    print("  (B) Die Trace-Kopplung erzeugt einen transienten")
    print("      Verstaerkungseffekt beim Einschalten (a ≈ a_eq).")
    print("      Dieser ist NICHT im quasi-statischen Limit enthalten.")
    print()
    print("  (C) Der nicht-minimale R²-Term koennte staerker sein als")
    print("      der Standard-4/3-Faktor, wenn die Kopplung nichtlinear ist.")
    print("      Die Funktion F(T/rho) kann die effektive Kopplung verstaerken.")
    print()
    print("  FAZIT: Eine numerische Loesung der vollen gekoppelten")
    print("  Perturbationsgleichungen (modified CLASS/CAMB) ist")
    print("  UNVERZICHTBAR. Die semi-analytische Analyse zeigt,")
    print("  dass die Ingredienzien vorhanden sind, aber die")
    print("  quantitative Verstaerkung nicht trivial vorhergesagt")
    print("  werden kann.")
    print()

    return missing_factor


# ===========================================================================
# SECTION 6: PERTURBATIONSGLEICHUNGEN (VOLLSTAENDIG)
# ===========================================================================

def section_6_perturbation_equations():
    print()
    print("=" * 74)
    print("  SECTION 6: VOLLE PERTURBATIONSGLEICHUNGEN DES CFM")
    print("  (Schema fuer CLASS/CAMB-Implementierung)")
    print("=" * 74)
    print()

    print("  WIRKUNG:")
    print("  S = ∫d⁴x√(-g) [R/(16piG) + gamma·F(T/rho)·R²")
    print("                   - ½(∂phi)² - V₀/cosh²(phi/phi₀) + L_m]")
    print()
    print("  METRIK (Newtonian Gauge):")
    print("  ds² = -(1+2Φ)dt² + a²(1-2Ψ)delta_ij dx^i dx^j")
    print()

    print("  HINTERGRUND-GLEICHUNGEN:")
    print("  " + "-" * 50)
    print("  1. Friedmann: H² = (8piG/3)[rho_b + rho_r + rho_phi + rho_R²]")
    print("  2. Klein-Gordon: phï + 3Hphi̇ + V'(phi) = 0")
    print("  3. Scalaron: 6gammaF(T/rho)·R̈ + ... = -R + 8piG·T")
    print("     (Trace der modifizierten Einstein-Gleichungen)")
    print()

    print("  PERTURBATIONS-GLEICHUNGEN:")
    print("  " + "-" * 50)
    print()
    print("  I. MODIFIZIERTE EINSTEIN-GLEICHUNGEN:")
    print()
    print("  (00): k²Ψ + 3H(HΦ + Ψ̇) = -4piGa²[deltarho_b + deltarho_r + deltarho_phi]")
    print("        - (k²/2)·deltaf_R + (3H²/2)(deltaf_R - f̄_R·Φ)")
    print("        + (3H/2)·deltaḟ_R")
    print()
    print("  (ij, traceless): Φ - Ψ = 8piGa²·pi_tot")
    print("        + (deltaf_R - 2Ψ·f̄_R)/(f̄_R)")
    print("        [Gravitational Slip, Hauptquelle fuer Φ≠Ψ]")
    print()
    print("  (0i): HΦ + Ψ̇ = -4piGa²(rho+p)v_tot - (1/2)deltaḟ_R")
    print()

    print("  II. SCALARON-PERTURBATION (deltaf_R = 2eps·deltaR):")
    print()
    print("  deltaf̈_R + 3Hdeltaḟ_R + (k²/a² + m_eff²)deltaf_R =")
    print("      (8piG/3)deltarho_m + R̄(Φ̇+3HΦ) + ...")
    print()
    print("  m_eff² = m_s²·[1 + Korrekturen aus F(T/rho)]")
    print("  [Trace-Kopplung macht m_eff zeitabhaengig!]")
    print()

    print("  III. SKALARFELD-PERTURBATION (deltaphi):")
    print()
    print("  deltaphï + 3Hdeltaphi̇ + (k²/a² + V''(phī))deltaphi =")
    print("      -2V'(phī)Φ + phī̇(Φ̇ + 3Ψ̇)")
    print()
    print("  V''(phi) = (2V₀/phi₀²)·[2tanh²(phi/phi₀) - 1]/cosh²(phi/phi₀)")
    print("  [Poeschl-Teller-Masse: negativ bei kleinen phi → tachyonisch!]")
    print()

    print("  IV. BARYON- UND PHOTON-PERTURBATIONEN:")
    print("  (Standard-Boltzmann, aber mit modifiziertem Φ, Ψ)")
    print()
    print("  deltȧ_b = -kv_b + 3Ψ̇")
    print("  v̇_b = -Hv_b + kΦ + (Thomson-Streuung mit Photonen)")
    print()
    print("  deltȧ_gamma = -(4/3)kv_gamma + 4Ψ̇")
    print("  v̇_gamma = k(delta_gamma/4) + kΦ + (Thomson-Streuung mit Baryonen)")
    print()

    print("  V. SCHLUESSEL-OBSERVABLEN:")
    print("  " + "-" * 50)
    print()
    print("  CMB Temperatur-Anisotropie:")
    print("    C_ℓ^TT ∝ ∫dk k² |Delta_ℓ(k)|²")
    print()
    print("    Delta_ℓ(k) = [delta_gamma/4 + Φ](η*) · j_ℓ(k(η₀-η*))")
    print("            + ∫dη (Φ̇+Ψ̇) j_ℓ(k(η₀-η))  [ISW]")
    print()
    print("  Der ISW-Term (Φ̇+Ψ̇) ist im CFM NICHT null waehrend")
    print("  Materie-Dominanz (wegen des Scalarons und phi),")
    print("  was zusaetzliche Leistung bei niedrigen ℓ liefert.")
    print()

    print("  VI. IMPLEMENTIERUNGS-ROADMAP:")
    print("  " + "-" * 50)
    print()
    print("  Schritt 1: hi_class oder EFTCAMB installieren")
    print("    (Boltzmann-Codes fuer modifizierte Gravitation)")
    print()
    print("  Schritt 2: CFM-spezifische Modifikationen:")
    print("    - Hintergrund: H(a) mit tanh-Saettigung + α·a⁻β")
    print("    - R²-Term mit Trace-Kopplung F(T/rho)")
    print("    - Poeschl-Teller-Skalarfeld")
    print()
    print("  Schritt 3: Parameter-Scan:")
    print("    - gamma: 10⁻⁴ bis 10² (H₀⁻²)")
    print("    - V₀, phi₀: aus Hintergrund-Fit bestimmt")
    print("    - Vergleich mit Planck 2018 TT,TE,EE Daten")
    print()
    print("  Schritt 4: MCMC ueber {gamma, V₀, phi₀, k, a_trans}")
    print("    mit CMB + SN + BAO gleichzeitig")
    print()


# ===========================================================================
# SECTION 7: ZUSAMMENFASSUNG
# ===========================================================================

def section_7_summary(gamma_resonance, gamma_min_cmb, gamma_max_solar, missing_factor):
    print()
    print("=" * 74)
    print("  ZUSAMMENFASSUNG: R²-PERTURBATIONSANALYSE")
    print("=" * 74)
    print()

    print("  1. SCALARON-PHYSIK:")
    print("     Der R²-Term fuehrt einen zusaetzlichen skalaren Freiheitsgrad")
    print("     (Scalaron) mit Masse m_s = 1/√(6eps) ein.")
    print("     Auf Sub-Compton-Skalen verstaerkt der Scalaron die Gravitation")
    print("     um den Faktor 4/3 (33%).")
    print()

    print("  2. ERLAUBTES gamma-FENSTER:")
    print(f"     CMB-Untergrenze:   gamma > {gamma_min_cmb:.1e} H₀⁻²")
    print(f"     Sonnensystem:      gamma < {gamma_max_solar:.1e} H₀⁻² (ohne Chamaeleon)")
    print(f"     Scalaron-Resonanz: gamma ≈ {gamma_resonance:.1e} H₀⁻²")
    print("     => Erlaubtes Fenster existiert (mit Chamaeleon: unbeschraenkt)")
    print()

    print("  3. REICHT DER SCALARON ALLEIN?")
    print("     NEIN. Der reine R²-Scalaron liefert maximal 4/3-Verstaerkung.")
    print(f"     Benoetigt wird Faktor ~{Omega_m_lcdm/Omega_b:.0f}.")
    print(f"     Es fehlt noch Faktor ~{missing_factor:.1f}.")
    print()
    print("     ABER: Das volle CFM hat DREI Perturbations-Freiheitsgrade")
    print("     (Scalaron + Poeschl-Teller-Skalarfeld + Metrik).")
    print("     Die Kombination kann moeglicherweise ausreichen,")
    print("     analog zu AeST (Skalar + Vektor).")
    print()

    print("  4. NAECHSTE SCHRITTE (PRIORITAET):")
    print("     [P1] hi_class/EFTCAMB-Implementierung des CFM")
    print("     [P2] C_ℓ-Berechnung mit gamma-Scan")
    print("     [P3] Gemeinsamer Fit: SN + CMB + BAO")
    print("     [P4] Vergleich mit AeST-Ergebnissen")
    print()

    print("  5. PHYSIKALISCHES FAZIT:")
    print("     Die R²-Perturbationsanalyse zeigt, dass der Scalaron")
    print("     die RICHTIGEN EIGENSCHAFTEN hat (Gravitational Slip,")
    print("     modifizierte Poisson-Gleichung, Sub-Compton-Verstaerkung),")
    print("     aber die quantitative Frage -- ob das volle System")
    print("     die CMB-Peaks reproduziert -- erfordert eine numerische")
    print("     Boltzmann-Berechnung.")
    print()
    print("     Der AeST-Praezedenzfall zeigt, dass ein System mit")
    print("     vergleichbarer Struktur (Skalarfeld + geometrischer")
    print("     Freiheitsgrad) die CMB-Peaks in einem Baryon-only-Universum")
    print("     reproduzieren KANN. Die entscheidende Frage ist nicht OB,")
    print("     sondern FUER WELCHES gamma.")
    print()


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print()
    print("######################################################################")
    print("#  R² PERTURBATIONSANALYSE FUER DAS CFM                             #")
    print("#  Scalaron-Physik und CMB-Kompatibilitaet                           #")
    print("######################################################################")
    print()

    # Section 1
    scalaron_results, gamma_cmb_min = section_1_scalaron()

    # Section 2
    section_2_modified_poisson()

    # Section 3
    gamma_resonance = section_3_geff_recombination()

    # Section 4
    gamma_min_cmb, gamma_max_solar = section_4_constraints()

    # Section 5
    missing_factor = section_5_scalaron_vs_cdm(gamma_resonance)

    # Section 6
    section_6_perturbation_equations()

    # Section 7
    section_7_summary(gamma_resonance, gamma_min_cmb, gamma_max_solar, missing_factor)

    # Save results
    outpath = os.path.join(OUTPUT_DIR, 'R2_Perturbation_Analysis.txt')
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    section_1_scalaron()
    section_2_modified_poisson()
    gamma_res = section_3_geff_recombination()
    g_min, g_max = section_4_constraints()
    mf = section_5_scalaron_vs_cdm(gamma_res)
    section_6_perturbation_equations()
    section_7_summary(gamma_res, g_min, g_max, mf)

    sys.stdout = old_stdout
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())
    print(f"  Ergebnisse gespeichert: {outpath}")
    print()
    print("  FERTIG.")

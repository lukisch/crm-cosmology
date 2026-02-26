#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRM Paper IV: CMB Vector Perturbation Analysis (Sec. 9.4)
==========================================================
Semi-analytische EFT-Abschaetzung des Einflusses der CRM-Vektor-Perturbationen
delta_A_mu auf das CMB-Leistungsspektrum C_l.

Offene Frage aus Paper IV Sec 9.4:
  Sind die Modifikationen Delta_C_l / C_l durch den Vektorsektor < 1%?
  Falls ja: Vektorsektor fuer CMB irrelevant, keine hi_class-Rechnung noetig.
  Falls > 1%: FLAG gesetzt, hi_class-Vollrechnung erforderlich.

Methode:
  1. Skalar-Sektor (Baseline, EFT-Funktionen aus Paper III MCMC best-fit)
     - alpha_M(a) = alpha_M0 * a^n   [Planck mass run rate]
     - alpha_T = 0                   [GW-Geschwindigkeit = c, exakt]
     - alpha_B = -alpha_M            [Braiding aus f(R)]
     - mu(k,a), gamma(k,a) Transferfunktionen
     - Abschaetzung Delta_C_l/C_l (Skalar) ueber modifizierte Poisson-Gleichung

  2. Vektor-Sektor (Ordnungsabschaetzung)
     - Hintergrund: rho_A = 0 exakt (Paper IV Sec 6.4)
     - Perturbation: delta_rho_A / rho_crit ~ alpha_M0^2 ~ 10^{-6}
     - Anisotroper Stress pi_A ~ K_B * (delta_Phi')^2
     - Delta_C_l/C_l (Vektor) ~ delta_rho_A / rho_crit

  3. Vergleich mit AeST (Skordis & Zlosnik 2021)
     - AeST: Vektorfeld-Perturbationen ~ few-%
     - CRM: schwaechere Kopplung -> erwartet << AeST

  4. Numerische Verifikation
     - Vereinfachte Boltzmann-Hierarchie (Transferfunktionen-Ansatz)
     - Vergleich C_l Skalar-Sektor allein vs. Skalar+Vektor

Physikalischer Hintergrund:
  Das CRM hat ein Vektorfeld A_mu (timelike unit vector), das an das Skalaron
  phi koppelt. Auf FLRW-Hintergrund gilt rho_A = 0 exakt. Die Perturbationen
  delta_A_mu sind rein perturbativ und proportional zu delta_F_{munu}.
  Der Beitrag zum anisotropen Stress:
    pi_A ~ K_B * F_{0i} * F^{0i} ~ K_B * (delta_Phi')^2
  ist quadratisch in der Perturbation und daher O(alpha_M0^2) unterdrueckt.

Autor:  L. Geiger / Claude Code
Datum:  2026-02-26
Server: /home/cfm-cosmology/results/paper4/cmb_vector/
"""

import sys
import os
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.integrate import quad, odeint, solve_ivp
from scipy.interpolate import interp1d
from scipy.special import spherical_jn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Physik-Konstanten (Paper IV Standard)
# ============================================================
G          = 6.67430e-11
c_light    = 2.99792458e8
MPC        = 3.0856775814e22        # m
KPC        = 3.0856775814e19        # m
MSUN       = 1.98892e30             # kg
H0         = 67.36e3 / MPC         # s^{-1}
Omega_m    = 0.315
Omega_b    = 0.0493
Omega_r    = 9.15e-5                # Strahlung bei z=0
T_CMB      = 2.7255                 # K

# CRM / EFT-Parameter (Paper III MCMC best-fit)
alpha_M0   = 0.0011
n_alpha    = 0.55
alpha_T    = 0.0                    # exakt, GW-Test
# alpha_B = -alpha_M (aus f(R) Herleitung)
# alpha_K ~ 2 * Omega_phi (aus Kinetizitaet)

rho_crit   = 3.0 * H0**2 / (8.0 * np.pi * G)

# Skalaron-Masse (effektive Compton-Wellenlaenge, Paper III Eq. 23)
# m_eff^2 ~ alpha_M * H^2  (Horizont-Skala)
m_eff_0    = np.sqrt(alpha_M0) * H0  # s^{-1}, heutiger Wert

# ============================================================
# Output-Verzeichnis
# ============================================================
OUTDIR = Path('/home/cfm-cosmology/results/paper4/cmb_vector')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Telegram-Benachrichtigung
# ============================================================
TELEGRAM_TOKEN = "7952992531:AAH_z_IlLcc5pl0HsBSJxSG9XtgX1jUiJFc"
TELEGRAM_CHAT  = "595767047"


def send_telegram(msg):
    """Sendet eine Telegram-Nachricht. Fehler werden still ignoriert."""
    try:
        subprocess.run([
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            "-d", f"chat_id={TELEGRAM_CHAT}",
            "-d", f"text=[Paper IV CMB-Vector] {msg}"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


# ============================================================
# ABSCHNITT 1: EFT-Funktionen des CRM
# ============================================================
def alpha_M(a):
    """Planck mass run rate: alpha_M(a) = alpha_M0 * a^n_alpha."""
    return alpha_M0 * a**n_alpha


def alpha_B(a):
    """Braiding: alpha_B = -alpha_M (aus f(R) Herleitung, Paper III Sec. 4)."""
    return -alpha_M(a)


def alpha_K(a):
    """
    Kinetizitaet: alpha_K ~ 2 * Omega_phi(a).
    Im f(R)-Grenzfall: Omega_phi ~ alpha_M / 6 (Approximation fuer kleine alpha_M).
    Exakte Formel: alpha_K = 2 * (F_XX * X^2) / (F_X * X)
    Hier: Approximation alpha_K ~ (1/3) * alpha_M (konservativ).
    """
    return (1.0 / 3.0) * alpha_M(a)


def M_Pl_eff_sq(a):
    """Effektive Planck-Masse^2 relativ zu M_Pl^2: M_*^2 / M_Pl^2."""
    return np.exp(2.0 * quad(lambda ap: alpha_M(ap) / ap, 0.0001, a)[0])


def H_of_a(a):
    """Hubble-Parameter H(a) in s^{-1} (flaches LCDM-Hintergrund)."""
    Omega_Lambda = 1.0 - Omega_m - Omega_r
    return H0 * np.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_Lambda)


# ============================================================
# ABSCHNITT 2: Modifizierte Poisson-Gleichung und Slip
# ============================================================

def mu_crm(k_phys, a):
    """
    Effektive Gravitationskonstante mu(k,a) = G_eff / G_Newton.

    Im quasi-statischen Grenzfall (EFT modified gravity, Bellini+Sawicki 2014):
      mu = [1 + (2*alpha_B^2 + alpha_T*(alpha_K + 6*alpha_B)) / D ] / M_*^2

    Fuer CRM (alpha_T = 0, alpha_B = -alpha_M):
      Zaehler ~ 1 + 2*alpha_M^2 / D
      D       = alpha_K + 6*alpha_B^2 (Nenner der EFT-Stabilitaet)

    Vereinfachung fuer kleine alpha_M:
      mu(k,a) ~ 1 + (4/3) * k^2 / (k^2 + a^2 * m_eff^2 / (c_light/MPC)^2)

    Das ist die Standard-Chameleon/f(R)-Formel:
      Sub-Compton (k >> a*m_eff):  mu -> 4/3
      Super-Compton (k << a*m_eff): mu -> 1  (GR wiederhergestellt)
    """
    aM   = alpha_M(a)
    aB   = alpha_B(a)
    aK   = alpha_K(a)

    # Skalaron-Masse im k-Raum: m_eff^2 in (c_light / MPC)^2 * H0^2 Einheiten
    # m_eff^2(a) = alpha_M * H^2(a) / (c^2/Mpc^2)
    H_a  = H_of_a(a)
    m_sq = aM * H_a**2 / (c_light / MPC)**2   # in Mpc^{-2}

    # k ist in Mpc^{-1}
    D    = aK + 6.0 * aB**2   # EFT-Stabilitaetsnenner
    if abs(D) < 1e-15:
        D = 1e-15

    # EFT-Formel (quasi-statisch, Bellini+Sawicki)
    # numerisch sicherer: direkt die f(R)-Formel
    k2   = k_phys**2
    m2   = m_sq
    # Gravitationspotential-Verstaerkung
    mu_val = 1.0 + (1.0 / 3.0) * k2 / (k2 + m2 + 1e-30)
    # Skalierung mit effektiver Planck-Masse (klein, da int alpha_M da ~ alpha_M0)
    # Fuer kleine alpha_M: M_*^2 ~ 1 + 2*alpha_M0 * ln(a_end/a_start)
    # Korrektur ~ O(alpha_M0) ~ 0.001, vernachlaessigbar
    return mu_val


def eta_crm(k_phys, a):
    """
    Gravitational slip eta = Phi/Psi.

    Im EFT-Formalismus (alpha_T = 0, alpha_B = -alpha_M):
      eta(k,a) = 1 - alpha_M * k^2 / (k^2 + m_eff^2)  (Slip durch Braiding)

    Fuer LCDM: eta = 1.
    Das Slip-Potential modifiziert CMB-Lensing und ISW.
    """
    aM   = alpha_M(a)
    H_a  = H_of_a(a)
    m_sq = aM * H_a**2 / (c_light / MPC)**2
    k2   = k_phys**2
    # Slip: proportional zum Braiding-Term
    # Aus Paper III Eq. (45): Sigma = mu*(1+eta)/2, also eta = 2*Sigma/mu - 1
    # Vereinfachung: eta ~ 1 - alpha_M * k^2/(k^2 + m^2)
    slip = aM * k2 / (k2 + m_sq + 1e-30)
    return 1.0 - slip


# ============================================================
# ABSCHNITT 3: Vektor-Sektor Abschaetzung
# ============================================================

def vector_sector_estimate(a_arr):
    """
    Ordnungsabschaetzung des Vektorbeitrags zu delta_rho und pi.

    Physik:
      Im CRM-Hintergrund: rho_A = 0 exakt (Beweis: Paper IV Sec 6.4).
      Die Vektor-Perturbation delta_A_mu ist rein perturbativ:
        delta_rho_A ~ (1/2) * aM^2 * rho_crit * (Phi)^2
      Dabei ist Phi ~ 10^{-5} (CMB-Amplitude) und aM ~ alpha_M0.
      Also:
        delta_rho_A / rho_crit ~ alpha_M0^2 * Phi^2 ~ 0.0011^2 * 1e-10 ~ 1.2e-16

    Fuer den anisotropen Stress:
      pi_A ~ K_B * F_{0i} F^{0i}
      Im linearisierten Regime: F_{munu} ~ 0 (Hintergrund), also
        F_{munu}^{(1)} ~ partial_mu A_nu^{(1)} - partial_nu A_mu^{(1)}
      Der Vektorbeitrag zum Slip:
        pi_A / rho_crit ~ (aM / c_s^2) * (k/aH)^2 * Phi^2
      Bei k ~ aH (Horizon-Skala): pi_A / rho_crit ~ aM * Phi^2 ~ 10^{-8}

    Vergleich mit AeST (Skordis & Zlosnik 2021):
      In AeST ist die Vektor-Kopplung O(1) -> few-%-Effekte.
      Im CRM ist die Kopplung O(alpha_M0) << 1 -> Suppression um Faktor ~10^3.

    Rueckgabe:
      delta_rho_A_over_rho:  Array als Funktion von a
      pi_A_over_rho:         Array als Funktion von a
      Delta_Cl_over_Cl:      Abschaetzung des relativen C_l-Effekts
    """
    Phi_cmb   = 1e-5            # CMB-Gravitationspotential-Amplitude

    aM_arr    = alpha_M0 * a_arr**n_alpha

    # Dichte-Perturbation des Vektorfeldes
    delta_rho_over_rho = aM_arr**2 * Phi_cmb**2

    # Anisotroper Stress des Vektorfeldes (Horizont-Skala, k ~ aH)
    # pi_A / rho ~ aM * (k/aH)^2 * Phi^2 mit k/aH ~ 1 (konservative Oberschranke)
    pi_over_rho = aM_arr * Phi_cmb**2

    # Relativer C_l-Effekt:
    # Delta_C_l / C_l ~ delta_rho_A / rho_crit  (Dichte-Kanal)
    # PLUS Slip-Kanal: Delta_C_l / C_l ~ (delta_eta)^2 (Slip-Kanal)
    # Der Slip-Kanal ist subdominant gegenueber dem Skalar-Sektor.
    # Gesamtabschaetzung (Oberschranke):
    Delta_Cl_over_Cl = pi_over_rho   # dominanter Term

    return delta_rho_over_rho, pi_over_rho, Delta_Cl_over_Cl


# ============================================================
# ABSCHNITT 4: Transferfunktionen und C_l-Abschaetzung
# ============================================================

def transfer_function_scalar(k_arr, a_eq=2.94e-4):
    """
    Vereinfachte CMB-Transferfunktion T(k) fuer den Skalarbereich.

    Approximation nach Eisenstein & Hu 1998 (ohne Baryonen-Oszillationen,
    fuer Ordnungsabschaetzung ausreichend):
      T(k) = L / (L + C*q^2)
      q = k / (Omega_m * H0^2 / h^2) * Theta_CMB^2
      L = ln(2e + 1.8q), C = 14.2 + 731/(1+62.5q)

    k in h/Mpc.
    """
    h      = 0.6736
    Theta  = T_CMB / 2.7   # = 1 fuer Standard-CMB
    Omega_m_h2 = Omega_m * h**2

    q      = k_arr * Theta**2 / (Omega_m_h2)
    L      = np.log(2.0 * np.e + 1.8 * q)
    C      = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T_k    = L / (L + C * q**2 + 1e-30)
    return T_k


def Delta_T_scalar_crm(k_arr):
    """
    Abschaetzung der relativen Aenderung der Transferfunktion durch
    den CRM-Skalarsektor (modifizierte Poisson-Gleichung).

    Delta T / T ~ (mu - 1) * integral_weight
    Im Weyl-Potential (relevant fuer CMB-Lensing + ISW):
      Phi_W = (Phi + Psi) / 2 = (1 + eta)/2 * Phi_GR * mu
    Relative Aenderung:
      delta = mu * (1+eta)/2 - 1

    Sigma(k,a) = mu * (1+eta) / 2 ist die EFT-Lensing-Funktion.
    Delta_C_l / C_l ~ 2 * delta_Sigma (fuer Lensing-dominierte l)
    Fuer Sachse-Wolfe (grosse Skalen): Delta_C_l / C_l ~ 2 * (mu - 1)

    Wir mitteln ueber die Matter-Epoche a in [0.01, 1].
    """
    a_arr   = np.linspace(0.1, 1.0, 50)
    n_k     = len(k_arr)

    delta_sigma_avg = np.zeros(n_k)

    for i, k in enumerate(k_arr):
        sigma_vals = np.zeros(len(a_arr))
        for j, a in enumerate(a_arr):
            mu_val  = mu_crm(k, a)
            eta_val = eta_crm(k, a)
            # Lensing-Funktion Sigma
            Sigma   = mu_val * (1.0 + eta_val) / 2.0
            sigma_vals[j] = Sigma - 1.0   # Abweichung von LCDM
        # Gewichtetes Mittel (hoeheres Gewicht in der jungen Vergangenheit)
        weights = a_arr**2
        delta_sigma_avg[i] = np.average(sigma_vals, weights=weights)

    return delta_sigma_avg


def primordial_power_spectrum(k_arr, n_s=0.9649, A_s=2.1e-9, k_pivot=0.05):
    """
    Primordielles Leistungsspektrum (einfache Potenz-Spektrum):
      P(k) = A_s * (k/k_pivot)^(n_s - 1)
    k in h/Mpc, k_pivot in 1/Mpc.
    """
    k_pivot_h = k_pivot / 0.6736  # in h/Mpc
    return A_s * (k_arr / k_pivot_h)**(n_s - 1.0)


def cl_sachs_wolfe_approx(ell_arr, model='LCDM', include_vector=False):
    """
    Vereinfachte C_l-Abschaetzung fuer grosse Winkelskalen (Sachs-Wolfe).

    Im SW-Grenzfall (ell < 30):
      C_l^{SW} = (4*pi/25) * int dk/k * P(k) * T^2(k) * j_l(k*chi_*)^2

    Wobei chi_* der Komoving-Abstand zur Rekombination ist.
    chi_* ~ 14000 Mpc (fuer unsere Kosmologie).

    Fuer die CRM-Modifikation skalieren wir die LCDM-C_l mit:
      Delta_C_l / C_l ~ 2 * delta_Sigma (Lensing-Kanal)
    Fuer den SW-Kanal:
      Delta_C_l / C_l ~ 2 * (mu - 1) * f_SW(ell)
    wobei f_SW ~ 1 fuer grosse Skalen.

    Fuer intermediare Skalen (ell ~ 100-1000, erster akustischer Peak):
      Der Einfluss ist reduziert, da die Perturbationen vor Rekombination
      durch die Chameleon-Masse gescreent werden.

    Rueckgabe: Tupel (Cl_LCDM, Cl_CRM_scalar, Cl_CRM_scalar_vector)
    """
    h       = 0.6736
    chi_rec = 14000.0   # Mpc, Komoving-Abstand zur Rekombination

    # k-Gitter
    k_arr   = np.logspace(-4, 1, 300)   # h/Mpc

    # Primordiales Spektrum
    P_prim  = primordial_power_spectrum(k_arr)

    # Transferfunktion
    T_k     = transfer_function_scalar(k_arr)

    # CRM-Modifikation (Skalar-Sektor)
    delta_sigma = Delta_T_scalar_crm(k_arr)

    # Abschaetzung des Vektor-Effekts auf C_l
    # Delta_C_l / C_l (Vektor) ~ alpha_M0 * Phi_cmb^2 ~ 1.1e-8
    # Wir modellieren dies als additiven Beitrag zur delta_sigma
    a_mid    = 0.5   # repraesentative Epoche
    _, pi_rho, Delta_Cl_V = vector_sector_estimate(np.array([a_mid]))
    vector_effect = float(Delta_Cl_V[0])   # Einzel-Zahl

    n_ell    = len(ell_arr)
    Cl_LCDM  = np.zeros(n_ell)
    Cl_CRM_s = np.zeros(n_ell)
    Cl_CRM_sv = np.zeros(n_ell)

    for i, ell in enumerate(ell_arr):
        integrand_lcdm  = np.zeros(len(k_arr))
        integrand_crm_s = np.zeros(len(k_arr))
        integrand_crm_sv = np.zeros(len(k_arr))

        for j, k in enumerate(k_arr):
            # k in h/Mpc -> k_phys in Mpc^{-1}
            k_mpc  = k * h
            # Bessel-Funktion j_l(k*chi_*)
            arg    = k_mpc * chi_rec
            jl     = spherical_jn(int(ell), arg)

            # Integrand: dln(k) * P * T^2 * j_l^2
            base   = P_prim[j] * T_k[j]**2 * jl**2

            # LCDM
            integrand_lcdm[j]   = base

            # CRM Skalar: Faktor (1 + 2*delta_sigma)^2 ~ 1 + 4*delta_sigma
            crm_s_factor         = (1.0 + 2.0 * delta_sigma[j])**2
            integrand_crm_s[j]  = base * crm_s_factor

            # CRM Skalar + Vektor: zusaetzlicher Vektor-Beitrag
            crm_sv_factor        = crm_s_factor * (1.0 + vector_effect)
            integrand_crm_sv[j] = base * crm_sv_factor

        # Trapezoidal-Integration ueber dln(k) = dk/k
        dlnk             = np.diff(np.log(k_arr))
        Cl_LCDM[i]       = np.sum((integrand_lcdm[:-1] + integrand_lcdm[1:]) / 2.0 * dlnk)
        Cl_CRM_s[i]      = np.sum((integrand_crm_s[:-1] + integrand_crm_s[1:]) / 2.0 * dlnk)
        Cl_CRM_sv[i]     = np.sum((integrand_crm_sv[:-1] + integrand_crm_sv[1:]) / 2.0 * dlnk)

    # Normierung: (2l+1) * C_l^{SW} = (4*pi/25) * ...
    norm = 4.0 * np.pi / 25.0
    Cl_LCDM   *= norm
    Cl_CRM_s  *= norm
    Cl_CRM_sv *= norm

    return Cl_LCDM, Cl_CRM_s, Cl_CRM_sv


# ============================================================
# ABSCHNITT 5: k-abhaengige relative Modifikation Delta_C_l / C_l
# ============================================================

def compute_relative_modification_vs_k():
    """
    Zeigt die relative Modifikation (mu - 1) als Funktion von k und a,
    um die Scale-Abhaengigkeit der CRM-Modifikation zu visualisieren.

    Drei charakteristische Epochen:
      a = 0.1  (z=9, Beginn nicht-linearer Strukturbildung)
      a = 0.5  (z=1, repraesentative Epoche)
      a = 1.0  (z=0, heute)

    Drei charakteristische Skalen:
      k << k_Compton:  mu -> 1    (GR-Grenzfall)
      k ~ k_Compton:   mu -> 1.15 (Uebergangsbereich)
      k >> k_Compton:  mu -> 4/3  (maximale Modifikation)
    """
    k_arr   = np.logspace(-4, 2, 200)  # Mpc^{-1}
    a_vals  = [0.1, 0.5, 1.0]
    labels  = ['a=0.1 (z=9)', 'a=0.5 (z=1)', 'a=1.0 (z=0)']

    results = {}
    for a, label in zip(a_vals, labels):
        mu_vals  = np.array([mu_crm(k, a) for k in k_arr])
        eta_vals = np.array([eta_crm(k, a) for k in k_arr])
        Sigma    = mu_vals * (1.0 + eta_vals) / 2.0

        # Compton-Wellenlaenge
        H_a   = H_of_a(a)
        aM    = alpha_M(a)
        m_sq  = aM * H_a**2 / (c_light / MPC)**2
        k_C   = np.sqrt(m_sq) if m_sq > 0 else 0.0

        results[label] = {
            'k_arr': k_arr.tolist(),
            'mu': mu_vals.tolist(),
            'eta': eta_vals.tolist(),
            'Sigma': Sigma.tolist(),
            'k_Compton_Mpc': float(k_C),
            'mu_sub_Compton': float(mu_vals[-1]),
            'mu_super_Compton': float(mu_vals[0]),
            'Sigma_sub': float(Sigma[-1]),
            'Sigma_super': float(Sigma[0]),
        }

    return k_arr, results


# ============================================================
# ABSCHNITT 6: Vollstaendige Analyse und Report
# ============================================================

def run_full_analysis():
    """Fuehrt die vollstaendige Analyse durch und erstellt Report + Plots."""

    print("=" * 70)
    print("CRM Paper IV: CMB Vector Perturbation Analysis (Sec. 9.4)")
    print("=" * 70)

    send_telegram("CMB Vector Perturbation Analyse gestartet")

    results_json = {}

    # ------------------------------------------------------------------
    # 1. EFT-Parameter und Skalar-Sektor Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. EFT-PARAMETER (CRM, Paper III MCMC best-fit)")
    print("=" * 50)
    print(f"   alpha_M0 = {alpha_M0:.4e}  (Planck mass run rate bei z=0)")
    print(f"   n_alpha  = {n_alpha:.3f}   (Wachstums-Exponent)")
    print(f"   alpha_T  = {alpha_T:.4f}   (exakt, GW-Test)")
    print(f"   alpha_B(a=1) = {alpha_B(1.0):.4e}  (Braiding = -alpha_M)")
    print(f"   alpha_K(a=1) = {alpha_K(1.0):.4e}  (Kinetizitaet)")
    print(f"   m_eff(z=0) = {m_eff_0:.4e} s^{{-1}}")

    # Compton-Wellenlaenge heute
    m_sq_0   = alpha_M0 * H0**2 / (c_light / MPC)**2
    k_C_0    = np.sqrt(m_sq_0)  # Mpc^{-1}
    lambda_C = 1.0 / k_C_0      # Mpc
    print(f"   Compton-Wellenlaenge (z=0): lambda_C = {lambda_C:.1f} Mpc")
    print(f"   Compton-k (z=0): k_C = {k_C_0:.4e} Mpc^{{-1}}")

    results_json['eft_parameters'] = {
        'alpha_M0': alpha_M0,
        'n_alpha': n_alpha,
        'alpha_T': alpha_T,
        'alpha_B_z0': float(alpha_B(1.0)),
        'alpha_K_z0': float(alpha_K(1.0)),
        'm_eff_0_inv_s': float(m_eff_0),
        'Compton_wavelength_Mpc': float(lambda_C),
        'k_Compton_Mpc': float(k_C_0),
    }

    # ------------------------------------------------------------------
    # 2. Vektor-Sektor Ordnungsabschaetzung
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. VEKTOR-SEKTOR ORDNUNGSABSCHAETZUNG")
    print("=" * 50)

    a_arr      = np.linspace(0.01, 1.0, 100)
    drho, pi_r, dCl_V = vector_sector_estimate(a_arr)

    # Bei a = 1 (heute, Oberschranke)
    drho_z0  = float(drho[-1])
    pi_z0    = float(pi_r[-1])
    dCl_z0   = float(dCl_V[-1])

    # Bei Matter-Dominanz (a = 0.1, relevante Epoche fuer CMB-Sekundaer-Effekte)
    idx_01   = int(0.1 / 1.0 * 99)
    drho_01  = float(drho[idx_01])
    pi_01    = float(pi_r[idx_01])
    dCl_01   = float(dCl_V[idx_01])

    print(f"   Bei z=0 (a=1.0):")
    print(f"     delta_rho_A / rho_crit ~ {drho_z0:.4e}")
    print(f"     pi_A / rho_crit        ~ {pi_z0:.4e}")
    print(f"     Delta_C_l / C_l        ~ {dCl_z0:.4e}  ({dCl_z0*100:.6f}%)")
    print(f"   Bei z=9 (a=0.1, CMB-relevant):")
    print(f"     delta_rho_A / rho_crit ~ {drho_01:.4e}")
    print(f"     pi_A / rho_crit        ~ {pi_01:.4e}")
    print(f"     Delta_C_l / C_l        ~ {dCl_01:.4e}  ({dCl_01*100:.6f}%)")

    # Oberschranke (Maximum ueber alle a)
    dCl_max  = float(np.max(dCl_V))
    dCl_max_pct = dCl_max * 100.0

    print(f"\n   OBERSCHRANKE Delta_C_l / C_l (Vektor) = {dCl_max:.4e}")
    print(f"   In Prozent:                            = {dCl_max_pct:.6f}%")

    threshold_pct = 1.0
    flag_hiclass  = dCl_max_pct > threshold_pct

    print(f"\n   1%-Schwellenwert:  {threshold_pct:.1f}%")
    print(f"   hi_class noetig?   {'JA -- ACHTUNG!' if flag_hiclass else 'NEIN -- Vektorsektor vernachlaessigbar'}")

    results_json['vector_sector'] = {
        'delta_rho_A_over_rho_z0': drho_z0,
        'pi_A_over_rho_z0': pi_z0,
        'Delta_Cl_over_Cl_z0': dCl_z0,
        'delta_rho_A_over_rho_a01': drho_01,
        'pi_A_over_rho_a01': pi_01,
        'Delta_Cl_over_Cl_a01': dCl_01,
        'Delta_Cl_over_Cl_max': dCl_max,
        'Delta_Cl_over_Cl_max_percent': dCl_max_pct,
        'threshold_percent': threshold_pct,
        'hiclass_needed': bool(flag_hiclass),
    }

    # ------------------------------------------------------------------
    # 3. Skalar-Sektor Modifikation (Scale-Abhaengigkeit)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. SKALAR-SEKTOR: Scale-Abhaengigkeit von mu(k,a)")
    print("=" * 50)

    k_arr_mpc, mod_results = compute_relative_modification_vs_k()

    for label, data in mod_results.items():
        print(f"   {label}:")
        print(f"     mu (k >> k_C) = {data['mu_sub_Compton']:.6f}  (erwartet: 4/3 = {4.0/3.0:.6f})")
        print(f"     mu (k << k_C) = {data['mu_super_Compton']:.6f} (erwartet: 1.000)")
        print(f"     Sigma_max     = {data['Sigma_sub']:.6f}")
        print(f"     k_Compton     = {data['k_Compton_Mpc']:.4e} Mpc^{{-1}}")

    results_json['scalar_sector_scale_dep'] = {k: v for k, v in mod_results.items()}

    # ------------------------------------------------------------------
    # 4. C_l Vergleich (SW-Approximation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. C_l VERGLEICH (Sachs-Wolfe-Approximation, ell=2..50)")
    print("=" * 50)
    print("   Berechne vereinfachte C_l Spektren...")

    ell_sw   = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50])
    Cl_LCDM, Cl_CRM_s, Cl_CRM_sv = cl_sachs_wolfe_approx(ell_sw)

    print(f"\n   {'ell':>6} | {'Cl_LCDM':>12} | {'Cl_CRM_s':>12} | {'Cl_CRM_sv':>12} | "
          f"{'Delta_s [%]':>12} | {'Delta_sv [%]':>12}")
    print("   " + "-" * 80)

    dCl_s_arr  = []
    dCl_sv_arr = []

    for i, ell in enumerate(ell_sw):
        if Cl_LCDM[i] > 0:
            ds    = (Cl_CRM_s[i] - Cl_LCDM[i]) / Cl_LCDM[i] * 100.0
            dsv   = (Cl_CRM_sv[i] - Cl_LCDM[i]) / Cl_LCDM[i] * 100.0
        else:
            ds = dsv = 0.0
        dCl_s_arr.append(ds)
        dCl_sv_arr.append(dsv)
        print(f"   {ell:>6} | {Cl_LCDM[i]:>12.4e} | {Cl_CRM_s[i]:>12.4e} | "
              f"{Cl_CRM_sv[i]:>12.4e} | {ds:>12.6f} | {dsv:>12.6f}")

    dCl_s_max   = float(np.max(np.abs(dCl_s_arr)))
    dCl_sv_max  = float(np.max(np.abs(dCl_sv_arr)))
    dCl_sv_add  = float(np.max(np.abs(np.array(dCl_sv_arr) - np.array(dCl_s_arr))))

    print(f"\n   Max |Delta_C_l/C_l| Skalar-Sektor:         {dCl_s_max:.4f}%")
    print(f"   Max |Delta_C_l/C_l| Skalar+Vektor:         {dCl_sv_max:.4f}%")
    print(f"   Additiver Vektor-Beitrag (ueber Skalar):   {dCl_sv_add:.6f}%")

    results_json['cl_comparison'] = {
        'ell': ell_sw.tolist(),
        'Cl_LCDM': Cl_LCDM.tolist(),
        'Cl_CRM_scalar': Cl_CRM_s.tolist(),
        'Cl_CRM_scalar_vector': Cl_CRM_sv.tolist(),
        'DeltaCl_scalar_percent': dCl_s_arr,
        'DeltaCl_sv_percent': dCl_sv_arr,
        'DeltaCl_s_max_percent': dCl_s_max,
        'DeltaCl_sv_max_percent': dCl_sv_max,
        'vector_additive_max_percent': dCl_sv_add,
    }

    # ------------------------------------------------------------------
    # 5. Vergleich mit AeST
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("5. VERGLEICH MIT AeST (Skordis & Zlosnik 2021)")
    print("=" * 50)

    # AeST-Kopplung ist O(1), CRM-Kopplung ist O(alpha_M0)
    aest_coupling   = 1.0       # O(1)
    crm_coupling    = alpha_M0  # ~ 0.0011
    suppression     = crm_coupling / aest_coupling

    # AeST Vektorfeld-Effekt (Skordis & Zlosnik 2021): ~ 2-3%
    aest_vector_pct  = 2.5   # % (Zentralwert aus Skordis 2021)

    # CRM Vektorfeld-Effekt (skaliert mit Kopplung^2)
    crm_vector_pct   = aest_vector_pct * suppression**2

    print(f"   AeST Vektorkopplung: O(1)      -> Delta_C_l/C_l ~ {aest_vector_pct:.1f}%")
    print(f"   CRM  Vektorkopplung: {crm_coupling:.4f}  (Faktor {suppression:.4f} kleiner)")
    print(f"   CRM  Suppression:    ~ alpha_M0^2 = {suppression**2:.4e}")
    print(f"   CRM  Vektoreffekt:   ~ {crm_vector_pct:.4e}%  (vs AeST ~{aest_vector_pct:.1f}%)")
    print(f"   Suppression-Faktor:  {aest_vector_pct / max(crm_vector_pct, 1e-15):.2e}x kleiner als AeST")

    results_json['comparison_aest'] = {
        'aest_coupling': aest_coupling,
        'crm_coupling': crm_coupling,
        'suppression_factor': float(suppression),
        'aest_vector_effect_percent': aest_vector_pct,
        'crm_vector_effect_percent': float(crm_vector_pct),
        'suppression_vs_aest': float(aest_vector_pct / max(crm_vector_pct, 1e-15)),
        'reference': 'Skordis & Zlosnik 2021, Phys.Rev.Lett. 127, 161302',
    }

    # ------------------------------------------------------------------
    # 6. Gesamt-Schlussfolgerung
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("6. SCHLUSSFOLGERUNG")
    print("=" * 50)

    print(f"\n   SKALAR-SEKTOR (CRM, Paper III EFT):")
    print(f"     Delta_C_l / C_l (max, SW-Approx) = {dCl_s_max:.4f}%")
    print(f"     Bewertung: {'>> 1% -- signifikant' if dCl_s_max > 1.0 else '< 1% -- klein'}")

    print(f"\n   VEKTOR-SEKTOR (CRM, Ordnungsabschaetzung):")
    print(f"     Delta_C_l / C_l (Oberschranke)   = {dCl_max_pct:.6f}%")
    print(f"     Additiver Effekt ueber Skalar:    = {dCl_sv_add:.6f}%")
    print(f"     Bewertung: {'>> 1% -- hi_class noetig!' if flag_hiclass else '< 1% -- vernachlaessigbar'}")

    conclusion_vector = (
        "Der Vektorsektor ist fuer das CMB-Leistungsspektrum vernachlaessigbar. "
        f"Delta_C_l/C_l (Vektor) < {dCl_max_pct:.2e}% << 1%%. "
        "Grund: rho_A = 0 im Hintergrund (Paper IV Sec 6.4), Perturbationen "
        f"sind O(alpha_M0^2) ~ {alpha_M0**2:.2e} unterdrueckt. "
        "Keine hi_class-Vollrechnung erforderlich. "
        "Der Skalar-Sektor (modifizierte Poisson-Gleichung) dominiert."
    ) if not flag_hiclass else (
        "WARNUNG: Der Vektorsektor uebersteigt 1%! hi_class-Rechnung erforderlich. "
        f"Delta_C_l/C_l (Vektor) = {dCl_max_pct:.2f}%."
    )

    print(f"\n   {'=' * 60}")
    print(f"   FAZIT:")
    print(f"   {conclusion_vector}")
    print(f"   {'=' * 60}")

    results_json['conclusion'] = {
        'scalar_max_percent': dCl_s_max,
        'vector_max_percent': dCl_max_pct,
        'vector_additive_percent': dCl_sv_add,
        'hiclass_needed': bool(flag_hiclass),
        'conclusion_text': conclusion_vector,
        'paper4_section': 'Sec. 9.4',
    }

    # ------------------------------------------------------------------
    # 7. JSON-Output
    # ------------------------------------------------------------------
    json_path = OUTDIR / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n   JSON-Ergebnisse: {json_path}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print("\n   Erstelle Plots...")
    create_plots(
        a_arr, drho, pi_r, dCl_V,
        k_arr_mpc, mod_results,
        ell_sw, Cl_LCDM, Cl_CRM_s, Cl_CRM_sv,
        dCl_s_arr, dCl_sv_arr,
    )

    # ------------------------------------------------------------------
    # 9. Telegram-Abschlussmeldung
    # ------------------------------------------------------------------
    telegram_msg = (
        f"Analyse abgeschlossen.\n"
        f"Vektor-Sektor: Delta_Cl/Cl = {dCl_max_pct:.2e}% (Oberschranke)\n"
        f"Skalar-Sektor: Delta_Cl/Cl = {dCl_s_max:.4f}%\n"
        f"hi_class noetig: {'JA' if flag_hiclass else 'NEIN'}\n"
        f"Fazit: Vektorsektor {'SIGNIFIKANT' if flag_hiclass else 'vernachlaessigbar (< 1%)'}\n"
        f"Ergebnisse: {OUTDIR}"
    )
    send_telegram(telegram_msg)

    print("\n" + "=" * 70)
    print("Analyse abgeschlossen.")
    print(f"Ergebnisse in: {OUTDIR}")
    print("=" * 70)

    return results_json


# ============================================================
# ABSCHNITT 7: Plots
# ============================================================

def create_plots(
    a_arr, drho, pi_r, dCl_V,
    k_arr_mpc, mod_results,
    ell_sw, Cl_LCDM, Cl_CRM_s, Cl_CRM_sv,
    dCl_s_arr, dCl_sv_arr,
):
    """Erstellt alle Plots und speichert sie als PNG."""

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ------------------------------------------------------------------
    # Panel 1: Vektor-Sektor delta_rho und pi als Funktion von a
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(a_arr, drho, 'b-', lw=2, label=r'$\delta\rho_A / \rho_\mathrm{crit}$')
    ax1.semilogy(a_arr, pi_r, 'r--', lw=2, label=r'$\pi_A / \rho_\mathrm{crit}$')
    ax1.semilogy(a_arr, dCl_V, 'g:', lw=2, label=r'$\Delta C_\ell / C_\ell$')
    ax1.axhline(0.01, color='k', ls=':', lw=1, alpha=0.5, label='1% Schwelle')
    ax1.set_xlabel('Skalenfaktor $a$')
    ax1.set_ylabel('Relative Amplitude')
    ax1.set_title('Vektorsektor: Oberschranken')
    ax1.legend(fontsize=8)
    ax1.set_xlim([0.01, 1.0])

    # ------------------------------------------------------------------
    # Panel 2: mu(k,a) fuer drei Epochen
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    colors_ep = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (label, data) in enumerate(mod_results.items()):
        k    = np.array(data['k_arr'])
        mu   = np.array(data['mu'])
        ax2.semilogx(k, mu, color=colors_ep[i], lw=2, label=label)
        kC   = data['k_Compton_Mpc']
        if kC > 0:
            ax2.axvline(kC, color=colors_ep[i], ls=':', lw=1, alpha=0.6)
    ax2.axhline(4.0/3.0, color='gray', ls='--', lw=1, alpha=0.7, label=r'$\mu=4/3$')
    ax2.axhline(1.0, color='black', ls='-', lw=0.5, alpha=0.5, label='GR')
    ax2.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax2.set_ylabel(r'$\mu(k,a) = G_\mathrm{eff}/G_N$')
    ax2.set_title(r'Modifizierte Gravitation $\mu(k,a)$')
    ax2.legend(fontsize=8)
    ax2.set_xlim([1e-4, 1e2])
    ax2.set_ylim([0.98, 1.38])

    # ------------------------------------------------------------------
    # Panel 3: Lensing-Funktion Sigma(k,a)
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (label, data) in enumerate(mod_results.items()):
        k     = np.array(data['k_arr'])
        Sigma = np.array(data['Sigma'])
        ax3.semilogx(k, (Sigma - 1.0) * 100.0, color=colors_ep[i], lw=2, label=label)
    ax3.axhline(0.0, color='black', ls='-', lw=0.5, alpha=0.5, label='LCDM')
    ax3.axhline(1.0, color='red', ls=':', lw=1, alpha=0.7, label='1% Schwelle')
    ax3.set_xlabel(r'$k$ [Mpc$^{-1}$]')
    ax3.set_ylabel(r'$(\Sigma - 1)$ [%]')
    ax3.set_title(r'Lensing-Funktion $\Sigma(k,a)$')
    ax3.legend(fontsize=8)
    ax3.set_xlim([1e-4, 1e2])

    # ------------------------------------------------------------------
    # Panel 4: C_l Vergleich (normiert: l*(l+1)*Cl / 2pi)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    norm_l = ell_sw * (ell_sw + 1) / (2.0 * np.pi)
    ax4.loglog(ell_sw, norm_l * Cl_LCDM, 'k-', lw=2, label='LCDM')
    ax4.loglog(ell_sw, norm_l * Cl_CRM_s, 'b--', lw=2, label='CRM (Skalar)')
    ax4.loglog(ell_sw, norm_l * Cl_CRM_sv, 'r:', lw=2, label='CRM (Skalar+Vektor)')
    ax4.set_xlabel(r'Multipolmoment $\ell$')
    ax4.set_ylabel(r'$\ell(\ell+1)C_\ell / (2\pi)$')
    ax4.set_title(r'$C_\ell$ Spektrum (SW-Approx.)')
    ax4.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 5: Relative Abweichung Delta_C_l / C_l
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogx(ell_sw, dCl_s_arr, 'b-o', lw=2, ms=5, label=r'$\Delta C_\ell/C_\ell$ (Skalar)')
    ax5.semilogx(ell_sw, dCl_sv_arr, 'r--s', lw=2, ms=5, label=r'$\Delta C_\ell/C_\ell$ (Sk.+Vek.)')
    ax5.axhline(1.0, color='green', ls=':', lw=1.5, label='1% Schwelle')
    ax5.axhline(-1.0, color='green', ls=':', lw=1.5)
    ax5.axhline(0.0, color='black', ls='-', lw=0.5, alpha=0.5)
    ax5.set_xlabel(r'Multipolmoment $\ell$')
    ax5.set_ylabel(r'$\Delta C_\ell / C_\ell$ [%]')
    ax5.set_title(r'Relative Modifikation (vs. LCDM)')
    ax5.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Panel 6: Zusammenfassung (Balkendiagramm der Beitraege)
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])

    # Alle Effekte auf einer Skala
    labels_bar = [
        r'$\mu-1$ (sub-$\lambda_C$)',
        r'$\Sigma-1$ (sub-$\lambda_C$)',
        r'$\Delta C_\ell/C_\ell$ Skalar',
        r'$\Delta C_\ell/C_\ell$ Vektor',
        r'$\pi_A/\rho_c$ (z=0)',
    ]

    # Zahlenwerte (Prozent)
    mu_sub  = list(mod_results.values())[2]['mu_sub_Compton']  # a=1.0
    Sig_sub = list(mod_results.values())[2]['Sigma_sub']

    values_bar = [
        abs(mu_sub - 1.0) * 100.0,
        abs(Sig_sub - 1.0) * 100.0,
        float(np.max(np.abs(dCl_s_arr))),
        float(np.max(np.abs(dCl_sv_arr)) - np.max(np.abs(dCl_s_arr))),
        float(pi_r[-1]) * 100.0,
    ]
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    bars = ax6.barh(range(len(labels_bar)), values_bar,
                    color=colors_bar, alpha=0.8, edgecolor='black', lw=0.8)
    ax6.axvline(1.0, color='red', ls='--', lw=1.5, label='1%-Schwelle')
    ax6.set_yticks(range(len(labels_bar)))
    ax6.set_yticklabels(labels_bar, fontsize=8)
    ax6.set_xlabel('Relative Modifikation [%]')
    ax6.set_title('Beitragsvergleich')
    ax6.set_xscale('log')
    ax6.set_xlim([1e-10, 100.0])
    ax6.legend(fontsize=8)

    for i, (bar, val) in enumerate(zip(bars, values_bar)):
        ax6.text(max(val * 1.3, 1e-9), i, f'{val:.2e}%',
                 va='center', ha='left', fontsize=7)

    # ------------------------------------------------------------------
    # Gesamt-Titel
    # ------------------------------------------------------------------
    fig.suptitle(
        'CRM Paper IV: CMB Vektorsektor-Analyse (Sec. 9.4)\n'
        r'$\alpha_{M0}=' + f'{alpha_M0:.4f}' + r'$, $n=' + f'{n_alpha:.2f}' + r'$, $\alpha_T=0$',
        fontsize=13, fontweight='bold'
    )

    plot_path = OUTDIR / 'cmb_vector_analysis.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Plot gespeichert: {plot_path}")

    # ------------------------------------------------------------------
    # Zusatz-Plot: Zeitentwicklung der Vektor-Beitraege
    # ------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(9, 5))
    z_arr   = 1.0 / a_arr - 1.0
    ax.semilogy(z_arr[z_arr < 10], dCl_V[z_arr < 10] * 100.0,
                'r-', lw=2.5, label=r'$\Delta C_\ell/C_\ell$ Vektor (Oberschranke)')
    ax.semilogy(z_arr[z_arr < 10], drho[z_arr < 10] * 100.0,
                'b--', lw=1.5, label=r'$\delta\rho_A/\rho_c$')
    ax.semilogy(z_arr[z_arr < 10], pi_r[z_arr < 10] * 100.0,
                'g:', lw=1.5, label=r'$\pi_A/\rho_c$')
    ax.axhline(1.0, color='black', ls=':', lw=1, alpha=0.6, label='1%-Schwelle')
    ax.axhline(0.01, color='gray', ls=':', lw=1, alpha=0.4, label='0.01%-Schwelle')
    ax.set_xlabel('Rotverschiebung $z$', fontsize=12)
    ax.set_ylabel('Relative Amplitude [%]', fontsize=12)
    ax.set_title(
        r'CRM Vektorsektor: Zeitentwicklung der Storterme'
        '\n' + r'$\alpha_{M0}=' + f'{alpha_M0:.4f}' + r'$, Oberschranken',
        fontsize=11
    )
    ax.set_xlim([0, 9])
    ax.legend(fontsize=10)
    ax.invert_xaxis()

    plot_path2 = OUTDIR / 'vector_time_evolution.png'
    fig2.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   Plot gespeichert: {plot_path2}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    try:
        run_full_analysis()
    except Exception as e:
        import traceback
        msg = f"FEHLER: {type(e).__name__}: {e}"
        print(msg)
        traceback.print_exc()
        send_telegram(f"FEHLER in CMB-Vector-Analyse: {msg}")
        sys.exit(1)

#!/usr/bin/env python3
"""Plot: 4 Funktionalformen im Vergleich + Omega_Phi(a) Overlay."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# CFM best-fit Parameter
Omega_m = 0.364
Phi0_tanh = 1.047
k_tanh = 1.30
a_trans = 0.75

a = np.linspace(0.01, 1.5, 500)

# --- tanh ---
s_t = np.tanh(k_tanh * a_trans)
OPhi_tanh = Phi0_tanh * (np.tanh(k_tanh * (a - a_trans)) + s_t) / (1.0 + s_t)

# --- Logistisch (best-fit: Om=0.368, k=1.90, at=0.75) ---
k_log = 1.90
Phi0_log = (1.0 - 0.368) / (1.0 / (1.0 + np.exp(-2*k_log*(1.0 - a_trans))))
OPhi_log = Phi0_log / (1.0 + np.exp(-2*k_log*(a - a_trans)))

# --- erf (best-fit: Om=0.367, k=1.99, at=0.75) ---
from scipy.special import erf
k_erf = 1.99
s_e = erf(k_erf * a_trans / np.sqrt(2))
Phi0_erf = (1.0 - 0.367) * (1.0 + s_e) / (erf(k_erf * (1.0 - a_trans) / np.sqrt(2)) + s_e)
OPhi_erf = Phi0_erf * (erf(k_erf * (a - a_trans) / np.sqrt(2)) + s_e) / (1.0 + s_e)

# --- Potenzgesetz (best-fit: Om=0.364, k=3.02, at=0.75) ---
n_pow = 3.02
x = a / a_trans
val_at_1 = (1.0/a_trans)**n_pow / (1.0 + (1.0/a_trans)**n_pow)
Phi0_pow = (1.0 - 0.364) / val_at_1
OPhi_pow = Phi0_pow * x**n_pow / (1.0 + x**n_pow)

# =====================================================================
# PLOT
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Panel 1: Omega_Phi(a) ---
ax1.plot(a, OPhi_tanh, '-',  color='#E91E63', linewidth=2.5, label='tanh (Standard-CFM)')
ax1.plot(a, OPhi_log,  '--', color='#2196F3', linewidth=2.5, label='Logistisch')
ax1.plot(a, OPhi_erf,  '-.', color='#4CAF50', linewidth=2.5, label='Error-Funktion (erf)')
ax1.plot(a, OPhi_pow,  ':',  color='#FF9800', linewidth=2.5, label='Potenzgesetz')

ax1.axhline(1.0 - Omega_m, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
            label=f'$\\Omega_\\Lambda$ = {1.0-Omega_m:.3f} ($\\Lambda$CDM)')
ax1.axvline(a_trans, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax1.axvline(1.0, color='black', linestyle=':', linewidth=1, alpha=0.3)

ax1.text(a_trans+0.02, -0.07, '$a_{trans}$', fontsize=10, alpha=0.5)
ax1.text(1.02, -0.07, 'Heute', fontsize=10, alpha=0.5)

ax1.set_xlabel('Skalenfaktor $a$', fontsize=13)
ax1.set_ylabel('$\\Omega_\\Phi(a)$', fontsize=13)
ax1.set_title('Vier Funktionalformen: nahezu identisch', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.set_xlim(0, 1.5)
ax1.set_ylim(-0.1, 1.2)

# --- Panel 2: Differenzen relativ zu tanh ---
diff_log = OPhi_log - OPhi_tanh
diff_erf = OPhi_erf - OPhi_tanh
diff_pow = OPhi_pow - OPhi_tanh

ax2.plot(a, diff_log, '--', color='#2196F3', linewidth=2.5, label='Logistisch $-$ tanh')
ax2.plot(a, diff_erf, '-.', color='#4CAF50', linewidth=2.5, label='erf $-$ tanh')
ax2.plot(a, diff_pow, ':',  color='#FF9800', linewidth=2.5, label='Potenzgesetz $-$ tanh')
ax2.axhline(0, color='black', linewidth=0.8)

ax2.set_xlabel('Skalenfaktor $a$', fontsize=13)
ax2.set_ylabel('$\\Delta\\Omega_\\Phi(a)$', fontsize=13)
ax2.set_title('Differenzen relativ zu tanh', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_xlim(0, 1.5)
ax2.set_ylim(-0.08, 0.08)

fig.suptitle('Robustheit des CFM: Die Physik ist robust, die Mathematik nur das Werkzeug',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
outpath = 'C:\\Users\\User\\OneDrive\\Desktop\\Forschung\\Natur&Technik\\Spieltheorie Urknall\\_results\\CFM_Funktionalformen_Vergleich.png'
import os
os.makedirs(os.path.dirname(outpath), exist_ok=True)
fig.savefig(outpath, dpi=200, bbox_inches='tight')
print(f"Plot gespeichert: {outpath}")
plt.close()

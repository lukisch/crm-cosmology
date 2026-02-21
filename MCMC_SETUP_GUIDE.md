# MCMC Setup Guide -- Extended Chain Run on New System

**Zweck:** Die MCMC-Chain aus `run_full_mcmc.py` muss mit mehr Samples neu gerechnet werden (Reviewer-Anforderung: mindestens 48 Walkers, 5000 Production Steps, Konvergenz-Diagnostik).

**Projektordner:** Dieses Projekt liegt in OneDrive unter:
```
C:\Users\User\OneDrive\Forschung\Natur&Technik\Spieltheorie Urknall\
```
Alle relevanten Dateien synchronisieren sich automatisch.

---

## 1. Voraussetzungen

### 1.1 WSL mit Ubuntu einrichten (falls nicht vorhanden)

```bash
# In Windows PowerShell (als Admin):
wsl --install -d Ubuntu-24.04

# Nach Installation, in WSL:
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gfortran cmake python3-pip python3-dev \
    libgsl-dev liblapack-dev wget git cython3
```

### 1.2 hi_class installieren

```bash
cd /home
sudo git clone https://github.com/miguelzuma/hi_class_public.git hi_class
cd /home/hi_class

# Python-Wrapper bauen:
pip3 install cython numpy scipy
make clean
make
cd python
python3 setup.py build
```

Der Python-Wrapper wird gebaut unter:
```
/home/hi_class/python/build/lib.linux-x86_64-cpython-3XX/
```
(XX = Python-Version, z.B. 312 fuer 3.12)

### 1.3 cfm_fR Gravity Model patchen

Das CFM-Modell muss in hi_class integriert werden BEVOR der Python-Wrapper gebaut wird:

```bash
cd /home/hi_class

# Pfad zum Projekt (OneDrive in WSL mounten):
# OneDrive ist typischerweise unter /mnt/c/Users/User/OneDrive/
PROJ="/mnt/c/Users/User/OneDrive/Forschung/Natur&Technik/Spieltheorie Urknall"

python3 "$PROJ/scripts/patch_cfm.py"

# WICHTIG: Nach dem Patchen hi_class NEU kompilieren:
make clean
make
cd python
python3 setup.py build
cd /home/hi_class
```

### 1.4 Verifizieren dass cfm_fR funktioniert

```bash
PROJ="/mnt/c/Users/User/OneDrive/Forschung/Natur&Technik/Spieltheorie Urknall"
python3 "$PROJ/scripts/test_cfm_fR_native.py"
```

Erwartete Ausgabe: Erfolgreiches C_l-Spectrum fuer LCDM und cfm_fR Modelle.

### 1.5 Python-Pakete

```bash
pip3 install emcee numpy scipy matplotlib
```

---

## 2. MCMC Script anpassen

Das bestehende Script `scripts/run_full_mcmc.py` muss fuer den erweiterten Lauf angepasst werden:

### 2.1 Pfad zum hi_class Python-Wrapper

Zeile 12 in `run_full_mcmc.py`:
```python
sys.path.insert(0, '/home/hi_class/python/build/lib.linux-x86_64-cpython-312')
```
**Anpassen an die tatsaechliche Python-Version** auf dem neuen System (z.B. cpython-310, cpython-311).

### 2.2 Parameter fuer den erweiterten Lauf

Folgende Einstellungen aendern:

```python
# Zeile 156: Mehr Walkers
nwalkers = 48  # war: 24, jetzt: 48 (ca. 10x ndim)

# Zeile 200: Laengerer Burn-in
state = sampler.run_mcmc(p0, 200, progress=False)  # war: 80, jetzt: 200

# Zeile 222: Mehr Production Steps
n_production = 5000  # war: 400, jetzt: 5000
chunk_size = 100     # war: 50, jetzt: 100
```

Dies ergibt: 48 Walkers x 5000 Steps = **240.000 Samples** (vs. vorher 9.600).

### 2.3 Geschaetzte Laufzeit

- Jede Likelihood-Evaluation: ~2-4 Sekunden (abhaengig von CPU)
- Burn-in: 48 x 200 = 9.600 Evaluationen ~ 6-10 Stunden
- Production: 48 x 5000 = 240.000 Evaluationen ~ 130-270 Stunden (5-11 Tage)

**Alternative: Parallelisierung mit multiprocessing:**

```python
import multiprocessing
# VOR dem Sampler:
with multiprocessing.Pool(processes=8) as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
    # ... rest wie gehabt
```

Mit 8 Kernen: ~16-33 Stunden (machbar ueber Nacht/Wochenende).

**ACHTUNG:** Die `Class()` Instanzen in `log_likelihood` muessen thread-safe sein. emcee + multiprocessing.Pool funktioniert out-of-the-box, da jeder Worker einen eigenen Process hat.

---

## 3. Konvergenz-Diagnostik hinzufuegen

**WICHTIG:** Der Reviewer verlangt explizit Gelman-Rubin R-hat, Effective Sample Size (ESS), und Autocorrelation Length. Fuege am Ende des Scripts (nach der Production-Loop, vor den Results) hinzu:

```python
# ================================================================
# 5b. CONVERGENCE DIAGNOSTICS
# ================================================================
print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS")
print("="*80)

# Autocorrelation time (emcee built-in)
try:
    tau = sampler.get_autocorr_time(quiet=True)
    print(f"\nAutocorrelation times:")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:>12s}: tau = {tau[i]:.1f}")
    print(f"\nMean autocorrelation time: {np.mean(tau):.1f}")
    print(f"Chain length / tau: {n_production / np.mean(tau):.1f} (should be >> 50)")
except:
    print("WARNING: Autocorrelation time estimation failed (chain too short?)")

# Effective Sample Size
chain_3d = sampler.get_chain()  # shape: (n_steps, n_walkers, n_dim)
n_steps_actual = chain_3d.shape[0]
try:
    ess = n_steps_actual * nwalkers / np.mean(tau)
    print(f"Effective Sample Size (ESS): {ess:.0f}")
except:
    print("ESS: could not compute (tau unavailable)")

# Gelman-Rubin R-hat (split each walker chain in half)
print(f"\nGelman-Rubin R-hat (split-chain):")
for i, name in enumerate(PARAM_NAMES):
    # Split each walker's chain into first/second half
    half = n_steps_actual // 2
    chains_split = []
    for w in range(nwalkers):
        chains_split.append(chain_3d[:half, w, i])
        chains_split.append(chain_3d[half:, w, i])
    chains_split = np.array(chains_split)

    # R-hat computation
    M = len(chains_split)
    N = len(chains_split[0])
    chain_means = np.mean(chains_split, axis=1)
    grand_mean = np.mean(chain_means)
    B = N / (M - 1) * np.sum((chain_means - grand_mean)**2)
    W = np.mean(np.var(chains_split, axis=1, ddof=1))
    var_hat = (N - 1) / N * W + B / N
    R_hat = np.sqrt(var_hat / W) if W > 0 else float('inf')
    status = "OK" if R_hat < 1.01 else "WARNING"
    print(f"  {name:>12s}: R-hat = {R_hat:.4f}  [{status}]")
```

---

## 4. Zwischenspeicherung (Checkpointing)

Um die Chain nicht zu verlieren (wie beim letzten Mal!), fuege Checkpointing in die Production-Loop ein:

```python
# In der Production-Loop (ersetze den bestehenden for-Loop):
n_production = 5000
chunk_size = 100
checkpoint_interval = 500  # Alle 500 Steps speichern

for chunk in range(n_production // chunk_size):
    state = sampler.run_mcmc(state, chunk_size, progress=False)
    step = (chunk + 1) * chunk_size
    elapsed = (time.time() - T_START) / 60
    accept = np.mean(sampler.acceptance_fraction)
    best_ll = np.max(sampler.get_log_prob(flat=True))
    print(f"  [{time.strftime('%H:%M:%S')}] Step {step}/{n_production}, "
          f"accept={accept:.3f}, best_logL={best_ll:.1f}, "
          f"evals={N_EVAL}, time={elapsed:.1f}min")
    sys.stdout.flush()

    # Checkpoint
    if step % checkpoint_interval == 0:
        _ckpt = dict(
            chain=sampler.get_chain(flat=True),
            log_prob=sampler.get_log_prob(flat=True),
            param_names=PARAM_NAMES,
            step=step,
            acceptance=np.mean(sampler.acceptance_fraction),
        )
        np.savez(_persistent_path.replace('.npz', f'_checkpoint_{step}.npz'), **_ckpt)
        np.savez(_persistent_path, **_ckpt)  # Auch Haupt-File updaten
        print(f"    -> Checkpoint saved ({step} steps)")
        sys.stdout.flush()
```

---

## 5. Speicherorte

**KRITISCH:** Die originale Chain ging verloren weil sie nur in `/tmp/` lag. Speichere IMMER an beiden Orten:

```python
# Persistent (OneDrive-synchronisiert):
_persistent_path = os.path.join(_project_dir, 'results', 'cfm_fR_mcmc_chain.npz')

# Lokal (schneller Zugriff, Backup):
_local_path = '/tmp/cfm_fR_mcmc_results.npz'

# Beide speichern:
np.savez(_persistent_path, **_save_args)
np.savez(_local_path, **_save_args)
```

Der `results/`-Ordner liegt unter:
```
C:\Users\User\OneDrive\Forschung\Natur&Technik\Spieltheorie Urknall\results\
```
In WSL:
```
/mnt/c/Users/User/OneDrive/Forschung/Natur&Technik/Spieltheorie Urknall/results/
```

---

## 6. Ausfuehrung

```bash
# In WSL:
PROJ="/mnt/c/Users/User/OneDrive/Forschung/Natur&Technik/Spieltheorie Urknall"

# Empfohlen: in screen/tmux laufen lassen (ueberlebt Terminal-Schliessung):
tmux new -s mcmc

cd /home/hi_class
python3 "$PROJ/scripts/run_full_mcmc.py" 2>&1 | tee "$PROJ/results/mcmc_run_log.txt"
```

---

## 7. Nach dem Lauf

### 7.1 Ergebnisse pruefen

```bash
python3 "$PROJ/scripts/analyze_mcmc_results.py"
```

### 7.2 Corner Plot generieren

```bash
python3 "$PROJ/scripts/generate_corner_plot.py"
```

**HINWEIS:** `generate_corner_plot.py` nutzt aktuell hardcodierte Summary-Statistiken. Nach dem neuen MCMC-Lauf muss es auf die echte Chain umgestellt werden. Der einfachste Weg: `analyze_mcmc_results.py` laedt die Chain automatisch aus `results/cfm_fR_mcmc_chain.npz`.

### 7.3 Ergebnisse in Paper eintragen

Die Schluesselwerte die aktualisiert werden muessen (falls sie sich aendern):
- `alpha_M_0` Best-fit und 68% CI (aktuell: 0.0013 +/- 0.0007)
- `n_exp` Best-fit und 68% CI (aktuell: 0.67 +/- 0.42)
- `Delta_chi2` (aktuell: -3.6)
- Detection significance (aktuell: 1.78 sigma)
- Korrelationsmatrix (besonders rho(alpha_M_0, n))

Diese Werte stehen in:
- `papers/Paper1_EN.tex` (Sections 6, 7)
- `papers/Paper1_DE.tex` (analog)
- `papers/Paper3_EN.tex` (Honest Assessment, Zeile ~872)
- `reviews/RESPONSE_TO_REVIEWER.md`

---

## 8. Erwartete Ergebnisse

Basierend auf dem vorherigen (kuerzeren) Lauf:
- **Best-fit chi2:** ~6625.2 (Delta_chi2 = -3.6 vs. LCDM's 6628.8)
- **alpha_M_0:** 0.0013 +/- 0.0007 (1.78 sigma detection)
- **n_exp:** 0.67 +/- 0.42
- **omega_cdm:** 0.11994 +/- 0.00029
- **logAs:** 3.044 +/- 0.002
- **n_s:** 0.9656 +/- 0.0024
- **Acceptance fraction:** ~0.45-0.50
- **Key correlation:** rho(alpha_M_0, n) = -0.633

Die laengere Chain sollte diese Werte bestaetigen oder leicht verschieben, aber die Konvergenz-Diagnostik (R-hat < 1.01, ESS > 1000) ist das primaere Ziel.

---

## 9. Troubleshooting

### "classy module not found"
-> Pfad in Zeile 12 pruefen. `ls /home/hi_class/python/build/` zeigt den korrekten Ordnernamen.

### "gravity_model cfm_fR not recognized"
-> `patch_cfm.py` wurde nicht ausgefuehrt oder hi_class wurde danach nicht neu kompiliert. Nochmal: patch -> make clean -> make -> python setup.py build.

### "Segfault bei cosmo.compute()"
-> Typisch fuer extreme Parameter-Kombinationen. Die Priors sind so gesetzt dass dies selten passiert. Die try/except-Bloecke in `log_likelihood` fangen dies ab.

### Chain-Datei ist 0 bytes
-> OneDrive-Sync-Konflikt. Pruefe ob `.npz.tmp` Dateien existieren. Speichere zusaetzlich lokal unter `/tmp/`.

---

*Erstellt am 15. Februar 2026. Bei Fragen: Die vollstaendige Projektdokumentation liegt in `reviews/RESPONSE_TO_REVIEWER.md`.*

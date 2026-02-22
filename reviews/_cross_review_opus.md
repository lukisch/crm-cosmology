# CFM 4-Paper Cross-Review (Opus)

**Datum:** 2026-02-21
**Reviewer-Rolle:** Strenger Physik-Referee (simuliert)
**Gepruefte Papers:**
- Paper I = `Paper1_EN.tex` (Game Theory, Nash, Pantheon+)
- Paper II = `Paper2_EN.tex` (MOND, Baryon-only, Joint-Fit)
- Paper III = `Paper3_EN.tex` (Lagrangian, f(R), QG, hi_class)
- Paper IV = `Paper4_EN.tex` (Vector Sector, galaktische Dynamik, DRAFT)

**Bewertungsskala:** CRITICAL / MAJOR / MINOR / SUGGESTION

---

## 1. Numerische Konsistenz zwischen den Papers

### 1.1 CRITICAL: Paper III Intro zitiert falsche Paper-II-Werte

**Datei:** `Paper3_EN.tex`, Zeile 61
**Problem:** Die Einleitung von Paper III fasst Paper II wie folgt zusammen:

> "achieving joint SN + CMB + BAO compatibility ($\Delta\chi^2 = -5.5$ vs. LCDM), with $H_0 = 67.3$ km/s/Mpc..."

Weiter heisst es aber auch:

> "...beta approximately 2.7..." und "$H_0 = 66$ km/s/Mpc" und "$\Delta\chi^2 = -5.7$"

**Tatsaechliche Werte in Paper II** (Table 7 / bevorzugte Variante, Zeile 487--507):
- $\Delta\chi^2 = -5.5$ (NICHT -5.7)
- $H_0 = 67.3$ km/s/Mpc (NICHT 66)
- $\beta_{\text{early}} = 2.82$ (NICHT "approximately 2.7")

**Bewertung:** CRITICAL -- Innerhalb derselben Zeile 61 werden sowohl korrekte als auch inkorrekte Werte genannt. Das Abstract ist korrekt ($\Delta\chi^2 = -5.5$, $H_0 = 67.3$), aber der Body-Text in Zeile 61 mischt zwei Varianten: Die $-5.7$ und $H_0 = 66$ scheinen von einer frueheren Variante mit konstantem $\mu$ und EDE zu stammen, nicht von der bevorzugten $\mu(a)$-Variante.

**Korrektur:** Zeile 61 konsistent mit der bevorzugten Variante aus Paper II, Table 7, ausfuehren: $\Delta\chi^2 = -5.5$, $H_0 = 67.3$, $\beta_{\text{early}} \approx 2.82$.

---

### 1.2 CRITICAL: Paper III Summary zitiert falsche Paper-II-Werte

**Datei:** `Paper3_EN.tex`, Zeile 808
**Problem:** Die Summary von Paper III fasst Paper II zusammen als:

> "Running curvature coupling beta_eff(a). Validated against Pantheon+ + Planck CMB + 9 BAO measurements jointly ($\Delta\chi^2 = -5.1$ vs. LCDM; $\ell_A = 301.477$, R = 1.7502)."

**Tatsaechliche Werte in Paper II:**
- $\Delta\chi^2 = -5.5$ (NICHT -5.1)
- $\ell_A = 301.471$ (NICHT 301.477)

**Bewertung:** CRITICAL -- Die Summary verwendet eine dritte, nochmals abweichende Zahlenkombination, die in keiner Tabelle von Paper II vorkommt. Die $\ell_A = 301.477$ erscheint in Paper II Zeile 680 im Discussion-Abschnitt zu CMB Challenge 1, wo sie sich auf eine Background-Level-Rechnung bezieht, aber nicht in der Hauptergebnistabelle (dort steht 301.471).

**Korrektur:** Konsistent $\Delta\chi^2 = -5.5$ und $\ell_A = 301.471$ verwenden.

---

### 1.3 MAJOR: beta_early Inkonsistenz ueber alle Papers

**Problem:** Der Wert von $\beta_{\text{early}}$ variiert innerhalb und zwischen den Papers:

| Stelle | Wert |
|--------|------|
| Paper II, Table 4 (Zeile 452) | 2.823 |
| Paper II, Zeile 523 (bevorzugte Variante) | 2.82 |
| Paper II, Zeile 703 (optimierter Wert) | 2.834 |
| Paper II, Zeile 712 (combined optimal) | 2.834 |
| Paper III, Zeile 61 (Intro) | "approx 2.7" |
| Paper III, Zeile 315 (Sec. 4 best-fit) | 2.78 |
| Paper III, Zeile 507 (Sec. 7, Relation to f(R)) | 2.823 und 2.834 |

**Bewertung:** MAJOR -- Es gibt mindestens 5 verschiedene Werte fuer $\beta_{\text{early}}$ in der Serie. Die Abweichungen in Paper II (2.82, 2.823, 2.834) sind zwar inhaltlich erklaerbar (verschiedene Optimierungsstufen), aber verwirrend. In Paper III wird "approx 2.7" verwendet -- eine Abweichung von ~4% gegenueber dem tatsaechlichen Wert 2.82, die in einer Praezisionswissenschaft nicht akzeptabel ist.

**Korrektur:** Einen kanonischen Wert fuer die bevorzugte Variante definieren (z.B. $\beta_{\text{early}} = 2.82$) und konsistent verwenden. Wo andere Werte aus frueheren Optimierungsstufen stammen, dies explizit kennzeichnen.

---

### 1.4 MAJOR: Paper IV Abstract zitiert nicht-bevorzugte Variante

**Datei:** `Paper4_EN.tex`, Zeile 37 (Abstract)
**Problem:** Das Abstract von Paper IV behauptet:

> "Papers I-III of the Curvature Feedback Model (CFM) series demonstrated that a single scalar degree of freedom [...] reproduces the cosmological dark sector ($\Delta\chi^2 = -6.1$ vs. LCDM on Pantheon+, CMB peak positions within 0.1%, alpha_T = 0 exactly)."

**Analyse:** Der Wert $\Delta\chi^2 = -6.1$ stammt aus Paper II, Zeile 520/865, wo er sich auf die constant-$\mu = \sqrt{\pi}$ Variante MIT EDE bezieht. Die bevorzugte Variante (scale-dependent $\mu(a)$, OHNE EDE) hat $\Delta\chi^2 = -5.5$.

**Bewertung:** MAJOR -- Das Abstract von Paper IV verwendet den guenstigeren, aber nicht-bevorzugten Wert. Ein Reviewer wuerde dies als "cherry-picking" auffassen.

**Korrektur:** Entweder den bevorzugten Wert $\Delta\chi^2 = -5.5$ (zero EDE) verwenden oder beide Varianten klar benennen: "$\Delta\chi^2 = -5.5$ (preferred zero-EDE variant) to $-6.1$ (constant-$\mu$ with EDE)".

---

### 1.5 MINOR: Leichte Inkonsistenz bei r_d

**Stellen:**
- Paper II, Table 7 (Zeile 494): $r_d = 146.9$ Mpc
- Paper III, Table theta_s (Zeile 766): $r_s = 147.10$ Mpc

**Bewertung:** MINOR -- Die Diskrepanz (146.9 vs. 147.1) ist gering und erklaerbar: Paper II berechnet $r_d$ aus dem phenomenologischen Background-Fit, Paper III aus hi_class. Dennoch sollte der Unterschied explizit erwaehnt werden.

---

### 1.6 MINOR: chi^2_total und chi^2_SN Diskrepanz

**Paper II** (bevorzugte Variante): $\chi^2_{\text{SN}} = 704.8$, $\chi^2_{\text{total}} = 704.8$ (mit $\chi^2_{\text{CMB}} = 0.0$, $\chi^2_{\text{BAO}} = 0.0$).

**Paper I**: $\chi^2_{\text{SN}} = 716.8$ (flat CFM, diagonal).

Die SN-only $\chi^2$ in Paper II (704.8) ist niedriger als in Paper I (716.8), obwohl Paper II nur $\Omega_b = 0.05$ hat. Das liegt am zusaetzlichen $\alpha \cdot a^{-\beta}$-Term. Zwar wird dies erklaert, aber der direkte Vergleich ist fuer Reviewer verwirrend.

**Bewertung:** MINOR -- Klar dokumentieren, dass die SN-$\chi^2$-Werte zwischen Paper I (diagonal-errors, Standard-CFM) und Paper II (extended CFM) nicht direkt vergleichbar sind.

---

## 2. Cross-Referenzen zwischen Papers

### 2.1 MAJOR: Paper I Forward-Referenz auf Paper III Lagrangian

**Datei:** `Paper1_EN.tex`, Zeile 659
**Problem:** Paper I sagt:

> "the following chain of deductions leads to the CFM action (Paper III, Eq. (8)):"

und listet dann den konkreten Lagrangian $R/(16\pi G) + \gamma R^2 + (\partial\phi)^2/2 - V_{\text{PT}}(\phi)$ auf (Zeile 669).

**Bewertung:** MAJOR -- Ein Paper, das als "Paper I" in der Serie positioniert ist, sollte nicht detailliert auf Ergebnisse verweisen, die erst in "Paper III" abgeleitet werden. Es handelt sich hier um eine Vorwegnahme des Lagrangians, der das Kernresultat von Paper III darstellt. Ein Reviewer koennte fragen: "Wenn der Lagrangian schon in Paper I steht, was ist der Beitrag von Paper III?"

**Korrektur:** In Paper I die Forward-Referenz abschwaechen: Statt den kompletten Lagrangian zu praesentieren, nur die Deduktionskette skizzieren und auf Paper III fuer die formale Ableitung verweisen.

---

### 2.2 MINOR: Asymmetrische Zitierung zwischen Paper III und Paper IV

Paper III zitiert Paper IV nicht (was korrekt ist, wenn Paper IV ein spaeterer DRAFT ist). Paper IV zitiert Paper III korrekt als [Geiger2026c]. Die Zitierung ist konsistent.

---

### 2.3 MINOR: Paper-Nummerierung und Beschreibung

- Paper I: "Paper I in the CFM series" (Zeile 50) -- korrekt
- Paper II: "Paper II in the CFM series -- Companion to [Geiger2026]" (Zeile 54) -- korrekt
- Paper III: "Paper III in the CFM series" (Zeile 48) -- korrekt
- Paper IV: "Paper IV in the CFM series" (Zeile 46) -- korrekt

Alle Companion-Papers werden korrekt cross-referenziert.

---

## 3. Physikalische Korrektheit

### 3.1 MAJOR: Dimensionsanalyse bei $a_0 = cH_0/(2\pi)$ Ableitung

**Datei:** `Paper4_EN.tex`, Zeile 346--371
**Problem:** Die Ableitung von $a_0 = cH_0/(2\pi)$ in Section 4.6 ist heuristisch. Die Aussage "The factor $2\pi$ arises from the Fourier relationship between the time-domain scalar dynamics $\dot{\phi}(t)$ and the spatial-domain gravitational potential $\Phi(r)$" (Zeile 365) wird nicht rigoros bewiesen. Die Dimensionsanalyse funktioniert ($[cH_0] = \text{m/s}^2$, korrekt), aber der Faktor $2\pi$ ist nicht aus den Feldgleichungen abgeleitet.

**Bewertung:** MAJOR -- Paper IV erkennt dies selbst an (Zeile 371: "A rigorous derivation from the field equations would require solving the coupled scalar-vector system"), aber im Abstract und der Conclusion wird $a_0 = cH_0/(2\pi)$ als "prediction" und nicht als "conjecture" praesentiert.

**Korrektur:** Im Abstract und der Conclusion das Wort "prediction" durch "conjecture" oder "order-of-magnitude prediction" ersetzen, bis die rigorose Ableitung vorliegt.

---

### 3.2 MAJOR: Deep-MOND-Skalierung nicht erreicht

**Datei:** `Paper4_EN.tex`, Zeile 700--722
**Problem:** Die linearisierte statische sphaerische Reduktion ergibt $g_A \propto g_N$ (konstante Verstaerkung), NICHT die MOND-Skalierung $g \sim \sqrt{g_N \cdot a_0}$. Paper IV identifiziert korrekt, dass die $\sqrt{\phantom{x}}$-Skalierung aus dem nichtlinearen Chameleon-Feedback entstehen muss (Sec. 7.6), aber dies ist nicht bewiesen.

**Bewertung:** MAJOR -- Die fundamentale Behauptung von Paper IV -- dass der Vektor-Sektor MOND-aehnliche Dynamik produziert -- ist auf der linearisierten Ebene NICHT gezeigt. Der Verweis auf "future numerical work" fuer den entscheidenden Beweis schwaecht das Paper erheblich. Ein Reviewer wuerde sagen: "Show me the interpolation function from your field equations."

**Korrektur:** Die Formulierungen in Abstract, Introduction und Conclusion abschwaechen: Statt "reproduces the RAR and flat rotation curves" sollte es heissen "is consistent with the RAR in the asymptotic limits, with the full interpolation function to be determined numerically."

---

### 3.3 MINOR: $15\%$-Diskrepanz bei $a_0$

**Datei:** `Paper4_EN.tex`, Zeile 366--369
**Problem:** $a_0^{\text{CFM}} = 1.04 \times 10^{-10}$ m/s$^2$ vs. $a_0^{\text{obs}} = 1.20 \times 10^{-10}$ m/s$^2$ -- eine $\sim 15\%$ Diskrepanz. Paper IV erklaert dies als "within the uncertainty introduced by the $\mathcal{O}(1)$ saturation factor $\mathcal{B}_0$".

**Bewertung:** MINOR -- Die $15\%$-Diskrepanz ist fuer eine "prediction" unangenehm gross, aber fuer eine Order-of-Magnitude-Rechnung akzeptabel. Die ehrliche Darstellung ist positiv.

---

### 3.4 MINOR: Phantom-Regime-Argumentation in Paper I

**Datei:** `Paper1_EN.tex`, Zeile 408--419
**Problem:** Die Argumentation, dass $w < -1$ im CFM unproblematisch sei, weil $\Omega_\Phi$ "not a physical field but a geometric property of spacetime" sei, ist konzeptionell richtig, aber ein strenger Reviewer koennte einwenden: In Paper III WIRD $\Omega_\Phi$ durch ein physikalisches Skalarfeld mit Poeschl-Teller-Potential realisiert. Dann gelten die Energiebedingungen und $w < -1$ muss sorgfaeltiger behandelt werden.

**Bewertung:** MINOR -- Die Papers I und III sind hier in leichter Spannung: Paper I interpretiert $\Omega_\Phi$ als rein geometrisch (keine Energiebedingungen), Paper III gibt einen Lagrangian mit explizitem Skalarfeld (Energiebedingungen relevant). Die Ghost-Freiheits-Analyse in Paper III (Sec. 2.4) adressiert dies, aber ein expliziter Verweis von Paper I auf Paper III waere hilfreich.

---

### 3.5 SUGGESTION: Saturation ODE und Poeschl-Teller-Potential

Die Verbindung zwischen der Saturation-ODE $d\Omega_\Phi/da = k[1 - (\Omega_\Phi/\Phi_0)^2]$ und dem Poeschl-Teller-Potential $V = V_0/\cosh^2(\phi/\phi_0)$ wird in Paper III (Sec. 2.2) korrekt dargestellt. Die mathematische Ableitung (Klein-Gordon -> tanh-Loesung -> Saturation) ist nachvollziehbar, allerdings fehlt ein expliziter Beweis, dass die Slow-Roll-Naeherung ($\ddot{\phi} \ll 3H\dot{\phi}$) fuer die gesamte relevante Zeitspanne gueltig bleibt.

---

## 4. Notation und Terminologie

### 4.1 MAJOR: $\mu$-Notation hat drei verschiedene Bedeutungen

Ueber die Paper-Serie hinweg bezeichnet $\mu$ mindestens drei verschiedene Groessen:

1. **Paper II:** $\mu_{\text{eff}} \approx \sqrt{\pi}$ -- Background-Level MOND-Enhancement
2. **Paper III:** $\mu(k,a) = 1 + \frac{1}{3}\frac{k^2}{k^2 + a^2 m^2}$ -- Perturbation-Level modifizierte Poisson-Gleichung
3. **Paper IV:** $\mu(x) = g_{\text{bar}}/g_{\text{obs}}$ -- Galaktische MOND-Interpolationsfunktion

Paper III adressiert dies in einer Fussnote (Zeile 396/418) und Paper II in Zeile 122, aber die Unterscheidung koennte klarer sein.

**Bewertung:** MAJOR -- Ein Reviewer, der alle Papers liest, wird durch die Dreifach-Belegung verwirrt. Die Fussnoten in Papers II und III sind hilfreich, aber ein konsistentes Notationssystem waere besser.

**Korrektur:** Empfehlung: $\mu_{\text{bg}}(a)$ fuer Background-Level, $\mu_{\text{pert}}(k,a)$ fuer Perturbation-Level, $\mu_{\text{gal}}(x)$ fuer galaktische Interpolation.

---

### 4.2 MINOR: $\Phi$ und $\Phi_0$ Doppelbelegung

- $\Phi$ = Newtonsches Gravitationspotential (in allen Papers bei Perturbationstheorie)
- $\Phi_0$ = Amplitude der Curvature-Return-Potential (in allen Papers)
- $\Phi$ = Game-theoretisches Potential (Paper I, Eq. 1)

Die Kontextabhaengige Verwendung ist in der Physik ueblich, aber die Dreifachbelegung erfordert Aufmerksamkeit.

**Bewertung:** MINOR

---

### 4.3 MINOR: "Curvature Feedback Model" vs. "cfm_fR"

Paper III verwendet intern die Bezeichnung "cfm_fR" fuer die hi_class-Implementation. Dies ist kein Problem fuer den Code, sollte aber im Text konsistent als "the native CFM gravity model" o.ae. bezeichnet werden.

---

## 5. Argumentstruktur (Bottom-Up-Kohaerenz)

### 5.1 MAJOR: Deduktive Kette vs. tatsaechliche Paper-Reihenfolge

Die behauptete deduktive Kette ist:
1. Axiome (Game Theory) -> 2. Saturation ODE -> 3. tanh-Potential -> 4. $R^2$ Lagrangian -> 5. Perturbationsgleichungen -> 6. Vorhersagen

**Problem:** In der tatsaechlichen Paper-Reihenfolge wird der Lagrangian bereits in Paper I (Zeile 659-669) praesentiert, BEVOR er in Paper III formal abgeleitet wird. Die "deduktive Struktur" ist also teilweise retroaktiv in Paper I eingefuegt worden.

**Bewertung:** MAJOR -- Die Darstellung suggeriert eine lineare Deduktion, die in Wirklichkeit iterativ war. Ein Reviewer koennte einwenden: "The deductive chain in Paper I Section 8.4 presupposes results from Paper III."

**Korrektur:** In Paper I Sec. 8.4 explizit sagen: "The following deductive chain is presented here in retrospect; its rigorous derivation is the subject of Paper III."

---

### 5.2 SUGGESTION: Paper IV als DRAFT kenntlich machen

Paper IV hat den Status DRAFT, aber der Tex-Code enthaelt keinen solchen Hinweis. Es waere sinnvoll, im Titel oder einer Fussnote "DRAFT" oder "Working Paper" zu vermerken, bis die offenen Fragen (numerische BVP-Loesung, rigorose $2\pi$-Ableitung) geklaert sind.

---

## 6. Overclaims und Ton

### 6.1 MAJOR: "No Lambda, no dark matter particles -- only geometry" (Paper IV)

**Datei:** `Paper4_EN.tex`, Zeile 37 und 959
**Problem:** Diese Behauptung im Abstract und in der Conclusion ist zu stark. Paper IV zeigt:
- $\alpha_T = 0$: bewiesen
- $\rho_A = 0$ auf FLRW: bewiesen
- Parasitisches Screening: bewiesen
- Flache Rotationskurven: NICHT bewiesen (nur linearisierter Limes, $g_A \propto g_N$)
- RAR: NICHT bewiesen (auf numerische Arbeit verwiesen)
- SPARC-Test: vorgeschlagen, NICHT durchgefuehrt

**Bewertung:** MAJOR -- Die Formulierung "only geometry" suggeriert, dass das Problem vollstaendig geloest sei. In Wirklichkeit sind die zentralen galaktischen Vorhersagen noch nicht bestaetigt.

---

### 6.2 MINOR: "dramatically outperforming both the standard CFM and LCDM"

**Datei:** `Paper1_EN.tex`, Zeile 691
**Problem:** In der Outlook-Sektion von Paper I wird ueber "preliminary analysis" gesprochen, die "$\Delta\chi^2 = -26.3$" ergibt. Das Wort "dramatically" ist fuer eine vorlaeufige Analyse unangemessen.

**Bewertung:** MINOR -- Abschwaechen zu "substantially outperforming".

---

### 6.3 MINOR: "exact Planck match" Formulierung

**Datei:** `Paper2_EN.tex`, Zeile 705, 714, 716
**Problem:** Die Formulierung "exact Planck match" wird mehrfach verwendet fuer $\ell_1 = 220$ und $\mathcal{P}_3/\mathcal{P}_1 = 0.4295$. Planck misst $\ell_1 = 220.0 \pm 0.5$ -- das CFM-Ergebnis ist also innerhalb des Fehlers, aber nicht "exakt" im physikalischen Sinne.

**Bewertung:** MINOR -- Besser: "within Planck uncertainties" oder "matches the Planck measurement".

---

### 6.4 SUGGESTION: Ton der Self-Assessment-Sektion

Paper II hat eine ausfuehrliche "Critical Self-Assessment: Too Good to Be True?" Sektion (Zeile 807-826). Das ist wissenschaftlich vorbildlich und sollte beibehalten werden.

---

## 7. Bibliographie und Referenzen

### 7.1 MINOR: DESI 2024 vs. DESI 2025

- Paper I zitiert `DESI2024` (arXiv:2404.03002) fuer "DESI 2024 VI"
- Paper I hat auch `DESI2025` (arXiv:2503.14738) fuer "DESI DR2 Results II"
- Paper III zitiert nur `DESI2025` (arXiv:2503.14738)

Die DESI-Zitierungen sind inhaltlich korrekt, aber die Benennung (2024 vs. 2025) koennte verwirrend sein.

---

### 7.2 MINOR: Planck2020 vs. PlanckMG2016

Papers II und III verwenden zwei verschiedene Planck-Referenzen:
- `Planck2020` fuer die kosmologischen Parameter
- `PlanckMG2016` fuer die Modified-Gravity-Constraints

Das ist korrekt, da es sich um unterschiedliche Publikationen handelt.

---

### 7.3 MINOR: Fehlender bibitem Khoury2004 in Paper III

**Datei:** `Paper3_EN.tex`, Zeile 223
**Problem:** Paper III referenziert `\cite{Khoury2004}` fuer den Chameleon-Mechanismus, aber `Khoury2004` ist nicht in der Bibliographie von Paper III enthalten.

**Bewertung:** MINOR -- Fehlende Referenz.

**Korrektur:** Hinzufuegen:
```
\bibitem{Khoury2004}
Khoury, J. \& Weltman, A. (2004).
Chameleon fields: Awaiting surprises for tests of gravity in space.
Physical Review Letters, 93(17), 171104.
DOI: 10.1103/PhysRevLett.93.171104.
```

---

### 7.4 MINOR: Fehlender bibitem Williams2004 in Paper III

**Datei:** `Paper3_EN.tex`, Zeile 231
**Problem:** Paper III zitiert `\cite{Williams2004}` fuer Lunar Laser Ranging, aber der bibitem fehlt.

**Korrektur:** Hinzufuegen.

---

### 7.5 MINOR: Fehlender bibitem Bertotti2003 in Paper III

**Datei:** `Paper3_EN.tex`, Zeile 231
**Problem:** `\cite{Bertotti2003}` wird fuer die Cassini-Messung zitiert, fehlt aber in der Bibliographie von Paper III. (In Paper IV ist er vorhanden.)

---

### 7.6 SUGGESTION: Foreman-Mackey2013 vs. ForemanMackey2013

Die bibitem-Keys variieren leicht zwischen den Papers (z.B. `Foreman-Mackey2013` in Paper III Zeile 648 vs. `ForemanMackey2013` im bibitem). Da jedes Paper eigenstaendig kompiliert wird, ist dies kein technisches Problem, aber die Vereinheitlichung waere sauber.

---

## 8. LaTeX-Fehler und technische Probleme

### 8.1 MINOR: Tabellen-Platzierung [H] statt [htbp]

**Datei:** `Paper2_EN.tex`, diverse Stellen (z.B. Zeile 360, 408, 443, 483)
**Problem:** Mehrere Tabellen verwenden `[H]` (erfordert das `float`-Paket) statt `[htbp]`. REVTeX 4.2 unterstuetzt `[H]` nicht standardmaessig.

**Bewertung:** MINOR -- Kann zu Kompilierungsfehlern fuehren, wenn `float` nicht geladen ist.

**Korrektur:** Entweder `\usepackage{float}` hinzufuegen oder `[H]` durch `[htbp]` ersetzen.

---

### 8.2 MINOR: Paper II laed `ulem` aber verwendet \sout nur einmal

**Datei:** `Paper2_EN.tex`, Zeile 9
Das Paket `ulem` (mit `normalem`) wird geladen, aber `\sout{}` wird nur einmal verwendet (Zeile 886, "BBN consistency check -- DONE"). Fuer ein Fachjournal ist Durchstreich-Text unueblich.

**Bewertung:** MINOR

---

### 8.3 MINOR: Figure-Referenzen in Paper III ohne Bilder

**Datei:** `Paper3_EN.tex`, Zeile 562--598
Paper III referenziert `cfm_cl_comparison.png`, `cfm_cl_peaks.png` und `cfm_contour.png`. Diese Bilder muessen im `../figures/`-Verzeichnis vorhanden sein.

**Bewertung:** MINOR -- Kompilierungsfehler, wenn Bilder fehlen.

---

### 8.4 MINOR: Inkonsistente MCMC-Sample-Zahlen

- Paper III, Zeile 554: "240,000 samples"
- Paper III, Zeile 582: "48 walkers, 100 burn-in + 5000 production steps, 240,000 samples total"
- Paper III, Zeile 845: "9,600 samples"

Die 9,600 in Zeile 845 (48 * 200?) scheint von einer anderen Run-Beschreibung zu stammen als die 240,000 in der Hauptanalyse.

**Bewertung:** MINOR

---

## 9. Zusammenfassung der Schweregrade

| Schweregrad | Anzahl | Beschreibung |
|-------------|--------|--------------|
| CRITICAL | 2 | Falsche Zahlenwerte bei Paper-III-Zitation von Paper-II-Ergebnissen |
| MAJOR | 9 | Inkonsistente beta_early-Werte; Paper-IV cherry-picking; Forward-Referenz Lagrangian; mu-Notation; deduktive Kette; Overclaim Paper IV; a_0-Ableitung; Deep-MOND nicht gezeigt |
| MINOR | 17 | r_d-Abweichung; chi^2-Vergleichbarkeit; Notation; Bibliographie-Luecken; LaTeX-Probleme; Ton |
| SUGGESTION | 4 | Slow-Roll-Beweis; DRAFT-Kennung; Ton; bibitem-Keys |

---

## 10. Priorisierte Handlungsempfehlungen

### Sofort zu beheben (vor jeder Einreichung):

1. **Paper3_EN.tex, Zeile 61:** $\Delta\chi^2$, $H_0$, $\beta$ korrigieren auf die bevorzugten Paper-II-Werte (-5.5, 67.3, 2.82)
2. **Paper3_EN.tex, Zeile 808:** $\Delta\chi^2$ korrigieren (-5.1 -> -5.5) und $\ell_A$ korrigieren (301.477 -> 301.471)
3. **Paper4_EN.tex, Abstract:** $\Delta\chi^2 = -6.1$ als bevorzugte Variante klarer kennzeichnen oder den Wert -5.5 verwenden
4. **Paper3_EN.tex:** Fehlende bibitems Khoury2004, Williams2004, Bertotti2003 hinzufuegen
5. **Alle Papers:** $\beta_{\text{early}}$-Wert vereinheitlichen (Empfehlung: 2.82 als kanonisch)

### Vor Peer Review zu ueberarbeiten:

6. **Paper1_EN.tex, Sec. 8.4:** Forward-Referenz auf Paper-III-Lagrangian abschwaechen
7. **Paper4_EN.tex:** "prediction" -> "conjecture" fuer $a_0 = cH_0/(2\pi)$ und flache Rotationskurven
8. **Paper4_EN.tex:** Overclaims in Abstract und Conclusion abschwaechen (Deep-MOND-Skalierung nicht bewiesen)
9. **Paper2_EN.tex:** "exact Planck match" -> "within Planck uncertainties"
10. **Notationssystem:** $\mu$-Dreifachbelegung durch indizierte Notation aufloesen

### Empfohlen, aber nicht zwingend:

11. Paper IV als DRAFT/Working Paper kennzeichnen
12. Paper II: `\usepackage{float}` hinzufuegen oder `[H]` durch `[htbp]` ersetzen
13. Einheitliche bibitem-Keys ueber alle Papers
14. Paper2_EN.tex: `ulem`-Paket und `\sout{}` entfernen

---

## 11. Einarbeitungsstatus (2026-02-21)

| # | Finding | Status | Aenderung |
|---|---------|--------|-----------|
| 1 | Paper3 Intro falsche Paper-II-Werte (CRITICAL) | DONE | Zeile 61: Deltachi2=-5.5, H0=67.3, beta_early=2.82, mu(a) statt mu_eff=1.77, zero-EDE Variante |
| 2 | Paper3 Summary falsche Paper-II-Werte (CRITICAL) | DONE | Zeile 808: Deltachi2=-5.5, ell_A=301.471, Dopplung "geometric dark sector replacement" entfernt |
| 3 | Paper4 Abstract cherry-picking Deltachi2 (MAJOR) | DONE | -6.1 durch -5.5 ersetzt, "preferred zero-EDE variant" ergaenzt |
| 4 | Paper1 Forward-Referenz Lagrangian (MAJOR) | DONE | "presented here in retrospect; its rigorous derivation is the subject of Paper III" eingefuegt |
| 5 | Paper1 "dramatically" Overclaim (MINOR) | DONE | "dramatically" -> "substantially" |
| 6 | Paper4 a0 "prediction" -> "order-of-magnitude prediction" (MAJOR) | DONE | Conclusion: "emerges as an order-of-magnitude prediction", offener Task fuer 2pi-Ableitung |
| 7 | Paper4 Deep-MOND Overclaim Conclusion (MAJOR) | DONE | Linearisiertes Resultat g_A propto g_N explizit benannt, sqrt-Skalierung als erwartet/offen |
| 8 | Paper4 "only geometry" Overclaim Conclusion (MAJOR) | DONE | Caveat ergaenzt: numerische Bestaetigung + SPARC-Test erforderlich |
| 9 | Paper3 fehlende bibitems (MINOR) | DONE | Khoury2004, Williams2004, Bertotti2003 hinzugefuegt |
| 10 | Paper2 "exact Planck match" (MINOR) | DONE | -> "matches Planck within uncertainties" / "matches Planck measurement" |
| 11 | Paper2 float-Paket (MINOR) | DONE | `\usepackage{float}` hinzugefuegt |
| 12 | Paper3 float-Paket (MINOR) | DONE | `\usepackage{float}` hinzugefuegt |
| 13 | Paper2 ulem + sout entfernt (MINOR) | DONE | `\sout{BBN...}` -> "BBN... (completed)", `\usepackage[normalem]{ulem}` entfernt |
| 14 | Paper3 MCMC 9,600 vs 240,000 (MINOR) | DONE | Zeile 845: "9,600" -> "240,000 (48 walkers x 5,000 steps)" |
| -- | beta_early Vereinheitlichung (MAJOR) | OFFEN | Paper-II interne Werte (2.82, 2.823, 2.834) sind verschiedene Optimierungsstufen -- redaktionell zu klaeren |
| -- | mu-Notation Dreifachbelegung (MAJOR) | OFFEN | Erfordert tiefgreifende Notationsaenderung in allen Papers -- fuer spaetere Revision |
| -- | Paper IV DRAFT-Kennung (SUGGESTION) | OFFEN | Entscheidung beim Autor |
| -- | Deduktive Kette vs. tatsaechliche Reihenfolge (MAJOR) | TEILWEISE | Forward-Referenz in Paper I abgeschwaecht; Paper III muss noch ggf. angepasst werden |

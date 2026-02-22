# PEER REVIEW: Eliminating the Dark Sector: Unifying the Curvature Feedback Model with MOND

Reviewer: Gemini (KI-simuliert)
Datum: 2026-02-22
Journal-Simulation: Journal of Cosmology and Astroparticle Physics (JCAP) / Physical Review D

## EMPFEHLUNG: MAJOR REVISION

Das Paper wagt den ambitionierten Schritt, sowohl Dunkle Materie als auch Dunkle Energie durch eine Kombination aus dem Curvature Feedback Model (CFM) und MOND aus dem kosmologischen Budget zu streichen (ein "Baryonen-only" Universum). Das vorgelegte empirische Fitting (SNe, CMB distance priors, BAO) ist methodisch auf hohem Niveau und zeigt, dass das Framework erstaunlich wettbewerbsfähig gegenüber $\Lambda$CDM ist ($\Delta\chi^2 = -5.5$ mit gleicher Parameterzahl). Gleichzeitig bedient sich das Modell stark gepatchter, rein phänomenologischer Funktionen (wie der Skalenabhängigkeit von $\mu(a)$ und $\beta(a)$), um strukturelle Katastrophen (z.B. im frühen Universum) zu vermeiden. Bevor ein solch radikales Paradigma akzeptiert wird, müssen diese Hilfskonstrukte stärker physikalisch begründet werden.

## 2. SUMMARY (Zusammenfassung)

Das Manuskript erweitert das CFM so, dass der Energieinhalt des Universums zu 100% aus baryonischer Materie und Strahlung besteht. Die gravitative Rolle der Dunklen Materie übernimmt dabei ein rein geometrischer Freiheitsgrad ($\alpha \cdot a^{-\beta}$), kombiniert mit einer modifizierten Hintergrund-Gravitationsstärke, inspiriert durch MOND ($\mu_{eff} = \sqrt{\pi}$). Um Diskrepanzen wie das CMB-Akustik-Skalen-Problem und das Early Dark Energy (EDE) Problem zu beheben, werden Parameter wie die Krümmungskopplung $\beta$ und die MOND-Verstärkung $\mu$ glattlaufend skalenabhängig gemacht. Die optimierte Parameterkombination kann $\Lambda$CDM basierend auf einem komprimierten Datensatz aus Pantheon+, BAO und CMB (ohne full power spectra) geringfügig "schlagen".

## 3. STRENGTHS (Stärken)

1. **Ambitionierte Synthese:** Der Versuch einer kompletten geometrischen Ersetzung des "Dark Sectors" überbrückt gekonnt die zwei sonst oft isolierten Anomalien von Rotationskurven (MOND) und beschleunigter Expansion (DE).
2. **Exzellentes Data-Fitting und Statistik:** Ein detailliertes Cross-Validation (5-fold) wird eingesetzt, um sicher das Overfitting-Risiko eines Modells mit teils mehr Parametern rigoros auszuschließen.
3. **Klare Lösung für den "CMB/EDE"-Konflikt:** Das Paper bietet mathematisch elegante Auswege aus den typischen Stolperfallen für MOND-artige Ansätze in der Kosmologie, speziell was das Beibehalten des exakten $r_d$ Scallings angeht.

## 4. WEAKNESSES (Schwächen) -- KERNPRÜFUNG

### 4.1 Methodische Schwächen

- **Gepatchte "Running Functions":** Die Übergangsfunktion für $\mu(a)$ und $\beta(a)$ sind logistisch und phänomenologisch gewählt, um den "Fit" zu zwingen, richtig herauszukommen. Ohne eine Herleitung dieser "Running"-Effekte aus der Dynamik eines relativistischen Ansatzes handelt es sich eher um eine Form von intelligentem Kurven-Fitting als um ab initio hergeleitete Vorhersagen.
- **Fehlende Power-Spektren Analyse:** Obwohl \texttt{hi\_class} angesprochen wird, beschränkt sich das Paper auf komprimierte Background-CMB-Observablen ($l_A$, $R$). Ohne die Berechnung der vollständigen Temperatur- ($TT$) und Polarisations-Spektren ($EE$, $TE$) bleibt die Passfähigkeit für das CMB höchst spekulativ.

### 4.2 Argumentative Schwächen

- **$\sqrt{\pi}$-Conjecture:** Das Argument, $\mu_{eff} = \sqrt{\pi}$ (Kosmologie) entstamme dem Verhältnis des Volumens der 2-Sphäre gegenüber der MOND $4/3$-Enhancement Rate, ist faszinierend, liest sich aber sehr numerologisch. Warum sollte eine derartige Projektion der Dimensionen die effektive Gravitationsstärke auf kosmologischen Hintergrundskalen definieren?
- **Bullet-Cluster-Resolution:** Das verbale Argument ("Die Linsenkonvergenz folgt dem geometrischen Gedächtnis") ist interessant, doch bei einem extrem asymmetrischen, nicht-homogenen, stark dynamischen Event wie dem Bullet-Cluster-Crash genügen simple Skalierungsabschätzungen des Hintergrundes nicht, um die Phasenverschiebung zwischen Röntgen- und Linsensignal ohne N-Body Simulation in einer MOND/f(R) Simulation zu validieren.

### 4.3 Literatur-Schwächen

- **Skordis & Złośnik (AeST):** Es wird stark von der Kompatibilität mit MOND ausgegangen und parallel das relativistische MOND (TeVeS/AeST) erwähnt. Die Verknüpfung der phänomenologischen Skalare hier (insbesondere des "geometric DM terms") mit dem Vektorfeld/Skalarfeld aus AeST ist aber nicht vorgenommen worden. Dies wirkt wie ein isolierter dritter Ansatz.

### 4.4 Formale Schwächen

- Wie im Paper I liegt vieles in der Verantwortung von "Paper III". Die Trennung in "Phänomenologie" hier und "Mikroskopische Derivation" dort schwächt dieses Paper signifikant im Standalone-Review-Prozess.

## 5. SPECIFIC COMMENTS (Detailkommentare)

- **[Eq. 7, Trace-Coupling $\mathcal{S}(a)$]:** Die Herleitung durch die Unterdrückung von Elementen bei konformer Symmetrie ist brilliant und elegant. Dies ist eine absolute Stärke des Papers und sollte isoliert betrachtet als eigenes wichtiges Resultat herausgestellt werden.
- **[Section 3.5.3, Eliminating EDE]:** Das "Ausschalten" von EDE durch die Einführung eines neuen Fits für $\mu(a)$ ist methodisch ein Nullsummenspiel bezüglich Einsparung von Parametern. Es verschiebt nur Unbekannte. Die Motivation warum $\mu$ so spät ($z > 4000$) erst das Standard Newtonian Regime betreten soll, während die Beschleunigung der Strahlungsdominanz massiv ist, ist unintuitiv.

## 6. QUESTIONS TO THE AUTHORS (Fragen an die Autoren)

1. Wie resultiert die angenommene "kosmologische" MOND-Verstärkung von $\sqrt{\pi}$ bei der makroskopischen Mittelung aus einer vollwertigen geometrischen MOND-Theorie (wie Bekensteins TeVeS oder AeST)?
2. Wenn der "geometrische DM-Term" essenziell wie eine Form von räumlicher Krümmung (Skalierung $a^{-2}$) wirkt, wie kann dieser Term auf der Skala von individuellen Galaxien klumpen, um dort als Quelle für modifizierte Rotationskurven oder Cluster-Dynamik zu wirken?
3. Die $\mu(a)$ Skalierung ist so angelegt, dass BBN geschützt wird. Führt jedoch ein starker Gradient im gravitativen Verhalten nahe Materie-Strahlungs-Gleichheit nicht zwingend zu signifikanten unnatürlichen Signaturen im Early Integrated Sachs-Wolfe (eISW) Effekt im CMB?

## 7. MINOR ISSUES (Kleinigkeiten)

- Einige Referenzen in Section 4.4.3 bezüglich MOND und Wachstumsstrukturen von Kosmischen Voids könnten ergänzt werden.

## Bewertungsskala

| Kriterium | Note (1-10) |
|-----------|-------------|
| Originalität / Neuheitswert | 10 |
| Methodische Qualität | 7 |
| Argumentative Stringenz | 6 |
| Literatureinbettung | 7 |
| Klarheit und Lesbarkeit | 8 |
| Relevanz für das Fachgebiet | 9 |
| **Gesamtnote** | 7.8 |

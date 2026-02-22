# PEER REVIEW: The Galactic-Cosmological Nexus: Deriving MOND Dynamics from Curvature Saturation

Reviewer: Gemini (KI-simuliert)
Datum: 2026-02-22
Journal-Simulation: Physical Review D / Classical and Quantum Gravity

## EMPFEHLUNG: MAJOR REVISION

Das vierte Paper der Serie zielt darauf ab, die auf kosmologischen Skalen funktionierende "Curvature Feedback Model" (CFM) Theorie auf galaktische Skalen zu erweitern, um das Phänomen der dunklen Materie (flache Rotationskurven) zu erklären, ohne neue Teilchen einzuführen. Es postuliert ein zusätzliches zeitartiges Vektorfeld ("Daughter 2"), um eine MOND-ähnliche (Modified Newtonian Dynamics) Phänomenologie zu erzeugen. Während die Motivation aus thermodynamischen Prinzipien intellektuell ansprechend ist und die Verknüpfung der kosmologischen mit der galaktischen Skala ($a_0 \sim cH_0 / 2\pi$) fasziniert, mangelt es dem aktuellen Manuskript an der mathematischen und phänomenologischen Strenge, die für eine Publikation in PRD erforderlich ist. Insbesondere die Behandlung der Vektorfeldkopplung und die rudimentäre SPARC-Analyse müssen signifikant verbessert werden.

## 2. SUMMARY (Zusammenfassung)

Das Paper adressiert die "galaktische Lücke" des skalaren CFM, das nur einen Faktor 4/3 an gravitativer Verstärkung liefert. Um galaktische Rotationskurven zu erklären, wird ein zeitartiges Vektorfeld $A_\mu$ eingeführt, das an das Skalarfeld und die Materie koppelt: $\mathcal{F} \propto (|T|/\rho_{crit}) \cdot \mathrm{sech}^2(\phi/\phi_0) \cdot A_\mu\partial^\mu\phi$. Der Autor leitet daraus ab, dass dieses Vektorfeld in Galaxien ("deep-MOND") reaktiviert wird, aber im Sonnensystem ("Chameleon screening") und bei BBN verschwindet. Die Kernthese ist, dass diese Konstruktion eine effektive MOND-Dynamik erzeugt, bei der die Grenzbeschleunigung $a_0$ kausal mit der Hubble-Konstante verknüpft ist. Es wird eine erste Überprüfung an rotierenden Galaxien aus der SPARC-Datenbank vorgenommen.

## 3. STRENGTHS (Stärken)

1. **Verknüpfung von Skalen:** Die analytische Ableitung von $a_0 \sim c H_0 / 2\pi$ aus der dynamischen Evolution eines Skalarfeldes anstelle einer simplen Parameter-Fassung ist ein konzeptionelles Highlight des Modells.
2. **Kreativer theoretischer Aufbau:** Die hierarchische thermodynamische Begründung ("Principal-Agent" Struktur) für das Vektorfeld als dissipatives stabilisierendes Element ist eine ausgesprochen originelle Idee für Modified Gravity Theorien.
3. **Beachtung von no-go Theoremen:** Die sorgfältige Argumentation, dass die Modifikation die Gravitationswellengeschwindigkeit ($c_T = c$) erhält und im Sonnensystem durch "parasitic screening" verschwindet, bewahrt das Modell vor dem schnellen Ausschluss durch bekannte Präzisionstests.

## 4. WEAKNESSES (Schwächen) -- KERNPRÜFUNG

### 4.1 Methodische Schwächen

- **Konstruktion von $\mathcal{F}$ (Gl. 17):** Die Kopplungsfunktion $\mathcal{F}$ wirkt willkürlich zusammengebaut ("ad-hoc"), um exakt die benötigten Phänomene hervorzubringen. Wenn ein Vektorfeld $A_\mu$ derart spezifisch und hochgradig nicht-linear (z.B. der Faktor $|T| / \rho_{crit}$) an Invarianten koppelt, ist das Risiko katastrophaler Instabilitäten extrem hoch. Es fehlt eine strenge Analyse der Stabilität (Cauchy-Problem, Subluminalität, Abwesenheit von Geistern und Tachyonen) im Vektorsektor abseits der reinen Tensormoden.
- **SPARC-Analyse:** Die SPARC-Analyse (Sec. 8.2) ist extrem oberflächlich ("Preliminary"). Für einen Artikel dieses Anspruchs ist die Nutzung von nur 10 "repräsentativen" (teils synthetischen) Galaxien unzureichend. Es muss die volle Datenbank verwendet werden, um eine robuste statistische Aussage im Vergleich zu MOND und $\Lambda$CDM zu treffen.

### 4.2 Argumentative Schwächen

- **Übergangsfunktion (Interpolation Function):** Der Übergang von Newton zu MOND wird auf das nicht-lineare "Operator Feedback" (Chameleon-Masse) geschoben. Das BVP wird gelöst, findet eine Steigung von 0.5, aber liefert keine analytisch greifbare Interpolationsfunktion $\mu(x)$. Im SPARC-Test wird dann einfach die empirische McGaugh-Funktion benutzt statt der modell-nativen Funktion. Das ist inkonsistent. Das Modell sollte an seinen *eigenen* Vorhersagen für die RAR (Radial Acceleration Relation) gemessen werden.

### 4.3 Formale Schwächen

- Der "Beweis" der Unterdrückung im Sonnensystem (Sec. 6.2, Gl. 42) extrapoliert die lineare Yukawa-Dämpfung $e^{-mr}$ um mehr als $3 \times 10^9$ Größenordnungen, ohne nicht-lineare "thin shell" Effekte (wie beim echten Chameleon) quantitativ abzugleichen. Die Argumentation ist suggestiv, aber nicht formal wasserdicht.

## 5. SPECIFIC COMMENTS (Detailkommentare)

- **[Abweichung bei $a_0$]:** Das Modell sagt $a_0 = c H_0 / 2\pi \approx 1.04 \times 10^{-10}$ m/s$^2$ voraus, beobachtet werden $1.20 \times 10^{-10}$ m/s$^2$. Eine $13\%$-Abweichung mag klein erscheinen, kann aber im $\chi^2$ exponentiell hart bestraft werden. Dies muss im SPARC-MCMC besser diskutiert werden. Ist der Fehler rein statistisch, oder systematisch im Modell?
- **[Tensor $T^{(A)}_{\mu\nu}$]:** In der Modifikation der Poissongleichung (Gl. 64ff) betrachten Sie nur den linearen Term. Generiert das Vektorfeld selbst keine nennenswerte effektive Energiedichte?

## 6. QUESTIONS TO THE AUTHORS (Fragen an die Autoren)

1. Die Kopplung $\mathcal{F}$ enthält den Betrag des Spurstensors $|T|$. Solche nicht-analytischen Terme (Betrag) können in Feldtheorien zu Singularitäten oder Unstetigkeiten in den Feldgleichungen an Nulldurchgängen führen. Wie rechtfertigen Sie diesen Term mathematisch rigoros? Wäre eine Funktion $\propto T^2$ nicht feldtheoretisch sauberer?
2. Warum haben Sie für den Pöschl-Teller-Faktor $\mathcal{B}(\phi)$ genau die Form $\mathrm{sech}^2(\phi)$ gewählt und nicht eine allgemeinere Funktion, die in die Sättigung läuft?
3. Ist geplant, die volle Vorhersage für die CMB-Spektren (unter Einschluss der Vektor-Störungen $A_\mu$) mit \texttt{hi\_class} zu berechnen? Die Aussage, dass die Vektordichte im Hintergrund verschwindet, bedeutet nicht zwangsläufig, dass deren Störungen im CMB irrelevant sind.

## 7. MINOR ISSUES (Kleinigkeiten)

- Die Struktur von Section 7 und 8 überlappt thematisch stark. Die Herleitung des MOND-Attraktors und seine numerische Bestätigung sollten in einem Abschnitt gebündelt werden.

## Bewertungsskala

| Kriterium | Note (1-10) |
|-----------|-------------|
| Originalität / Neuheitswert | 9 |
| Methodische Qualität | 6 |
| Argumentative Stringenz | 7 |
| Literatureinbettung | 8 |
| Klarheit und Lesbarkeit | 8 |
| Relevanz für das Fachgebiet | 8 |
| **Gesamtnote** | 7.6 |

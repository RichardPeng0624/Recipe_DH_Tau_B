# Discussion (Sect. 5) — outline

**Revision note**: the previous version of this outline mis-identified "Snellen
2025" as Mulder et al. 2025 (SupJup VI). The user has now supplied the correct
paper: **Snellen, I. A. G. 2025, ARA&A, 63, 83** ("Exoplanet Atmospheres at
High Spectral Resolution" — confirmed on disk,
`paper/references/Snellen - Exoplanet Atmospheres at High Spectral Resolution.pdf`),
a review article. Its Fig. 12 compiles literature $^{12}$C/$^{13}$C ratios
across field brown dwarfs, young brown dwarfs, super-Jupiters, and hot
Jupiters — **DH Tau b is already one of the plotted points**, at Xuan et al.
(2024)'s value ($53^{+50}_{-24}$), sitting near the low end of the
super-Jupiter cluster. This is the actual "current sample" the user wants
5.1 built around. Following Snellen (2025)'s own citations back to the
primary literature (the ESO SupJup Survey papers III, IV — this paper is
itself SupJup Survey **XI**, per the title in `dh_tau_b.tex` line 25, so
these are sibling papers, not just generic comparisons) turned up
substantially more precise, directly citable material than the first draft
had — see §5.1 below, fully rewritten.

Scope note (unchanged): `dh_tau_b.tex` has four empty Discussion subsections
(lines 575-581): 5.1 "The $^{12}$CO/$^{13}$CO ratio", 5.2 "The presence of
clouds", 5.3 "Retrievals with the absolute-calibrated spectrum", 5.4
"Multi-wavelength SED fitting". 5.3/5.4 remain placeholders.

All fiducial-model numbers are job **430784** (cloud-free, 19 params, Na/Ca
opacity present but not free); cloudy-model numbers are job **1001064** (24
params) — the standing benchmark pair confirmed 2026-08-05. Items not
verified against on-disk posteriors/primary sources this session are marked
**[VERIFY]**.

---

## Issues found in the *existing* Results text while researching this — flag before drafting Discussion

Sect. 4.2 (`dh_tau_b.tex` lines 527-539) already discusses the
$^{12}$CO/$^{13}$CO comparison sample, and rereading it against the primary
sources (Zhang et al. 2021b; Zhang et al. 2024, arXiv:2409.16660 = ESO
SupJup Survey III) surfaced problems worth fixing regardless of the
Discussion rewrite:

1. **"TYC 8998-760-1 b" and "YSES 1 b" are the same object**, written as if
   they were two ("...the substantially $^{13}$C-enriched super-Jupiter
   TYC~8998-760-1~b (...) and YSES 1~b; \citealt{zhang21a}, \citealt{zhang21b}"
   — line 535). Confirmed via the SupJup III abstract itself: "two wide-orbit
   super-Jupiters in YSES 1 (or TYC 8998-760-1)." Needs to become one name,
   one value, one citation.
2. **The cited value, $31^{+17}_{-10}$, is stale.** That is the *original*
   2021 SINFONI discovery value (Zhang et al. 2021b, per the SupJup III
   paper's own citation of "our previous SINFONI result of $31^{+17}_{-10}$").
   Higher-quality CRIRES+ data (Zhang et al. 2024, SupJup III) **revised
   this upward to $88\pm13$** — a value that is *no longer* a clean example
   of strong $^{13}$C-enrichment (it is close to the terrestrial/solar
   value of $\approx89$, and consistent with the host star TYC 8998-760-1's
   own isotope ratio, $66\pm5$, within $1.6\sigma$). Citing $31^{+17}_{-10}$
   without noting the revision risks using a superseded number as "the"
   core-accretion prototype.
3. **`\citealt{xxxx}` placeholder** at line 530 for the ISM ($\approx68$)
   and solar ($\approx89$–94) values is still unfilled.
4. **The `zhang21a`/`zhang21b` bib-key assignment needs verification once
   `reference.bib` is populated** (currently 0 bytes, per
   [[project_retrieval_framework_paper_notes]]) — "a" vs. "b" suffixes are
   assigned per-citing-paper, not universal, so don't assume our keys point
   to the same real papers as the "Zhang et al. (2021a)"/"(2021b)" labels
   used inside the SupJup III/IV text without checking directly against
   whichever two real Zhang 2021 papers end up in our own bib file.

These are Results-section fixes, not Discussion content — flagged here
because 5.1 depends on getting the comparison-sample facts right, and I
already had to resolve them to write the outline below. Not applied to the
.tex; needs a decision from the user/Giancarlo on whether to fix now or in a
later pass.

---

## 5.1 The $^{12}$CO/$^{13}$CO ratio — formation history of DH Tau B

### 1. Restate the measurement (already in Results, don't re-derive)

Fiducial $^{12}$CO/$^{13}$CO $=141^{+42}_{-32}$; CCF S/N at rv=0 for
$^{13}$CO is only 2.41 (not a clean detection at the companion's velocity).
Carry this caveat forward rather than re-litigating it — but it is fair
game to build the formation argument on top of it here in Discussion,
unlike in Results (per the established "no hedging in Results" convention).

### 2. DH Tau b's place in the *published* comparison sample (Snellen 2025, Fig. 12)

Open with this, since it is the direct answer to what the user asked for.
Snellen (2025) Fig. 12 — a literature compilation spanning field BDs, young
BDs, super-Jupiters, and hot Jupiters — already includes DH Tau b, plotted
at Xuan et al. (2024)'s KPIC value ($53^{+50}_{-24}$), sitting near the
*low* end of the super-Jupiter cluster (alongside HIP 79098 b, VHS 1256 b —
all below the ISM/solar lines drawn on the figure). **Our independent,
higher-resolution CRIRES+ measurement revises this to $141^{+42}_{-32}$** —
above ISM and solar, and above every other point Snellen (2025) plots in
the super-Jupiter category. This single sentence is the paper's actual news
on this axis: DH Tau b moves from the $^{13}$C-enriched tail of the
super-Jupiter distribution to above the entire compiled sample.

**Explicit caution to carry over from Snellen (2025)'s own text** (worth
quoting/paraphrasing directly, since it's the review's own verdict on this
exact kind of comparison): the compiled sample is "very heterogeneous...
could be biased in multiple ways," "these measurements are still
challenging," and "it is too early to draw any conclusions" from the
population trend alone. Frame DH Tau b's revision in that spirit — a
genuine update to the literature value, not a decisive formation
verdict by itself.

### 3. A revision of this size is not unusual in this literature — cite the direct precedents

Two documented cases, both from the same SupJup collaboration whose papers
this project already cites, show comparably large method-driven swings for
*the same object*:

- **YSES 1 b / TYC 8998-760-1 b**: $31^{+17}_{-10}$ (Zhang et al. 2021b,
  SINFONI, $R\sim4500$) $\rightarrow$ $88\pm13$ (Zhang et al. 2024, SupJup
  III, CRIRES+, $R\sim100{,}000$) — a $\sim2.8\times$ upward revision with
  better spectral resolution/SNR, explicitly consistent with the host
  star's own ratio ($66\pm5$) after the revision, not before.
- **GQ Lup B**: two *contemporaneous* independent measurements disagree at
  $\geq3\sigma$ — Xuan et al. (2024a, KPIC, $R\sim35{,}000$,
  three reddest orders, equilibrium chemistry): $153^{+43}_{-31}$; González
  Picos et al. (2025, SupJup IV, CRIRES+ full K2166, $R\sim120{,}000$, free
  chemistry): $53^{+7}_{-6}$. González Picos et al. attribute the gap to
  wavelength coverage, resolution, S/N ($\sim20$ vs. $\sim12$), and the
  equilibrium-vs-free-chemistry modelling choice, and show it is *not*
  fully explained by differing P-T profiles alone.
- **Note the sign flips between objects**: for GQ Lup B, KPIC gives the
  *higher* value; for DH Tau b, KPIC (Xuan et al. 2024) gives the *lower*
  value relative to our own CRIRES+ result. This argues against a simple
  one-directional instrumental bias and for genuine per-object modelling
  sensitivity — useful to state explicitly so the DH Tau b revision doesn't
  read as cherry-picked in the direction that favours a nicer story.

### 4. The fuller comparison sample, with values checked against primary sources (not just read off Snellen's Fig. 12)

| Object | $^{12}$CO/$^{13}$CO | Source | Category |
|---|---|---|---|
| ISM | $68\pm15$ | Milam et al. 2005 | anchor |
| Solar/terrestrial | $89.3\pm0.2$ (Meija et al. 2016) or $93.5\pm3.1$ (Lyons et al. 2018) — two values in circulation, **[VERIFY which the paper should standardise on]** | anchor |
| YSES 1 b (orig.) | $31^{+17}_{-10}$ | Zhang et al. 2021b | super-Jupiter, superseded |
| YSES 1 b (revised) | $88\pm13$ | Zhang et al. 2024 (SupJup III) | super-Jupiter |
| TYC 8998-760-1 (host of YSES 1 b) | $66\pm5$ | Zhang et al. 2024 (SupJup III) | host star |
| 2MASS J0355 (isolated BD) | $97^{+25}_{-18}$ | Zhang et al. 2021b | isolated young BD |
| TWA 28, 2M J0856, 2M J1200 (3 young BDs, age $\leq10$ Myr) | $\approx80$–$120$ | González Picos et al. 2024 (SupJup II) | young BD |
| DENIS J0255 (old field T dwarf) | $184^{+40}_{-61}$ | de Regt et al. 2024 (SupJup I) | field BD |
| GQ Lup B | $153^{+43}_{-31}$ (KPIC) / $53^{+7}_{-6}$ (CRIRES+) | Xuan et al. 2024a / González Picos et al. 2025 (SupJup IV) | super-Jupiter |
| GQ Lup A (host) | $51^{+10}_{-8}$ | González Picos et al. 2025 (SupJup IV) | host star |
| VHS 1256 b | $62\pm2$ | Gandhi et al. 2023 (JWST) | super-Jupiter |
| HIP 55507 A/B | consistent host/companion pair (values not extracted this pass) **[VERIFY exact numbers if used]** | Xuan et al. 2024b | host+companion |
| DH Tau b (lit.) | $53^{+50}_{-24}$ | Xuan et al. 2024a | super-Jupiter |
| **DH Tau b (this work)** | $141^{+42}_{-32}$ | — | — |
| Full literature range, as characterised by Zhang et al. (2024) themselves | "$\approx30$ to $180$" | SupJup III §1 | — |

DH Tau b's revised value sits **at the high end of the entire compiled
range**, above the revised YSES 1 b value, close to GQ Lup B's KPIC (X24)
outlier, and inside/above the young-BD cluster — no longer resembling the
low, $^{13}$C-enriched tail its own earlier measurement had placed it in.

### 5. The compositional formation-tracer framework (from SupJup III/§1, directly applicable — use this instead of the flatter "high ratio = no ice accretion" framing from the first draft)

Zhang et al. (2024, SupJup III §1) lay out the theoretical mapping cleanly,
and it is worth reproducing (paraphrased) rather than re-deriving:

- **Gravitational instability / cloud fragmentation**: rapid top-down
  collapse occurs while disk/cloud material is still pristine → resets
  companion composition to **protostellar values** (near-stellar C/O,
  near-stellar/ISM $^{12}$CO/$^{13}$CO, no metal enrichment).
- **Core accretion beyond the CO iceline**, outcome depends on gas- vs.
  solid-dominated accretion:
  - *Gas-dominated*: sub-stellar $[\mathrm{C/H}]$, $[\mathrm{O/H}]$ (most
    volatiles condensed out of the gas already), but **super-stellar C/O**
    (the residual gas-phase disk material is O-poor). Isotope ratio
    relatively unprocessed (gas-phase CO not strongly fractionated).
  - *Solid/ice-dominated*: **super-solar metallicity** (ices carry the
    bulk C, O budget), near-stellar C/O, but **significant $^{13}$C
    enrichment** (low $^{12}$CO/$^{13}$CO) from isotope-selective
    photodissociation/ion exchange in the ice-forming region — this is the
    YSES 1 b (2021)/TYC 8998-760-1 b-type signature.

**Map DH Tau B's own retrieved values onto this** (all already in Results,
just reused here): $[\mathrm{C/H}]=-0.26^{+0.11}_{-0.12}$ (sub-solar,
$\sim2.3\sigma$), $\mathrm{C/O}=0.59^{+0.02}_{-0.02}$ (marginally
super-solar, $\sim1.6\sigma$), $^{12}\mathrm{CO}/^{13}\mathrm{CO}=141^{+42}_{-32}$
(no significant $^{13}$C-enrichment). This specific
combination — sub-solar metals *and* a high isotope ratio — does **not**
match the solid/ice-dominated core-accretion signature (which predicts
metal enrichment, not depletion). It is instead consistent with *either*
(i) gravitational instability, or (ii) gas-dominated core accretion beyond
the CO iceline — the isotope ratio alone cannot separate these two, but it
does rule out the metal-rich, strongly-$^{13}$C-enriched signature that
would be the clean fingerprint of substantial icy-solid/pebble accretion.

### 6. Host-star comparison — and an honest limitation relative to the sibling SupJup papers

DH Tau A's C/O ($0.555\pm0.063$, Hejazi et al. 2025) agrees with the
companion's C/O at $<0.5\sigma$ (already in Results) — consistent with,
but not proof of, a shared-origin/near-stellar composition.

**Important limitation to state explicitly**: every one of the
sibling-paper precedents above (YSES 1 A/b, GQ Lup A/B, HIP 55507 A/B) runs
the sharper test — comparing the companion's $^{12}$CO/$^{13}$CO
*directly* against the host star's own isotope ratio, since (per SupJup
III §6.3) "a deviation of the $^{12}$CO/$^{13}$CO ratio from the stellar
value would be an essential probe for distinguishing core accretion from
gravitational instability." **DH Tau A's own carbon isotope ratio has not
been measured** — our host comparison rests only on C/O, a weaker test.
State this as a genuine limitation and a natural pointer to future work
(a host-star HRS isotope measurement for DH Tau A), not something to paper
over.

### 7. Weigh the three formation hypotheses the user named, using all of the above

- **(a) In-situ disk formation via solid/ice-dominated core accretion beyond
  the CO snowline** (the Xuan et al. 2024 / Bonnefoy et al. 2014-motivated
  picture): predicts super-solar metallicity *and* strong $^{13}$C-enrichment.
  **Disfavoured on both axes** — our metallicity is sub-solar, not enriched,
  and the isotope ratio shows no enrichment. Weakened somewhat by the
  $^{13}$CO CCF non-detection at rv=0, and by Xuan et al. (2024)'s own
  parallel caveat that their $^{13}$CO detection for this object is
  "tentative" — a lean, not a clean rejection, but a *double* line of
  evidence (not just isotopes) against this specific sub-case now.
- **(b) Formed like an isolated brown dwarf (gravitational instability /
  cloud fragmentation, star-like), either in-situ around DH Tau A or
  independently and later gravitationally captured**: consistent with the
  measured isotope ratio (now sitting with the young/field-BD population)
  and with the sub-solar-metals/near-stellar-C/O pattern. **Chemistry
  cannot split this into "in-situ GI" vs. "captured"** — both predict the
  same pristine-composition signature; that distinction needs orbital/
  kinematic evidence (capture typically leaves a dynamically distinct
  signature — e.g. non-coplanarity — just as scattering does; see (c)).
- **(c) Formed closer to the star (core accretion) and later dynamically
  scattered/ejected onto the current wide orbit**: chemically
  **indistinguishable from (a)** on this data — scattering doesn't erase a
  natal isotopic/metallicity signature, so this hypothesis inherits (a)'s
  compositional disfavouring only if the pre-scattering formation mode was
  solid-dominated; a gas-dominated-accretion-then-scattered history would
  look identical to (b) chemically. Zhang et al. (2024, SupJup III §6.3)
  make exactly this point for the analogous case of YSES 1 c: discriminating
  in-situ formation from scattering requires **orbital dynamics**
  (eccentricity, obliquity, additional companions), not atmospheric
  composition, and note this is the literature's standard alternative
  specifically invoked *because* wide separations (see below) are not
  cleanly compatible with in-situ core accretion (long core-assembly
  timescale, Lambrechts & Johansen 2012) or often with in-situ GI either
  (which tends to produce higher masses) — flag explicitly as needing
  follow-up (astrometry/RV monitoring), not something this retrieval can
  settle.

### 8. Tie to the wide separation

`dh_tau_b.tex` already states a projected separation of $\approx2\farcs3$
(line 87-88). At $d=135.2$~pc (Gaia DR3; currently only in a commented-out
line, 335 — **needs reinstating/citing properly before use**), this is
$\approx310$~AU projected **[VERIFY — recompute/cite properly; not
currently live in the compiled draft]**. Per the SupJup III framing (§1),
wide-orbit super-Jovian companions are "not easily compatible with
formation via either in-situ core accretion or gravitational collapse"
individually (core accretion: assembly-timescale problem; GI: tends to
produce BD/stellar masses, though DH Tau B's own mass, $\sim12\,M_{\rm
Jup}$ per \citet{xuan24}, sits right at that boundary, which is itself
worth one sentence — the mass alone doesn't disfavour GI here the way it
would for a clearly planetary-mass object) — motivating scattering
(hypothesis c) as a standing alternative *independent* of the isotope data.
Worth one sentence connecting the two lines of evidence rather than
treating them separately.

### 9. Synthesis / closing paragraph

The isotope ratio and bulk composition together disfavour solid/ice-dominated
core accretion beyond the CO snowline (hypothesis a in its strong form) and
are consistent with either gravitational instability or gas-dominated core
accretion (hypothesis b, or a gas-dominated version of a/c) — but this
rests on a single, CCF-marginal $^{13}$CO measurement, a host comparison
that (unlike the sibling SupJup systems) only reaches C/O rather than the
sharper isotope-to-isotope test, and cannot exclude a scattered-then-migrated
history on chemistry alone. State it at this hedged strength, echoing
Snellen (2025)'s own explicit caution about drawing conclusions from this
kind of comparison. Matches the established "no hedging in Results, but
Discussion is where the caveats belong" convention
([[project_retrieval_framework_paper_notes]]).

---

## 5.2 The presence of clouds

Unaffected by the Snellen (2025) correction — unchanged from the previous
draft of this outline.

Keep this section short, per the user's framing ("brief section to
announce the presence of clouds cannot be proved").

1. **State the comparison**: fiducial (cloud-free, job 430784) vs. cloudy
   (EddySed MgSiO$_3$+Fe, job 1001064) — same data, normalisation,
   likelihood; cloud model adds 5 params (log_X_MgSiO3, log_X_Fe, fsed,
   log_Kzz, sigma_lnorm).

2. **Evidence-ratio result — needs a clean recompute before drafting**
   **[VERIFY — not yet confirmed for this exact standing pair]**: memory
   only has a clean NS-vs-INS $\ln Z$ comparison for the *predecessor*
   mednorm-era pair (3160171 vs 3708914: NS $\Delta\ln Z\approx+1$,
   essentially indifferent; INS $\Delta\ln Z\approx-9$, favouring
   cloud-free) — the sign flip between estimators was flagged there as
   needing resolution before quoting any Bayes factor, and was never closed
   out for 430784 vs 1001064. **Pull `final_params_dict.pickle`'s NS and
   INS $\ln Z$ (with errors) for both jobs directly before writing this.**

3. **Posterior-consistency argument** (verified 2026-08-05): every shared
   parameter between 1001064 and its no-opacity/no-cloud sibling agrees to
   $<0.4\sigma$ — adding clouds doesn't pull anything else, part of the
   "clouds are not required" case.

4. **Condensation-curve check** (already computed for the Results P-T
   figure, reuse directly): Ackerman & Marley (2001) Fe/MgSiO$_3$
   saturation curves at the cloudy model's own retrieved
   $[\mathrm{C/H}]=-0.333$, C/O$=0.579$:
   - **MgSiO$_3$**: retrieved profile stays hotter than the condensation
     curve across essentially the whole probed range (+150 K at 0.1 bar,
     +408 K at 1 bar) — **not thermodynamically favoured**.
   - **Fe**: tracks within ~20-75 K of the profile through
     $10^{-4}$–$3\times10^{-2}$ bar, crossing right at the profile's own
     kink ($\approx0.08$ bar) — **only marginally/locally plausible**.
   - Useful estimator-independent cross-check explaining *why* a null
     result is physically reasonable.

5. **Cloud-parameter posteriors — check before writing** **[VERIFY]**: are
   fsed/log_Kzz/sigma_lnorm/log_X_MgSiO3/log_X_Fe well-constrained or
   prior-dominated for 1001064? Don't assert without checking
   `final_posterior.npy`.

6. **Closing sentence**: given (i) a small/estimator-dependent evidence
   ratio (pending recompute), (ii) an otherwise-unchanged posterior, and
   (iii) condensation curves that put MgSiO$_3$ outside and Fe only
   marginally inside the favourable regime, clouds are **not demonstrated
   one way or the other**. This is the "cannot be proved" conclusion the
   user asked for — don't overbuild beyond it.

---

## 5.3 Retrievals with the absolute-calibrated spectrum — placeholder

Unchanged. Likely source: demoted absolute-flux benchmark job **2027997**
(22 params incl. Na/Ca as free params, predates the atomopac decision) vs.
the standing benchmark (430784). Note the SINFONI absolute-flux-cal
paragraph was already removed from Sect. 2.2 as unused — this subsection
needs to justify discussing it at all.

## 5.4 Multi-wavelength SED fitting — placeholder

Unchanged. No source material identified yet.

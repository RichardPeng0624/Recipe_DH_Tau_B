# DH Tau B Retrieval Framework — Notes for Paper Writing (Methods Section)

> **Purpose of this file**: self-contained briefing for whichever Claude agent
> drafts the "Atmospheric Retrieval Framework" (Methods) section of the DH Tau B
> paper. Every numeric value below was either read directly from the code on
> disk or verified against `final_params_dict.pickle` / `final_posterior.npy`
> in the relevant `/data2/peng/retrievals/<job_id>_*/` output directory on
> **2026-07-16**. Where a number comes from memory/log paraphrase rather than a
> direct disk check, it is marked "(unverified, log-only)".
>
> **Companion file**: `retrieval_framework_priors_table.tex` — ready-to-include
> LaTeX deluxetable of the adopted free parameters and priors.
>
> **Do not re-derive project history from scratch** — read this file plus the
> two source files below before writing anything; they contain the full
> derivation trail already digested from ~4400 lines of research log.

## 0. Source files this document distills

| File | What it is |
|---|---|
| `/data2/peng/Recipe_DH_Tau_B/Tasting_guidebook/Guidebook_GAStronomy_Piette_v4.2.py` | **The forward-model + retrieval engine.** All classes (`Target`, `Parameters`, `Covariance`, `CovarianceGP`, `LogLikelihood`, `pRT_spectrum`, `Retrieval`) and default prior sets live here. 2443 lines; treat as ground truth over any paraphrase below. |
| `/data2/peng/Recipe_DH_Tau_B/Tasting/tasting_retrieval_equa_chem_v6.1_piette_cloudfree.py` | The **primary/benchmark** retrieval driver script (007PM-EQ-GP): absolute flux calibration + Standard Gaussian likelihood + GP covariance, equilibrium chemistry, cloud-free. This is "the paper's main result" per the reduction notes. |
| `/data2/peng/Recipe_DH_Tau_B/Tasting/tasting_retrieval_equa_chem_v6.1_piette_cloudfree_mednorm.py` | The **calibration-sensitivity** driver (008PM-EQ-MEDNORM): sidesteps SINFONI absolute flux calibration entirely via per-chip-median normalisation of data *and* model, same GP + Standard Gaussian likelihood. |
| `/data2/peng/Recipe_DH_Tau_B/Tasting/tasting_retrieval_equa_chem_v6.1_piette_cloud_mednorm.py` | Cloud extension of the mednorm series (008PM-EQ-CLD-MEDNORM): adds EddySed clouds on top of the mednorm setup. |
| `/data2/peng/Recipe_DH_Tau_B/Cookbook.md` | Decision-tree flowchart + full retrieval-series registry (every job ID ever run, what changed, what was abandoned and why). Use this to trace *why* a given choice was made if the paper needs a justification beyond what's here. |
| `/data2/peng/Recipe_DH_Tau_B/observation_data_reduction_notes.md` | The **data reduction** section (upstream of this one) — extraction, telluric correction, flux calibration. §11-14 overlap with this document but are **STALE**: they still quote job 2528367 (pre-fix) as the primary result; job 2027997 (post-fix, same config) is the one verified against disk here and should supersede it. |
| `/data2/peng/recording_recipe.md` | Full chronological research log (4400+ lines). Only consult directly for something not already distilled here. |
| `/home/peng/.claude/projects/-data2-peng/memory/project_dhtaub.md` | Cross-session memory; same content as recording_recipe but pre-digested. Point-in-time, may lag behind disk — this document supersedes it for retrieval-framework facts. |

---

## 1. One-paragraph summary (for the Methods section opener)

We retrieve the atmospheric properties of DH Tau B from CRIRES+ K-band (K2166,
R≈100,000) spectra using a Bayesian framework built on **petitRADTRANS v3**
(pRT3; line-by-line radiative transfer, `opacity_sampling=3`) as the forward
model and **PyMultiNest** (nested sampling, `MultiNest`) as the sampler. The
forward model uses the Piette & Madhusudhan (2020) pressure-referenced
temperature profile, equilibrium chemistry (with three exceptions handled as
free parameters), free gravity or mass+radius, rotational and instrumental
broadening, and an optional Ackerman & Marley (2001) EddySed cloud model. The
two nights of data (2022-12-31, 2023-01-01) are fit **jointly** with a shared
parameter vector but independent per-night radial velocities, `ln L = ln
L(N1) + ln L(N2)`, each night evaluated on its own native (barycentric)
wavelength grid — no interpolation/co-addition is performed. Correlated noise
is modelled with a heteroscedastic Gaussian-process (GP) kernel per detector
chip (González Picos et al. 2025 formulation), implemented as a banded
Cholesky factorization for tractability. Three independent
calibration/likelihood strategies (absolute-flux Standard Gaussian; per-chip
Ruffio-marginalized-φ; per-chip-median-normalized Standard Gaussian) are run
as cross-checks of whether the absolute SINFONI flux calibration biases the
composition/kinematics results.

---

## 2. Software stack

- **Forward model**: petitRADTRANS v3 (`petitRADTRANS.radtrans.Radtrans`),
  opacities from `/net/lem/data2/pRT3_formatted` (set via
  `petitradtrans_config_parser.set_input_data_path`).
  - `line_opacity_mode='lbl'`, `line_by_line_opacity_sampling=3` (i.e. native
    R~1e6 grid sampled every 3rd point → effective input resolving power
    `in_res = 1e6/3 ≈ 333,333`).
  - Continuum opacity: `rayleigh_species=['H2','He']`,
    `gas_continuum_contributors=['H2--H2','H2--He']`.
  - Pressure grid: `np.logspace(-5, 1, 50)` (50 layers, 10⁻⁵–10 bar); extends
    slightly beyond the deepest P-T node (10^0.7≈5 bar) as a safety margin.
  - Wavelength boundary: min/max of the data ± 7 nm padding (for the RV
    shift), converted nm→µm for pRT3.
  - `Radtrans` atmosphere objects are pickle-cached per job ID
    (`atmosphere_objects_<job_id>_N<N_live>[_cloud_<species>].pickle`) since
    construction (opacity loading) is the single most expensive step (~1.4 GB
    HDF5 read for the equilibrium-chemistry table alone).
- **Sampler**: PyMultiNest (`pymultinest.run`), constant-efficiency mode
  (`const_efficiency_mode=True`, `sampling_efficiency=0.5`), MPI-parallel
  (`mpiexec -n 20`), `n_iter_before_update=10`.
- **Rotational broadening**: `PyAstronomy.pyasl.fastRotBroad` with a free
  limb-darkening coefficient ε (see §5).
- **Environment pinning**: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
  set before any MPI-using import (required for MPI-safe determinism/perf).

---

## 3. Forward-model components

### 3.1 Pressure–Temperature profile (Piette & Madhusudhan 2020)

Implemented in `pRT_spectrum.make_pt()`. Eight **fixed** pressure nodes,
customised for DH Tau B following Xuan et al. (2024) Table C1 — denser
sampling through the K-band photosphere (emission contribution function peaks
≈0.1 bar), coarser at the deep/upper extremes:

| Node index | log₁₀ P (bar) | P (bar) | Role |
|---|---|---|---|
| 0 (deepest)   | +0.7 | 5.0     | dT_1 interval starts here |
| 1             |  0.0 | 1.0     | dT_2 interval |
| 2             | −0.3 | 0.5     | dT_3 interval |
| 3 (**anchor**)| −0.7 | 0.2     | **T_anchor** — free parameter, ECF peak |
| 4             | −1.0 | 0.10    | dT_4 interval |
| 5             | −1.5 | 0.032   | dT_5 interval |
| 6             | −3.0 | 0.001   | dT_6 interval |
| 7 (shallowest)| −5.0 | 10⁻⁵    | dT_7 interval |

Temperature at each node is built outward from the anchor:
`T(node j) = T(node j+1) + dT_j` for nodes deeper than the anchor, and
`T(node j) = T(node j-1) - dT_{j-1}` for nodes shallower. **All seven `dT_i`
priors have a strictly non-negative (and mostly non-zero) lower bound**, which
alone enforces monotonicity — no explicit ordering constraint is coded. The
non-zero lower bounds (rather than a naive `[0, X]`) were added deliberately:
an earlier retrieval (job 2898115) showed `dT_3`/`dT_6` piling up at exactly 0,
i.e. isothermal segments, once `lb=0` allowed it.

Interpolation onto the 50-layer pRT pressure grid uses a **monotonic cubic
PCHIP spline** (`scipy.interpolate.PchipInterpolator`) — explicitly *not*
Gaussian-smoothed, per Rowland et al. (2023)'s finding that smoothing biases
retrieved P-T profiles, a recommendation Xuan et al. (2024, §4.3.4) also
follow.

### 3.2 Chemistry

Two mutually exclusive modes, selected by the constant parameter
`chemistry ∈ {'equilibrium', 'free'}`. **Equilibrium chemistry is the adopted
mode for all "current" retrieval series** (007PM-EQ-*, 008PM-EQ-*); free
chemistry (003PM-FR / 006PM-FR) was used only in early exploratory/validation
work (era 0–2, and the free-chemistry CCF validation cross-check).

**Equilibrium chemistry** (`pRT_spectrum.equilibrium_chemistry()`):
uses pRT3's pre-calculated equilibrium-chemistry table
(`PreCalculatedEquilibriumChemistryTable`, loaded once per process — a 1.4 GB
HDF5 read), interpolated over two free parameters:
- `C_H` ≡ [C/H] (dex relative to solar; free parameter directly)
- `C/O` (free parameter directly)
- internally, `[Z/H] = [C/H] - log10((C/O)/(C/O)_solar)` with
  `(C/O)_solar = 0.5495` (Lodders 2020) is what's actually passed to the table
  interpolator as `log10_metallicities`.

Three species are **not** present in the pRT3 equilibrium table and are
injected by hand after the table lookup:
1. **¹³CO** — derived from the table's ¹²CO mass fraction via a free parameter
   `log_12CO_13CO` (prior on log₁₀ of the ratio): `X(¹³CO) = X(¹²CO) ×
   (29.002355/28.009999) / ratio`. Default/solar reference ratio if the
   parameter is absent: 70 (Asplund et al. 2021).
2. **HF** — parameterised via `[F/H]` (free parameter `F_H`), following Picos
   et al. (2024) Eq. 3: `log VMR_HF = [F/H] + log10(n_HF/n_H)_solar +
   log10(2 × VMR_H2)`, with the solar reference `log10(n_F/n_H)_solar = -7.6`
   (Maiorca et al. 2014; Asplund et al. 2021).
3. **Na, Ca** (formerly also Ti) — direct free log-VMR parameters (`log_Na`,
   `log_Ca`), **historically** included but **removed from the model as of
   2026-07-07** (see §7 "Species-removal history" — important for reconciling
   which jobs include them).

**Free chemistry** (`pRT_spectrum.free_chemistry()`): each species gets a
direct `log_<species>` VMR free parameter (uniform in log VMR, typically
[-12,-2]); C/O and [C/H] are *derived* quantities computed from the summed
C/O/H budget across all included species (not free parameters themselves).
Solar reference `log10(n_C/n_H)_solar = -3.54` (Asplund et al. 2021).

**Species/line-list identity** (`EQ_SPECIES_PRT3` in the Guidebook, current
state — Na/Ca/Ti commented out):
H₂O (POKAZATEL), ¹²CO (HITEMP), ¹³CO (HITEMP), CH₄ (ExoMol MM), NH₃ (CoYuTe),
H₂S (AYT2), HCN (Harris), CO₂ (HITEMP), FeH (MoLLIST), HF (Coxon-Hajig/Wilzewski+2016)
— 10 line species total in the adopted (post-07-07) configuration.
`species_info.csv` (repo root) carries the mass/C-O-H stoichiometry table used
by both chemistry modes; `PRT3_SPECIES_OVERRIDES` in the Guidebook hard-codes
disambiguated line-list names to avoid interactive prompts crashing MPI jobs.

### 3.3 Clouds (optional — EddySed, Ackerman & Marley 2001)

Enabled by setting `constant_params['cloud_species']` to a non-empty list
(`CLOUD_SPECIES_DEFAULT = ['MgSiO3(s)_crystalline_000', 'Fe(s)_crystalline_000']`,
following Xuan et al. 2024 §4.3.3). Cloud mass fraction per species:
`X = X_eq(metallicity, C/O) × 10^(log_X)`, where `X_eq` is pRT3's
`return_cloud_mass_fraction` chemical-equilibrium cloud abundance and `log_X`
is the free scaling parameter (`log_X_MgSiO3`, `log_X_Fe`). The vertical
condensate profile is a step function at the condensation temperature via
`simple_cdf_free`. Structural parameters `fsed`, `log_Kzz`, `sigma_lnorm` are
**shared** across cloud species (pRT3's `calculate_flux` only accepts a
scalar `cloud_f_sed`, so a per-species fsed as in Xuan+2024 is not directly
supported — noted as a modelling simplification). Adds 5 free parameters.
Model comparison (ΔlnZ) between cloudy and cloud-free variants is a
first-class question in this project (see §8) — clouds are **not** currently
favoured by the evidence under either calibration strategy, but the two
MultiNest evidence estimators disagree on the *sign* of the preference (see
caveat in §9).

### 3.4 Gravity / mass / radius

Two mutually exclusive parameterisations, chosen by which keys are present in
`free_params`:
- **Absolute-flux mode** (007PM-EQ-GP, 007PM-EQ-CLD): `log_M` (Gaussian,
  Xuan+2024 evolutionary-model prior) and `log_R` (Gaussian, same source);
  gravity derived as `g = GM/R²` (cgs). `log_R` also sets the (R/d)² flux
  normalisation (§4).
- **Normalised/no-absolute-flux mode** (008PM-EQ-MEDNORM/RUFFIO/CLD-MEDNORM):
  `log_g` directly, Gaussian **N(3.64, 0.20)** (derived from Xuan+2024 Table 1
  M=12 M_Jup via g=GM/R²; **note**: the in-code comment inconsistently cites
  R=2.9 R_Jup for this derivation while the log_M/log_R Gaussian priors used
  elsewhere in the same file cite R=2.6±0.6 R_Jup — flag for the paper author
  to pick one consistent R and re-derive log_g's σ if this is scrutinised by a
  referee; the discrepancy is in a code comment, not in any actual fitted
  result). No M or R free parameter in this mode — R is simply not retrieved.

### 3.5 Kinematics and broadening

- `rv_N1`, `rv_N2` — independent topocentric radial velocity per night,
  Gaussian priors N(31.5, 1) km/s (current default in the Guidebook) — note
  the reduction-notes document (§12.2, possibly more current for this
  specific number) instead quotes slightly offset priors N(31.7, 0.5) for N1
  and N(31.9, 0.5) for N2, derived from the CCF RV-prior analysis
  (`tasting_RV_prior.ipynb`) and the measured inter-night barycentric drift
  (Δv_bary = −0.193 km/s); **verify which literal prior a specific job used
  from its own `final_params_dict.pickle`/script before quoting a number in
  the paper** — the Guidebook's `make_free_params_equilibrium()` default may
  not match what every historical driver script actually passed.
- `vsini` — uniform [1,20] km/s (equilibrium mode) or [0,20] (free-chem
  default).
- `epsilon` — linear limb-darkening coefficient for `fastRotBroad`, uniform
  [0,1] free parameter (physically bounded; historically was a fixed value
  0.5, then briefly a bad literature value −0.39 before being corrected —
  see `[[feedback_epsilon_sign]]` memory: valid range is 0 ≤ ε ≤ 1, correct
  Landman et al. (2024) literature value is ε=0.84, consistent with what the
  retrievals actually recover, ε≈0.5–0.85 across jobs).
- **RV shift** applied in the pRT rest-frame wavelength grid; **rotational
  broadening** via `fastRotBroad` on an evenly-resampled grid; **instrumental
  broadening** via Gaussian LSF convolution,
  `sigma_LSF = sqrt(1/out_res² - 1/in_res²) / (2√(2 ln 2))` converted to
  pixels via the local wavelength spacing.
  - **Known minor bug** (flagged in project memory, not yet fixed): `out_res`
    passed to `instr_broadening` is `self.instrument_res[i]`, which is
    `estimate_spectral_resolution(wave)` = median(λ/Δλ_pixel) — i.e. the
    **pixel-sampling** resolution (~3.0–3.3×10⁵), *not* the true instrumental
    resolving power (CRIRES+ K2166 nominal R≈100,000). This under-broadens
    (over-deepens) lines slightly. Quantified as negligible at the binnings
    relevant to the (separate, larger, still partly unexplained) line-depth
    deficit investigation — see §9 — but is a genuine, unfixed inconsistency
    worth a one-line caveat if the paper discusses line depths in detail.

---

## 4. Data ingestion and normalisation strategies

Three calibration/normalisation strategies are used across the retrieval
series, each paired with a specific likelihood mode (this pairing is not
arbitrary — see the derivation in §5):

| Strategy | Data file | Model treatment | Likelihood | Series |
|---|---|---|---|---|
| **Absolute flux** | `extracted_spectra_combined_flux_cal.npy` (W m⁻² μm⁻¹, SINFONI-calibrated) | Model × (R/d)² (Gaia DR3 d=135.2 pc), no normalisation | Standard Gaussian, φ=1 fixed | 007PM-EQ-GP (**benchmark**), 007PM-EQ-CLD |
| **Ruffio marginal φ** | `extracted_spectra_combined_sigmaclipper*.npy` (uncalibrated, sigma-clipped) | Data only divided by per-chip median before `Target` creation; model left as pRT native units | Ruffio+2019 marginalised-φ (one φ per (order,detector) chip, analytically marginalised) | 008PM-EQ-RUFFIO |
| **Per-chip median (mednorm)** | Same sigmaclipper files | **Both** data and model divided by their own per-(order,detector)-chip median (`normalize_flux='per_chip_median'`, applied inside `_load_night()` for data and `pRT_spectrum._apply_normalization()` for the model) | Standard Gaussian, φ=1 fixed | 008PM-EQ-MEDNORM, 008PM-EQ-CLD-MEDNORM, 008PM-EQ-MEDNORM-NO13CO |

**Why the phi/normalisation pairing is not interchangeable** (this is a real
methodological point worth stating explicitly in the paper): under Ruffio's
analytic optimum `φ = (mᵀC⁻¹d)/(mᵀC⁻¹m)`, rescaling the model by any fixed
per-chip constant k is a no-op — `m'=m/k` gives `φ'=kφ` so `φ'm' ≡ φm`
identically. So normalising the model in addition to the data under a
Ruffio-φ likelihood would do nothing. Conversely, once **both** sides are
median-normalised to 1, an additional marginalised φ would spend
`N_φ=15` degrees of freedom "fitting" a factor that is fixed by construction
— hence per-chip-median normalisation is deliberately paired with the
**fixed**-φ Standard Gaussian, never with Ruffio marginalisation. (Full
derivation: `tasting_retrieval_equa_chem_v6.1_piette_cloudfree_mednorm.py`
docstring, lines 40–57.)

Also available but not currently used in any active series: `'savgol'`
(per-chip Savitzky-Golay continuum division, window=301 px, polyorder=2) and
`'median'` (single global median) — these were the Era 1–3 (pre-GP,
pre-flux-cal) normalisation choices; see Cookbook.md registry for that
history. `chip_gap_nm=0.5` nm (used by `CovarianceGP` to split contiguous
pixel runs into sub-chips) is unrelated to these — see §6.

The **purpose of running three calibration strategies in parallel** is a
sensitivity/robustness test: if composition and kinematics parameters agree
across all three, that is evidence the absolute SINFONI flux calibration
(which has one documented systematic — the det2/ord0 chip, see
`observation_data_reduction_notes.md` §11 — plausibly biased 10-18% high) is
not distorting the science. **The result of this test is genuinely mixed, not
a clean confirmation** — see §8/§9.

---

## 5. Likelihood functions

Implemented in `LogLikelihood.__call__` (`Guidebook_GAStronomy_Piette_v4.2.py`
lines ~788–992). Three modes, selected by `scale_flux`:

1. **`scale_flux=False`** (or `'physical'` for the absolute-flux case) —
   **Standard multivariate Gaussian**, φ=1 fixed:
   `ln L = -0.5 N_d ln(2π) - 0.5(ln|Σ| + χ²₀)`, `χ²₀ = rᵀΣ⁻¹r`. Used by both
   the absolute-flux benchmark and the mednorm series.
2. **`scale_flux=True`** — **Ruffio et al. (2019) marginalised-φ likelihood**,
   one φ per (order, detector) chip (`N_φ = n_orders × n_dets = 15`):
   `ln L = -0.5(N_d-N_φ)ln(2π) + lnΓ(0.5(N_d-N_φ+α-1)) - 0.5(ln|Σ| +
   ln(mᵀΣ⁻¹m) + (N_d-N_φ+α-1)ln χ²₀)`, α=2 (Ruffio+2019 hyperparameter). Used
   by 008PM-EQ-RUFFIO.
3. **`scale_flux=None/'Single'`** — same Ruffio form but with a single global
   φ (`N_φ=1`) — implemented but not used by any current active series.

An error-scaling factor `s2 = sqrt(χ²₀/N_d)` is applied post-hoc when
`scale_err=True` (i.e. whenever `scale_flux is not False`) to rescale the
reported data errors to match the realised chi-square — **note this means
`s2` is only meaningfully ≠1 for Ruffio-mode runs**; all Standard-Gaussian
runs report `s2=1.0` by construction (verified: every absolute-flux and
mednorm job checked has `s2=1.0000` exactly in its `final_params_dict.pickle`).

### 5.1 GP correlated-noise covariance (González Picos et al. 2025, §4.2)

`cov_mode='GP'` (used by **every** current active series). Model:

Σ₀,ᵢⱼ = δᵢⱼσᵢ² + a²·((σᵢ+σⱼ)/2)²·exp(−Δλᵢⱼ²/2l²)

with **two free parameters**, `log_a ∈ [-1,1]` (GP amplitude relative to
photon noise) and `log_l ∈ [-3,0]` (GP length scale in nm, 0.001–1 nm range).
Applied **per detector chip** (block-diagonal across chip boundaries, so
cross-chip correlations are structurally zero, not just numerically small).
Recovered posterior values are essentially identical across every job checked
(`log_a ≈ 0.266–0.267`, `log_l ≈ −2.28 to −2.29`, i.e. `a≈1.85`,
`l≈0.0052 nm`) regardless of calibration strategy or cloud model — this GP
amplitude/length-scale pair appears to be a robust, calibration-independent
feature of the noise, not a fitting artefact of any one series (cross-checked
independently against a nod-difference RMS excess in
`cooking_reduction_diagnostics.ipynb`, √(1+a²)≈2.1 matching the ~2.35–2.5×
nod A−B RMS/σ excess measured directly from the reduced 2D frames — i.e. this
is very likely tracing a genuine reduction-level (PSF-subtraction/background)
systematic, not overfitting).

**Computational implementation** (relevant for a Methods "numerics"
paragraph, or just for whoever needs to defend runtime/scaling choices):
dense N×N Cholesky is replaced by a **banded Cholesky**
(`scipy.linalg.cholesky_banded`/`cho_solve_banded`), bandwidth
`B = min(ceil(5l/δλ), 100)` pixels (hard cap), truncation error
`< exp(−12.5) ≈ 3.7×10⁻⁶` (negligible vs float64 precision). This gives an
~18× memory reduction and ~300× Cholesky-cost reduction per chip versus the
dense approach, which was necessary to avoid OOM/timeout at the ~50k-pixel,
two-night, MPI-20-rank scale of this problem. **This is purely a numerics/
performance detail with no effect on the reported science** — worth at most
one sentence in the paper (e.g. "...evaluated via a banded Cholesky
factorization for computational tractability (bandwidth capped at 100 pixels,
truncation error <4×10⁻⁶)"), not a subsection.

**Four correctness bugs were found and fixed** in this GP implementation
during development (2026-06-12); listed here only so the paper text doesn't
need to relitigate them, but they are resolved and do not affect any of the
posterior numbers quoted in §8 (all of which post-date the fixes):
wrong Cholesky `lower` flag; missing `a²` self-covariance term on the
diagonal; a chip-boundary detector that only caught positive Δλ jumps
(missed inter-order boundaries because the wavelength array decreases across
orders); and a silently-swallowed `LinAlgError` in the MultiNest callback
that would have returned garbage instead of −∞ on a failed Cholesky. Full
detail in memory file `project_guidebook_gp_bugs.md` if needed.

---

## 6. Two-night joint retrieval architecture

`PMN_lnL` in `Retrieval` evaluates `ln L = ln L(N1) + ln L(N2)` — one
`pRT_spectrum` object per call (the expensive pRT radiative-transfer step is
cached on that instance via `_make_prt_flux()`, so it runs **once** per
likelihood evaluation even though `make_spectrum()` is called twice, once per
night, with the night's own `rv_key` and `out_res`). Each night keeps its own
native (barycentric-corrected) wavelength grid — **no interpolation or
co-addition**, precisely to avoid introducing correlated noise from
resampling (a combined/co-added spectrum was prototyped and validated to work
correctly — √2 SNR gain achieved — but set aside for this reason, not a
data-quality problem; see `observation_data_reduction_notes.md` §15 point 6).
This is equivalent to a simultaneous fit of both datasets under one shared
parameter vector (all atmospheric/kinematic-except-RV parameters are shared;
`rv_N1`/`rv_N2` are the only genuinely per-night free parameters; `vsini` and
`epsilon` are shared, consistent with them being intrinsic stellar/companion
properties rather than per-observation quantities).

---

## 7. Retrieval series naming decoder + species-removal history

Series names follow `<NNN><PM>-<CHEM>[-<VARIANT>]`, e.g. `007PM-EQ-GP`:

- `NNN` — sequential era number (007, 008, ...), see Cookbook.md registry.
- `PM` — "Piette-Madhusudhan" P-T parameterisation (used by every current
  series; distinguishes from earlier gradient-T-P eras).
- `EQ`/`FR` — equilibrium vs free chemistry.
- `-GP` — GP correlated-noise covariance (as opposed to a plain diagonal
  covariance, used only in early/legacy jobs).
- `-CLD` — EddySed clouds added.
- `-RUFFIO` — Ruffio marginalised-φ likelihood variant.
- `-MEDNORM` — per-chip-median (data+model) normalisation variant.
- `-NO13CO` — ¹³CO removed as a model-comparison test.

**Species-removal history (important for reconciling which job used which
model)**: Ti was removed from the equilibrium-mode species/free-parameter set
on **2026-06-07** (cloud retrievals were failing with it included; not
re-investigated since — treat as settled). Na and Ca were removed on
**2026-07-07**, confirmed via an explicit A/B test (job 2968924 [with] vs
3497146 [without], and the cloud-model equivalent 3160171 vs 3708914) that
**removing them changes nothing** — every shared parameter shifted <1σ,
ΔlnZ consistent with zero within ~2σ across both MultiNest evidence
estimators. **The currently-adopted model (as encoded in the Guidebook file's
present state, and used by every job from 3497146/3708914/170416 onward)
excludes Na and Ca.** However: **the quoted absolute-flux benchmark, job
2027997, and its cloud-model companion 2204420/2499181, all predate this
removal and still include `log_Na`/`log_Ca` as free parameters** (confirmed
directly: `2027997`'s `final_params_dict.pickle` has 22 keys including both).
**This is a real, unresolved inconsistency for the paper**: the "final"
adopted model has 19 free params (cloud-free) / 24 (cloudy) without Na/Ca,
but the only completed run of the absolute-flux/Standard-Gaussian/GP
configuration still has them. Two options for whoever finalizes the paper:
(a) re-run 007PM-EQ-GP without Na/Ca for full consistency (cheap given the
demonstrated A/B-null result under mednorm — likely <1σ shift expected), or
(b) explicitly state in the paper that Na/Ca were retrieved but found
consistent with upper limits / not used further, and quote the benchmark
as-is. Given the A/B result was null under the *other* calibration strategy,
(b) is defensible without a re-run, but this should be a conscious choice,
not an oversight.

---

## 8. Verified benchmark numbers (read directly from disk, 2026-07-16)

All values are posterior medians with 16th/84th-percentile uncertainties from
`final_posterior.npy`, computed directly for this document (not copied from
earlier log paraphrases, which in a couple of cases used slightly different
central-tendency conventions — e.g. mean vs median — and are superseded by
the numbers below). χ² values are the GP-covariance-weighted reduced
chi-square (`chi2_0_red` = `chi2_0/N_d`), **not** a diagonal-only reduced
chi-square (diagonal-only χ² is ~4–5× larger — see
`tasting_retrieval_results_analysis_4.ipynb` header note — because the GP
kernel absorbs ~78% of residual variance under this noise model; **be careful
which χ² definition the paper quotes**).

### 8.1 Primary benchmark — job 2027997 (007PM-EQ-GP: absolute flux, Standard Gaussian, GP, cloud-free, equilibrium chem, WITH Na/Ca — 22 free params, N_live=600, 5054 posterior samples)

| Parameter | Median | +1σ | −1σ |
|---|---|---|---|
| rv_N1 (km/s) | 31.641 | 0.096 | 0.089 |
| rv_N2 (km/s) | 31.546 | 0.088 | 0.088 |
| vsini (km/s) | 7.539 | 0.138 | 0.152 |
| ε (limb darkening) | 0.820 | 0.096 | 0.133 |
| log M/M_Jup | 1.213 | 0.032 | 0.038 |
| log R/R_Jup | 0.479 | 0.006 | 0.005 |
| T_anchor (K, at 0.2 bar) | 2076 | 25 | 27 |
| [C/H] (= C_H) | −0.030 | 0.086 | 0.070 |
| C/O | 0.640 | 0.021 | 0.018 |
| log(¹²CO/¹³CO) | 1.692 | 0.112 | 0.112 |
| [F/H] | −0.174 | 0.237 | 0.253 |
| log VMR Na | −4.393 | 0.221 | 0.252 |
| log VMR Ca | −4.936 | 0.236 | 0.306 |
| log a (GP) | 0.2658 | 0.0011 | 0.0011 |
| log l (GP, nm) | −2.2848 | 0.0014 | 0.0015 |

Derived: M = 16.3 M_Jup, R = 3.02 R_Jup, g_derived ≈ 4450 cm/s² (log g ≈
3.648 — consistent with the log_g≈3.64 Gaussian prior mean used in the other
calibration strategies, a useful internal-consistency check). ¹²CO/¹³CO =
49.2. χ²(N1)=1.041, χ²(N2)=0.962. lnZ (nested-sampling global log-evidence,
as stored in `final_params_dict['lnZ']`) = 1,762,366.6 (this is the
**importance nested sampling** value; see §9 caveat on NS vs INS before
quoting a Bayes factor against another job).

### 8.2 Cloud comparison, absolute flux — jobs 2204420 (N=600) / 2499181 (N=1000, robustness re-run)

Both agree with 2027997 (cloud-free) on every shared parameter to well within
1σ (e.g. C/O 0.641/0.657 vs 2027997's 0.640; T_anchor 2128/2098 K vs 2076 K).
Cloud parameters (2499181): log_X_MgSiO3 = −0.306, log_X_Fe = −1.300,
fsed = 5.69, log_Kzz = 11.05, sigma_lnorm = 1.99. **lnZ difference is small
and its sign flips between estimators** — see §9; do not quote a cloud-vs-
cloud-free Bayes factor without picking one estimator and saying so
explicitly.

### 8.3 Calibration-sensitivity series — mednorm, WITHOUT Na/Ca (adopted model) — job 3497146 (008PM-EQ-MEDNORM, sigmaclipper, per-chip-median norm, Standard Gaussian, GP, log_g direct, 19 free params, N_live=600, 4970 samples)

| Parameter | Median | +1σ | −1σ |
|---|---|---|---|
| rv_N1 (km/s) | 31.461 | 0.077 | 0.073 |
| rv_N2 (km/s) | 31.506 | 0.100 | 0.074 |
| vsini (km/s) | 5.936 | 0.212 | 0.199 |
| ε | 0.542 | 0.196 | 0.223 |
| T_anchor (K) | 1856 | 58 | 49 |
| [C/H] | −0.306 | 0.141 | 0.131 |
| C/O | 0.581 | 0.024 | 0.026 |
| log(¹²CO/¹³CO) | 2.104 | 0.120 | 0.115 |
| [F/H] | −0.920 | 0.386 | 0.346 |
| log g | 3.678 | 0.095 | 0.108 |
| log a (GP) | 0.2670 | 0.0011 | 0.0011 |
| log l (GP, nm) | −2.2890 | 0.0014 | 0.0016 |

¹²CO/¹³CO = 127.0. χ²(N1)=1.042, χ²(N2)=0.960. lnZ (INS) = −18,064.29.

**Head-to-head with the absolute-flux benchmark (§8.1)** — this is the
central open question of the calibration-sensitivity program and should be
stated candidly in the paper, not smoothed over:

| Quantity | Abs. flux (2027997) | Mednorm (3497146) | Agreement? |
|---|---|---|---|
| rv_N1, rv_N2 | 31.64, 31.55 | 31.46, 31.51 | ✅ robust, <0.3σ-ish shifts |
| vsini | 7.54 | 5.94 | ❌ ~7–8σ formal disagreement |
| C/O | 0.640 | 0.581 | ⚠️ ~2σ, moderate tension |
| [C/H] | −0.030 | −0.306 | ❌ ~0.28 dex apart, several σ |
| ¹²CO/¹³CO | 49.2 | 127.0 | ❌ ~2.6× — large, **still unexplained** |
| T_anchor | 2076 K | 1856 K | ❌ several σ |

The ¹²CO/¹³CO discrepancy is the sharpest single number. CCF validation
(`tasting_retrieval_validation_clean.ipynb` §14, GP-independent, using the
retrieved best-fit model as a matched-filter template) shows H₂O, ¹²CO, and
"totalCO" are all robust rv=0 detections in the mednorm run, but **¹³CO is NOT
a CCF detection** (peaks at rv≈−543 km/s, nowhere near the planet), while a
model-comparison run (170416, ¹³CO removed entirely) still shows **moderate
evidence FOR including ¹³CO** in the fit (ΔlnZ = +4.90±0.29 NS / +3.36±0.51
INS versus the same model without it). In other words: ¹³CO is statistically
required by the mednorm data, its best-fit ratio is likely biased high by
~2.5×, and it is not independently confirmed by a direct CCF — **the discrepancy
is a property of the mednorm calibration strategy interacting with the
weakest-lined species in the fit, not a sign that ¹³CO isn't present at all**.
Candidate mechanisms considered but not yet confirmed: GP-normalisation
interaction, per-chip noise-weighting differences. **This should be reported
in the paper as an open systematic** affecting the isotopologue ratio
specifically, with the absolute-flux value (49±13, CCF-confirmed at the
correct RV) recommended as the primary quoted ¹²CO/¹³CO result, and the
mednorm value reported as a systematics-affected cross-check, not averaged
together.

The vsini and T_anchor/[C/H] shifts are less thoroughly chased down;
Cookbook.md's 5-way and later comparisons found "every disagreement splits
along the calibration axis (absolute flux vs per-chip median); the cloud axis
and atomic-species axis are inert" — i.e. whatever is driving these
differences is specifically the calibration strategy, not clouds or Na/Ca.

### 8.4 Comparison — 008PM-EQ-RUFFIO, job 657632 (per-chip Ruffio φ, flux_cal data but PerChipScaleTrue — note this uses the *flux_cal* files with Ruffio normalisation applied on top, per the folder name `NormNone_PerChipScaleTrue`, not the sigmaclipper files described in the original Cookbook plan — verify data source against this job's own driver script before citing, since Cookbook.md's Era-8 table describes a sigmaclipper-based plan that may not match what this specific completed job actually ran)

C/O = 0.512, [C/H] = −0.274 (using the params_dict's own `[C/H]` key, which
differs slightly from `C_H` here — check both fields), ¹²CO/¹³CO = 50.9,
vsini = 6.90 km/s, χ²(N1) = 1.87, χ²(N2) = 1.74 (notably worse fit than the
Gaussian-likelihood series — consistent with s2=1.369 ≠ 1, i.e. this run's
own error-rescaling found the nominal errors needed inflating by ~37%,
whereas both Gaussian-likelihood series show s2=1.0 exactly by construction).
**This job's C/O sits between the absolute-flux and mednorm values** — worth
noting as a third data point in the calibration-sensitivity discussion, but
flag the data-source ambiguity above before treating it as a clean sigmaclipper
cross-check.

---

## 9. Known caveats / open issues relevant to the Methods or Discussion section

1. **NS vs INS evidence estimators disagree by ~20–30 nats and can flip the
   sign of a model-comparison conclusion.** `stats['nested sampling global
   log-evidence']` (NS) and `stats['nested importance sampling global
   log-evidence']` (INS, what `PMN_analyse()` actually stores as
   `self.lnZ`/`params_dict['lnZ']`) are genuinely different estimators, not
   interchangeable read-outs of "the" evidence. Concretely, for the cloud-vs-
   cloud-free comparison under mednorm: NS gives cloud−cloudfree ≈ +1 nat
   (indifferent), INS gives ≈ −9 nats (moderately against clouds). **Before
   the paper quotes any Bayes factor, pick one estimator explicitly and say
   which** — do not silently mix NS-derived numbers from one log entry with
   INS-derived numbers from another (this exact mixing happened during
   development and was caught only in retrospect).
2. **det2/ord0 flux-calibration chip (absolute-flux mode only)**: one of 15
   chips (2457–2472 nm, ~7% of the K-band bandpass) has its SINFONI/CRIRES+
   scale factor frozen flat rather than fit (below the 20% SINFONI-coverage
   threshold), plausibly biasing that chip's flux level 10–18% high. No
   independent photometric anchor exists redward of 2290 nm (2MASS K_s) to
   check it directly. Expected impact: sub-percent on log_R (1/15 chips,
   lower SNR), negligible on line-shape-driven parameters — but **not yet
   demonstrated with an explicit det2/ord0-masked sensitivity retrieval**.
   Full derivation in `observation_data_reduction_notes.md` §11.
3. **`out_res` LSF bug** (§3.5): pixel-sampling resolution used in place of
   true instrumental R in the LSF convolution. Quantified as a small,
   negative-going (over-deepening) effect, not the cause of the line-depth
   deficit in point 4. Unfixed; low priority given its demonstrated small
   size, but should not be silently assumed away if a referee asks about
   line-depth accuracy specifically.
4. **Global line-depth deficit (unresolved, pre-dates GP)**: models
   under-predict data RMS by roughly a factor of 2–3 across essentially every
   order (15–28% of data RMS captured natively, climbing to ~30–65% after
   heavy binning but still not closing fully), for **both** H₂O- and
   ¹²CO-dominated regions, in **both** a 2026-06-05 pre-GP diagonal-covariance
   retrieval (368546) and the current GP/mednorm retrievals. A uniform
   order-of-magnitude single-species abundance boost (H₂O ×15-30, ¹²CO ×7-10)
   closes the gap in isolated tests, suggesting the shared equilibrium-
   chemistry (C_H, C/O) parameterization (and/or the shared T-P profile) is
   being pulled to a joint compromise across a heterogeneous many-order fit,
   rather than any single species/band being poorly modelled. **Root cause
   not yet identified** — flagged here so the paper doesn't need to resolve
   it, but any by-eye spectral-fit figure in the paper should be captioned
   honestly about residual line-depth under-prediction rather than implying a
   perfect fit. Full trail: `recording_recipe.md` 2026-07-11 entries (note
   there is an explicit self-correction in the log that day — the first
   write-up wrongly attributed most of this to the GP kernel; a direct
   pre-GP-era check disproved that attribution — read both entries, not just
   the first, if tracing this history).
5. **Nod A−B residual broadband noise excess** (RMS/σ ≈ 2.35–2.5, matching
   the GP amplitude √(1+a²)≈2.1 to ~15–20%): circumstantial evidence the GP
   kernel is absorbing a genuine reduction-level systematic (PSF-subtraction
   or background residual), not simply overfitting. Consistent with, but not
   proven to be, related to point 4 above — do not conflate them in the
   paper without the caveat that the connection is circumstantial.

---

## 10. MultiNest configuration reference

| Setting | Value |
|---|---|
| N_live_points | 600 (cloud-free); 1000 (cloudy — more free params) |
| evidence_tolerance | 0.5 |
| sampling_efficiency | 0.5 |
| const_efficiency_mode | True |
| n_iter_before_update | 10 |
| Parallelisation | MPI, 20 ranks (`mpiexec -n 20`) |
| ln L cap | `min(ln_L, 1e10)` — prevents Fortran underflow in MultiNest importance weights at extreme-precision absolute-flux likelihoods (N~50k pixels, σ~1.5e-16 W/m²/μm ⇒ normalisation constant alone ~1.77e6); an earlier cap of 1e6 was too low and caused instant/flat convergence |
| ln L failure value | −1e101 (any exception, non-finite ln L, or Cholesky `LinAlgError`) |

---

## 11. File map (for pulling figures / additional numbers)

- **Corner plots**: `<output_dir>/final_cornerplot.pdf` (also `.png` and
  `_rasterized.pdf` fallbacks for the 22+ param plots that some PDF viewers
  render incorrectly — verified as a viewer limitation, not file corruption).
- **Spectrum fit figures**: `<output_dir>/retrieval_data_model_spectrum.png`,
  `retrieval_data_model_residuals.png`, `retrieval_PT_profile.png`.
- **Results-analysis notebooks** (posterior comparisons, CCF validation,
  benchmark tables — likely the fastest source of any additional derived
  number needed): `Reminisce/tasting_retrieval_results_analysis_3.ipynb`
  (absolute-flux + mednorm + Ruffio comparisons through job 3708914),
  `Reminisce/tasting_retrieval_results_analysis_4.ipynb` (sigmaclipper/
  mednorm-only benchmark with the `plot_model_vs_data_scatter` spectrum-fit
  figure function, derived from `analysis.py`, intended for paper figures),
  `Reminisce/tasting_retrieval_validation_clean.ipynb` (per-species CCF
  validation, §10–§15 cover all post-fix benchmarks).
- **Paper figures already prepared**: `Reminisce/paper_figures/` (flux
  calibration figure, companion-trace/PSF figure); `Reminisce/plot_paper.ipynb`
  is the source notebook.
- **Reduction diagnostics** (if the paper needs to defend the noise model or
  answer "did you check the reduction wasn't at fault"):
  `Reminisce/cooking_reduction_diagnostics.ipynb`.

---

## 12. Reference list for this section (BibTeX keys are illustrative; verify against the paper's actual .bib)

- Piette, A. A. A. & Madhusudhan, N. (2020). MNRAS 497, 5136. — P-T parameterisation.
- Xuan, J. W. et al. (2024). ApJ 970:71. — DH Tau b system parameters (M, R), P-T node/prior customisation, cloud priors (Table 3), P-T priors (Table C1).
- Rowland, M. J. et al. (2023). — Gaussian-smoothing bias in P-T retrievals (justifies PCHIP-only, no smoothing).
- Lodders, K. (2020). — Solar (C/O)_solar = 0.5495.
- Asplund, M. et al. (2021). — Solar log(C/H), ¹²C/¹³C=70 reference values.
- Picos et al. / González Picos, D. et al. (2024). A&A, Survey II. — Atomic species (HF, Na, Ca, Ti) treatment, §4.2, Table 3; per-chip mean-flux normalisation, §3.2.
- Maiorca et al. (2014). — Solar [F/H] reference.
- González Picos, D. et al. (2025). A&A 693, A298. — GP correlated-noise covariance kernel, §4.1–4.2.
- de Regt et al. (2024). A&A 688, A116. — No-data-normalisation / per-chip-φ likelihood formulation, §3.2 (equivalent GP kernel formulation).
- Ruffio, J.-B. et al. (2019). ApJ 881, 1. — Marginalised-φ likelihood, Eq. A1–A6.
- Ackerman, A. S. & Marley, M. S. (2001). ApJ 556, 872. — EddySed cloud model.
- Landman, R. et al. (2024). — Limb-darkening coefficient ε=0.84 for rotational broadening.
- Gaia DR3 — d(DH Tau) = 135.2 pc.

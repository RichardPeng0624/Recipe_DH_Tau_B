# DH Tau B — Observations and Data Reduction Notes

**Purpose**: Reference document for a Claude agent writing the "Observations and Data Reduction" section of the DH Tau B atmospheric retrieval paper.
**Standard reduction workflow**: Raw 2D frames → PSF subtraction → blaze correction → sigma-clipping → telluric correction → barycentric correction → A+B nod combination → **absolute flux calibration** → joint two-night retrieval. The flux-calibrated spectra are the primary science data products.
**Working directory**: `/data2/peng/Recipe_DH_Tau_B/`
**Data root**: `/data2/peng/` (per-night subdirectories `2022-12-31/` and `2023-01-01/`)
**Primary log**: `/data2/peng/recording_recipe.md`
**Notebooks**: `Cooking/cooking_subtraction.ipynb` (Night 1), `Cooking/cooking_subtraction_0101.ipynb` (Night 2), `Cooking/cooking_combine_spectra.ipynb` (Night 1), `Cooking/cooking_combine_spectra_0101.ipynb` (Night 2), `Cooking/cooking_combine_spectra_inter_night.ipynb` (barycentric-correction computation + prototype two-night combination, both nights)

---

## Pipeline Overview

| Step | Section | Notebook / Script | Output |
|------|---------|-------------------|--------|
| 1. 2D frame reduction | §3 | excalibuhr | `Extr2D_PRIMARY__COMBINED_*.npz` |
| 2. Wavelength solution | §4 | CRIRES+ pipeline | `WLEN_K2166_*.fits` |
| 3. PSF subtraction + optimal extraction | §5 | `cooking_subtraction.ipynb` | `extracted_spectra_position_{A,B}_sigmaclipper.npy` |
| 4. Blaze correction | §6 | `cooking_subtraction.ipynb` | (in-memory; applied before saving) |
| 5. Sigma-clipping | §7 | `cooking_subtraction.ipynb` | `extracted_spectra_position_{A,B}_sigmaclipper.npy` |
| 6. Telluric correction | §8 | `cooking_combine_spectra.ipynb` | `extracted_spectra_position_{A,B}_masked.npy` |
| 7. Barycentric correction | §9 | `cooking_combine_spectra_inter_night.ipynb` | `barycentric_wavelengths_night{1,2}.npy` |
| 8. A+B nod combination | §10 | `cooking_combine_spectra.ipynb` | `extracted_spectra_combined_sigmaclipper.npy` |
| **9. Absolute flux calibration** | **§11** | `cooking_combine_spectra.ipynb` | **`extracted_spectra_combined_flux_cal.npy`** ← **primary product** |
| 10. CCF / RV validation | §12 | `tasting_RV_prior.ipynb` | RV prior for retrieval |
| 11. Joint two-night retrieval | §13 | `tasting_retrieval_equa_chem_v6.1_piette_cloudfree.py` | Posteriors in `retrievals/{job_id}/` |

---

## 1. The System: DH Tau A+B

DH Tau is a young T Tauri star in the Taurus star-forming region (~140 pc). DH Tau B is a directly-imaged substellar companion (super-Jupiter / brown dwarf):

- Projected separation from primary: **2.3 arcsec** → ~39 pixels at CRIRES+ plate scale of 0.059 arcsec/pixel
- DH Tau A: K ≈ 8.5 mag (primary star); DH Tau A is ~250× brighter than DH Tau B
- DH Tau B: K_s ≈ 14.19 mag (Patience et al. 2012, Table 2, 2MASS)
- Estimated T_eff: ~2000–2200 K; log g ~3.5–4.0
- Systemic radial velocity: ~31–32 km/s (barycentric; from CCF analysis, see §9)

---

## 2. Observations

### Instrument and Setting

- **Instrument**: CRIRES+ at VLT-UT3 (Paranal Observatory)
- **Setting**: K2166 (K-band), slit width **0.4 arcsec** (`w_0.4`)
- **Detector array**: 3 Hawaii-2RG detectors, 2048 × 2048 pixels; plate scale 0.059 arcsec/pixel
- **Spectral format**: 7 echelle orders imaged per detector. Only **5 orders are used** (echelle orders 23–27, array indices 0–4). Orders 5–6 fall outside the useful wavelength range.
- **Wavelength coverage (used)**: 2063.7–2472.4 nm (K-band), split across 5 × 3 = **15 chips**
- **Spectral resolving power**: R ≈ 100,000 (estimated per night from the wavelength solution as median(λ / Δλ_pixel) across all chips)
- **Pixel spacing**: ~0.05 nm/pixel

### Observing Log

Two consecutive nights were observed:

| Night | UT Date | Science frames | DIT (s) | Total exp. | Airmass range | Seeing (arcsec) |
|---|---|---|---|---|---|---|
| Night 1 | 2022-12-31 (starts 2023-01-01T02:56–04:14) | 16 | 300 | 4800 s (~1.33 h) | 1.612–1.856 | 0.55–1.18 |
| Night 2 | 2023-01-01 (starts 2023-01-02T00:45–01:52) | **14** | 300 | 4200 s (~1.17 h) | 1.611–1.799 | 0.39–0.76 |

(Ranges verified against `header_info.txt` DPR CATG=SCIENCE rows in each night's data folder; Night 2's airmass/seeing range is materially lower than Night 1's — the two nights are **not** photometrically/seeing-equivalent, contrary to an earlier version of this table that listed identical ranges for both.)

Night 2 has 14 frames (not 16) because the last 2 frames were acquired at increasing airmass and seeing > 1 arcsec. Combined on-source integration: **30 frames × 300 s = 9000 s (~2.5 h)**.

- **Nodding pattern**: ABBA (standard CRIRES+ nodding), jitter = 0.0 arcsec (no dithering)
- **NABCYCLES**: 18 nod cycles defined per night (8 complete ABBA cycles per night were actually obtained)
- **Target identifier in headers**: "V DH Tau A+B center" (slit centered between primary and companion)
- **Object RA/Dec (J2000)**: 04h29m41.56s, +26°32′56.2″ = RA 67.4232°, Dec 26.5490° (SIMBAD; Taurus, ~140 pc)

### Telluric Standard Star

- **Star**: χ Tau (Chi Tau), K2166, `w_0.4` slit
- **Each night has its own genuine χ Tau observation**, taken immediately after that night's DH Tau B science block:

| Night | UT date/time | Frames | DIT | Nodding | Effective airmass | Seeing (arcsec) |
|---|---|---|---|---|---|---|
| Night 1 | 2023-01-01T04:21–04:27 (~8 min after science ended 04:13:38) | 4 | 20 s | ABBA (1 cycle) | 1.9038 (simple mean of 4 AIRM_END values: 1.9050, range 1.899–1.911) | 0.89–1.14 |
| Night 2 | 2023-01-02T00:30–02:05 (bracketing science block 00:45–01:52) | 6 | 100 s | — | 1.7251 (range 1.567–1.814) | — |

**Correction history**: `chi_tau/2022-12-31/` originally (erroneously) contained a byte-for-byte duplicate of the Night 2 raw data and reduction — an earlier version of this document incorrectly concluded from that duplication that only one χ Tau epoch existed for the whole run. The genuine Night 1 χ Tau raw frames (ESO archive templates `CRIRES_SPEC_OBS001_0034–0037.fits`, downloaded via the ESO archive on 2026-07-02) were not in the local file tree until they were located, downloaded, and reduced through the same excalibuhr `CriresPipeline` recipe chain used for Night 2 (see `Cooking/cooking_chi_tau_2022-12-31.ipynb`), overwriting the erroneous duplicate. Night 1's own `TELLURIC_DATA.fits` (43008 points, 1921.3–2472.4 nm) now exists at `/data2/peng/2022-12-31/chi_tau/2022-12-31/out/molecfit/TELLURIC_DATA.fits`.
- **Science target effective airmass**: Night 1 = 1.7007; Night 2 = 1.6946 (printed values; rounds to the 1.701 / 1.695 used in the airmass-rescaling formula below)

**RESOLVED (2026-07-02/03)**: Because Night 1's telluric template was wrong (silently reused Night 2's) until the correction above, the Night 1 telluric-corrected products (`extracted_spectra_position_{A,B}_masked.npy` and everything downstream — combined spectra, flux-calibrated spectra) had been produced with the **wrong** telluric template and airmass-rescaling factor (α = 0.9858 using the wrong Chi Tau airmass 1.725, instead of the correct α = 0.8933 using 1.9038 — see §8.2). A comparison of the two candidate templates (`compare_chi_tau_telluric_night1_night2.py`) showed the error was coherently structured within H₂O absorption bands (up to several percent in flux, driven by a real 3.4× difference in precipitable water vapor between the two nights — not just an airmass effect), judged large enough to matter for the retrieval rather than safe to ignore. `Cooking/cooking_combine_spectra.ipynb` (Night 1) has since been re-run with the corrected template — see §8.2, §11, and §12.1 for the updated numbers. **The primary retrieval (job 2528367, §13) has not yet been re-run and its posteriors still reflect the old, incorrectly-telluric-corrected Night 1 data** — this is the next open step.

### Calibration Frames (from CRIRES+ pipeline, per night)

- Dark frames at multiple DITs (1.427, 10, 30 s) → `DARK_MASTER_DIT*.fits`, `DARK_BPM_DIT*.fits`, `DARK_RON_DIT*.fits`
- Flat frames → `FLAT_MASTER_K2166.fits`, `FLAT_NORM_K2166.fits`, `BLAZE_K2166.fits`, `FLAT_BPM_K2166.fits`
- Wavelength calibration (UNe arc + FPET etalon) → `WLEN_K2166_*.fits`, `INIT_WLEN_K2166.fits`
- Slit geometry → `SLIT_TILT_K2166.fits`, `TW_FLAT_K2166.fits`

---

## 3. Initial 2D Reduction — excalibuhr Pipeline

Raw CRIRES+ frames are first reduced with the **excalibuhr** pipeline (Zhang et al. 2024, arXiv:2409.16660), which performs dark subtraction, flat-fielding, bad-pixel masking, order tracing, slit-tilt correction, nodding subtraction (A−B for each pair), and stacking of all nod-subtracted frames.

**Output files consumed by our custom pipeline** (from `/data2/peng/{night}/out/combined/`):

```
Extr2D_PRIMARY__COMBINED_VDHTauA+Bcenter_K2166_NOD_A.npz
Extr2D_PRIMARY__COMBINED_VDHTauA+Bcenter_K2166_NOD_B.npz
```

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `flux` | (2541, 2048) | Stacked 2D PSF-subtracted spectrum (all orders/detectors) |
| `var` | (2541, 2048) | Variance from excalibuhr (not used for optimal extraction; see §5) |
| `psf` | (2541, 2048) | PSF model |
| `id_dets` | [847, 1694] | Detector boundary indices in the stacked spatial axis |
| `id_orders` | 6 values | Order boundary indices within each detector |

The stacked spatial dimension: 2541 = 3 detectors × 7 orders × 121 spatial rows. Loaded via `excalibuhr.data.DETECTOR` and restructured into shape **(3, 7, 121, 2048)**.

**Only orders 0–4 (5 of 7)** are carried forward. The companion appears at **spatial row ~19** in the 35-row sub-array extracted from each chip (nod position A); it appears at the mirrored position in nod B.

---

## 4. Wavelength Solution

- **File**: `WLEN_K2166_V_DH_Tau_A+B_center.fits` (same file for both nights; in `/data2/peng/2022-12-31/cal/`)
- **Shape**: (3, 7, 2048), extension `WAVE`, **units: nm**
- **Full range**: 1921.32–2472.35 nm; **science range (5 orders)**: 2063.7–2472.4 nm
- **Calibration source**: combined UNe arc lamp + FPET Fabry-Pérot etalon (`ESO DPR TYPE = WAVE,FPET`)
- Used as `wave_cal[:,0:5,:]` to select the 5 science orders

---

## 5. Stellar PSF Subtraction and Optimal Extraction

This custom step (notebooks `cooking_subtraction.ipynb` and `cooking_subtraction_0101.ipynb`) isolates the companion's 1D spectrum from the bright DH Tau A stellar halo. **Method 5.A.2** (polynomial halo + Moffat PSF) was selected after evaluating alternatives, which remain in the notebook (unexecuted/superseded) as a record of the selection process:

- 5.A.1 Single-kernel Moffat fitting
- **5.A.2 Two-kernel Moffat fitting + polynomial residual removal — adopted**
- 5.A.3 Two-kernel Moffat fitting, ignoring residuals
- 5.A.4 Double-kernel simultaneous fitting
- 5.A.5 Double-Gaussian simultaneous fitting
- 5.B "β Pic strategy" — simultaneously fit both the star and companion components (sub-variants 5.B.0–5.B.3, including a cross-correlation-based centroiding step)

### 5.1 Spatial Sub-array

A sub-array of **35 spatial rows** (indices 0–34 of 121 per chip) is extracted for all subsequent processing. This captures the stellar halo and the companion PSF while discarding noisy detector edges.

### 5.2 Polynomial Stellar Halo Model (per wavelength column)

For each wavelength pixel independently, a **degree-3 polynomial** is fitted to the 2D spatial profile using rows free of companion signal:

| Nod position | Sky rows used for polynomial fit |
|---|---|
| A (Night 1) | rows 0–10 and 27–34 |
| A (Night 2) | rows 0–9 and 27–34 |
| B | symmetric/mirrored sky rows |

The companion aperture (~rows 14–24) is excluded from the polynomial fit. The spatial axis is normalized to [0, 1] for numerical stability. The fit uses `np.linalg.lstsq`.

The per-column polynomial is **subtracted from the full spatial profile**, yielding the PSF-subtracted residual frame `spectra_clean` in physical units [e⁻/s]. This removes the smooth, slowly varying stellar halo.

### 5.3 Moffat Profile Fit (companion centroid)

A **Moffat profile** is fitted to the clean residual to locate the companion centroid precisely:

```
PSF(x) = A × [1 + ((x − x₀)/α)²]^(−β)
```

- Initial guess: x₀ = 19 px, FWHM = 8.0 px, β = 3.0
- Bounds: β ∈ [0.5, 50], α ≥ 10⁻⁶, x₀ within ±10 px of initial guess
- Fitted with `scipy.optimize.curve_fit`, `max_iter = 20000`
- Fitted centroid x₀ defines the center of the extraction aperture

### 5.4 Optimal Extraction (Horne 1986)

**Library**: `excalibuhr.utils.optimal_extraction`
**Half-aperture**: **5 pixels** (total aperture = 10 px; captures ~95% of companion flux)

**Sky-variance error array** — critical fix applied 2026-03-28:

The excalibuhr `var` array is dominated by DH Tau A's stellar photon noise at the companion's spatial position. Passing it to `optimal_extraction` caused χ²_r >> 1 inside the algorithm, inflating output errors by 10–100× and yielding SNR < 0.1 throughout. The correct variance is estimated empirically from companion-free sky rows in the PSF-subtracted frame `spectra_clean`:

| Nod | Sky rows for variance |
|---|---|
| A | rows 0:5 and 25:30 |
| B | rows 5:10 and 26:30 |

```python
Var_col = np.var(sky_rows, axis=0, ddof=1)     # unbiased; shape (2048,)
Var_2D  = np.tile(Var_col, (n_aperture_rows, 1))
```

**Optimal extraction parameters**: `aper_half=5`, `filter_width=11`, `obj_cen=5`, `badpix_clip=3`, `max_iter=100`, plus the 2D NaN bad-pixel mask from the excalibuhr data cleaning step.

**Output**: 1D extracted spectrum [e⁻/s] and formal error array, shape **(3, 5, 2048)** per nod position.

---

## 6. Blaze Correction

Applied immediately after optimal extraction, before any further steps.

- **File**: `BLAZE_K2166.fits`, shape (3, 7, 2048)
- Both extracted flux and error arrays are divided pixel-by-pixel by `blaze[det, order, :]` for orders 0–4
- Zero/negative blaze values replaced with NaN before division

Note: excalibuhr's internal 1D output (`Extr1D`) already corrects for the blaze; the `Extr2D` product we use does not, so this step is required.

---

## 7. Sigma-Clipping (Cosmic Ray / Bad Pixel Removal)

Applied after blaze correction, **per order** (all 2048 pixels of one det-order combination treated together):

- **Threshold**: 3σ
- **Iterations**: 100
- **Method**: median and std computed with `np.nanmedian` / `np.nanstd` on the full order; pixels outside median ± 3σ replaced with the median (NOT set to NaN)
- Applied identically to nod positions A and B, and to both nights

**Intermediate files saved** (shape: 3, 5, 2048; location: `/data2/peng/{night}/`):

| File | Content |
|---|---|
| `extracted_spectra_position_A_sigmaclipper.npy` | Sigma-clipped A nod flux (Night 1) |
| `extracted_spectra_position_B_sigmaclipper.npy` | Sigma-clipped B nod flux (Night 1) |
| `extracted_spectra_position_A_err.npy` | Formal error array, nod A (Night 1) |
| `extracted_spectra_position_B_err.npy` | Formal error array, nod B (Night 1) |
| `extracted_spectra_position_A_sigmaclipper_0101.npy` | Night 2 equivalents |
| `extracted_spectra_position_B_sigmaclipper_0101.npy` | Night 2 equivalents |

---

## 8. Telluric Correction

### 8.1 Telluric Transmission Template

Source: molecfit fit to the Chi Tau standard star spectrum.

- **File per night**: `chi_tau/{night}/out/molecfit/TELLURIC_DATA.fits`
- The FITS extension `TELLURIC_DATA` contains the fitted transmission model `mtrans ∈ [0, 1]`
- Shape after loading and re-ordering: (3, 5, 2048) for the 5 science orders

### 8.2 Airmass Rescaling (Beer-Lambert Correction)

Chi Tau effective airmass differs from the DH Tau B science airmass on both nights — and, since each night has its own genuine χ Tau observation (§2, Telluric Standard Star), the two nights use **different** Chi Tau airmasses. A Beer-Lambert power-law correction is applied:

```
T_target(λ) = T_ChiTau(λ)^α     where α = X_DHTauB / X_ChiTau
```

- Night 1: α = 1.7007 / 1.9038 = **0.8933** (corrected 2026-07-02/03; printed directly by the re-run `cooking_combine_spectra.ipynb`, cell 3, using Night 1's own χ Tau data)
- Night 2: α = 1.6946 / 1.7251 = 0.9823

**Correction history**: Until 2026-07-02, Night 1's telluric correction used Night 2's χ Tau template (airmass 1.7251) because Night 1's own χ Tau raw data was missing from the local file tree (see §2). This gave an incorrect α = 1.7007/1.7251 = 0.9858 for Night 1. The true Night 1 χ Tau airmass (1.9038, from 4 genuine Night 1 frames) gives α = 0.8933 instead — a much larger correction, since Night 1's χ Tau observation was taken at higher airmass (~1.90) than Night 1's own science block (~1.70). **`cooking_combine_spectra.ipynb` has been re-run for Night 1 with this corrected α; the telluric-corrected, combined, and flux-calibrated Night 1 products on disk are now up to date (as of 2026-07-03).**

For Night 2, this was added after finding that the Chi Tau FITS header records only the last frame airmass (1.568), while the true effective airmass of 6 co-added frames is 1.7251 — molecfit used the wrong value but the fitted `rel_mol_col` compensated, so the template itself is correct at ~1.725 effective airmass. Both α values are printed directly by the reduction notebooks (`cooking_combine_spectra.ipynb` / `_0101.ipynb`, cell 3).

**Quantified impact of the correction (2026-07-02, `compare_chi_tau_telluric_night1_night2.py`)**: Beer-Lambert rescaling only corrects for airmass (optical path length) assuming an *identical* atmospheric column; it cannot correct for a genuinely different atmosphere. Molecfit's own fitted precipitable water vapor differs by **3.43×** between the nights (Night 1: 1.223 mm; Night 2: 4.198 mm — Night 2 was far more humid), while well-mixed gases differ far less (CO2/CH4/N2O ratios ~1.10×), confirming this is a real weather difference, not a fitting artifact. Comparing the two candidate "Night 1 correction templates" (T_new = Night 1's own template^0.8928 vs. T_old = Night 2's template^0.9858 — computed with a slightly earlier α estimate of 0.8928 rather than the final pipeline value 0.8933; the 0.05% difference between them is immaterial to the conclusion below — both evaluated at Night 1's science airmass, over 2063.7–2472.4 nm at 100,000-point resolution):

- Mean fractional flux-correction error if T_old (the old, wrong template) had been used: **+4.0%** (median +0.6%), concentrated almost entirely in H₂O-band wavelength windows (redward of ~2320 nm, and 2140–2200 nm, 2060–2110 nm); H₂O-poor continuum windows (~2200–2320 nm) show near-zero difference — the pattern is coherent with H₂O opacity, not noise.
- Restricted to pixels with mtrans > 0.8 in both templates (i.e. **not** already masked by §8.4): mean fractional error +1.7% (median +0.4%), but 7.5% of these unmasked pixels differ by >5% and 2.0% by >10%, in the same H₂O-band locations.
- Plot: `Recipe_DH_Tau_B/chi_tau_telluric_night1_vs_night2_comparison.png`.

**Assessment**: the error is small in the bulk of the spectrum but coherently structured within H₂O bands — exactly the region diagnostic of the retrieved H₂O column / C/O ratio — rather than randomly distributed noise that would average out. This is a plausible source of systematic bias for Night 1's contribution to the joint retrieval, not merely a cosmetic difference.

### 8.3 Sentinel Pixel Masking

excalibuhr marks unreliable pixels with a sentinel error value ~10¹⁰ (reduced to ~6×10⁵–9×10⁵ after sigma-clipping). Any pixel with formal error > 10 (typical good-pixel errors < 0.013 in file units) is masked:

```python
bad = (extracted_spectra_err > 10)
flux[bad] = np.nan
err[bad]  = np.nan
```

**Pixels masked** (out of 3×5×2048 = 30720 total per nod position; printed by the reduction notebooks, cell 5):

| Night | Position A | Position B |
|---|---|---|
| Night 1 | 3941 (12.8%) | 2870 (9.3%) |
| Night 2 | 3942 (12.8%) | 2866 (9.3%) |

### 8.4 Telluric Masking Threshold

Pixels where the telluric transmission falls below **mtrans < 0.80** are set to NaN in both flux and error arrays. This threshold was tuned from 0.70 (earlier runs): at 0.70, spurious emission-like artefacts appeared at partially-absorbed telluric lines, caused by division by a slightly under-estimated transmission. At 0.80, these artefacts disappear; the stricter mask removes ~10% more pixels.

### 8.5 Telluric Division

```python
flux_corrected = flux_masked / mtrans_rescaled
err_corrected  = err_masked  / mtrans_rescaled
```

Error propagated by the same division (telluric model treated as noiseless).

**Intermediate files saved** (shape: 3, 5, 2048):
`extracted_spectra_position_{A,B}_masked.npy`, `extracted_spectra_position_{A,B}_err_masked.npy` (Night 2: `_0101` suffix)

---

## 9. Barycentric Correction

**Computed in `Cooking/cooking_combine_spectra_inter_night.ipynb`** (not in the per-night `cooking_combine_spectra*.ipynb` notebooks, which only contain a commented-out load of the pre-computed file — an earlier version of this document misattributed this step). The retrieval scripts (`Tasting/tasting_retrieval_*.py`) simply load the cached `.npy` outputs of this notebook at run time.

The wavelength grid is corrected to the barycentric rest frame using **PyAstronomy `helcorr`** (Cerro Paranal: lon=−70.4042°, lat=−24.6272°, alt=2635 m; DH Tau RA/Dec = 67.4232°/26.5490°, J2000), cross-validated against Astropy `radial_velocity_correction` (independent ERFA/SOFA ephemeris):

| Night | Date | Science midpoint (JD) | v_bary helcorr (km/s) | v_bary astropy (km/s) | helcorr − astropy |
|---|---|---|---|---|---|
| Night 1 | 2022-12-31 | 2459945.650894 | −15.37101 | −15.36733 | −3.68 m/s |
| Night 2 | 2023-01-01 | 2459946.556409 | −15.56411 | −15.56045 | −3.66 m/s |

helcorr values are adopted (consistent with the retrieval-script convention). Inter-night barycentric drift Δv = −0.193 km/s agrees between the two methods to < 0.1 m/s.

The correction is applied to the **wavelength array** (not the spectrum):

```python
wave_bary = wave_obs × (1 + v_bary / c)
```

Output files:
- `/data2/peng/2022-12-31/barycentric_wavelengths_night1.npy` — shape (3, 5, 2048), nm
- `/data2/peng/2023-01-01/barycentric_wavelengths_night2.npy` — shape (3, 5, 2048), nm

**Residual inter-night instrumental drift** (after barycentric correction, before any interpolation) was checked per chip: 14 of 15 chips are ≤ 0.5 pixels, but chip (det=0, ord=0) — the same chip later found to need the det2/ord0-adjacent flux-cal fix — measures **0.536 pixels**, marginally above the 0.5-pixel threshold used to justify skipping flux interpolation. At R≈100,000 this is still < 1/6 of the LSF width and was judged negligible; no interpolation was applied. The two nights are kept on their **separate barycentric wavelength grids** in the primary joint retrieval; no interpolation to a common grid is performed, avoiding correlated noise from resampling.

### 9.1 Prototype Inter-Night Combined Spectrum (not used in the primary retrieval)

The same notebook also prototyped co-adding the two nights onto the night-1 barycentric grid (LPU error propagation, identical formula to §10), producing files in `/data2/peng/combined_two_nights/` (regenerated 2026-06-18 alongside the flux-cal fix): `extracted_spectra_combined_two_nights{,_err}.npy`, `extracted_spectra_combined_two_nights_flux_cal{,_err}.npy`, `WLEN_combined_two_nights_bary.npy`. Diagnostics:

| Quantity | Night 1 | Night 2 | Combined |
|---|---|---|---|
| Median per-pixel SNR (sigma-clipper) | 5.46 | 7.23 | 8.90 |
| Median per-pixel SNR (flux-cal) | 5.46 | 7.23 | 8.62 |
| Median flux-cal level (W m⁻² μm⁻¹) | 8.984×10⁻¹⁶ | 9.080×10⁻¹⁶ | 9.030×10⁻¹⁶ |

Expected SNR gain from co-adding two nights is √2 ≈ 1.41×; achieved gain is consistent with this. 24950/30720 pixels have both nights contributing (N_valid=2); 14 pixels have only one night finite. This dataset exists on disk but is **not used in the primary or comparison retrieval series** — see design decision 15.6.

---

## 10. A+B Nod Combination

The two nod positions are combined after telluric correction.

**Combination** (nanmedian = nanmean for N = 2):
```python
flux_combined = np.nanmedian([flux_A_corr, flux_B_corr], axis=0)
```

**Error propagation** — Law of Propagation of Uncertainty (GUM §5):
```python
n_valid = np.sum(np.isfinite([flux_A, flux_B]), axis=0)   # 0, 1, or 2
err_combined = np.sqrt(nansum([err_A², err_B²])) / n_valid
```

For pixels where one nod is NaN-masked (e.g., deep telluric cores), `n_valid = 1` and the remaining nod's error is used directly. Pixels masked in both nods yield NaN.

**Note**: An earlier incorrect formula used `np.nanstd([A, B]) / sqrt(2)`, which for N = 2 computes `|A−B| / (2√2)` — not a standard uncertainty formula. Corrected to formal LPU propagation (2026-03-27).

**Intermediate science files** (shape: 3, 5, 2048; input to §11 flux calibration):

| File | Location | Units |
|---|---|---|
| `extracted_spectra_combined_sigmaclipper.npy` | `/data2/peng/2022-12-31/` | e⁻/s |
| `extracted_spectra_combined_err_sigmaclipper.npy` | `/data2/peng/2022-12-31/` | e⁻/s |
| `extracted_spectra_combined_sigmaclipper_0101.npy` | `/data2/peng/2023-01-01/` | e⁻/s |
| `extracted_spectra_combined_err_sigmaclipper_0101.npy` | `/data2/peng/2023-01-01/` | e⁻/s |

---

## 11. Absolute Flux Calibration

Absolute flux calibration converts the extracted spectra from detector units [e⁻/s] to physical flux units [W m⁻² μm⁻¹], enabling retrieval of absolute physical parameters (mass log_M and radius log_R) through the angular diameter factor (R/d)². This step is **standard** and produces the primary science data products used in the main retrieval series. The calibration was updated on **2026-06-18** to fix three compounding bugs in the scale-factor computation (see "Known issue" below).

### Reference Spectrum

- **Source**: SINFONI spectral library — `/data2/peng/DHTaub_SINFONIspeclib_JHK.fits`
- **Coverage**: 1100–2458.5 nm (non-zero); overlap with K2166: 2063.7–2458.5 nm
- **Units**: W m⁻² μm⁻¹; wavelength axis in μm (converted in pipeline: × 10³ → nm)
- **Photometric anchor**: K_s = 14.19 mag (Patience et al. 2012); zero-point F₀(K, Vega) = 4.29 × 10⁻¹⁰ W m⁻² μm⁻¹
- SINFONI median in [2.0–2.5 μm] is scaled to match the expected K_s flux before use: absolute correction factor ≈ 1.3346×10⁸ (raw SINFONI library median 6.7781×10⁻²⁴ → expected K=14.19 flux of 9.0460×10⁻¹⁶ W m⁻² μm⁻¹). Printed identically (to 4 s.f.) for both nights.

### Per-Chip Scale Computation

For each of the 15 chips:
1. SINFONI interpolated onto the CRIRES+ wavelength grid (cubic spline)
2. CRIRES+ combined spectrum smoothed with SavGol (window = 201 px ≈ 20 nm, polyorder = 2); NaN gaps filled by linear interpolation before filtering
3. Per-pixel ratio = SINFONI_interp / CRIRES_smooth; 3σ clip (3 iterations) removes telluric residuals
4. Chips with SINFONI coverage < 20% of valid CRIRES+ pixels are excluded from the polynomial fit

### Per-Detector Degree-3 Polynomial

A degree-3 polynomial is fitted per detector (3 separate fits) to the ratio values from all covered chips, with 3-pass iterative 3σ clipping. The polynomial is clamped at the last wavelength with SINFONI data to prevent cubic extrapolation artefacts beyond 2458.5 nm.

Point retention after 3-pass σ-clipping (printed diagnostics, both nights ~96–99.5%): Night 2 det0 ≈ 8100/8250 pts, det1 ≈ 7800/7900 pts, det2 ≈ 7200/7260 pts; Night 1 (re-run 2026-07-03 with the corrected χ Tau template) det0 = 8033/8396, det1 = 8073/8349, det2 = 7302/7351 pts (det2 includes the 1 extrapolated chip). Det2 ord0 (2457–2472 nm) SINFONI coverage is 5.8% (Night 2) / 4.5% (Night 1, updated) — below the 20% threshold in both cases — and is excluded from the direct ratio fit, relying entirely on the clamped polynomial extrapolation described above.

**Effect of the Night 1 telluric-correction fix on flux calibration (2026-07-03)**: re-running with the corrected χ Tau template shifted Night 1's per-chip SINFONI/CRIRES+ ratio medians, most visibly in the reddest, most H₂O-affected chips — e.g. det0 ord0 (2429.9 nm): 3.1630×10⁻¹³ → 3.4049×10⁻¹³ (+7.6%); det1 ord0 (2446.9 nm): 3.2430×10⁻¹³ → 3.4393×10⁻¹³ (+6.1%) — consistent with the H₂O-band-concentrated bias quantified in §8.2. Bluer, H₂O-poor chips changed negligibly (e.g. det0 ord2 at 2236.6 nm: 4.7470×10⁻¹³ → 4.7475×10⁻¹³). More chip pixels also survived the 3σ clip after the fix (fewer spurious telluric-residual outliers), consistent with a better telluric correction.

**Known issue — det2 ord0 factor-3 under-calibration (fixed 2026-06-18)**:
Night 1 detector 2 order 0 (2457–2472 nm) was under-calibrated by a factor of ~3 in the original flux-cal data due to three compounding bugs in the ratio/polynomial computation (SINFONI wavelength unit mismatch, missing polynomial clamping, incorrect chip coverage threshold). After correction, all 5 orders on all 3 detectors show consistent median flux levels (~5.5–7.2 × 10⁻¹⁶ W m⁻² μm⁻¹). All retrievals use the recalibrated files from 2026-06-18.

### Systematic caveat: det2 ord0 flat-extrapolated scale (anticipated reviewer concern)

Det2 ord0 (chip center ≈2457.6 nm, night 2; ≈2457.9 nm, night 1) is the single chip out of 15 whose scale factor is **not fit from real SINFONI overlap** — it fails the 20% coverage threshold (§"Per-Chip Scale Computation," item 4) because its blue edge only marginally enters SINFONI's 1100–2458.5 nm range before SINFONI runs out of signal. Mechanically, what is applied to this chip is not an extrapolated polynomial value but a **flat freeze**: `det_clamp_wl[det]` is set to the center wavelength of the reddest chip on that detector that *does* have ≥20% coverage (for det2, night 2, this is ord1 at 2362.5 nm), and every pixel redward of that wavelength — which is all of ord0 — is assigned the constant scale value the degree-3 polynomial takes at `det_clamp_wl`, rather than the polynomial's own (diverging) extrapolation. This was a deliberate choice to avoid cubic runaway, not an attempt to model the true chip response.

**Is this justified, and by how much could it be wrong?** The per-chip ratio computed directly from the data shows a clear, smooth *decrease* toward redder wavelengths within det2 itself (ord4 2100.1 nm: 4.25×10⁻¹³ → ord3 2180.7 nm: 3.80×10⁻¹³ → ord2 2267.9 nm: 3.35×10⁻¹³ → ord1 2362.5 nm: 3.02×10⁻¹³), and the same trend is directly measured on det0 and det1's own reddest chips, which *do* have SINFONI coverage all the way to ord0 (det0: ord1 2329.8 nm 2.93×10⁻¹³ → ord0 2429.8 nm 2.47×10⁻¹³, a −15.5% step; det1: ord1 2346.1 nm 3.18×10⁻¹³ → ord0 2444.8 nm 2.62×10⁻¹³, a −17.7% step). Linearly extrapolating det2's own local slope (ord2→ord1) out to ord0's center wavelength predicts a ratio ≈2.7×10⁻¹³, roughly **10% below** the frozen value (3.02×10⁻¹³) actually applied; using the steeper det0/det1 fractional step as a proxy instead predicts more like **15–18% below**. In short, the flat freeze most likely **overestimates** det2 ord0's true flux calibration scale by something in the ballpark of **10–18%**, i.e. that one chip's absolute flux level is probably biased high by a similar fraction. This is a plausibility estimate from a 2-point local slope, not a rigorous error bar — but it is the best available check, because **no independent absolute-flux measurement covers this range at all**: the 2MASS K_s bandpass (2028–2290 nm) ends well short of 2425 nm, and SINFONI itself is only 5.8% (night 2) / 4.5% (night 1) covered there.

**Scope of the impact**: this is 1 of 15 (det, order) chips, ≈47 nm out of the ≈450 nm K2166 bandpass (~7% of the wavelength range, night 2). Line-shape-driven retrieval parameters (abundances, vsini, RV) are locally self-calibrated against each chip's own continuum and should be insensitive to a chip-wide multiplicative offset of this size. The absolute-flux-dependent parameter (log_R, via the (R/d)² factor) draws on the whole bandpass jointly, so a ~10–18% bias confined to 1/15 of the chips, on a chip that is also intrinsically lower-SNR (order edge, reddest/faintest part of K-band), should dilute to a sub-percent effect on the fitted radius — but this has not been explicitly demonstrated with a sensitivity run.

**Recommended text/action for the paper** (for whoever drafts the flux-calibration section): (1) state explicitly that this chip's scale factor is extrapolated rather than measured, with the quantified ~10–18% plausible bias above; (2) note the absence of any independent photometric anchor redward of 2290 nm; (3) ideally, run the primary retrieval (007PM-EQ-GP) once with det2 ord0 masked out and report that the posteriors are unchanged within their quoted precision, which would pre-empt the obvious referee question rather than just asserting insensitivity. Figure 2 (bottom panel) in `Reminisce/plot_paper.ipynb` plots the per-detector scale actually applied together with the underlying per-pixel ratio scatter, with the frozen segment on det2 ord0 directly annotated — useful either as a paper figure or as supporting material for a referee response.

### Primary Science Output Files (shape: 3, 5, 2048; units W m⁻² μm⁻¹)

| File | Location |
|---|---|
| `extracted_spectra_combined_flux_cal.npy` | `/data2/peng/2022-12-31/` |
| `extracted_spectra_combined_err_flux_cal.npy` | `/data2/peng/2022-12-31/` |
| `extracted_spectra_combined_flux_cal.npy` | `/data2/peng/2023-01-01/` |
| `extracted_spectra_combined_err_flux_cal.npy` | `/data2/peng/2023-01-01/` |

---

## 12. Cross-Correlation Validation and RV Prior

Two independent CCF analyses exist. (1) A **per-species detection/data-quality check** embedded directly in the reduction notebooks (`cooking_combine_spectra.ipynb` / `_0101.ipynb`, §3, cells 16–18), run immediately after flux calibration, single-species templates, topocentric RV grid. (2) The dedicated **`tasting_RV_prior.ipynb`**, using multi-species SONORA-grid forward models, which supplies the actual RV prior used in the retrieval (described below).

### 12.1 In-notebook single-species CCF check (§3 of the combine notebooks)

**Templates**: single-species pRT forward models at T_eff = 2300 K, log g = 4.0, [Fe/H] = 0.0 (`planet_2300_4.0_-0.0_0_{co,h2o_pokazatel,ch4}_only.dat`), RV lag grid −100 to +100 km/s (201 steps, 1 km/s). SNR = peak / std(CCF outside the peak region).

| Species | Night 1 peak SNR | Night 1 RV | Night 2 peak SNR | Night 2 RV |
|---|---|---|---|---|
| CO | 11.14 | 31.0 km/s | 13.92 | 31.0 km/s |
| H₂O | 20.12 | 32.0 km/s | 17.38 | 32.0 km/s |
| CH₄ | 2.21 | 97.0 km/s | 2.58 | −63.0 km/s |

CO and H₂O both peak cleanly at the expected systemic RV (~31–32 km/s, topocentric) on both nights. CH₄ shows no coherent detection (SNR ≲ 2.6, peak RV inconsistent between nights), consistent with no CH₄ expected in a ~2000 K atmosphere.

**Night 1 values updated 2026-07-03** after re-running `cooking_combine_spectra.ipynb` with the corrected χ Tau telluric template (§8.2): CO SNR rose 9.81→11.14 and H₂O SNR fell 23.43→20.12, consistent with removing a spurious H₂O-band systematic from the old (wrong) telluric correction. CH₄'s peak RV also shifted (50.0→97.0 km/s) but both are far from the systemic RV and the SNR remains low (~2.2) in both cases — still no coherent CH₄ detection either way.

### 12.2 Dedicated RV-prior analysis (`tasting_RV_prior.ipynb`)

**Templates**: Cloud-free pRT3 forward models at T_eff = 2000–2400 K (100 K steps), log g = 3.7, [Fe/H] = +0.5, C/O = 0.54, vsini = 5.7 km/s. Generated using SONORA Diamondback P-T profiles; convolved from LBL sampling R = 333,333 to instrumental R = 100,000.

**CCF method**: Pearson cross-correlation vs. RV lag grid −100 to +100 km/s (1 km/s steps). SNR = peak / std(CCF at |v| > 50 km/s). Significance tested via 200 spectral-shuffle trials.

**Detection**: CCF peaks at **RV ≈ 31–32 km/s** (barycentric), consistent with the systemic RV of the DH Tau system. H₂O and ¹²CO signals detected at SNR > 3 — consistent with, and more conservative than, the higher per-species SNRs found in §12.1.

**Retrieval RV prior**: Gaussian N(μ = 31.7 km/s, σ = 0.5 km/s) for rv_N1; N(μ = 31.9 km/s, σ = 0.5 km/s) for rv_N2. The 0.2 km/s centre offset between nights corresponds to the measured inter-night barycentric drift (Δv_bary = −0.193 km/s; see §9).

---

## 13. Final Data Products and Retrieval Configuration

### 13.1 Primary Retrieval — Series 007PM-EQ-GP (Standard Gaussian, Absolute Flux)

The primary retrieval series uses the absolute-flux-calibrated spectra with a Standard Gaussian likelihood and GP correlated noise. This is the paper's main result.

**STALE (as of 2026-07-03) — re-run IN PROGRESS**: job 2528367 below was run on Night 1 flux-calibrated data produced with the *wrong* χ Tau telluric template (§2, §8.2). Night 1's `extracted_spectra_combined_flux_cal.npy`/`_err_flux_cal.npy` have since been regenerated with the corrected template. **A new job, 2027997, was launched 2026-07-03 (~01:11) against the corrected data**, same configuration (N600, evidence_tol=0.5, NormNone, PerChipScaleFalse) — PyMultiNest actively sampling as of this note; not yet converged. The posteriors below (job 2528367) still reflect the old, incorrectly-telluric-corrected Night 1 spectrum and should be treated as the pre-fix reference until job 2027997 completes and this section is updated with its results.

**Script**: `Recipe_DH_Tau_B/Tasting/tasting_retrieval_equa_chem_v6.1_piette_cloudfree.py`
**Guidebook**: `Guidebook_GAStronomy_Piette_v4.2.py`
**Reference run**: job **2528367** (pre-fix) — `/data2/peng/retrievals/2528367_N600_ev0.5_NormNone_PerChipScaleFalse/`
**In-progress re-run**: job **2027997** (post-fix) — `/data2/peng/retrievals/2027997_N600_ev0.5_NormNone_PerChipScaleFalse/`

| Night | Flux file | Error file | Location |
|---|---|---|---|
| 2022-12-31 | `extracted_spectra_combined_flux_cal.npy` | `extracted_spectra_combined_err_flux_cal.npy` | `/data2/peng/2022-12-31/` |
| 2023-01-01 | `extracted_spectra_combined_flux_cal.npy` | `extracted_spectra_combined_err_flux_cal.npy` | `/data2/peng/2023-01-01/` |

Wavelength grid per night: `WLEN_K2166_V_DH_Tau_A+B_center.fits` with barycentric correction applied from `barycentric_wavelengths_night{1,2}.npy`. Shape: **(3, 5, 2048)**.

**Retrieval configuration**:

| Parameter | Value |
|---|---|
| Likelihood | Standard Gaussian (scale_flux=False, use_absolute_flux=True) |
| Normalisation | None (absolute flux) |
| Per-chip scaling | False |
| Flux parameters | log_M N(1.079, 0.145), log_R N(0.415, 0.100) |
| Covariance | GP block-diagonal per chip (Guidebook v4.2 banded Cholesky) |
| P-T model | Piette+2020 PCHIP — T_anchor at 0.2 bar + dT_1…dT_7 |
| Chemistry | Equilibrium (pRT3 table) + F_H, log_Na, log_Ca; no Ti |
| Free parameters | 22 total (4 kinematics + 2 physical + 8 P-T + 3 chem + 3 atomic + 2 GP) |
| N_live | 600; evidence_tol = 0.5 |
| Nights | Joint two-night (rv_N1, rv_N2 as separate free parameters) |

**Posterior results (job 2528367; posterior mean ± 1σ)**:

| Parameter | Value | Note |
|---|---|---|
| rv_N1 | 31.62 ± 0.12 km/s | Topocentric |
| rv_N2 | 31.54 ± 0.09 km/s | Topocentric |
| vsini | 7.61 ± 0.16 km/s | |
| ε (limb-darkening) | 0.841 ± 0.101 | |
| log_M → M | 1.190 → 15.5 MJup | |
| log_R → R | 0.485 → 3.05 RJup | σ = 0.007 (data-dominated; 15× tighter than prior) |
| T_anchor | 2078 ± 28 K | At 0.2 bar |
| [C/H] | +0.127 dex | C_H = +0.020 |
| C/O | 0.646 ± 0.025 | |
| ¹²CO/¹³CO | 49 (log = 1.686 ± 0.121) | |
| lnZ (NS) | 1,714,455 ± 0.23 | |
| χ² (Night 1 / Night 2) | 1.039 / 0.967 | |
| Posterior samples | 5507 | |

### 13.2 Comparison Retrieval — Series 008PM-EQ-RUFFIO (Ruffio Likelihood, Sigma-Clipper)

A comparison retrieval uses the non-flux-calibrated sigma-clipper data with the Ruffio per-chip φ marginal likelihood (González Picos et al. 2024), confirming that the reduction choice and likelihood formulation do not bias the posterior.

**STALE (as of 2026-07-03)**: same issue as §13.1 — job 3286785 used Night 1's `extracted_spectra_combined_sigmaclipper.npy` from before the χ Tau telluric-template fix. That file has also been regenerated by the same `cooking_combine_spectra.ipynb` re-run, so this comparison job is stale too and would need re-launching to remain a valid check against the primary series.

**Script**: `Recipe_DH_Tau_B/Tasting/tasting_retrieval_equa_chem_v6.1_piette_cloudfree_ruffio.py`
**Reference run**: job **3286785** (pre-fix) — `/data2/peng/retrievals/3286785_*/`

| Night | Flux file | Error file |
|---|---|---|
| 2022-12-31 | `extracted_spectra_combined_sigmaclipper.npy` | `extracted_spectra_combined_err_sigmaclipper.npy` |
| 2023-01-01 | `extracted_spectra_combined_sigmaclipper_0101.npy` | `extracted_spectra_combined_err_sigmaclipper_0101.npy` |

**Retrieval configuration**:

| Parameter | Value |
|---|---|
| Likelihood | Ruffio marginal (scale_flux=True, per-chip φ) |
| Normalisation | Per-chip median (data only) — each chip divided by nanmedian(flux) before Target creation |
| Gravity | log_g N(3.64, 0.20) — no absolute flux (no log_M, log_R) |

### 13.3 Additional Data-Quality Handling (Comparison Retrieval, Night 2 Sigma-Clipper Only)

Before creating the `Target` object in the Ruffio comparison retrieval, two per-pixel masking steps are applied to Night 2 sigma-clipper data:

1. **Zero-error pixels** (135 pixels in Night 2 sigmaclipper): set flux = NaN, err = NaN. These arise from the sigma-clipping replacement-with-median step producing pixels with formally zero uncertainty in some chips.

2. **Gap pixels** (12 pixels in Night 2 sigmaclipper): pixels whose wavelengths fall 0.001–0.019 nm outside all K2166 chip boundary definitions are **deleted** from the wavelength, flux, and error arrays before Target creation. These are detector-edge pixels where the wavelength calibration extends slightly beyond the nominal K2166 limits. Deleting them prevents a sub-chip size mismatch in the GP covariance matrix solve. Night 1 and both flux-cal nights have zero gap pixels.

---

## 14. Summary of Key Parameters

| Parameter | Value |
|---|---|
| Instrument | CRIRES+, VLT-UT3, K2166 setting |
| Slit width | 0.4 arcsec |
| DIT per frame | 300 s |
| Total science frames | 30 (16 Night 1 + 14 Night 2) |
| Total on-source time | 9000 s (~2.5 h) |
| Spectral resolving power | R ≈ 100,000 |
| Wavelength coverage (used) | 2063.7–2472.4 nm; 15 chips (5 orders × 3 detectors) |
| Plate scale | 0.059 arcsec/pixel |
| Companion projected separation | 2.3 arcsec (~39 pixels) |
| Companion position in extraction subarray | row ~19 of 35 |
| PSF subtraction | Degree-3 polynomial (per wavelength column) + Moffat centroid fit |
| Sky rows for polynomial (nod A, Night 1) | rows 0–10 and 27–34 |
| Extraction aperture | ±5 pixels (10 pixels total; Horne 1986) |
| Variance source | Empirical sky variance from companion-free rows |
| Sigma-clipping | 3σ, 100 iterations, replace with median |
| Telluric standard | χ Tau, molecfit transmission template; separate genuine observation each night (Night 1: 4×20s; Night 2: 6×100s) |
| Telluric mask threshold | mtrans < 0.80 |
| Telluric airmass correction | Beer-Lambert power law T^α, α = X_DHTauB / X_ChiTau |
| Barycentric correction | PyAstronomy helcorr; Night 1: −15.371 km/s; Night 2: −15.564 km/s |
| A+B nod combination | nanmedian (= nanmean for N = 2); LPU error propagation |
| Flux calibration | SINFONI reference, degree-3 polynomial per detector; recalibrated 2026-06-18 |
| Photometric anchor | K_s = 14.19 mag (Patience+2012); F₀ = 4.29×10⁻¹⁰ W m⁻² μm⁻¹ |
| Primary data product | `extracted_spectra_combined_flux_cal.npy` (both nights), W m⁻² μm⁻¹ |
| Primary retrieval | Series 007PM-EQ-GP; job 2528367; Standard Gaussian + absolute flux + GP |

---

## 15. Key Design Decisions and Their Rationale

1. **Custom extraction over excalibuhr 1D SECONDARY product**: The excalibuhr `Extr1D_SECONDARY` product exists on disk but uses the full-frame variance (dominated by DH Tau A photon noise), causing SNR < 0.1. Our custom pipeline replaces the variance with sky-row empirical estimates, recovering SNR ~ 2–5 per pixel.

2. **Polynomial + Moffat rather than PSF-only subtraction**: At 2.3 arcsec separation, the stellar PSF halo varies smoothly across the companion aperture and is well-described by a low-degree polynomial. The companion contributes a distinct Moffat-shaped peak above this smooth background. Fitting both simultaneously in two steps avoids degeneracy between the halo model and the companion flux.

3. **Telluric mask threshold 0.80 instead of 0.70**: At 0.70, spurious emission artefacts appeared at partially-absorbed telluric lines (over-correction residuals). At 0.80, these disappear. The stricter mask removes ~10% more pixels but eliminates a systematic that would bias the chi-squared landscape.

4. **Separate barycentric grids per night, no co-adding**: Interpolating two nights onto a common wavelength grid introduces correlated noise from resampling. The retrieval framework evaluates a joint log-likelihood across both nights on their own grids, avoiding this entirely. The 0.5-pixel residual instrumental drift between nights (after barycentric correction) is sub-LSF-width and does not require interpolation.

5. **Absolute flux calibration as the standard reduction step**: The flux-calibrated dataset provides a physically-grounded data product in W m⁻² μm⁻¹, enabling direct retrieval of mass (log_M) and radius (log_R) via the (R/d)² angular diameter factor. Retrieved log_R = 0.485 ± 0.007 is data-dominated (15× tighter than the prior), confirming that absolute flux information is well-constrained and scientifically informative. The sigma-clipper dataset (comparison series 008PM-EQ-RUFFIO) serves as a validation check to confirm that the SINFONI-based polynomial response correction does not bias the atmospheric posteriors.

6. **Joint two-night retrieval rather than combined spectrum**: A combined two-night spectrum was prototyped in `cooking_combine_spectra_inter_night.ipynb` (outputs still on disk at `/data2/peng/combined_two_nights/`, regenerated 2026-06-18) but abandoned in favor of the joint-likelihood approach. Co-adding requires assigning both nights to a single (night-1) barycentric wavelength grid without interpolation, which is only justified because residual inter-night instrumental drift is ≤0.5 px on 14/15 chips (one chip, det0/ord0, is marginally over at 0.536 px — see §9). The joint likelihood approach sums log L_N1 + log L_N2 evaluated independently on each night's native barycentric grid, which is exactly equivalent to fitting both datasets simultaneously under a shared parameter vector, at no interpolation cost and without needing to accept the det0/ord0 drift approximation. The prototyped combination achieved the expected √2 SNR gain (median SNR 5.46 → 7.23 → 8.90 per night vs. combined) confirming the co-addition itself was implemented correctly; it was set aside for the interpolation-avoidance reason above, not because of a data-quality problem.

---

## 16. References for Data Reduction Section

- Zhang, Y. et al. (2024). "excalibuhr: A Python package for CRIRES+ data reduction." *arXiv:2409.16660*
- Horne, K. (1986). "An optimal extraction algorithm for CCD spectroscopy." *PASP*, 98, 609–617.
- Ruffio, J.-B. et al. (2019). "Radial velocity measurements of HR 8799 b and c." *ApJ*, 881, 1.
- Patience, J. et al. (2012). "The widest-separation substellar companion candidate to a binary T Tauri star." *A&A*, 540, A85.
- González Picos, D. et al. (2024). "Atmospheric characterisation of GQ Lup b with CRIRES+." *A&A*, Survey II.
- Xuan, J.W. et al. (2024). "A KPIC survey of directly imaged companions." *AJ*, 167, 5.
- Piette, A.A.A. & Madhusudhan, N. (2020). "Considerations for atmospheric retrieval of high-precision brown dwarf spectra." *ApJ*, 904, 1.
- Marley, M.S. et al. (2021). "The Sonora Brown Dwarf Atmosphere and Evolution Models. I." *ApJ*, 920, 85.
- JCGM 100:2008. "Evaluation of measurement data — Guide to the expression of uncertainty in measurement (GUM)." BIPM/ISO. §5.

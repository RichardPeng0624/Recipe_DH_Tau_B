# Plan: Fix SNR < 1 Bug in Spectral Extraction Pipeline
**Date:** 2026-04-16  
**Context:** Retrieval 1461589 returned an isothermal P-T profile and extremely low C/O ratio,
suggesting the data fed to the retrieval is noise-dominated. A full audit of the cooking
notebooks traced the root cause to a normalization domain error in the optimal extraction step.

---

## Rationale

The pipeline processes two nights (2022-12-31, 2023-01-01) through four notebook stages:

```
cooking_subtraction[_0101].ipynb
  → cooking_combine_spectra[_0101].ipynb
    → cooking_combine_spectra_inter_night.ipynb
      → retrieval
```

The SNR check in `cooking_combine_spectra_inter_night.ipynb` (Cell 18 + Cell 22) already
shows median per-pixel SNR < 1 for both nights before inter-night combination. This low SNR
propagates to the retrieval, which then cannot constrain temperature structure or molecular
abundances.

The cause is **not** intrinsic faintness of DH Tau B. The 2D SNR in
`cooking_first_look.ipynb` shows the stellar trace at SNR ~837. The companion at contrast
~1/160 should still yield ~5–15 per spectral pixel in the optimal-extraction output — enough
for a meaningful retrieval. But the saved error arrays are 10–50× too large.

---

## Bug 1 (ROOT CAUSE): Optimal Extraction on Wavelength-Normalized Data

**File:** `Cooking/cooking_subtraction.ipynb`  
**Cells:** Cell 25 (spectra_clean construction) + Cell 28 (optimal_extraction call)

### What goes wrong

In Cell 25, `spectra_clean_A` is built by **normalizing each wavelength column** of the 2D
spectrum by its spatial sum (dominated by stellar flux):

```python
for i in range(spec2d.shape[-1]):          # i = wavelength pixel
    norm_channel = normalize_profile(spec2d[:, i])   # divide by Σ over 45 rows
    spectra_clean_A[iDet, iOrder, :, i] = norm_channel  # fit_star_component=False → no subtraction
```

After this step `spectra_clean_A` is **dimensionless** (values ~6×10⁻³ at companion peak,
tiny fractions elsewhere). It is NOT in physical counts (e⁻).

This normalized array is then passed directly to `utils.optimal_extraction` in Cell 28:

```python
opt_extraction_A = utils.optimal_extraction(
    D_full=spectra_clean_A[det, order, center-half:center+half, :].T,
    V_full=Var[center-half:center+half, :].T,
    aper_half=half, obj_cen=half, ...)
```

Inside `excalibuhr/src/excalibuhr/utils.py` line 1627, the function adds a Poisson term:

```python
V_new = V + np.abs(D) / gain / NDIT    # gain=2 (default), NDIT=1 (default)
```

This assumes D is in **photoelectrons**. Since D is actually a dimensionless fraction
(~6×10⁻³), the Poisson term `|D|/2 ~ 3×10⁻³` completely dominates over the sky variance
(which is also in the normalized domain, ~10⁻¹⁰ to 10⁻⁸).

The result: variance is proportional to the signal itself, so **err ∝ √signal** and
**SNR ≈ √(normalized_companion_flux) ≈ 0.08–0.15** — far below the photon-noise-limited
value of ~5–10.

The sky variance array `Var` is computed from sky rows of `spectra_clean_A` (rows 0:5 and
25:30), which also contain only tiny normalized values. So both V_sky and the Poisson term
are self-consistently wrong in the same domain; the Poisson term still dominates by ~10⁴×.

### Why the signal estimate f_opt is unbiased (signal is OK, error is wrong)

Mathematically, when V_new ∝ |D| ∝ PSF_profile × f_true, the optimal estimator
`f_opt = Σ(P·D/V) / Σ(P²/V)` simplifies to f_true (the companion flux). So the extracted
**flux values are correct**, but the **error bars are inflated by ~10–50×**.

### The fix

Run `optimal_extraction` on the **raw counts** (in e⁻ or e⁻/s), not on the normalized array.
The per-column normalization is needed only to build the PSF model; the actual flux and
variance should stay in physical units.

**Option A (cleanest): Use raw 2D data for extraction**

```python
# Cell 28 replacement
for det in range(3):
    for order in range(5):
        # Use the raw 2D data (in physical counts/s), not the normalized spectra_clean_A
        D_raw = fit_workflow_A.spectra[det, order, center-half:center+half, :]   # (4, 2048) e-/s

        # Sky-based read+dark variance in physical units (from raw sky rows)
        _sky_raw = np.concatenate([
            fit_workflow_A.spectra[det, order, 0:5,  :],
            fit_workflow_A.spectra[det, order, 25:30, :]], axis=0)
        _sky_var_raw = np.var(_sky_raw, axis=0, ddof=1)   # shape (2048,)  [e-/s]²
        Var_raw = np.tile(_sky_var_raw, (D_raw.shape[0], 1))   # (4, 2048)

        opt_extraction_A = utils.optimal_extraction(
            D_full=D_raw.T, V_full=Var_raw.T,
            aper_half=half, filter_width=half*2+1,
            bpm_full=mask_areas_A[det, order, center-half:center+half, :].T,
            obj_cen=half, badpix_clip=3, max_iter=100)

        extracted_spectra_A[det, order, :] = opt_extraction_A[0]
        extracted_spectra_A_err[det, order, :] = opt_extraction_A[1]
```

Apply the **same fix** in `cooking_subtraction_0101.ipynb` (identical structure).

**Option B (quick diagnostic): Disable Poisson term**

If you want to test quickly without restructuring, pass a large fake gain to suppress the
Poisson term:

```python
opt_extraction_A = utils.optimal_extraction(
    D_full=spectra_clean_A[...].T, V_full=Var.T,
    aper_half=half, obj_cen=half,
    gain=1e30,   # effectively disables Poisson term → uses sky variance only
    ...)
```

Note: Option B still has the wrong signal units (normalized fractions), so the absolute
flux calibration step in `cooking_combine_spectra` needs to rescale accordingly. Option A
is the proper fix.

---

## Bug 2 (Moderate): Telluric Error Propagation Uses Wrong Template

**File:** `Cooking/cooking_combine_spectra.ipynb` (same in `_0101` version)  
**Cell:** `aosje5wea4d` (the flux calibration cell)

The flux is divided by the **airmass-scaled** telluric template `telluric_mtrans`, but the
error is divided by the **unscaled** original template `telluric_template['mtrans']`:

```python
# Flux: uses airmass-scaled template  (alpha_airmass ~ 0.98, so ~2% shallower) ✓
extracted_spectra_A_corrected = extracted_spectra_A_masked / telluric_mtrans[:, 0:5, :]

# Error: uses RAW template  ← inconsistent, should match flux ✗
extracted_spectra_A_err_corrected = extracted_spectra_A_err / telluric_template[:, 0:5, :]['mtrans']
```

Since `alpha_airmass < 1`, `telluric_mtrans > telluric_template['mtrans']` everywhere, so
the error is divided by a smaller number → error is inflated. Maximum effect ~25% at the
masking boundary (mtrans = 0.8). Pixels below 0.8 are already masked to NaN in flux.

**Fix:**

```python
extracted_spectra_A_err_corrected = extracted_spectra_A_err / telluric_mtrans[:, 0:5, :]
extracted_spectra_B_err_corrected = extracted_spectra_B_err / telluric_mtrans[:, 0:5, :]
```

---

## Bug 3 (Moderate): Error Arrays Not Masked in Sync with Flux Arrays

**File:** `Cooking/cooking_combine_spectra.ipynb` (same in `_0101` version)  
**Cell:** `aosje5wea4d` + combining cell `76d9c6c3`

After telluric masking, the flux arrays have NaN where `telluric_mtrans < 0.8`, but the
error arrays are never masked — they retain finite values everywhere. In the combination:

```python
_n_valid_fc = np.sum(np.isfinite(_stack_fc), axis=0)         # counts finite FLUX pixels
extracted_spectra_combined_err = np.where(
    _n_valid_fc > 0,
    np.sqrt(np.nansum(_err_stack_fc**2, axis=0)) / _n_valid_fc,  # nansum includes NaN-flux errors!
    np.nan)
```

`np.nansum(_err_stack_fc**2)` sums squared errors from both A and B, even when one of them
has NaN flux (masked) but finite error. This inflates the combined error by up to √2 at
telluric boundary pixels.

**Fix:** Apply the telluric mask to error arrays immediately after masking the flux:

```python
# After masking flux:
extracted_spectra_A_err_corrected = np.where(mask_A, extracted_spectra_A_err_corrected, np.nan)
extracted_spectra_B_err_corrected = np.where(mask_B, extracted_spectra_B_err_corrected, np.nan)
```

---

## Bug 4 (Minor): Aperture of Only 4 Spatial Pixels

**File:** `Cooking/cooking_subtraction.ipynb`  
**Cell:** 28

```python
half = 2
# Passes center-2:center+2 → 4 spatial pixels to optimal_extraction
```

`optimal_extraction` is called with `aper_half=half=2, obj_cen=half=2`, so it internally
crops `D_full[:, 0:5]` from a 4-column array (numpy silently clips to 4). With a Moffat
PSF of FWHM ≈ 5–8 px, 4 pixels captures ~60–80% of the companion flux.

Consider increasing to `half = 4` (8 pixels, captures ~95%) after fixing Bug 1.

---

## Action Plan (Sequential)

1. **Fix `cooking_subtraction.ipynb` (Bug 1 — Option A)**
   - In Cell 28: replace `D_full=spectra_clean_A[...].T` with `D_full=fit_workflow_A.spectra[...].T`
   - Build `Var_raw` from sky rows of `fit_workflow_A.spectra` (not `spectra_clean_A`)
   - Also consider `half = 3` or `4` while here
   - Apply identical fix to `cooking_subtraction_0101.ipynb`

2. **Re-run `cooking_subtraction.ipynb` and `cooking_subtraction_0101.ipynb`**
   - Check: the saved error files should now have much larger values (in physical e⁻/s)
   - The SNR plot in Cell 32/33 should now show SNR ≈ 3–15 per spectral pixel

3. **Fix `cooking_combine_spectra.ipynb` (Bugs 2 & 3)**
   - Cell `aosje5wea4d`: change error division to use `telluric_mtrans`, not `telluric_template['mtrans']`
   - After that same cell: mask error arrays with `mask_A`, `mask_B`
   - Apply identical fix to `cooking_combine_spectra_0101.ipynb`

4. **Re-run `cooking_combine_spectra.ipynb` and `_0101.ipynb`**
   - Check the SNR plot (Cell `3e89371f`): median SNR should now be > 2
   - Check the CO and H₂O CCF peaks: should sharpen and increase in amplitude

5. **Re-run `cooking_combine_spectra_inter_night.ipynb`**
   - Cell 18 will print median SNR per night and combined — confirm improvement
   - Check inter-night spectral alignment plot (Cell 22) is unchanged

6. **Run a new retrieval** on the fixed combined spectrum
   - Expected improvement: P-T structure constraints, physically reasonable C/O ratio
   - Compare with retrieval 1461589 to quantify how much the P-T and composition changed

---

## Key File Paths

| File | What it contains |
|---|---|
| `Cooking/cooking_subtraction.ipynb` | Main fix location (Bug 1) — 2022-12-31 |
| `Cooking/cooking_subtraction_0101.ipynb` | Same fix — 2023-01-01 |
| `Cooking/cooking_combine_spectra.ipynb` | Bugs 2 & 3 — 2022-12-31 |
| `Cooking/cooking_combine_spectra_0101.ipynb` | Same — 2023-01-01 |
| `Cooking/cooking_combine_spectra_inter_night.ipynb` | Re-run after above; check SNR |
| `/data2/peng/2022-12-31/extracted_spectra_position_A_err.npy` | Currently bad (10–50× too large) |
| `/data2/peng/2022-12-31/extracted_spectra_position_B_err.npy` | Currently bad |
| `/data2/peng/2023-01-01/extracted_spectra_position_A_err.npy` | Currently bad |
| `/data2/peng/2023-01-01/extracted_spectra_position_B_err.npy` | Currently bad |
| `/data2/peng/combined_two_nights/extracted_spectra_combined_two_nights_err.npy` | Stacks bad errors |
| `excalibuhr/src/excalibuhr/utils.py:1627` | Poisson term line in optimal_extraction |

---

## What to Verify After Fixing

- Median per-pixel SNR in `cooking_combine_spectra_inter_night.ipynb` Cell 18 should increase
  from ~0.5 to ~3–10
- CO CCF peak SNR in `cooking_combine_spectra.ipynb` should be detectable (> 3σ)
- New retrieval should show a non-isothermal P-T profile and C/O in the range 0.3–0.8
- The flux values in the extracted 1D spectra should now be in raw e⁻/s units
  (pre-flux calibration), consistent with what `cooking_combine_spectra` expects to scale
  via SINFONI

---

*Audit performed 2026-04-16 by reviewing all six cooking notebooks and excalibuhr utils.py.*

# Plan: Metallicity [C/H] Investigation for DH Tau B Retrieval

## Context

Our equilibrium-chemistry retrieval (v3.4.5) consistently finds [C/H] ≈ −0.91 and C/O ≈ 0.29 for DH Tau B. This is markedly lower than Xuan et al. (2024, ApJ 970:71), who report metallicity = 0.5 × solar ([C/H] ≈ −0.30 dex) and C/O = 0.54 for the same object, using KPIC/K-band at R~35,000. The discrepancy (~0.6 dex in [C/H], factor ~1.9 in C/O) needs to be understood before conclusions about atmospheric composition can be drawn.

---

## Findings: Code Inspection

### File: `Recipe_DH_Tau_B/tasting_retrieval_equa_chem_v3.4.5.py`

**`free_chemistry()` (lines 420–425):**
```python
log_CH_solar = -3.54  # = 8.46 − 12 (Asplund 2021 photospheric)
C_H = np.log10(C/H) - log_CH_solar
```
✅ Solar normalization is correct (A(C) = 8.46 from Asplund 2021).

**`equilibrium_chemistry()` (lines 452–497):**
```python
CO_SOLAR = 0.5495   # Lodders 2020 (matches pRT3 internal solar abundances)
ZH = C_H - np.log10(CO / CO_SOLAR)   # [Z/H] passed to pRT3 table
```
✅ The [C/H] → [Z/H] conversion is physically self-consistent.
⚠️ **Minor inconsistency**: `log_CH_solar` uses Asplund 2021 (C/O_solar = 0.59), but `CO_SOLAR = 0.5495` uses Lodders 2020. Mixed solar references for reporting vs. table conversion. Effect: ~0.03 dex on ZH. Not the main cause of the discrepancy.

**pRT3 table solar reference (verified):**
- `/net/lem/data2/pRT3_formatted/.../equilibrium_chemistry.chemtable.petitRADTRANS.h5`
- Uses Lodders 2020: A(C) = 8.47 → log(C/H)_solar = −3.53
- Difference from Asplund 2021 (−3.54): **0.01 dex — negligible**

**Prior on [C/H] (line 1398):** `[-1.5, 1.5]` — posterior (−0.91) is safely inside; prior not the cause.

**No mathematical bug** was found in either function. The formula is correct and matches González Picos et al. (2025, A&A 693 A298), Eq. 10, including their solar reference value of log(C/H)_solar = −3.54.

---

## Findings: Physical Analysis

### 1. The low C/O is the primary driver of low [C/H]

All retrievals across every version find C/O ≈ 0.24–0.37 (strongly sub-solar), while Xuan 2024 finds C/O = 0.54 (near solar). This is persistent and version-independent:

| Run | [C/H] | C/O |
|-----|--------|-----|
| 1296501 | −0.89 | 0.295 |
| 1379189 | −0.75 | 0.286 |
| 1691251 | −0.83 | 0.315 |
| **1831329 (latest)** | **−0.97** | **0.291** |

The [C/H] → [Z/H] conversion amplifies this: at C/O = 0.29, ZH = C_H + 0.28 dex. This couples C/O and [C/H] — lower C/O forces lower [C/H] for a given [Z/H]. But even comparing [Z/H] ≈ −0.63 directly to Xuan's [Z/H] ≈ −0.30 leaves a ~0.33 dex difference.

### 2. [C/H] − log_g degeneracy is critical

González Picos et al. (2025) retrieved [C/H] = +0.50 with their DG T-P profile (log g ≈ 3.8) but [C/H] = −0.12 with their SG profile (lower log g) for GQ Lup B — **a 0.6 dex shift from T-P profile alone**, with Pearson correlation r = 0.977 between [C/H] and log_g. Our retrieval finds log_g = 3.98 (relatively high). If log_g is overestimated, [C/H] is driven negative by this degeneracy.

### 3. Different methodology from Xuan 2024

- Xuan uses **pRT2** (petitRADTRANS 2.x) equilibrium chemistry, which passes [Z/H] = [Fe/H] directly to the grid. They implicitly define [C/H] ≈ [Fe/H] (valid near solar C/O but not at C/O = 0.29).
- We use **pRT3** with the [C/H] → [Z/H] conversion. At C/O = 0.29, this gives [Z/H] > [C/H] by 0.28 dex, making our [C/H] definition stricter.
- Picos 2025 uses **pRT3 free chemistry** (individual VMRs), which avoids the [Z/H] assumption and finds different C/O from Xuan for the same object (0.50 vs 0.70).

### 4. The chi² values are very small (~0.04)

This implies the noise model (s2, error bars) is generous relative to the residuals, consistent with limited constraining power per pixel at this S/N.

---

## Issues by Severity

| # | Issue | Severity | Effect |
|---|-------|----------|--------|
| 1 | Persistent C/O ≈ 0.29 (vs Xuan's 0.54) drives [C/H] low | **Major** | ~0.3 dex on [C/H] |
| 2 | log_g − [C/H] degeneracy (r ≈ 0.98); log_g = 3.98 may be too high | **Major** | ~0.6 dex potential shift |
| 3 | Mixed solar references: Asplund 2021 for log_CH_solar, Lodders 2020 for CO_SOLAR | **Minor** | ~0.03 dex |
| 4 | No free-chemistry comparison run yet for v3.3/v3.4 | **Diagnostic gap** | unknown |

---

## Recommended Procedure (Scientific Roadmap)

### Step 1 — Immediate diagnostic: run free chemistry (already scripted)
`tasting_retrieval_free_chem_v3.3.py` or `tasting_retrieval_free_chem_v3.4.py` already exists. Submit a free-chemistry retrieval on the same two-night CRIRES+ data with N_live ≥ 600. Compare:
- C/O from free chem vs equilibrium chem
- [C/H] computed from free-chem VMRs vs equilibrium [C/H]
- If free chem also gives C/O ≈ 0.29 → the bias is in the data/normalization, not the equilibrium model
- If free chem gives C/O ≈ 0.50 → the equilibrium [C/H] → [Z/H] coupling or table is introducing a bias

### Step 2 — Investigate the log_g − [C/H] degeneracy
From the current v3.4.5 posterior:
- Make a corner plot showing log_g vs [C/H] (C_H) correlation
- Check if the log_g posterior is consistent with evolutionary model predictions for DH Tau b (m = 12 ± 4 M_Jup, age = 0.7 Myr → log_g ~ 3.5–3.7 from BHAC15/AMES-Dusty in Table 1 of Xuan 2024)
- Consider adding a Gaussian prior on log_g from evolutionary models: `log_g ~ N(3.65, 0.2²)` (centered between 3.5–3.8)

### Step 3 — Report [Z/H] alongside [C/H]
Add to `params_dict` the derived [Z/H]:
```python
self.params_dict['[Z/H]'] = self.model_object.C_H - np.log10(self.model_object.CO / 0.5495)
```
This is directly comparable to Xuan's "metallicity" ([Fe/H] ≈ [Z/H]) and closes the apples-to-oranges comparison.

### Step 4 — Test with consistent solar reference (optional but clean)
In `equilibrium_chemistry()`, replace:
```python
CO_SOLAR = 0.5495   # Lodders 2020
```
with:
```python
CO_SOLAR = 10**(8.46 - 8.69)  # ≈ 0.589, Asplund 2021 photospheric (matches log_CH_solar)
```
Predicted shift: +0.03 dex on ZH → +0.03 dex on [C/H] if posterior shifts. Likely negligible, but makes the solar reference self-consistent throughout.

### Step 5 — Sensitivity test: fix C/O and log_g to Xuan's values
Run a forward model (not a full retrieval) with [C/H] = −0.30, C/O = 0.54, log_g = 3.7, and the retrieved T-P profile. Plot the residuals against the data. If the residuals are significantly worse than the best-fit model, the data genuinely prefer our lower [C/H]. If residuals are comparable, the data cannot discriminate and Xuan's result is consistent with our data.

### Step 6 — Scientific interpretation (after steps 1–5)
- If free chem and equa chem agree: trust the low C/O and [C/H] as a real atmospheric signal.
- If log_g prior shifts [C/H] substantially: report both and note the degeneracy.
- Compare C/O and [C/H] with the stellar host DH Tau A (M2.3, [Fe/H] ≈ 0.0 from Taurus associations; see Xuan 2024 Section 6.4.1): a sub-stellar metallicity for the companion would be unusual if formed via fragmentation.
- Note that Picos 2025 explicitly caution that different T-P parameterizations change [C/H] by up to 0.6 dex for GQ Lup B — a similar caveat applies here.

---

## Critical Files

| File | Role |
|------|------|
| `Recipe_DH_Tau_B/tasting_retrieval_equa_chem_v3.4.5.py` | Current equa-chem script; lines 420–497 (chemistry), 1395–1398 (priors) |
| `Recipe_DH_Tau_B/tasting_retrieval_free_chem_v3.3.py` or `v3.4.py` | Free-chem comparison |
| `retrievals/1831329_N700_.../final_params_dict.pickle` | Latest equa-chem posterior |
| `Recipe_DH_Tau_B/Recipe_reference/Xuan_2024_ApJ_970_71.pdf` | Reference: Table 5 (DH Tau b: C/O=0.54, met=0.5×solar) |
| `Recipe_DH_Tau_B/Recipe_reference/González Picos et al. - 2025 ....pdf` | Reference: log_g−[C/H] degeneracy; Eq. 10 |

---

## Verification

1. **Free-chem retrieval**: Check that the posterior C/O and [C/H] from free chem converges, and compare with equa-chem (Step 1 above)
2. **Forward model at Xuan values**: Confirm residuals are not substantially worse, quantifying data sensitivity
3. **log_g corner plot**: Visualize correlation before and after applying an evolutionary model prior
4. **[Z/H] output**: Confirm `[Z/H] = [C/H] − log10(C/O / C/O_solar)` is stored and printed in the summary

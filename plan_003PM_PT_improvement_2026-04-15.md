# Plan: Piette+2020 P-T Improvement — Series 003PM
**Date**: 2026-04-15

## Context

Retrieval 885739 (002PM-EQ, Piette+2020 P-T + equilibrium chemistry) shows an unrealistic P-T
profile: nearly isothermal from 10⁻⁵ to 10⁻¹ bar ("vertical line in P-T plot"), very low
C/O = 0.30 ([C/H] = −0.91), and log g = 4.0.  Both free-chemistry (002PM-FR) and
equilibrium-chemistry (002PM-EQ) give the same behavior, so the issue is in the P-T setup,
not the chemistry.

---

## Code Inspection Findings

### 1 — No arithmetic bug in `make_pt()`
The PCHIP implementation, node structure `LOG_P_NODES = [2.0, 1.5, 1.0, 0.5, 0.0, −1.0, −2.0,
−3.0]`, anchor at 1 bar (Xuan+2024 convention), and pRT3 interface are all internally correct.

### 2 — Root cause: upper ΔT priors too tight vs. Piette+2020 original

From Piette+2020 Table 1: all upper-atmosphere ΔT parameters use **[0, 1000] K**.
Guidebook currently uses [0, 500–600] K — nearly half as wide.

### 3 — Root cause: 14 "dead" isothermal layers above 10⁻³ bar

The pRT pressure grid is `np.logspace(-5, 2, 50)`, but the shallowest PCHIP node is at 10⁻³
bar.  All 14 layers from 10⁻⁵ to 10⁻³ bar are forced isothermal by `make_pt()`'s extrapolation:
```python
temperature = np.where(log_P_atm < log_P_asc[0], T_asc[0], temperature)
```
These layers:
- contribute negligibly to K2166 flux (optically thin at ~2 µm)
- dilute photosphere pressure resolution (0.001–100 bar gets only 36 of 50 layers)
- are the visual cause of the "vertical line" from 10⁻⁵ to 10⁻³ bar in the P-T plot

**Piette+2020 themselves used 100 layers between 10⁻³ and 100 bar — not 10⁻⁵.**

### 4 — Actual posterior from 885739

```
T_anchor  = 1998.6 K    dT_5 =  321.6 K  (1 → 0.1 bar)
log_g     =   3.998     dT_6 =   91.8 K  (0.1 → 0.01 bar)  ← small; Xuan expects ~250–400 K
dT_1      =  831.6 K    dT_7 =   11.1 K  (0.01 → 0.001 bar) ← essentially zero
chi²      =   0.044  ← very small: many solutions fit equally well

T(0.01 bar) = 1585 K   (Xuan: ~1200 K — 385 K too warm)
T(100 bar)  = 3787 K   (Xuan: ~2800 K — ~1000 K too hot)
```

dT_7 ≈ 0 and dT_6 << prior mean → posterior is likely flat (prior-dominated) for upper layers.

### 5 — Stale comment (trivial)
`pRT_spectrum.__init__` says `# --- P-T profile (DG, Picos+2025) ---` — copy-paste artifact.

---

## Changes to Implement

### Change A — Widen upper ΔT priors to match Piette+2020 Table 1

**File**: `Guidebook_GAStronomy_Piette_v1.0.py`
Functions: `make_free_params_equilibrium()` AND `make_free_params_free_chem()`

```python
# Current → Proposed
'dT_5': ([0,  600], ...),  →  'dT_5': ([0, 1000], ...),  # 1 → 0.1 bar
'dT_6': ([0,  600], ...),  →  'dT_6': ([0, 1000], ...),  # 0.1 → 0.01 bar
'dT_7': ([0,  500], ...),  →  'dT_7': ([0, 1000], ...),  # 0.01 → 0.001 bar
```

### Change B — Narrow pRT pressure grid to (10⁻³, 100) bar

**File**: `Guidebook_GAStronomy_Piette_v1.0.py`, `Retrieval.__init__` (~line 899)

```python
# Current:
self.pressure = np.logspace(-5, 2, self.n_atm_layers)   # 10⁻⁵ to 100 bar

# Proposed:
self.pressure = np.logspace(-3, 2, self.n_atm_layers)   # 10⁻³ to 100 bar
```

**Effect table:**

| Aspect | Current (10⁻⁵–100) | Proposed (10⁻³–100) |
|--------|-------------------|-------------------|
| Layers in PCHIP range | 36 / 50 | **50 / 50** |
| Resolution in photosphere | ~7 layers/dex | **~10 layers/dex** |
| Dead isothermal layers | 14 | **0** |
| Aligns with Piette+2020 | ✗ | **✓** |
| Isothermal extrapolation code | Active (14 layers) | Becomes no-op |

The `make_pt()` extrapolation code remains in place (correct to keep for safety) but has zero
effect since `min(log_P_atm) = −3.0 = log_P_asc[0]`.

### Change C — Remove dT_7 as free parameter (conditional)

**Condition**: inspect `retrievals/885739_.../final_cornerplot.pdf` first.
- If dT_7 marginal is flat → remove it
- If peaked away from 0 → keep it

**Implementation** in tasting scripts (NOT Guidebook), after `make_free_params_*()`:
```python
del free_params['dT_7']
constant_params['dT_7'] = 0.0   # atmosphere isothermal above 0.01 bar
```

K2166 has essentially no sensitivity to 0.001–0.01 bar.  With Change B in place, the 0.001 bar
level is the topmost pRT layer; setting dT_7 = 0 is equivalent to an isothermal top layer.

### Change D — Fix stale comment (trivial)

**File**: `Guidebook_GAStronomy_Piette_v1.0.py`, `pRT_spectrum.__init__` (~line 461)
```python
# Current (stale):
# --- P-T profile (DG, Picos+2025) ---

# Proposed:
# --- P-T profile (Piette+2020 / Xuan+2024) ---
```

---

## Parameters NOT Changed

| Parameter | Decision |
|-----------|----------|
| Anchor at 1 bar (log P = 0.0) | Keep — Xuan+2024 convention |
| `log_g`: Gaussian(3.7, 0.2) | Keep |
| dT_1: [0, 1500] | Keep |
| dT_2: [0, 1000] | Keep |
| dT_3: [0, 700] | Keep |
| dT_4: [0, 500] | Keep |
| `T_anchor`: [1000, 3000] | Keep |

---

## Workflow

1. **Inspect 885739 corner plot** (`final_cornerplot.pdf`) BEFORE coding:
   - dT_7 marginal flat? → implement Change C
   - log_g × C_H anti-correlation: quantify degeneracy

2. **Implement Changes A + B + D** in `Guidebook_GAStronomy_Piette_v1.0.py`

3. **Implement Change C** in tasting scripts if corner plot confirms flat dT_7

4. **Run `testing=True`** to verify no pRT3 errors with the new pressure grid

5. **Submit 003PM-EQ and 003PM-FR** (parallel equilibrium + free chemistry)
   - Label series **003PM** in script docstrings and recording_recipe.md

---

## Verification

After 003PM retrievals finish:
- dT_6 posterior mean > 200 K? (was 92 K in 885739)
- T(0.01 bar) < 1400 K? (was 1585 K; Xuan+2024 finds ~1200 K)
- C/O closer to 0.50–0.54? (was 0.30)
- P-T plot no longer shows "vertical line" from 10⁻³ to 10⁻¹ bar?

---

## Critical Files

| File | Change |
|------|--------|
| `Guidebook_GAStronomy_Piette_v1.0.py` | Changes A, B, D |
| `Tasting/tasting_retrieval_equa_chem_v4.0_piette.py` | Change C (conditional), series → 003PM |
| `Tasting/tasting_retrieval_free_chem_v4.0_piette.py` | Change C (conditional), series → 003PM |
| `retrievals/885739_.../final_cornerplot.pdf` | Inspect BEFORE coding |
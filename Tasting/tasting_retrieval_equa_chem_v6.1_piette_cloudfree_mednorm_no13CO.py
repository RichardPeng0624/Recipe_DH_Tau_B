"""
tasting_retrieval_equa_chem_v6.1_piette_cloudfree_mednorm_no13CO.py
====================================================================
Series 008PM-EQ-MEDNORM-NO13CO — Equilibrium chemistry, cloud-free, per-chip
median normalisation (data AND model), Standard Gaussian likelihood, GP
correlated noise, **13CO removed from the model**.

Purpose of the NO13CO variant (2026-07-09)
------------------------------------------
Bayesian model-comparison test of whether 13CO is required to explain the
data under per-chip-median normalisation. Every mednorm retrieval
(2968924/3160171/3497146/3708914) retrieves 12CO/13CO ~ 130-145 while its
13CO CCF is a NON-detection (SNR ~4 at rv ~ -543 to -767 km/s, nowhere near
the planet RV; De Regt-style knockout validation, validation_clean §9/§13-15)
— unlike the absolute-flux runs (~50, CCF at +47 km/s). So the mednorm
posterior constraint on the isotopolog ratio is not CCF-backed and may be
driven by something other than genuine 13CO lines. This run drops 13CO
entirely and compares evidence against 3497146 (identical config otherwise):

    Delta lnZ = lnZ(3497146, with 13CO) - lnZ(this run, no 13CO)

Jeffreys reading: Delta lnZ > 5 -> 13CO strongly required; |Delta lnZ| <= 2.5
-> data indifferent to 13CO (the ratio posterior is then prior/systematics
-driven and should not be quoted as a detection-backed constraint under this
normalisation). Quote BOTH MultiNest estimators (NS and INS) — they disagree
by ~20-30 nats globally on this series and have flipped the sign of small
comparisons before (recording_recipe.md 2026-07-09).

Implementation: '13C-16O' is removed from the imported Guidebook module's
EQ_SPECIES_PRT3 (and '13CO' from EQ_LABEL_NAMES) IN THIS SCRIPT, after import
— Retrieval.get_species() takes a copy of the module-level list, and
equilibrium_chemistry()'s 13CO branch only triggers if '13C-16O' is in
self.species, so no Guidebook file change is needed (and the Guidebook on
disk stays correct for every other series). log_12CO_13CO is likewise deleted
from the free params (18 free params vs 3497146's 19); its only Guidebook use
is a params.get() with a solar default inside the (now-dead) 13CO branch.
Na/Ca remain removed via the Guidebook state (2026-07-07 patch), matching
3497146.

Everything below the next section is the 008PM-EQ-MEDNORM (no Na/Ca) setup
of job 3497146, unchanged.

Original series purpose (008PM-EQ-MEDNORM)
------------------------------------------
Calibration-strategy sensitivity test, sibling to 008PM-EQ-RUFFIO but using a
different (simpler, more literal) way of removing dependence on absolute
flux calibration. observation_data_reduction_notes.md section 11 ("Systematic
caveat: det2 ord0 flat-extrapolated scale") flags that one of 15
flux-calibration chips (det2, ord0, ~2425-2472 nm, both nights) has its
SINFONI/CRIRES+ scale factor frozen flat rather than fit from real overlap --
a plausibility estimate put the resulting bias at roughly 10-18% too high for
that one chip. Rather than testing that specific bias directly, this series
sidesteps the SINFONI absolute-flux calibration ENTIRELY: it retrieves on the
non-flux-calibrated, sigma-clipped combined spectrum
(extracted_spectra_combined_sigmaclipper.npy), with both the data and the
model spectrum divided by their own per-(order,detector)-chip median before
the likelihood is evaluated. If the composition/kinematics parameters agree
with 007PM-EQ-GP (absolute flux, Standard Gaussian, job 2027997) and
008PM-EQ-RUFFIO (sigmaclipper, Ruffio marginal phi, job 3286785), that is
evidence the SINFONI flux-calibration strategy is not distorting the science
-- three independent ways of handling (or entirely avoiding) absolute flux
calibration converging on the same answer.

Differences from 008PM-EQ-RUFFIO
-----------------------------------------------------------------------------
| Item              | 008PM-EQ-RUFFIO                | 008PM-EQ-MEDNORM (this)      |
|-------------------|----------------------------------|--------------------------------|
| Data              | sigmaclipper (data only norm'd)  | sigmaclipper (data AND model)  |
| Normalisation     | per-chip median, data only       | per-chip median, data + model   |
| Likelihood        | Ruffio marginal (phi/chip)       | Standard Gaussian (phi=1 fixed) |
| Flux scaling      | phi per chip; no M/R              | none; no M/R                    |
| Gravity           | 10^log_g (free param)             | 10^log_g (free param)           |
| Covariance        | GP                                | GP                              |

Why Standard Gaussian here (and not Ruffio)
------------------------------------------------
This is the flip side of 008PM-EQ-RUFFIO's design choice. Under Ruffio's
per-chip phi = (m^T C^-1 d) / (m^T C^-1 m), rescaling the model by any fixed
per-chip constant k is a no-op (m' = m/k gives phi' = k*phi, and
phi'*m' = phi*m identically) -- so normalising the model in addition to the
data would have been pointless there. Here we do the opposite: we fix the
per-chip level match explicitly (each chip's own median, for both data and
model) and do NOT let the likelihood refit an independent scale on top of
that. That means scale_flux must be False (phi = 1 fixed, plain multivariate
Gaussian on the residuals) -- using Ruffio's marginal phi here would try to
refit a scale that, after both sides are already median-normalised to 1,
carries no information, and would incorrectly spend N_phi=15 degrees of
freedom marginalising a factor that's supposed to be fixed by construction.
Data-only vs. data+model normalisation is therefore not an implementation
detail but tied directly to which likelihood is statistically appropriate:
Ruffio-refit-per-chip pairs with data-only normalisation; explicit
match-both-sides normalisation pairs with a fixed-phi Gaussian.

Guidebook change (v4.2 patch, 2026-07-04): added 'per_chip_median' as a new
normalize_method / normalize_flux option -- in _load_night() for the data,
and in pRT_spectrum._apply_normalization() for the model -- so this no longer
needs an ad hoc per-tasting-script helper (as 008PM-EQ-RUFFIO's
_per_chip_median_norm() was). See Guidebook_GAStronomy_Piette_v4.2.py's
top-of-file changelog for the exact diff.

Data-version note: uses the CURRENT (post-2026-06-18) sigmaclipper files for
both nights (extracted_spectra_combined_sigmaclipper.npy, no per-night
suffix) -- verified directly to have zero exact-zero-error pixels and zero
gap pixels outside the K2166 chip boundaries for both nights. 008PM-EQ-RUFFIO
used a stale night-2 file (..._sigmaclipper_0101.npy, 135 zero-error pixels)
that needed an explicit workaround; not needed here.

Interpretation guide (for whoever analyses the output)
---------------------------------------------------------
log_R is NOT a free parameter in this series (no absolute-flux mode), so
this run cannot test whether the calibration bias affects the retrieved
radius -- only the composition/kinematics parameters that don't depend on
absolute flux level (C/O, [C/H], 12CO/13CO, vsini, rv_N1, rv_N2, P-T
profile). Compare directly against 007PM-EQ-GP and 008PM-EQ-RUFFIO for those.

References
----------
De Regt et al. (2024). A&A 688, A116. Section 3.2 (no data normalisation, per-chip phi).
Gonzalez Picos et al. (2024). A&A, Survey II, Section 3.2 (mean-flux normalisation per chip).
Gonzalez Picos et al. (2025). A&A 693, A298. Section 4.1-4.2 (GP covariance).
Xuan et al. (2024). ApJ 970:71. Section 4.3.4, Table C1 (P-T priors for DH Tau b).
Piette & Madhusudhan (2020). MNRAS 497, 5136.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Import Guidebook v4.2 (patched 2026-07-04 with 'per_chip_median' normalisation)
# ---------------------------------------------------------------------------
import importlib.util

_guidebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Tasting_guidebook', 'Guidebook_GAStronomy_Piette_v4.2.py',
)
_spec = importlib.util.spec_from_file_location('Guidebook_GAStronomy_Piette_v4_2', _guidebook_path)
_gb   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gb)

Target                       = _gb.Target
Parameters                   = _gb.Parameters
Retrieval                    = _gb.Retrieval
_load_night                  = _gb._load_night
make_free_params_equilibrium = _gb.make_free_params_equilibrium

# ---------------------------------------------------------------------------
# NO13CO: remove 13CO from the model at the module level, BEFORE any Retrieval
# construction. Retrieval.get_species() returns list(EQ_SPECIES_PRT3), and the
# 13CO mass-fraction branch in equilibrium_chemistry() is conditional on
# '13C-16O' being in self.species — so this is a complete removal (no opacity
# loaded, no mass fraction injected). The Guidebook FILE is untouched.
# ---------------------------------------------------------------------------
assert '13C-16O' in _gb.EQ_SPECIES_PRT3, 'Guidebook state unexpected: 13C-16O missing'
_gb.EQ_SPECIES_PRT3.remove('13C-16O')
_gb.EQ_LABEL_NAMES.remove('13CO')
# Na/Ca must already be absent (Guidebook 2026-07-07 patch) to match 3497146
assert '23Na' not in _gb.EQ_SPECIES_PRT3 and '40Ca' not in _gb.EQ_SPECIES_PRT3, \
    'Guidebook state unexpected: Na/Ca re-enabled — this run must match 3497146 minus 13CO only'
print(f'NO13CO: line species after removal ({len(_gb.EQ_SPECIES_PRT3)}): {_gb.EQ_SPECIES_PRT3}')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURATION
# ===========================================================================

Normalize_method  = 'per_chip_median'  # divides data AND model by their own per-chip median
scaling_parameter = False              # Standard Gaussian likelihood (phi = 1 fixed) — see docstring
use_absolute_flux = False              # no (R/d)² — level already fixed by per-chip median match
cov_mode          = 'GP'               # GP correlated noise kernel (Picos+2024 §5.2)

N_points     = 600
evidence_tol = 0.5


# ===========================================================================
# DATA LOADING (per-chip median normalisation is applied inside _load_night)
# ===========================================================================
# K2166 wavelength boundaries table kept locally only for the night-2 gap/zero
# checks below (diagnostics), not for normalisation — that now happens inside
# _load_night() itself via normalize_method='per_chip_median'.
_K2166 = np.array([
    [[2063.711, 2077.942], [2078.967, 2092.559], [2093.479, 2106.392]],
    [[2143.087, 2157.855], [2158.914, 2173.020], [2173.983, 2187.386]],
    [[2228.786, 2244.133], [2245.229, 2259.888], [2260.904, 2274.835]],
    [[2321.596, 2337.568], [2338.704, 2353.961], [2355.035, 2369.534]],
    [[2422.415, 2439.061], [2440.243, 2456.145], [2457.275, 2472.388]],
])

night1 = '2022-12-31'
wave_N1, flux_N1, err_N1, R_N1 = _load_night(
    night1,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)
print(f'Night 1 ({night1}): estimated R = {R_N1:.0f}')

night2 = '2023-01-01'
wave_N2, flux_N2, err_N2, R_N2 = _load_night(
    night2,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)
# Defensive checks (see data-version note in module docstring): the current
# sigmaclipper files have zero exact-zero-error pixels and zero gap pixels
# outside the K2166 chip boundaries for both nights, verified directly before
# writing this script — these are expected to be no-ops here.
_bad_N2 = (err_N2 == 0)
if _bad_N2.any():
    flux_N2[_bad_N2] = np.nan
    err_N2[_bad_N2]  = np.nan
print(f'Night 2 ({night2}): masked {_bad_N2.sum()} zero-error pixels')
_in_k2166_N2 = np.zeros(len(wave_N2), dtype=bool)
for _o in range(5):
    for _d in range(3):
        _in_k2166_N2 |= (wave_N2 >= _K2166[_o, _d, 0]) & (wave_N2 <= _K2166[_o, _d, 1])
_gap_N2 = ~_in_k2166_N2
print(f'Night 2 ({night2}): removing {_gap_N2.sum()} gap pixels outside K2166 boundaries')
if _gap_N2.any():
    wave_N2 = wave_N2[_in_k2166_N2]
    flux_N2 = flux_N2[_in_k2166_N2]
    err_N2  = err_N2[_in_k2166_N2]
print(f'Night 2 ({night2}): estimated R = {R_N2:.0f}')

print(f'Data source            : extracted_spectra_combined_sigmaclipper.npy (both nights)')
print(f'Normalisation method   : {Normalize_method} (data AND model — model side via pRT_spectrum)')
print(f'Per-chip flux scaling  : {scaling_parameter} (Standard Gaussian, phi=1 fixed)')
print(f'Absolute flux (R/d)²   : {use_absolute_flux}')
print(f'Covariance mode        : {cov_mode}')


# ===========================================================================
# FREE PARAMETERS — 008PM-EQ-MEDNORM
# Swap log_M + log_R → log_g (Gaussian prior from Xuan+2024 Table 1)
# ===========================================================================

constant_params = {
    'chemistry': 'equilibrium',
}

free_params = make_free_params_equilibrium()

# NO13CO: drop the isotopolog-ratio parameter — with '13C-16O' removed from
# the species list it would be a dead (likelihood-flat) dimension. 18 params.
del free_params['log_12CO_13CO']

# Replace absolute-flux params (log_M, log_R) with direct log_g
del free_params['log_M']
del free_params['log_R']
# log_g Gaussian N(3.64, 0.20) — DH Tau B from Xuan+2024 Table 1 (M=12 MJup, R=2.9 RJup)
free_params['log_g'] = ({'type': 'gaussian', 'mu': 3.64, 'sigma': 0.20}, r'$\log g$')

# GP correlated noise parameters (González Picos+2025 §4.2)
free_params['log_a'] = ([-1, 1], r'$\log a_{\rm GP}$')
free_params['log_l'] = ([-3, 0], r'$\log l_{\rm GP}$')


# ===========================================================================
# INITIALISE
# ===========================================================================

parameters = Parameters(free_params, constant_params)

cube = np.random.rand(parameters.ndim)
parameters(cube)

T1 = Target(wl=wave_N1, fl=flux_N1, err=err_N1, name='dh_tau_b_N1')
T2 = Target(wl=wave_N2, fl=flux_N2, err=err_N2, name='dh_tau_b_N2')
T  = T1

retrieval = Retrieval(
    parameters         = parameters,
    N_live_points      = N_points,
    evidence_tolerance = evidence_tol,
    targets            = [T1, T2],
    testing            = False,
    normalize_flux     = Normalize_method,
    per_chip_scaling   = scaling_parameter,
    instrument_res     = [R_N1, R_N2],
    use_absolute_flux  = use_absolute_flux,
    cov_mode           = cov_mode,
)


# ===========================================================================
# RUN RETRIEVAL
# ===========================================================================

retrieval.run_retrieval()


# ===========================================================================
# POST-PROCESSING
# ===========================================================================

retrieval.evaluate()
retrieval.get_params_and_spectrum()

params_dict, model_flux, model_flux_scaled, model_flux_err = \
    retrieval.get_params_and_spectrum()

np.place(retrieval.data_flux, retrieval.data_flux == 0, np.inf)
np.place(model_flux,          model_flux          == 0, np.inf)
np.place(model_flux_scaled,   model_flux_scaled   == 0, np.inf)

np.save(retrieval.output_dir / 'retrieval_model_flux.npy',        model_flux)
np.save(retrieval.output_dir / 'retrieval_model_flux_scaled.npy', model_flux_scaled)
np.save(retrieval.output_dir / 'retrieval_model_wave.npy',        retrieval.data_wave)

if retrieval.two_night_mode and hasattr(retrieval, 'model_flux2'):
    np.save(retrieval.output_dir / 'retrieval_model_flux_N2.npy',  retrieval.model_flux2)
    np.save(retrieval.output_dir / 'retrieval_model_wave_N2.npy',  retrieval.data_wave2)


# ===========================================================================
# PLOTS
# ===========================================================================

mask  = np.isfinite(retrieval.data_flux)
K2166 = T.K2166

fig, ax = plt.subplots(3, 5, figsize=(20, 10))
for order in range(5):
    for det in range(3):
        wave_border    = K2166[4 - order][det]
        sel            = mask & (retrieval.data_wave >= wave_border.min()) & \
                                 (retrieval.data_wave <= wave_border.max())
        wave_cut       = retrieval.data_wave[sel]
        data_flux_cut  = retrieval.data_flux[sel]
        model_flux_cut = model_flux_scaled[sel]
        residuals      = data_flux_cut - model_flux_cut
        std_res        = np.std(residuals)
        ax[det][order].scatter(wave_cut, residuals, color='blue', s=2, alpha=0.7)
        ax[det][order].axhline(0,             color='black', linestyle='--', linewidth=0.8)
        ax[det][order].axhline( 3 * std_res,  color='red',   linestyle=':',  linewidth=0.8)
        ax[det][order].axhline(-3 * std_res,  color='red',   linestyle=':',  linewidth=0.8)
        ax[det][order].set_title(f'Order {27 - order} - Det {det + 1}', fontsize=7)
plt.tight_layout()
plt.savefig(retrieval.output_dir / 'retrieval_data_model_residuals.png', dpi=300)
plt.close()

fig, ax = plt.subplots(3, 5, figsize=(20, 10))
for order in range(5):
    for det in range(3):
        wave_border    = K2166[4 - order][det]
        sel            = mask & (retrieval.data_wave >= wave_border.min()) & \
                                 (retrieval.data_wave <= wave_border.max())
        wave_cut       = retrieval.data_wave[sel]
        data_flux_cut  = retrieval.data_flux[sel]
        model_flux_cut = model_flux_scaled[sel]
        ax[det][order].plot(wave_cut, data_flux_cut,  color='darkgray', linewidth=0.6, label='data')
        ax[det][order].plot(wave_cut, model_flux_cut, color='red',      linewidth=2,   label='model')
        ax[det][order].set_ylim(np.nanmin(data_flux_cut) * 0.8, np.nanmedian(data_flux_cut) * 3)
        ax[det][order].set_title(f'Order {27 - order} - Det {det + 1}', fontsize=7)
        ax[det][order].legend(fontsize=5)
plt.tight_layout()
plt.savefig(retrieval.output_dir / 'retrieval_data_model_spectrum.png', dpi=300)
plt.close()

plt.figure(figsize=(6, 8))
plt.plot(retrieval.model_object.temperature, retrieval.model_object.pressure)
plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (bar)')
plt.title('P-T profile (Piette+2020 PCHIP) — 008PM-EQ-MEDNORM-NO13CO')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

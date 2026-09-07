"""
tasting_retrieval_equa_chem_v6.1_piette_cloud_mednorm.py
==========================================================
Series 008PM-EQ-CLD-MEDNORM — Equilibrium chemistry + EddySed clouds, per-chip
median normalisation (data AND model), Standard Gaussian likelihood, GP
correlated noise.

Based on 008PM-EQ-MEDNORM (tasting_retrieval_equa_chem_v6.1_piette_cloudfree_mednorm.py);
adds the same EddySed cloud parameterisation used by 007PM-EQ-CLD
(tasting_retrieval_equa_chem_v6.1_piette_cloud.py). Everything else (data
source, normalisation, likelihood, GP kernel) is unchanged from the
cloud-free mednorm series -- see that script's docstring for the full
calibration-strategy-sensitivity rationale, which applies identically here.

Cloud parameters (prior ranges from Xuan+2024 Table 3, same as 007PM-EQ-CLD)
-----------------------------------------------------------------------------
  log_X_MgSiO3 : U(-2.3, 1)  — MgSiO3 cloud mass fraction (log10 scale vs. equil.)
  log_X_Fe     : U(-2.3, 1)  — Fe cloud mass fraction (log10 scale vs. equil.)
  fsed         : U(0, 10)    — sedimentation efficiency (shared, pRT3 scalar API)
  log_Kzz      : U(5, 13)    — log10(Kzz / cm^2 s^-1), eddy diffusion coefficient
  sigma_lnorm  : U(1.05, 3)  — log-normal particle size distribution width

Differences from 008PM-EQ-MEDNORM (cloud-free)
-----------------------------------------------------------------------------
| Item              | 008PM-EQ-MEDNORM (cloud-free)  | 008PM-EQ-CLD-MEDNORM (this)     |
|-------------------|----------------------------------|-----------------------------------|
| Chemistry         | make_free_params_equilibrium()  | make_free_params_equil_chem_cloudy |
| Clouds            | None                             | EddySed, MgSiO3 + Fe (Xuan+2024)   |
| N_live            | 600                              | 1000 (matches 007PM-EQ-CLD; more   |
|                   |                                  | free params -> more live points)   |
| Data / norm / lik | sigmaclipper, per-chip median    | sigmaclipper, per-chip median      |
|                   | (data+model), Standard Gaussian  | (data+model), Standard Gaussian    |
| Gravity           | direct log_g                     | direct log_g                       |
| Covariance        | GP                                | GP                                  |

Free-parameter order note: make_free_params_equil_chem_cloudy() builds on
make_free_params_equilibrium() (same log_M/log_R placement) and appends the
five cloud params at the end, BEFORE log_M/log_R are deleted here. Deleting
a dict key in Python does not reorder the remaining keys, but appending
log_g afterwards inserts it as the last key -- so the final posterior column
order is: rv_N1, rv_N2, vsini, epsilon, T_anchor, dT_1..7, C_H, C/O,
log_12CO_13CO, F_H, log_Na, log_Ca, log_X_MgSiO3, log_X_Fe, fsed, log_Kzz,
sigma_lnorm, log_g, log_a, log_l. Confirmed by the same construction pattern
already validated against 2968924's real final_params_dict.pickle for the
cloud-free series; re-verify against this run's own final_params_dict.pickle
before indexing its posterior array in analysis notebooks.

Purpose (same calibration-strategy sensitivity question, cloudy branch)
---------------------------------------------------------------------------
observation_data_reduction_notes.md section 11 flags a ~10-18% plausible bias
in the det2/ord0 flux-calibration chip. 008PM-EQ-MEDNORM already tests
whether that bias affects the cloud-free retrieval by sidestepping the
SINFONI calibration entirely (sigmaclipper + per-chip median on both data and
model + Standard Gaussian). This run repeats that same test in the cloud
model, to check the calibration-strategy conclusion is not contingent on
assuming a clear atmosphere. Additionally pairs naturally with a
cloud-vs-cloud-free Bayesian model comparison (lnZ) purely WITHIN this
calibration-agnostic series, complementing the existing cloud-vs-cloud-free
comparison already run under absolute flux calibration (2499181 vs 2027997,
Delta lnZ = -1.7 nats, no significant preference).

Interpretation guide
------------------------
log_R is NOT a free parameter in this series (no absolute-flux mode) -- same
limitation as 008PM-EQ-MEDNORM. Compare composition/kinematics/cloud
parameters against 007PM-EQ-CLD (2499181, absolute flux) for the calibration
question, and against 008PM-EQ-MEDNORM (this series' own cloud-free sibling)
for the cloud-vs-cloud-free question under calibration-agnostic normalisation.

References
----------
De Regt et al. (2024). A&A 688, A116. Section 3.2 (no data normalisation, per-chip phi).
Gonzalez Picos et al. (2024). A&A, Survey II, Section 3.2 (mean-flux normalisation per chip).
Gonzalez Picos et al. (2025). A&A 693, A298. Section 4.1-4.2 (GP covariance).
Ackerman & Marley (2001). ApJ 556, 872. (EddySed cloud model)
Xuan et al. (2024). ApJ 970:71. Section 4.3.3, Table 3 (cloud priors); Section 4.3.4, Table C1 (P-T priors).
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

Target                             = _gb.Target
Parameters                         = _gb.Parameters
Retrieval                          = _gb.Retrieval
_load_night                        = _gb._load_night
make_free_params_equil_chem_cloudy = _gb.make_free_params_equil_chem_cloudy
CLOUD_SPECIES_DEFAULT              = _gb.CLOUD_SPECIES_DEFAULT

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURATION
# ===========================================================================

Normalize_method  = 'per_chip_median'  # divides data AND model by their own per-chip median
scaling_parameter = False              # Standard Gaussian likelihood (phi = 1 fixed)
use_absolute_flux = False              # no (R/d)² — level already fixed by per-chip median match
cov_mode          = 'GP'               # GP correlated noise kernel (Picos+2024 §5.2)

N_points     = 1000  # matches 007PM-EQ-CLD — more free params (5 cloud) than cloud-free series
evidence_tol = 0.5


# ===========================================================================
# DATA LOADING (per-chip median normalisation is applied inside _load_night)
# ===========================================================================
# K2166 wavelength boundaries table kept locally only for the night-2 gap/zero
# checks below (diagnostics), not for normalisation — that happens inside
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
# Defensive checks (see 008PM-EQ-MEDNORM's docstring): the current sigmaclipper
# files have zero exact-zero-error pixels and zero gap pixels outside the
# K2166 chip boundaries for both nights — verified before writing that script.
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
print(f'Cloud species          : {CLOUD_SPECIES_DEFAULT}')


# ===========================================================================
# FREE PARAMETERS — 008PM-EQ-CLD-MEDNORM
# Equilibrium chem + EddySed clouds; swap log_M + log_R -> log_g
# ===========================================================================

constant_params = {
    'chemistry'     : 'equilibrium',
    'cloud_species' : CLOUD_SPECIES_DEFAULT,
}

free_params = make_free_params_equil_chem_cloudy(CLOUD_SPECIES_DEFAULT)

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
plt.title('P-T profile (Piette+2020 PCHIP) — 008PM-EQ-CLD-MEDNORM EddySed GP')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

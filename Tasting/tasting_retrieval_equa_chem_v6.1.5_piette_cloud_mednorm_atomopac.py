"""
tasting_retrieval_equa_chem_v6.1.5_piette_cloud_mednorm_atomopac.py
=====================================================================
Series 008PM-EQ-CLD-MEDNORM-ATOMOPAC — Equilibrium chemistry + EddySed
clouds, per-chip median normalisation (data AND model), Standard Gaussian
likelihood, GP correlated noise, Na+Ca opacity restored to the model
spectrum WITH ZERO NEW FREE PARAMETERS.

Direct sibling of 008PM-EQ-CLD-MEDNORM (job 3708914, this script's
un-suffixed v6.1 ancestor) -- identical data, normalisation, likelihood, GP
kernel, and cloud parameterisation. The ONLY change is that this run's model
spectrum includes Na and Ca opacity via Guidebook v4.2.5, whereas 3708914
has neither. Cloud counterpart of
tasting_retrieval_equa_chem_v6.1.5_piette_cloudfree_mednorm_atomopac.py
(job TBD, cloud-free sibling of this run) -- see that script's docstring for
the full Na/Ca-opacity-vs-Na/Ca-as-free-parameter rationale, which applies
identically here.

Cloud parameters (prior ranges from Xuan+2024 Table 3, same as 3708914)
-----------------------------------------------------------------------------
  log_X_MgSiO3 : U(-2.3, 1)  — MgSiO3 cloud mass fraction (log10 scale vs. equil.)
  log_X_Fe     : U(-2.3, 1)  — Fe cloud mass fraction (log10 scale vs. equil.)
  fsed         : U(0, 10)    — sedimentation efficiency (shared, pRT3 scalar API)
  log_Kzz      : U(5, 13)    — log10(Kzz / cm^2 s^-1), eddy diffusion coefficient
  sigma_lnorm  : U(1.05, 3)  — log-normal particle size distribution width

Free-parameter order note: make_free_params_equil_chem_cloudy() builds on
make_free_params_equilibrium() (same log_M/log_R placement) and appends the
five cloud params at the end, BEFORE log_M/log_R are deleted here -- same
construction pattern validated against 3708914's real final_params_dict.
pickle. Expected free-parameter count: 24 (same as 3708914 — Na/Ca add
opacity, not dimensions). Re-verify against this run's own
final_params_dict.pickle before indexing its posterior array.

Purpose
-------
Repeats the cloud-free sibling's calibration/opacity test in the cloud
model, so the Na/Ca-opacity conclusion is not contingent on assuming a
clear atmosphere -- mirrors how 3708914 repeated 3497146's test in the
cloud model for the original (parameter-only) Na/Ca A/B. Also pairs with a
cloud-vs-cloud-free Bayesian model comparison (lnZ) WITHIN this
Na/Ca-opacity-restored series, directly comparable to the existing
cloud-vs-cloud-free comparisons under both other calibration strategies
(2499181 vs 2027997, absolute flux; 3708914 vs 3497146, mednorm no-opacity).

Interpretation guide
------------------------
log_R is NOT a free parameter in this series (no absolute-flux mode) — same
limitation as its ancestors. Compare directly against 3708914 (this run's
mednorm/no-Na-Ca-opacity twin) for the opacity question, and against 3497146
/ this run's own cloud-free sibling for the cloud-vs-cloud-free question.

References
----------
De Regt et al. (2024). A&A 688, A116. Section 3.2 (no data normalisation, per-chip phi).
Gonzalez Picos et al. (2024). A&A, Survey II, Section 3.2 (mean-flux normalisation per chip).
Gonzalez Picos et al. (2025). A&A 693, A298. Section 4.1-4.2 (GP covariance).
Ackerman & Marley (2001). ApJ 556, 872. (EddySed cloud model)
Xuan et al. (2024). ApJ 970:71. Section 4.3.3, Table 3 (cloud priors); Section 4.3.4, Table C1 (P-T priors).
Piette & Madhusudhan (2020). MNRAS 497, 5136.
Lodders, K. (2020). Solar elemental abundances. arXiv:1912.00844.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Import Guidebook v4.2.5 (Na/Ca opacity restored, zero new free params)
# ---------------------------------------------------------------------------
import importlib.util

_guidebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Tasting_guidebook', 'Guidebook_GAStronomy_Piette_v4.2.5.py',
)
_spec = importlib.util.spec_from_file_location('Guidebook_GAStronomy_Piette_v4_2_5', _guidebook_path)
_gb   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gb)

Target                             = _gb.Target
Parameters                         = _gb.Parameters
Retrieval                          = _gb.Retrieval
_load_night                        = _gb._load_night
make_free_params_equil_chem_cloudy = _gb.make_free_params_equil_chem_cloudy
CLOUD_SPECIES_DEFAULT              = _gb.CLOUD_SPECIES_DEFAULT
EQ_SPECIES_PRT3                    = _gb.EQ_SPECIES_PRT3
EQ_LABEL_NAMES                     = _gb.EQ_LABEL_NAMES

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

N_points     = 600  # matches 3708914 — more free params (5 cloud) than cloud-free series
evidence_tol = 0.5

print(f'Equilibrium-chemistry opacity species (v4.2.5): {EQ_SPECIES_PRT3}')
print(f'Equilibrium-chemistry label names     (v4.2.5): {EQ_LABEL_NAMES}')
assert '23Na' in EQ_SPECIES_PRT3 and '40Ca' in EQ_SPECIES_PRT3, \
    'v4.2.5 must carry Na and Ca in EQ_SPECIES_PRT3 — wrong Guidebook loaded?'


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
# FREE PARAMETERS — 008PM-EQ-CLD-MEDNORM-ATOMOPAC
# Equilibrium chem + EddySed clouds; swap log_M + log_R -> log_g
# Na/Ca get OPACITY (via Guidebook v4.2.5) but NO free parameter — asserted
# below to guard against a future Guidebook edit silently reintroducing them.
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

assert 'log_Na' not in free_params and 'log_Ca' not in free_params, \
    'This series must have ZERO new free params for Na/Ca — found one.'
print(f'Free parameters ({len(free_params)}): {list(free_params.keys())}')
print('Expected count: 24 (same as 3708914 — Na/Ca add opacity, not dimensions)')


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
plt.title('P-T profile (Piette+2020 PCHIP) — 008PM-EQ-CLD-MEDNORM-ATOMOPAC')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

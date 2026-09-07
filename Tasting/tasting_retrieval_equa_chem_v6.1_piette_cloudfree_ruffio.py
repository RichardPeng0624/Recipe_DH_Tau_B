"""
tasting_retrieval_equa_chem_v6.1_piette_cloudfree_ruffio.py
============================================================
Series 008PM-EQ-RUFFIO — Equilibrium chemistry, cloud-free, no continuum
normalisation, per-chip Ruffio (φ) likelihood, GP correlated noise.

Follows the Picos+2024 / De Regt+2024 / Xuan+2024 approach for direct comparison
with SupJup literature:
  - Per-chip median normalisation applied to DATA ONLY (in-place after _load_night,
    before Target construction), following Picos+2024 §3.2: "divided by the mean
    flux of each order-detector pair".  Median used instead of mean for robustness
    against masked pixels.  Model is NOT normalised; per-chip φ absorbs the ratio.
  - Per-chip flux scaling (φ) via Ruffio+2019 marginal likelihood (scale_flux=True).
  - No absolute flux mode: (R/d)² not applied; gravity from log_g directly.
  - GP correlated noise (cov_mode='GP') following Picos+2024 §5.2.

Differences from 007PM-EQ-GP (retrieval 3062330 / v6.1_piette_cloudfree.py)
-----------------------------------------------------------------------------
| Item              | 007PM-EQ-GP (3062330)       | 008PM-EQ-RUFFIO (this)      |
|-------------------|-----------------------------|------------------------------|
| Normalisation     | None (absolute flux)        | None (φ absorbs mean/chip)   |
| Likelihood        | Standard Gaussian (φ=1)     | Ruffio marginal (φ per chip) |
| Flux scaling      | (R/d)² via log_M + log_R   | φ per chip; no M/R           |
| Gravity           | G·M/R² (derived)           | 10^log_g (free param)        |

Guidebook change (v4.2): pRT_spectrum.__init__ gravity block now conditional —
falls back to 10^log_g when log_M/log_R are absent from params.

Normalisation note
------------------
De Regt+2024 retain absolute flux and anchor the first chip to R[R_Jup].
Picos+2024 (§3.2) divide data by the per-chip mean flux (simple scalar, NOT SG),
then use per-chip φ via the Ruffio likelihood.  We follow Picos+2024: no SG
normalisation, with per-chip φ (Normalize_method=None, scaling_parameter=True).
SG normalisation was incorrectly used in the initial setup of this script — it
removes continuum SHAPE within each chip, which is far more aggressive than the
simple mean-flux division used by De Regt/Picos.

Data note (retrieval 2777844 → new)
-------------------------------------
Previous runs used flux-calibrated data (extracted_spectra_combined_flux_cal.npy),
which applies a SINFONI-based polynomial response correction.  This introduced
artefacts (blaze residuals, SINFONI resolution mismatch).  This run uses the
non-flux-calibrated combined spectra (extracted_spectra_combined_sigmaclipper.npy /
_0101.npy) directly — closer to the SupJup approach, where only the telluric
standard provides throughput correction.  Absolute flux scale per chip is handled
entirely by per-chip φ; log_M / log_R are absent from free_params.

References
----------
De Regt et al. (2024). A&A 688, A116. §3.2 (no data normalisation, per-chip φ).
González Picos et al. (2024). A&A, Survey II, §3.2 (mean-flux normalisation per chip),
    §5.1 (Ruffio marginal φ per chip), §5.2 (GP kernel).
González Picos et al. (2025). A&A 693, A298. §4.1–4.2 (GP covariance).
Xuan et al. (2024). ApJ 970:71. §4.3.4, Table C1 (P-T priors for DH Tau b).
Ruffio et al. (2019). ApJ 881:1. Eq. A1–A6 (marginal likelihood with φ and s²).
Piette & Madhusudhan (2020). MNRAS 497, 5136.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Import Guidebook v4.2
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================================
# CONFIGURATION
# ===========================================================================

Normalize_method  = None   # no continuum normalisation — φ absorbs per-chip mean
scaling_parameter = True   # per-chip φ via Ruffio+2019 marginal likelihood
use_absolute_flux = False  # no (R/d)² — φ absorbs total flux scaling per chip
cov_mode          = 'GP'   # GP correlated noise kernel (Picos+2024 §5.2)

N_points     = 600
evidence_tol = 0.5


# ===========================================================================
# DATA LOADING + PER-CHIP MEDIAN NORMALISATION (Picos+2024 §3.2)
# ===========================================================================
# Picos+2024 divide each order-detector pair by its mean flux before retrieval.
# We do the same here (using median for robustness against outlier pixels),
# applied to data only.  The model is NOT normalised; per-chip φ absorbs the
# ratio model_flux / data_chip_median at every likelihood evaluation.
# This is done in-place on the 1-D arrays returned by _load_night, before
# passing them to Target — no Guidebook change required.

# K2166 wavelength boundaries (nm) for 5 orders × 3 detectors
_K2166 = np.array([
    [[2063.711, 2077.942], [2078.967, 2092.559], [2093.479, 2106.392]],
    [[2143.087, 2157.855], [2158.914, 2173.020], [2173.983, 2187.386]],
    [[2228.786, 2244.133], [2245.229, 2259.888], [2260.904, 2274.835]],
    [[2321.596, 2337.568], [2338.704, 2353.961], [2355.035, 2369.534]],
    [[2422.415, 2439.061], [2440.243, 2456.145], [2457.275, 2472.388]],
])

def _per_chip_median_norm(wave, flux, err):
    """Divide each K2166 order-detector chip by its median flux (in-place)."""
    for order in range(5):
        for det in range(3):
            mask = (wave >= _K2166[order, det, 0]) & (wave <= _K2166[order, det, 1])
            if mask.sum() < 10:
                continue
            med = np.nanmedian(flux[mask])
            if med > 0:
                flux[mask] /= med
                err[mask]  /= med


night1 = '2022-12-31'
wave_N1, flux_N1, err_N1, R_N1 = _load_night(
    night1,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)
_per_chip_median_norm(wave_N1, flux_N1, err_N1)
print(f'Night 1 ({night1}): estimated R = {R_N1:.0f}')

night2 = '2023-01-01'
wave_N2, flux_N2, err_N2, R_N2 = _load_night(
    night2,
    flux_file='extracted_spectra_combined_sigmaclipper_0101.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper_0101.npy',
    normalize_method=Normalize_method,
)
# Night 2 sigmaclipper has 135 pixels with err=0 (not NaN) — mask them to
# prevent division-by-zero in the likelihood.
_bad_N2 = (err_N2 == 0)
flux_N2[_bad_N2] = np.nan
err_N2[_bad_N2]  = np.nan
print(f'Night 2 ({night2}): masked {_bad_N2.sum()} zero-error pixels')
_in_k2166_N2 = np.zeros(len(wave_N2), dtype=bool)
for _o in range(5):
    for _d in range(3):
        _in_k2166_N2 |= (wave_N2 >= _K2166[_o, _d, 0]) & (wave_N2 <= _K2166[_o, _d, 1])
_gap_N2 = ~_in_k2166_N2
print(f'Night 2 ({night2}): removing {_gap_N2.sum()} gap pixels outside K2166 boundaries')
wave_N2 = wave_N2[_in_k2166_N2]
flux_N2 = flux_N2[_in_k2166_N2]
err_N2  = err_N2[_in_k2166_N2]
_per_chip_median_norm(wave_N2, flux_N2, err_N2)
print(f'Night 2 ({night2}): estimated R = {R_N2:.0f}')

print(f'Normalisation method  : per-chip median (data only, Picos+2024 §3.2)')
print(f'Per-chip flux scaling : {scaling_parameter}')
print(f'Absolute flux (R/d)²  : {use_absolute_flux}')
print(f'Covariance mode       : {cov_mode}')


# ===========================================================================
# FREE PARAMETERS — 008PM-EQ-RUFFIO
# Swap log_M + log_R → log_g (Gaussian prior from Xuan+2024 Table 1)
# ===========================================================================

constant_params = {
    'chemistry': 'equilibrium',
}

free_params = make_free_params_equilibrium()

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
plt.title('P-T profile (Piette+2020 PCHIP) — 008PM-EQ-RUFFIO')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

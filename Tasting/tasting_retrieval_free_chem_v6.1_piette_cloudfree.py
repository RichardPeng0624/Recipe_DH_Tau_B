"""
tasting_retrieval_free_chem_v6.2_piette_cloudfree.py
=====================================================
Series 007PM-FR — Free chemistry, cloud-free, with GP correlated noise

Based on 007PM-FR (tasting_retrieval_free_chem_v6.1_piette_cloudfree.py).
Uses Guidebook_GAStronomy_Piette_v4.2.

Changes from v6.1
-----------------
- Imports Guidebook v4.2 (banded GP Cholesky, _ChipCovProxy, species-info
  CSV cache — ~18× memory reduction, ~300× Cholesky speedup, OOM fix).
  No API change; log_l prior [-3, 0] unchanged and already correct.

Changes from v6.0 (v6.1)
--------------------------
- Imports Guidebook v4.1 (adds CovarianceGP / cov_mode support).
- cov_mode = 'GP' — correlated noise modelled with heteroscedastic GP kernel
  following González Picos et al. (2025) §4.2 (= de Regt+2024 formulation):
    Σ₀,ij = δᵢⱼσᵢ²  +  a²((σᵢ+σⱼ)/2)² exp(−Δλ²/2l²)
- Two new free parameters added to free_params:
    log_a : U(−1, 1)   — log10 GP amplitude (dimensionless, relative to noise)
    log_l : U(−3, 0)   — log10 GP length scale (nm); 0.001–1 nm covers
                          sub-pixel to inter-order correlation scales.
- cov_mode='GP' passed to Retrieval constructor.

Species retrieved
-----------------
Molecular : H2O, 12CO, 13CO, CH4
Atomic/HF : HF, Na, Ca  (Ti commented out)

Two-night joint retrieval for DH Tau B (CRIRES+ K2166).
Data: extracted_spectra_combined_flux_cal.npy  (W/m²/μm, absolute flux)
      extracted_spectra_combined_err_flux_cal.npy
NO normalisation — absolute flux level preserved for (R/d)² scaling.

P-T model
---------
Piette & Madhusudhan (2020) / Xuan+2024 PCHIP parameterisation.
  8 pressure nodes: log P = [+0.7, 0.0, -0.3, -0.7*, -1.0, -1.5, -3.0, -5.0] bar.
  T_anchor at log P = -0.7 (0.2 bar), free params: T_anchor + dT_1…dT_7.

References
----------
González Picos et al. (2025). A&A, 693, A298. §4.2.
Picos et al. (2024). A&A. §4.2, Table 3.
Piette & Madhusudhan (2020). MNRAS 497, 5136.
Xuan et al. (2024). ApJ 970:71.
Gaia DR3: d(DH Tau) = 135.2 pc.
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

Target                     = _gb.Target
Parameters                 = _gb.Parameters
Retrieval                  = _gb.Retrieval
_load_night                = _gb._load_night
make_free_params_free_chem = _gb.make_free_params_free_chem

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ===========================================================================
# CONFIGURATION
# ===========================================================================

Normalize_method  = None     # DISABLED — absolute flux mode requires no normalisation
scaling_parameter = False    # per-chip flux scaling disabled; absolute (R/d)² used
use_absolute_flux = True     # enable (R/d)² physical scaling in pRT_spectrum
cov_mode          = 'GP'     # 'diag' → diagonal only | 'GP' → correlated noise kernel

N_points     = 800
evidence_tol = 0.5

# ===========================================================================
# DATA LOADING (no normalisation)
# ===========================================================================

night1 = '2022-12-31'
wave_N1, flux_N1, err_N1, R_N1 = _load_night(
    night1,
    flux_file='extracted_spectra_combined_flux_cal.npy',
    err_file ='extracted_spectra_combined_err_flux_cal.npy',
    normalize_method=Normalize_method,
)
print(f'Night 1 ({night1}): estimated R = {R_N1:.0f}')

night2 = '2023-01-01'
wave_N2, flux_N2, err_N2, R_N2 = _load_night(
    night2,
    flux_file='extracted_spectra_combined_flux_cal.npy',
    err_file ='extracted_spectra_combined_err_flux_cal.npy',
    normalize_method=Normalize_method,
)
print(f'Night 2 ({night2}): estimated R = {R_N2:.0f}')

print(f'Normalisation method: {Normalize_method}  (DISABLED — absolute flux mode)')
print(f'Per-chip flux scaling: {scaling_parameter}')
print(f'Absolute flux scaling (R/d)²: {use_absolute_flux}')
print(f'Covariance mode: {cov_mode}')
print(f'Mean flux N1: {np.nanmean(flux_N1):.4e} W/m²/μm')
print(f'Mean flux N2: {np.nanmean(flux_N2):.4e} W/m²/μm')

# Legacy alias
wave_flat           = wave_N1
spectra_AB_flat     = flux_N1
spectra_AB_err_flat = err_N1


# ===========================================================================
# FREE PARAMETERS — 007PM-FR (free chem, cloud-free, GP correlated noise)
# ===========================================================================

constant_params = {}   # no chemistry key → defaults to free VMR mode

free_params = make_free_params_free_chem()   # includes log_HF, log_Na, log_Ca

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
        ax[det][order].set_ylim(np.nanmin(data_flux_cut)*0.8, np.nanmedian(data_flux_cut)*3)
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
plt.title('P-T profile (Piette+2020 PCHIP) — 007PM-FR cloud-free GP')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

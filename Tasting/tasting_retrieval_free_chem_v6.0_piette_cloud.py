"""
tasting_retrieval_free_chem_v6.0_piette_cloud.py
=================================================
Series 006PM-FR-CLD — Free chemistry + EddySed clouds (MgSiO3 + Fe)
                       with atomic species Na, Ca and HF  (Ti removed)

Based on 005PM-FR-CLD (tasting_retrieval_free_chem_v5.0_piette_cloud.py).
Uses Guidebook_GAStronomy_Piette_v4.0.

Changes from v5.0
-----------------
- Imports Guidebook v4.0 (adds HF, Na, Ca to opacity species list; Ti removed)
- make_free_params_free_chem_cloudy() includes log_HF, log_Na, log_Ca
  with prior U(-12, -2) following Picos et al. (2024) §4.2, Table 3.
  log_Ti commented out (not retrieved in this run).

Species retrieved
-----------------
Molecular : H2O, 12CO, 13CO, CH4
Atomic/HF : HF, Na, Ca  [new in v4.0; Ti removed from this run]

Cloud parameters (prior ranges from Xuan+2024 Table 3):
  log_X_MgSiO3 : U(-2.3, 1)  — MgSiO3 cloud mass fraction (log10 scale vs. equil.)
  log_X_Fe     : U(-2.3, 1)  — Fe cloud mass fraction (log10 scale vs. equil.)
  fsed         : U(0, 10)    — sedimentation efficiency (shared, pRT3 scalar API)
  log_Kzz      : U(5, 13)    — log10(Kzz / cm² s⁻¹), eddy diffusion coefficient
  sigma_lnorm  : U(1.05, 3)  — log-normal particle size distribution width

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
Picos et al. (2024). A&A. §4.2, Table 3.
Ackerman & Marley (2001). ApJ 556, 872.
Xuan et al. (2024). ApJ 970:71. §4.3.3, Table 3.
Piette & Madhusudhan (2020). MNRAS 497, 5136.
Gaia DR3: d(DH Tau) = 135.2 pc.
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Import Guidebook v4.0
# ---------------------------------------------------------------------------
import importlib.util

_guidebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Tasting_guidebook', 'Guidebook_GAStronomy_Piette_v4.0.py',
)
_spec = importlib.util.spec_from_file_location('Guidebook_GAStronomy_Piette_v4_0', _guidebook_path)
_gb   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gb)

Target                            = _gb.Target
Parameters                        = _gb.Parameters
Retrieval                         = _gb.Retrieval
_load_night                       = _gb._load_night
make_free_params_free_chem_cloudy = _gb.make_free_params_free_chem_cloudy
CLOUD_SPECIES_DEFAULT             = _gb.CLOUD_SPECIES_DEFAULT

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
print(f'Mean flux N1: {np.nanmean(flux_N1):.4e} W/m²/μm')
print(f'Mean flux N2: {np.nanmean(flux_N2):.4e} W/m²/μm')

# Legacy alias
wave_flat           = wave_N1
spectra_AB_flat     = flux_N1
spectra_AB_err_flat = err_N1


# ===========================================================================
# FREE PARAMETERS — 006PM-FR-CLD (free chem + EddySed clouds + HF/Na/Ca/Ti)
# ===========================================================================

constant_params = {
    'chemistry'     : 'free',
    'cloud_species' : CLOUD_SPECIES_DEFAULT,
}

free_params = make_free_params_free_chem_cloudy(CLOUD_SPECIES_DEFAULT)


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
plt.title('P-T profile (Piette+2020 PCHIP) — 006PM-FR-CLD EddySed clouds')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

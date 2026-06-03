"""
tasting_retrieval_free_chem_v4.0_picos_DG.py
=============================================
Series 002DG-FR — Free chemistry + Dynamic Gradients (DG) P-T profile
(González Picos et al. 2025)

Two-night joint retrieval for DH Tau B (CRIRES+ K2166).
Calls Guidebook_GAStronomy_Picos_DG_v1.0 for all classes and RT machinery.

Data
----
Night 1: 2022-12-31
Night 2: 2023-01-01
Files  : extracted_spectra_combined_sigmaclipper.npy  (flux)
         extracted_spectra_combined_err_sigmaclipper.npy (error)

P-T model
---------
Dynamic Gradients (DG) parameterisation (Picos+2025 §4.4.2, Eqs. 11-12).
  6 pressure nodes (100 → 10⁻⁵ bar) with piecewise-gradient interpolation.
  pRT grid: 50 layers from 10⁻⁵ to 100 bar.
  Free params: T_bottom + nabla_RCE, nabla_0 – nabla_4, log_P_RCE,
               dlog_P_bot, dlog_P_top.
  Convective-zone gradients constrained ≥ 0.04 and ≤ nabla_RCE;
  radiative-zone gradients constrained 0 ≤ ∇ ≤ nabla_RCE.

Chemistry: FREE (log VMR per species)
Species: H2O, 12CO, 13CO, CH4

References
----------
González Picos, D. et al. (2025). GQ Lup b atmospheric retrieval with CRIRES+.
  A&A (in press). §4.4.2 (DG P-T profile), Table C.1 (priors).
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Import Guidebook module (sister file in the same Recipe directory)
# File name contains dots (v1.0), so use importlib rather than a plain import.
# ---------------------------------------------------------------------------
import importlib.util

_guidebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'Guidebook_GAStronomy_Picos_DG_v1.0.py',
)
_spec = importlib.util.spec_from_file_location('Guidebook_GAStronomy_Picos_DG_v1_0', _guidebook_path)
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

Normalize_method  = 'savgol'   # 'median' | 'savgol' | None
scaling_parameter = False       # per-chip flux scaling (phi); False = no scaling

N_points     = 700
evidence_tol = 0.5

# ===========================================================================
# DATA LOADING AND NORMALISATION
# ===========================================================================

# -----------------------------------------------------------------------
# Night 1: 2022-12-31
# -----------------------------------------------------------------------
night1 = '2022-12-31'
wave_N1, flux_N1, err_N1, R_N1 = _load_night(
    night1,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)
print(f'Night 1 ({night1}): estimated R = {R_N1:.0f}')

# -----------------------------------------------------------------------
# Night 2: 2023-01-01
# -----------------------------------------------------------------------
night2 = '2023-01-01'
wave_N2, flux_N2, err_N2, R_N2 = _load_night(
    night2,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)
print(f'Night 2 ({night2}): estimated R = {R_N2:.0f}')

print(f'Normalisation method: {Normalize_method}')
print(f'Per-chip flux scaling: {scaling_parameter}')

# Legacy alias (night 1) used by plotting code below
wave_flat           = wave_N1
spectra_AB_flat     = flux_N1
spectra_AB_err_flat = err_N1


# ===========================================================================
# FREE PARAMETERS — DG P-T (Picos+2025) + free chemistry (002DG-FR)
# ===========================================================================

constant_params = {}   # chemistry flag absent → pRT_spectrum defaults to 'free'

# Default free-chemistry parameter set from Guidebook (nabla_* + T_bottom +
# log VMRs for H2O, 12CO, 13CO, CH4).  Modify priors here if needed.
free_params = make_free_params_free_chem()

# Uncomment to extend with additional species:
# free_params['log_NH3']  = ([-12, -2], r'$\log$ NH$_3$')
# free_params['log_H2S']  = ([-12, -2], r'$\log$ H$_2$S')
# free_params['log_HCN']  = ([-12, -2], r'$\log$ HCN')


# ===========================================================================
# INITIALISE PARAMETERS, TARGETS, RETRIEVAL
# ===========================================================================

parameters = Parameters(free_params, constant_params)

# Draw a random initial cube to populate parameters.params
cube = np.random.rand(parameters.ndim)
parameters(cube)

# Target objects — one per night
T1 = Target(wl=wave_N1, fl=flux_N1, err=err_N1, name='dh_tau_b_N1')
T2 = Target(wl=wave_N2, fl=flux_N2, err=err_N2, name='dh_tau_b_N2')

# Backward-compat alias used by the plotting code below
T = T1

# Joint two-night Retrieval — instrument_res estimated per-night from wavelength solution
retrieval = Retrieval(
    parameters         = parameters,
    N_live_points      = N_points,
    evidence_tolerance = evidence_tol,
    targets            = [T1, T2],
    testing            = False,
    normalize_flux     = Normalize_method,
    per_chip_scaling   = scaling_parameter,
    instrument_res     = [R_N1, R_N2],
)


# ===========================================================================
# RUN RETRIEVAL
# ===========================================================================

retrieval.run_retrieval()


# ===========================================================================
# POST-PROCESSING: extract best-fit spectrum and save outputs
# ===========================================================================

retrieval.evaluate()
retrieval.get_params_and_spectrum()

params_dict, model_flux, model_flux_scaled, model_flux_err = \
    retrieval.get_params_and_spectrum()

# Replace exact zeros with inf so they are excluded from plots
np.place(retrieval.data_flux, retrieval.data_flux == 0, np.inf)
np.place(model_flux,          model_flux          == 0, np.inf)
np.place(model_flux_scaled,   model_flux_scaled   == 0, np.inf)

np.save(retrieval.output_dir / 'retrieval_model_flux.npy',        model_flux)
np.save(retrieval.output_dir / 'retrieval_model_flux_scaled.npy', model_flux_scaled)
np.save(retrieval.output_dir / 'retrieval_model_wave.npy',        retrieval.data_wave)

# Save night-2 model flux for joint analysis
if retrieval.two_night_mode and hasattr(retrieval, 'model_flux2'):
    np.save(retrieval.output_dir / 'retrieval_model_flux_N2.npy',  retrieval.model_flux2)
    np.save(retrieval.output_dir / 'retrieval_model_wave_N2.npy',  retrieval.data_wave2)


# ===========================================================================
# PLOTS
# ===========================================================================

mask  = np.isfinite(retrieval.data_flux)
K2166 = T.K2166

# -----------------------------------------------------------------------
# Plot 1: residuals per order/detector
# -----------------------------------------------------------------------
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

        ax[det][order].scatter(wave_cut, residuals,
                               label='residuals', color='blue',
                               linewidth=1, alpha=0.7, s=2)
        ax[det][order].axhline(0,            color='black', linestyle='--', linewidth=0.8)
        ax[det][order].axhline( 3 * std_res, color='red',   linestyle=':',  linewidth=0.8, label='3σ')
        ax[det][order].axhline(-3 * std_res, color='red',   linestyle=':',  linewidth=0.8)
        ax[det][order].set_title(f'Order {27 - order} - Det {det + 1}', fontsize=7)
        ax[det][order].legend(fontsize=5)

plt.tight_layout()
plt.savefig(retrieval.output_dir / 'retrieval_data_model_residuals.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------
# Plot 2: data vs model per order/detector
# -----------------------------------------------------------------------
fig, ax = plt.subplots(3, 5, figsize=(20, 10))

for order in range(5):
    for det in range(3):
        wave_border    = K2166[4 - order][det]
        sel            = mask & (retrieval.data_wave >= wave_border.min()) & \
                                 (retrieval.data_wave <= wave_border.max())
        wave_cut       = retrieval.data_wave[sel]
        data_flux_cut  = retrieval.data_flux[sel]
        model_flux_cut = model_flux_scaled[sel]

        ax[det][order].plot(wave_cut, data_flux_cut,
                            label='data', color='darkgray', linewidth=0.6, alpha=0.7)
        ax[det][order].plot(wave_cut, model_flux_cut,
                            label='model', color='red', linewidth=2, alpha=0.7)
        ax[det][order].set_ylim(
            np.nanmin(data_flux_cut) * 0.8,
            np.nanmedian(data_flux_cut) * 3,
        )
        ax[det][order].set_title(f'Order {27 - order} - Det {det + 1}', fontsize=7)
        ax[det][order].legend(fontsize=5)

plt.tight_layout()
plt.savefig(retrieval.output_dir / 'retrieval_data_model_spectrum.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------
# Plot 3: P-T profile
# -----------------------------------------------------------------------
plt.figure(figsize=(6, 8))
plt.plot(retrieval.model_object.temperature, retrieval.model_object.pressure)
plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (bar)')
plt.title('DG P-T profile (Picos+2025)')
plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
plt.close()

# -----------------------------------------------------------------------
# Plot 4: smoothed data vs model per order/detector
# -----------------------------------------------------------------------
fig, ax = plt.subplots(3, 5, figsize=(20, 10))

data_flux_bin  = savgol_filter(retrieval.data_flux[mask], 51, 2)
model_flux_bin = savgol_filter(model_flux_scaled[mask],   51, 2)

for det in range(3):
    for order in range(5):
        wave_border = K2166[4 - order][det]
        sel2 = (retrieval.data_wave[mask] >= wave_border.min()) & \
               (retrieval.data_wave[mask] <= wave_border.max())

        wave_cut_data  = retrieval.data_wave[mask][sel2]
        data_flux_cut  = data_flux_bin[sel2]
        model_flux_cut = model_flux_bin[sel2]

        d_mean = np.nanmean(data_flux_cut)
        m_mean = np.nanmean(model_flux_cut)

        ax[det][order].scatter(wave_cut_data, data_flux_cut  / (d_mean or 1),
                               label='data',  s=4, color='darkgray')
        ax[det][order].scatter(wave_cut_data, model_flux_cut / (m_mean or 1),
                               label='model', s=4, color='red')
        ax[det][order].set_title(f'Order {27 - order} - Det {det + 1}', fontsize=7)
        ax[det][order].legend(fontsize=5)

plt.tight_layout()
plt.savefig(retrieval.output_dir / 'retrieval_data_model_spectrum_binned.png', dpi=300)
plt.close()


print('+++++++++++ Retrieval and plotting complete +++++++++++')
print('Arrivederci! ^_^')

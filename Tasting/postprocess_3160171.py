"""
postprocess_3160171.py
======================
Post-processing only for retrieval 3160171 (008PM-EQ-CLD-MEDNORM,
cloud + GP + per-chip-median normalisation run).

MultiNest sampling completed but the job was killed before post-processing
finished.  This script regenerates the missing outputs:
  - final_params_dict.pickle (overwritten with correct median values)
  - retrieval_model_flux.npy / retrieval_model_flux_scaled.npy
  - retrieval_model_wave.npy / retrieval_model_flux_N2.npy / retrieval_model_wave_N2.npy
  - retrieval_data_model_residuals.png
  - retrieval_data_model_spectrum.png
  - retrieval_PT_profile.png

Configuration copied from tasting_retrieval_equa_chem_v6.1_piette_cloud_mednorm.py
(the script that produced run 3160171):
  - Normalize_method = 'per_chip_median', use_absolute_flux = False
  - sigmaclipper data files (not _flux_cal)
  - log_M/log_R replaced by log_g Gaussian N(3.64, 0.20)
  -> 26 free params, matching final_posterior.npy shape (n, 26)

Requires Guidebook_GAStronomy_Piette_v4.2.py with the
get_params_and_spectrum() fix (parameters.params.update(params_dict) before
pRT_spectrum is created).

Run with a SINGLE rank — no MPI needed:
  python postprocess_3160171.py
"""

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')

retrieval_id = 3160171

# ---------------------------------------------------------------------------
# Import Guidebook v4.2
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Configuration — must match the original run exactly
# ---------------------------------------------------------------------------
Normalize_method  = 'per_chip_median'  # divides data AND model by their own per-chip median
scaling_parameter = False              # Standard Gaussian likelihood (phi = 1 fixed)
use_absolute_flux = False              # no (R/d)² — level already fixed by per-chip median match
cov_mode          = 'GP'
N_points          = 1000
evidence_tol      = 0.5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
night1 = '2022-12-31'
wave_N1, flux_N1, err_N1, R_N1 = _load_night(
    night1,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)

night2 = '2023-01-01'
wave_N2, flux_N2, err_N2, R_N2 = _load_night(
    night2,
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file ='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=Normalize_method,
)

# ---------------------------------------------------------------------------
# Build retrieval object (identical to original run)
# ---------------------------------------------------------------------------
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

parameters = Parameters(free_params, constant_params)

T1 = Target(wl=wave_N1, fl=flux_N1, err=err_N1, name='dh_tau_b_N1')
T2 = Target(wl=wave_N2, fl=flux_N2, err=err_N2, name='dh_tau_b_N2')

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

# ---------------------------------------------------------------------------
# Point at the existing completed run's output directory
# ---------------------------------------------------------------------------
from pathlib import Path
retrieval.output_dir = Path('/data2/peng/retrievals/%s_N1000_ev0.5_Normper_chip_median_PerChipScaleFalse'%retrieval_id)
assert retrieval.output_dir.exists(), "Output directory not found"
assert (retrieval.output_dir / 'final_posterior.npy').exists(), \
    "final_posterior.npy missing — MultiNest run did not complete"
print(f'Output dir: {retrieval.output_dir}')

# Sanity check: posterior column count must match the free-parameter count
_post = np.load(retrieval.output_dir / 'final_posterior.npy')
assert _post.shape[1] == parameters.ndim, \
    f'posterior has {_post.shape[1]} columns but Parameters has {parameters.ndim} free params'
del _post

# ---------------------------------------------------------------------------
# Post-processing
# evaluate() calls: PMN_analyse() → get_params_and_spectrum() → cornerplot()
# get_params_and_spectrum() now syncs parameters.params before building
# pRT_spectrum, so no KeyError on T_anchor.
# ---------------------------------------------------------------------------
print('Running evaluate() ...')
retrieval.evaluate()
print('evaluate() complete.')

# evaluate() only makes the cornerplot; the residual/spectrum/PT PNGs come
# from save_diagnostic_plots()
print('Running save_diagnostic_plots() ...')
retrieval.save_diagnostic_plots()
print('Diagnostic plots saved.')

# ---------------------------------------------------------------------------
# Save model flux arrays (not saved by evaluate/cornerplot)
# ---------------------------------------------------------------------------
np.place(retrieval.data_flux,  retrieval.data_flux  == 0, np.inf)
np.place(retrieval.model_flux, retrieval.model_flux == 0, np.inf)
np.save(retrieval.output_dir / 'retrieval_model_flux.npy', retrieval.model_flux)
np.save(retrieval.output_dir / 'retrieval_model_wave.npy', retrieval.data_wave)

# use_absolute_flux=False: model_flux_scaled may not exist in this mode
if hasattr(retrieval, 'model_flux_scaled') and retrieval.model_flux_scaled is not None:
    np.place(retrieval.model_flux_scaled, retrieval.model_flux_scaled == 0, np.inf)
    np.save(retrieval.output_dir / 'retrieval_model_flux_scaled.npy', retrieval.model_flux_scaled)

if retrieval.two_night_mode and hasattr(retrieval, 'model_flux2'):
    np.save(retrieval.output_dir / 'retrieval_model_flux_N2.npy',  retrieval.model_flux2)
    np.save(retrieval.output_dir / 'retrieval_model_wave_N2.npy',  retrieval.data_wave2)

print('Model flux arrays saved.')
print()
print('+++++++++++ Post-processing complete +++++++++++')

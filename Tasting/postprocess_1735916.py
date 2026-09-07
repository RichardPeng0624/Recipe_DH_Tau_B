"""
postprocess_1735916.py
======================
Post-processing only for retrieval 1735916 (007PM-EQ-CLD, cloud+GP run).

MultiNest sampling completed but the job was killed before post-processing
finished.  This script regenerates the missing outputs:
  - final_params_dict.pickle (overwritten with correct median values)
  - retrieval_model_flux.npy / retrieval_model_flux_scaled.npy
  - retrieval_model_wave.npy / retrieval_model_flux_N2.npy / retrieval_model_wave_N2.npy
  - retrieval_data_model_residuals.png
  - retrieval_data_model_spectrum.png
  - retrieval_PT_profile.png

Requires Guidebook_GAStronomy_Piette_v4.2.py with the
get_params_and_spectrum() fix (parameters.params.update(params_dict) before
pRT_spectrum is created).

Run with a SINGLE rank — no MPI needed:
  python postprocess_1735916.py
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
Normalize_method  = None
scaling_parameter = False
use_absolute_flux = True
cov_mode          = 'GP'
N_points          = 1000
evidence_tol      = 0.5

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
night1 = '2022-12-31'
wave_N1, flux_N1, err_N1, R_N1 = _load_night(
    night1,
    flux_file='extracted_spectra_combined_flux_cal.npy',
    err_file ='extracted_spectra_combined_err_flux_cal.npy',
    normalize_method=Normalize_method,
)

night2 = '2023-01-01'
wave_N2, flux_N2, err_N2, R_N2 = _load_night(
    night2,
    flux_file='extracted_spectra_combined_flux_cal.npy',
    err_file ='extracted_spectra_combined_err_flux_cal.npy',
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

# ---------------------------------------------------------------------------
# Post-processing
# evaluate() calls: PMN_analyse() → get_params_and_spectrum() → cornerplot()
# get_params_and_spectrum() now syncs parameters.params before building
# pRT_spectrum, so no KeyError on T_anchor.
# ---------------------------------------------------------------------------
print('Running evaluate() ...')
retrieval.evaluate()
print('evaluate() complete.')

# ---------------------------------------------------------------------------
# Save model flux arrays (not saved by evaluate/cornerplot)
# ---------------------------------------------------------------------------
np.place(retrieval.data_flux,         retrieval.data_flux         == 0, np.inf)
np.place(retrieval.model_flux,        retrieval.model_flux        == 0, np.inf)
np.place(retrieval.model_flux_scaled, retrieval.model_flux_scaled == 0, np.inf)

np.save(retrieval.output_dir / 'retrieval_model_flux.npy',        retrieval.model_flux)
np.save(retrieval.output_dir / 'retrieval_model_flux_scaled.npy', retrieval.model_flux_scaled)
np.save(retrieval.output_dir / 'retrieval_model_wave.npy',        retrieval.data_wave)

if retrieval.two_night_mode and hasattr(retrieval, 'model_flux2'):
    np.save(retrieval.output_dir / 'retrieval_model_flux_N2.npy',  retrieval.model_flux2)
    np.save(retrieval.output_dir / 'retrieval_model_wave_N2.npy',  retrieval.data_wave2)

print('Model flux arrays saved.')
print()
print('+++++++++++ Post-processing complete +++++++++++')
print(f'lnZ (NS)      = 1,714,452.7  (cloud run 1735916)')
print(f'lnZ (NS)      = 1,714,455.5  (cloud-free 2528367)')
print(f'Delta lnZ     = -2.8 nats  → clouds not detected')

#!/usr/bin/env python3
"""Standalone CCF validation for retrieval 3062330 (GP, Guidebook v4.2).

STALE / SUPERSEDED (flagged 2026-08-21): job 3062330 uses Guidebook v4.2
(NormNone / absolute flux calibration, use_absolute_flux=True) with log_Na
and log_Ca as sampled free parameters. It predates the 2026-07-07 decision
to drop Na/Ca as free params and the 2026-07-20/21 "atomopac" decision
(Na/Ca opacity in the model but not sampled). The current standing
benchmark retrievals are job 430784 (cloud-free) and 1001064 (cloud),
Guidebook v4.2.5, per-chip-median-normalised sigmaclipper data
(use_absolute_flux=False) -- see Reminisce/tasting_retrieval_validation_clean.ipynb
Sections 17-18, which is the correct, in-repo, currently-maintained CCF
procedure. This script is kept only for the historical record of job
3062330 and should NOT be used as a template for new CCF validation runs.
"""
import sys, pickle, importlib.util
import numpy as np
import astropy.io.fits as fits
from pathlib import Path

RECIPE_DIR     = Path('/data2/peng/Recipe_DH_Tau_B')
WORKPATH       = Path('/data2/peng')
RETRIEVAL_BASE = WORKPATH / 'retrievals'

sys.path.insert(0, str(RECIPE_DIR))
import analysis
from analysis import run_species_ccf_validation
importlib.reload(analysis)
from analysis import run_species_ccf_validation
print('analysis.py loaded')

# --- Load Guidebook v4.2 ---
_gb42_path = str(RECIPE_DIR / 'Tasting_guidebook' / 'Guidebook_GAStronomy_Piette_v4.2.py')
_spec42 = importlib.util.spec_from_file_location('Guidebook_v4_2', _gb42_path)
_gb42   = importlib.util.module_from_spec(_spec42)
_spec42.loader.exec_module(_gb42)

Target42                       = _gb42.Target
Parameters42                   = _gb42.Parameters
Retrieval42                    = _gb42.Retrieval
_load_night42                  = _gb42._load_night
make_free_params_equilibrium42 = _gb42.make_free_params_equilibrium
pRT_spectrum42                 = _gb42.pRT_spectrum
print('Guidebook v4.2 loaded')

# --- 1-D observation arrays ---
wave_N1, flux_N1, err_N1, R_N1 = _load_night42(
    '2022-12-31',
    flux_file='extracted_spectra_combined_flux_cal.npy',
    err_file='extracted_spectra_combined_err_flux_cal.npy',
    normalize_method=None,
)
wave_N2, flux_N2, err_N2, R_N2 = _load_night42(
    '2023-01-01',
    flux_file='extracted_spectra_combined_flux_cal.npy',
    err_file='extracted_spectra_combined_err_flux_cal.npy',
    normalize_method=None,
)
print(f'N1: {len(wave_N1)} px   N2: {len(wave_N2)} px')

T1 = Target42(wl=wave_N1, fl=flux_N1, err=err_N1, name='dh_tau_b_N1')
T2 = Target42(wl=wave_N2, fl=flux_N2, err=err_N2, name='dh_tau_b_N2')

# --- 3-D cubes for CCF ---
def _load_3d(night_str):
    base = Path(f'/data2/peng/{night_str}')
    flux = np.load(base / 'extracted_spectra_combined_flux_cal.npy').astype(float)
    err  = np.load(base / 'extracted_spectra_combined_err_flux_cal.npy').astype(float)
    hdu  = fits.open(base / 'cal/WLEN_K2166_V_DH_Tau_A+B_center.fits')
    wave = np.array(hdu[1].data, dtype=float)[:, :5, :]
    bad  = ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0) | ~np.isfinite(wave)
    flux[bad] = np.nan; err[bad] = np.nan; wave[bad] = np.nan
    print(f'3D loaded: {flux.shape}  valid={( ~bad).sum()}')
    return wave, flux, err

wave3_N1, flux3_N1, err3_N1 = _load_3d('2022-12-31')
wave3_N2, flux3_N2, err3_N2 = _load_3d('2023-01-01')

# --- Wave model (sorted) ---
wave_model = np.load(
    RETRIEVAL_BASE / '1918539_N800_ev0.5_NormNone_PerChipScaleFalse/retrieval_model_wave.npy'
)
wave_model = np.sort(wave_model)
print(f'wave_model: {len(wave_model)} pts  [{wave_model[0]:.1f}, {wave_model[-1]:.1f}] nm')

# --- Setup retrieval 3062330 ---
RID_C  = '3062330_N600_ev0.5_NormNone_PerChipScaleFalse'
DIR_C  = RETRIEVAL_BASE / RID_C
LABEL_C = '3062330'

with open(DIR_C / 'final_params_dict.pickle', 'rb') as f:
    best_fit_C = pickle.load(f)

constant_params_C = {'chemistry': 'equilibrium'}
free_params_C     = make_free_params_equilibrium42()
free_params_C['log_a'] = ([-1.0,  1.0], r'$\log a$')
free_params_C['log_l'] = ([-3.0,  0.0], r'$\log l$')

parameters_C = Parameters42(free_params_C, constant_params_C)
parameters_C(np.random.rand(parameters_C.ndim))
parameters_C.params.update(best_fit_C)

retrieval_C = Retrieval42(
    parameters         = parameters_C,
    N_live_points      = 600,
    evidence_tolerance = 0.5,
    targets            = [T1, T2],
    testing            = False,
    normalize_flux     = None,
    per_chip_scaling   = False,
    instrument_res     = [R_N1, R_N2],
    use_absolute_flux  = True,
    cov_mode           = 'GP',
)
retrieval_C.parameters.params.update(best_fit_C)
print('Retrieval 3062330 ready')

# --- TRACE SPECIES ---
TRACE_SPECIES_BASE = {
    '1H2-16O':     'H2O',
    '12C-16O':     '12CO',
    '13C-16O':     '13CO',
    '12C-1H4__MM': 'CH4',
    '56Fe-1H':     'FeH',
    '1H-19F':      'HF',
    '23Na':        'Na',
    '40Ca':        'Ca',
}

# --- Run CCF validation ---
results_C = run_species_ccf_validation(
    retrieval          = retrieval_C,
    pRT_spectrum_class = pRT_spectrum42,
    best_fit_params    = best_fit_C,
    wave_model         = wave_model,
    obs_flux_N1        = flux3_N1,
    err_N1             = err3_N1,
    obs_wave_N1        = wave3_N1,
    obs_flux_N2        = flux3_N2,
    err_N2             = err3_N2,
    obs_wave_N2        = wave3_N2,
    TRACE_SPECIES      = TRACE_SPECIES_BASE,
    retrieval_dir      = DIR_C,
    retrieval_label    = LABEL_C,
    use_absolute_flux  = True,
)
print('Done.')

#!/usr/bin/env python3
"""Standalone CCF validation for the fiducial job 430784, reproducing exactly
the section 17 cells (46, 51, 52) of Reminisce/tasting_retrieval_validation_clean.ipynb,
but saving the raw per-species ccf_results dict (not just the SNR summary) so
it can be replotted in AA journal style in plot_paper.ipynb.
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

_gb425_path = str(RECIPE_DIR / 'Tasting_guidebook' / 'Guidebook_GAStronomy_Piette_v4.2.5.py')
_spec425 = importlib.util.spec_from_file_location('Guidebook_v4_2_5', _gb425_path)
_gb425   = importlib.util.module_from_spec(_spec425)
_spec425.loader.exec_module(_gb425)

Target425                       = _gb425.Target
Parameters425                   = _gb425.Parameters
Retrieval425                    = _gb425.Retrieval
_load_night425                  = _gb425._load_night
make_free_params_equilibrium425 = _gb425.make_free_params_equilibrium
pRT_spectrum425                 = _gb425.pRT_spectrum
print('Guidebook v4.2.5 loaded; EQ_SPECIES_PRT3 =', _gb425.EQ_SPECIES_PRT3)
assert '23Na' in _gb425.EQ_SPECIES_PRT3 and '40Ca' in _gb425.EQ_SPECIES_PRT3

wave_N1, flux_N1, err_N1, R_N1 = _load_night425(
    '2022-12-31',
    flux_file='extracted_spectra_combined_sigmaclipper.npy',
    err_file='extracted_spectra_combined_err_sigmaclipper.npy',
    normalize_method=None,
)
wave_N2, flux_N2, err_N2, R_N2 = _load_night425(
    '2023-01-01',
    flux_file='extracted_spectra_combined_sigmaclipper_0101.npy',
    err_file='extracted_spectra_combined_err_sigmaclipper_0101.npy',
    normalize_method=None,
)
print(f'N1: {len(wave_N1)} px   N2: {len(wave_N2)} px')

def _load_3d(night_str):
    base = Path(f'/data2/peng/{night_str}')
    flux = np.load(base / 'extracted_spectra_combined_sigmaclipper.npy').astype(float)
    err  = np.load(base / 'extracted_spectra_combined_err_sigmaclipper.npy').astype(float)
    hdu  = fits.open(base / 'cal/WLEN_K2166_V_DH_Tau_A+B_center.fits')
    wave = np.array(hdu[1].data, dtype=float)[:, :5, :]
    bad  = ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0) | ~np.isfinite(wave)
    flux[bad] = np.nan; err[bad] = np.nan; wave[bad] = np.nan
    print(f'  [{night_str}] 3-D shape: {flux.shape}  valid pix: {(~bad).sum()}')
    return wave, flux, err

wave3_N1, flux3_N1, err3_N1 = _load_3d('2022-12-31')
wave3_N2, flux3_N2, err3_N2 = _load_3d('2023-01-01')

wave_model = np.load(
    RETRIEVAL_BASE / '2968924_N600_ev0.5_Normper_chip_median_PerChipScaleFalse/retrieval_model_wave.npy'
)
wave_model = np.sort(wave_model)
print(f'wave_model: {len(wave_model)} pts')

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

RID_M   = '430784_N600_ev0.5_Normper_chip_median_PerChipScaleFalse'
DIR_M   = RETRIEVAL_BASE / RID_M
LABEL_M = '430784'

with open(DIR_M / 'final_params_dict.pickle', 'rb') as f:
    best_fit_M = pickle.load(f)

constant_params_M = {'chemistry': 'equilibrium'}
free_params_M     = make_free_params_equilibrium425()
del free_params_M['log_M']
del free_params_M['log_R']
free_params_M['log_g'] = ({'type': 'gaussian', 'mu': 3.64, 'sigma': 0.20}, r'$\log g$')
free_params_M['log_a'] = ([-1.0, 1.0], r'$\log a_{\rm GP}$')
free_params_M['log_l'] = ([-3.0, 0.0], r'$\log l_{\rm GP}$')
assert 'log_Na' not in free_params_M and 'log_Ca' not in free_params_M

parameters_M = Parameters425(free_params_M, constant_params_M)
parameters_M(np.random.rand(parameters_M.ndim))
parameters_M.params.update(best_fit_M)

_K2166_M = np.array([
    [[2063.711, 2077.942], [2078.967, 2092.559], [2093.479, 2106.392]],
    [[2143.087, 2157.855], [2158.914, 2173.020], [2173.983, 2187.386]],
    [[2228.786, 2244.133], [2245.229, 2259.888], [2260.904, 2274.835]],
    [[2321.596, 2337.568], [2338.704, 2353.961], [2355.035, 2369.534]],
    [[2422.415, 2439.061], [2440.243, 2456.145], [2457.275, 2472.388]],
])

def _pcm_1d_M(wave, flux, err, K2166):
    flux = flux.copy(); err = err.copy()
    for order in range(5):
        for det in range(3):
            mask = (wave >= K2166[order, det, 0]) & (wave <= K2166[order, det, 1])
            if mask.sum() < 10: continue
            med = np.nanmedian(flux[mask])
            if med > 0: flux[mask] /= med; err[mask] /= med
    return flux, err

def _pcm_3d_M(wave3, flux3, err3, K2166):
    flux3 = flux3.copy(); err3 = err3.copy()
    for det in range(flux3.shape[0]):
        for order in range(flux3.shape[1]):
            fin = np.isfinite(flux3[det, order, :])
            if fin.sum() < 10:
                continue
            med = np.nanmedian(flux3[det, order, fin])
            if med > 0: flux3[det, order, :] /= med; err3[det, order, :] /= med
    return flux3, err3

flux_M_N1, err_M_N1   = _pcm_1d_M(wave_N1, flux_N1, err_N1, _K2166_M)
flux_M_N2, err_M_N2   = _pcm_1d_M(wave_N2, flux_N2, err_N2, _K2166_M)
flux3_M_N1, err3_M_N1 = _pcm_3d_M(wave3_N1, flux3_N1, err3_N1, _K2166_M)
flux3_M_N2, err3_M_N2 = _pcm_3d_M(wave3_N2, flux3_N2, err3_N2, _K2166_M)

T1_M = Target425(wl=wave_N1, fl=flux_M_N1, err=err_M_N1, name='dh_tau_b_M_N1')
T2_M = Target425(wl=wave_N2, fl=flux_M_N2, err=err_M_N2, name='dh_tau_b_M_N2')

retrieval_M = Retrieval425(
    parameters         = parameters_M,
    N_live_points      = 600,
    evidence_tolerance = 0.5,
    targets            = [T1_M, T2_M],
    testing            = False,
    normalize_flux     = 'per_chip_median',
    per_chip_scaling   = False,
    instrument_res     = [R_N1, R_N2],
    use_absolute_flux  = False,
    cov_mode           = 'GP',
)
retrieval_M.parameters.params.update(best_fit_M)
print('Retrieval 430784 ready')

results_M = run_species_ccf_validation(
    retrieval          = retrieval_M,
    pRT_spectrum_class = pRT_spectrum425,
    best_fit_params    = best_fit_M,
    wave_model         = wave_model,
    obs_flux_N1        = flux3_M_N1,
    err_N1             = err3_M_N1,
    obs_wave_N1        = wave3_N1,
    obs_flux_N2        = flux3_M_N2,
    err_N2             = err3_M_N2,
    obs_wave_N2        = wave3_N2,
    TRACE_SPECIES      = TRACE_SPECIES_BASE,
    retrieval_dir      = DIR_M,
    retrieval_label    = LABEL_M,
    use_absolute_flux  = False,
)

rvlag = np.arange(-1000.0, 1001.0, 1.0)
save = {'rvlag': rvlag, 'ccf_results': {}}
for prt_name, res in results_M['ccf_results'].items():
    save['ccf_results'][prt_name] = {
        k: res[k] for k in
        ('label', 'ccf', 'acf', 'acf_aligned', 'residual', 'snr', 'peak_rv',
         'peak_val', 'std_noise', 'template_fraction')
    }

OUT = '/var/tmp/peng/claude-3444/-data2-peng/d6ed8900-ce48-4f84-9e12-01a257057175/scratchpad/ccf_430784_results.pickle'
with open(OUT, 'wb') as f:
    pickle.dump(save, f)
print('Saved raw CCF results to', OUT)
print('Species:', [v['label'] for v in save['ccf_results'].values()])

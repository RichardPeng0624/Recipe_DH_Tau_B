import os
import sys
import pathlib

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.ndimage import gaussian_filter
from PyAstronomy.pyasl import fastRotBroad

import petitRADTRANS as prt
from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.config import petitradtrans_config_parser
from petitRADTRANS.chemistry.pre_calculated_chemistry import pre_calculated_equilibrium_chemistry_table

petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')
pre_calculated_equilibrium_chemistry_table.load()   # load once at import time


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Sonora Diamondback cloud-free P-T profile directory
SONORA_GRIDPATH = '/net/lem/data2/SONORA_Diamondback/pressure-temperature_profiles'

# Nearest gravity in the Diamondback grid to log_g=3.7 (grid: 31,100,316,1000,3160 cm/s²)
# log10(3160) ≈ 3.5  — closest available to log_g = 3.7
SONORA_GRAVITY  = 3160   # cm/s²  (used only for P-T profile file lookup)

LOG_G     = 3.7    # log10(surface gravity / [cm/s²])
CO_RATIO  = 0.54   # C/O ratio
FEH       = 0.5    # [Fe/H] metallicity
VSINI     = 5.7    # km/s rotational velocity

TEFF_GRID = np.arange(2000, 2401, 100)   # [2000, 2100, 2200, 2300, 2400] K

LBL_OPACITY_SAMPLING = 3
N_ATM_LAYERS         = 50
SPECTRAL_RESOLUTION  = 100_000           # CRIRES+ resolving power

WL_PAD = 7     # nm padding around the K-band edges
WL_MIN = 1921 - WL_PAD   # nm
WL_MAX = 2475 + WL_PAD   # nm

# pRT line species (pRT_name from species_info.csv)
LINE_SPECIES = ['1H2-16O', '12C-16O', '13C-16O', '12C-1H4']

OUTPUT_DIR = pathlib.Path('/data2/peng/Recipe_DH_Tau_B/sonora_model_spectra')


# -----------------------------------------------------------------------
# pRT atmosphere object  (built once, reused for every Teff)
# -----------------------------------------------------------------------

pressure   = np.logspace(-6, 2, N_ATM_LAYERS)   # 50 log-spaced levels, 1e-6–100 bar
wlen_range = np.array([WL_MIN, WL_MAX]) * 1e-7   # nm → cm
boundary   = wlen_range * 1e4                     # cm → micron

print('Setting up pRT Radtrans object...')
print(f'  Line species       : {LINE_SPECIES}')
print(f'  Wavelength range   : {WL_MIN}–{WL_MAX} nm')
print(f'  lbl opacity sampling: {LBL_OPACITY_SAMPLING}')

atmosphere = Radtrans(
    line_species=LINE_SPECIES,
    rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2-H2', 'H2-He'],
    wavelength_boundaries=boundary,
    line_opacity_mode='lbl',
    line_by_line_opacity_sampling=LBL_OPACITY_SAMPLING,
    pressures=pressure,
)


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def get_pt_profile(teff, pressure_grid):
    """
    Read a Sonora Diamondback cloud-free P-T profile and interpolate it onto
    the pRT pressure grid.

    File naming: t{teff}g{SONORA_GRAVITY}nc_m+0.5_co1.0.pt
    Format: 2 header lines, then rows of (n, P[bar], T[K], ...)

    Parameters
    ----------
    teff          : effective temperature [K]
    pressure_grid : 1-D array of pressure levels [bar] used by pRT

    Returns
    -------
    temperature : 1-D array (same length as pressure_grid) [K]
    """
    fname = os.path.join(
        SONORA_GRIDPATH,
        f't{int(teff)}g{SONORA_GRAVITY}nc_m+0.5_co1.0.pt'
    )
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f'Sonora Diamondback PT file not found: {fname}\n'
            f'Check SONORA_GRIDPATH and SONORA_GRAVITY.'
        )
    # Skip 2 header lines; columns: n, P[bar], T[K], ...
    data = np.genfromtxt(fname, skip_header=2)
    p_sonora = data[:, 1]   # pressure [bar]
    t_sonora = data[:, 2]   # temperature [K]

    # Sort by ascending pressure (some files may be top-down)
    sort_idx  = np.argsort(p_sonora)
    p_sonora  = p_sonora[sort_idx]
    t_sonora  = t_sonora[sort_idx]

    # Interpolate onto our pRT pressure grid (in log-pressure space)
    temperature = np.interp(
        np.log10(pressure_grid),
        np.log10(p_sonora),
        t_sonora,
    )
    return temperature


def get_mass_fractions(temperature, pressure_grid, feh, co_ratio):
    """
    Compute equilibrium-chemistry mass fractions using pRT3's pre-calculated table.

    The table keys ('CO', 'H2O', 'CH4') are mapped to pRT isotopologue names.
    13CO is not in the table; its mass fraction is derived from CO using the
    solar 12C/13C isotope ratio (≈ 89).

    Parameters
    ----------
    temperature    : 1-D array [K], shape (N_ATM_LAYERS,)
    pressure_grid  : 1-D array [bar], shape (N_ATM_LAYERS,)
    feh            : [Fe/H] metallicity  (= log10 metallicity)
    co_ratio       : C/O ratio

    Returns
    -------
    mass_fractions : dict  (ready for Radtrans.calculate_flux, includes 'MMW')
    MMW            : 1-D array of mean molecular weights [g/mol]
    """
    # table returns (mass_frac_dict, MMW_array, nabla_ad_array) when full=True
    mf_table, MMW, _ = pre_calculated_equilibrium_chemistry_table.interpolate_mass_fractions(
        co_ratios=np.full_like(temperature, co_ratio),
        log10_metallicities=np.full_like(temperature, feh),
        temperatures=temperature,
        pressures=pressure_grid,
        full=True,
    )

    # Mapping from equilibrium-table keys → pRT line-species names
    table_to_prt = {
        'H2O': '1H2-16O',
        'CO':  '12C-16O',
        'CH4': '12C-1H4',
    }

    mass_fractions = {}
    for table_key, prt_key in table_to_prt.items():
        if table_key in mf_table:
            mass_fractions[prt_key] = mf_table[table_key]
        else:
            print(f'WARNING: {table_key} not in equilibrium table – skipping {prt_key}.')

    # 13CO: scale from 12CO using solar isotope ratio  12C/13C ≈ 89
    # mass_frac ratio = VMR ratio × (m_13CO / m_12CO) = (1/89) × (29/28)
    if '12C-16O' in mass_fractions:
        mass_fractions['13C-16O'] = mass_fractions['12C-16O'] * (29.0 / 28.0) / 89.0

    mass_fractions['H2']  = mf_table['H2']
    mass_fractions['He']  = mf_table['He']
    mass_fractions['MMW'] = MMW

    return mass_fractions, MMW


def instr_broadening(wave, flux, out_res, in_res):
    """Gaussian LSF broadening from in_res to out_res (mirrored from reference file)."""
    sigma_LSF = np.sqrt(1 / out_res**2 - 1 / in_res**2) / (2 * np.sqrt(2 * np.log(2)))
    spacing   = np.mean(2 * np.diff(wave) / (wave[1:] + wave[:-1]))
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    return gaussian_filter(flux, sigma=sigma_LSF_gauss_filter, mode='nearest')


def convolve_to_resolution(in_wlen, in_flux, out_res, in_res=None):
    """Convolve spectrum to a target resolving power (mirrored from reference file)."""
    if in_res is None:
        in_res = np.mean(in_wlen[:-1] / np.diff(in_wlen))
    sigma_LSF = np.sqrt(1.0 / out_res**2 - 1.0 / in_res**2) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    spacing   = np.mean(2.0 * np.diff(in_wlen) / (in_wlen[1:] + in_wlen[:-1]))
    sigma_LSF_gauss_filter = sigma_LSF / spacing
    out_flux = np.tile(np.nan, in_flux.shape)
    nans     = np.isnan(in_flux)
    out_flux[~nans] = gaussian_filter(in_flux[~nans], sigma=sigma_LSF_gauss_filter, mode='reflect')
    return out_flux


# -----------------------------------------------------------------------
# Main spectrum generator
# -----------------------------------------------------------------------

def generate_spectrum(teff):
    """
    Generate a single cloud-free Sonora-based spectrum at the given Teff.

    Fixed parameters: LOG_G, CO_RATIO, FEH, VSINI (module-level constants).
    No RV shift applied; the cross-correlation step will determine RV.

    Returns
    -------
    waves_even : 1-D wavelength array [nm], evenly spaced
    flux       : median-normalised flux array (same length)
    """
    print(f'\n--- Generating spectrum for Teff={teff} K ---')

    # 1. P-T profile from Sonora Diamondback cloud-free grid
    temperature = get_pt_profile(teff, pressure)
    print(f'  T range: {temperature.min():.1f} – {temperature.max():.1f} K')

    # 2. Equilibrium chemistry
    gravity = 10**LOG_G
    mass_fractions, MMW = get_mass_fractions(temperature, pressure, FEH, CO_RATIO)
    print(f'  MMW (mean): {np.mean(MMW):.2f} g/mol')

    # 3. pRT radiative transfer
    wl, flux, _ = atmosphere.calculate_flux(
        temperatures=temperature,
        mass_fractions=mass_fractions,
        reference_gravity=gravity,
        mean_molar_masses=MMW,
        return_contribution=True,
        frequencies_to_wavelengths=True,   # returns wl in cm
    )
    wl   *= 1e7   # cm → nm
    print(f'  pRT wl range: {wl.min():.1f} – {wl.max():.1f} nm')
    print(f'  Raw flux range: {flux.min():.2e} – {flux.max():.2e}')

    # 4. Interpolate onto evenly-spaced wavelength grid (required by fastRotBroad)
    waves_even = np.linspace(wl.min(), wl.max(), wl.size)
    flux_even  = np.interp(waves_even, wl, flux)

    # 5. Rotational broadening  (limb-darkening ε=0.5; valid range 0–1)
    spec = fastRotBroad(waves_even, flux_even, 0.5, VSINI)

    # 6. Convolve to CRIRES+ spectral resolution
    spec = convolve_to_resolution(waves_even, spec, SPECTRAL_RESOLUTION)

    # 7. Instrumental broadening (lbl native → lbl sampling resolution, mirrors reference)
    resolution_lbl = int(1e6 / LBL_OPACITY_SAMPLING)   # 333 333
    flux_out = instr_broadening(waves_even, spec, out_res=resolution_lbl, in_res=500_000)

    # 8. Median normalise for cross-correlation
    flux_out /= np.nanmedian(flux_out)

    print(f'  Final flux range (normalised): {np.nanmin(flux_out):.3f} – {np.nanmax(flux_out):.3f}')
    return waves_even, flux_out


# -----------------------------------------------------------------------
# Run and save
# -----------------------------------------------------------------------

if __name__ == '__main__':

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_flux = []
    wl_ref   = None

    for teff in TEFF_GRID:
        wl, flux = generate_spectrum(teff)

        fname = (OUTPUT_DIR /
                 f'spectrum_Teff{teff}_logg{LOG_G}_CO{CO_RATIO}_FeH{FEH}_vsini{VSINI}.npz')
        np.savez(fname, wl=wl, flux=flux)
        print(f'  Saved: {fname.name}')

        all_flux.append(flux)
        if wl_ref is None:
            wl_ref = wl

    # Combined grid file
    combined = OUTPUT_DIR / 'all_spectra.npz'
    np.savez(
        combined,
        wl=wl_ref,
        teff_grid=TEFF_GRID,
        flux_grid=np.array(all_flux),   # shape: (n_teff, n_pixels)
    )
    print(f'\nDone. All spectra saved to {OUTPUT_DIR}')
    print(f'Combined file: {combined}')
    print(f'flux_grid shape: {np.array(all_flux).shape}')

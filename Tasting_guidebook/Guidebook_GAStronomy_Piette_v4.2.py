"""
Guidebook_GAStronomy_Piette_v4.2.py
=====================================
Series 005PM — Absolute flux mode: (R/d)² physical scaling  [updated from v2.0]

Changes from v4.1 (v4.2)
--------------------------
v4.2 patch (2026-07-07) — atomic species Na, Ca removed from equilibrium mode
  - EQ_SPECIES_PRT3: '23Na' and '40Ca' commented out (same treatment as '48Ti');
    EQ_LABEL_NAMES reduced accordingly. log_Na / log_Ca priors were already
    commented out in make_free_params_equilibrium() / make_free_params_free_chem(),
    which would otherwise leave the two line species with no mass fractions
    supplied. HF (F_H) is retained. The equilibrium_chemistry() injection block
    is unchanged — it is conditional on species AND param presence, so it is
    inert while these are commented.

v4.2 patch (2026-07-04) — new 'per_chip_median' normalisation option
  - _load_night(normalize_method='per_chip_median'): divides each (order,
    detector) chip's flux/err by that chip's own median (min. 10 finite px;
    skipped if median <= 0) — same semantics as the ad hoc
    _per_chip_median_norm() helper duplicated in tasting_retrieval_equa_chem_
    v6.1_piette_cloudfree_ruffio.py, now available as a built-in option so
    tasting scripts don't need to reimplement it.
  - pRT_spectrum._apply_normalization(normalize_flux='per_chip_median'):
    same per-chip median division applied to the model spectrum, using
    self.target.K2166 chip boundaries (same pattern as the existing 'savgol'
    branch). Together these let a tasting script median-normalise BOTH data
    and model per chip, then use the plain Standard Gaussian likelihood
    (scale_flux=False) — as opposed to per_chip_scaling=True (Ruffio+2019
    marginal phi), which normalises data only and is provably invariant to
    any additional fixed rescaling of the model (see
    tasting_retrieval_equa_chem_v6.1_piette_cloudfree_mednorm.py docstring
    for the derivation). The two are different, deliberate choices: Ruffio
    marginalises the per-chip scale factor's own uncertainty into the
    likelihood; 'per_chip_median' + Standard Gaussian treats each chip's
    level as exactly known from the data median, which is simpler and more
    transparent at the cost of not propagating that uncertainty.
  - Purpose: lets a retrieval be run on non-flux-calibrated data
    (extracted_spectra_combined_sigmaclipper.npy) while remaining directly
    comparable in shape to the model, without depending at all on the
    SINFONI-based absolute flux calibration (see
    observation_data_reduction_notes.md §11 for why that calibration's
    det2/ord0 chip is a documented caveat this sidesteps entirely).

v4.2 (2026-06-12) — GP performance optimisation (memory, speed, OOM prevention)
v4.2 patch (2026-06-19) — flux_scaling_rolling correctness + additional speedups
  - CovarianceGP.chip_gap_nm default changed from 2.0 → 0.5 nm.  With the
    old 2 nm threshold, all three K2166 detectors within an order formed
    ONE sub-chip (inter-detector gaps ≈ 0.92–1.18 nm < 2 nm).  This caused
    solve_for_wl_range() to apply a ~5000-pixel Cholesky to a ~1800-pixel
    vector → ValueError / silently wrong answer.  The 0.5 nm threshold is
    below the minimum K2166 inter-detector gap (≈ 1.07 nm) so each detector
    becomes its own sub-chip.  Scientific effect: negligible (cross-detector
    correlations at 1 nm are already zero within the 100-pixel bandwidth cap
    at typical l values; banded approximation already treats detectors as
    independent).  Night 1 sub-chips: 8 → 27; build() overhead +9 ms/call.
  - CovarianceGP.precompute_K2166_ranges(K2166_bounds): new method; call
    once after __init__ to cache which sub-chips (and pixel counts) serve
    each K2166 (order, det) range.  solve_for_wl_range() uses an O(1) dict
    lookup instead of scanning all ~27 sub-chips and creating boolean arrays
    per call.  Eliminates ~540 in_range boolean ops per PMN_lnL call.
  - LogLikelihood.__init__: precomputes _chip_masks (15 bool arrays) and
    _chip_d_cuts (15 data slices) when scale_flux=True.  flux_scaling_rolling
    reads precomputed masks instead of recomputing 15 × 3 boolean array ops
    per call.
  - Combined speedup for flux_scaling_rolling (one night): 51 ms → 6.9 ms
    (7.4×).  Net gain over full retrieval (2 nights, ~100k calls): ≈117 min.
  - Retrieval.__init__: calls Cov[2].precompute_K2166_ranges(K2166) for
    each night after CovarianceGP is built.
  - Bug (original v4.2): _RangeCovProxy was correct in concept but broken
    in practice — chip_gap_nm=2.0 caused sub-chips to span multiple K2166
    detectors, so solve_for_wl_range mismatched Cholesky dimensions.  Fixed
    by chip_gap_nm=0.5 change above.

v4.2 (2026-06-12) — GP performance optimisation (memory, speed, OOM prevention)
  - CovarianceGP: dense N×N Cholesky replaced by banded Cholesky using
    scipy.linalg.cholesky_banded / cho_solve_banded.  Bandwidth B is
    computed per build() call as B = min(ceil(5l/δλ), MAX_BANDWIDTH=100)
    where δλ is the per-chip median pixel spacing and l is the sampled GP
    length scale.  Hard cap of 100 pixels prevents GP kernel from sampling
    distant uncorrelated pixels regardless of the log_l prior upper bound.
    Memory: ~18× reduction in stored GP factors per night (≈390 MB → ≈22 MB
    for 1800 valid px/chip, 15 chips); Cholesky cost: ~300× reduction
    (O(N³) → O(N×B²)).  Statistical truncation error: < exp(−12.5) ≈
    3.7×10⁻⁶ per entry (below float64 noise floor at any realistic S/N).
  - CovarianceGP.__init__: precomputes per-chip pixel count and median
    pixel spacing (avoids np.median(np.diff(wl)) on every build() call).
  - CovarianceGP.solve_for_wl_range(): applies stored banded Cholesky
    factors to a vector restricted to a wavelength window [lo, hi] nm,
    reusing _banded_factors already computed by build() — no redundant
    factorisation.
  - New _RangeCovProxy class: lightweight wrapper exposing .solve() that
    delegates to CovarianceGP.solve_for_wl_range(); used by
    flux_scaling_rolling to eliminate the 2× Cholesky redundancy (one
    build() per chip per night was being re-run inside flux_scaling_rolling
    even though Cov.build() had already factorised all chips).  With two
    nights and 18 sub-chips/night this removes 36 redundant factorisations
    per likelihood call (~2× speedup in the GP-dominated cost).
  - Bug: _ChipCovProxy / _chip_index_for_mask were described in v4.2
    changelog but never wired up; _chip_index_for_mask also had an
    index-space mismatch (full-array indices vs compressed-array slices).
    Both are superseded by the _RangeCovProxy approach above.
  - Module-level _get_species_info() cache: species_info.csv is parsed
    once per process (first call) and reused across all subsequent calls;
    eliminates ~12 pd.read_csv() invocations per likelihood evaluation
    (free_chemistry, read_species_info, get_species).
  - scipy.linalg: cholesky_banded, cho_solve_banded added to imports.
    cho_factor / cho_solve retained (used by diagonal Covariance class).

Changes from v4.0 (v4.1)
--------------------------
v4.1 (2026-06-09) — Correlated noise GP kernel (González Picos+2025 §4.2)
  - New CovarianceGP class: Σ₀,ij = δᵢⱼσᵢ² + a²((σᵢ+σⱼ)/2)² exp(−Δλ²/2l²)
    following González Picos et al. (2025) Eq. 5–6 (= de Regt+2024 formulation).
  - Retrieval.__init__ accepts cov_mode='diag'|'GP' (default 'diag').
    When 'GP': creates CovarianceGP(err, wl) instead of Covariance(err).
  - PMN_lnL rebuilds GP covariance each call with sampled log_a, log_l.
  - get_params_and_spectrum rebuilds GP covariance at median params.
  - flux_scaling_rolling patched: per-chip sub-covariance correctly
    propagates CovarianceGP with current a, l when GP mode is active.
  - scipy.linalg.cho_factor / cho_solve added to imports.
  - GP free params (log_a, log_l) must be added in the tasting file and
    cov_mode='GP' passed to Retrieval; no default params added here.

Changes from v3.0 (v4.0)
--------------------------
v4.0 (2026-06-04) — Atomic species Na, Ca, Ti and HF added (Picos+2024 §4.2)
  - EQ_SPECIES_PRT3 / EQ_LABEL_NAMES extended with '1H-19F', '23Na', '48Ti', '40Ca'
  - equilibrium_chemistry(): HF, Na, Ca, Ti injected from free params after eq-table
    lookup (these atoms are absent from the pRT3 equilibrium table)
  - make_free_params_free_chem(): added log_HF, log_Na, log_Ca, log_Ti — all U(-12,-2)
  - make_free_params_equilibrium(): same four params added (hybrid: eq table for main
    molecular species, free VMR for atomic species not in the table)
  - species_info.csv updated: HF→1H-19F, Na→23Na, Ti→48Ti, Ca→40Ca (Kurucz/Coxon-Hajig)
  References: Picos et al. (2024). A&A. §4.2, Table 3.

Changes from v2.0 (v3.0)
--------------------------
v3.0 (2026-05-05) — EddySed cloud model (Ackerman & Marley 2001 / Xuan+2024 §4.3.3)
  - Added cloud mass fraction computation via simple_cdf_free (pRT3 chemistry)
  - Added _add_cloud_mass_fractions() to pRT_spectrum (free + equil chemistry)
  - Added _build_cloud_kwargs() to pRT_spectrum (passes fsed, Kzz, sigma to pRT3)
  - Modified Retrieval.get_atmosphere_objects() to pass cloud_species to Radtrans
  - Added make_free_params_free_chem_cloudy() and make_free_params_equil_chem_cloudy()
  - Cloud species: MgSiO3(s)_crystalline_000, Fe(s)_crystalline_000 (Xuan+2024 §4.3.3)
  - Prior ranges from Xuan+2024 Table 3; shared fsed (pRT3 scalar API limitation)

Changes from v1.1 (v2.0)
--------------------------
- New physical flux scaling: the analytic φ marginalization is DISABLED.
  Instead, the model spectrum is multiplied by (R/d)² at the Earth, using
  the retrieved log_R and the known Gaia distance to DH Tau (135.2 pc).
- The continuum normalisation step in _load_night() is also disabled
  (normalize_method=None) — absolute flux information must be preserved.
- LogLikelihood supports scale_flux='physical' mode: N_phi=0, phi=1.
- pRT_spectrum.make_spectrum() applies (R/d)² scaling when
  use_absolute_flux=True (new kwarg in __init__).
- log_R prior changed to Gaussian N(0.415, 0.100) from Xuan+2024 Table 1 /
  Figure A1 (R = 2.6 ± 0.6 RJup; σ_log = 0.434 × 0.6/2.6 ≈ 0.100).
  Previously uniform U(−0.1, 0.5) in v2.0; reverted to evolutionary-model
  Gaussian so that M and R priors are both physically motivated.
- log_M prior remains Gaussian N(1.079, 0.145) from Xuan+2024 Table 1.
- New module-level constants: D_DH_TAU_PC, PC_M, R_JUP_M, M_JUP_KG, G_SI.

Purpose: test whether the SINFONI-calibrated absolute flux level independently
constrains R, providing an additional physical anchor beyond evolutionary models.

Changes from v1.0 (inherited via v1.1)
---------------------------------------
- log_g replaced by log_M + log_R; gravity derived as g = G*M/R² (cgs).

Series 003PM — Piette & Madhusudhan (2020) P-T Profile  [updated from 002PM]

Retrieval workflow entry point for the DH Tau B CRIRES+ atmospheric retrieval
pipeline using the Piette & Madhusudhan (2020) P-T parameterisation.
All classes, helper functions, and the main pipeline entry point are defined
here so that downstream scripts can import and call them without duplicating
code.

Methodology Summary
-------------------
003PM  Piette & Madhusudhan (2020) P-T parameterisation
       Temperature-pressure profile parameterised via temperature *increments*
       (ΔT) between 8 fixed pressure nodes spanning 5–0.00001 bar.  Monotonicity
       is enforced by a non-negative lower bound on every ΔT prior (no explicit
       constraint code required).  Interpolation uses a monotonic cubic PCHIP
       spline (PchipInterpolator).  No Gaussian smoothing is applied (Rowland
       et al. 2023 showed it biases retrieval results; Xuan et al. 2024 §4.3.4
       follow this recommendation).

       Free P-T parameters: T_anchor (at 0.2 bar, log P = −0.7) + dT_1 … dT_7
       = 8 total.  Pressure nodes are customised for DH Tau B following
       Xuan+2024 §4.3.4: at least 4 nodes within the 90%-flux region
       (ECF peak ≈ 0.1 bar for DH Tau B), others equally spaced in log P.

       Two chemistry sub-modes share the same P-T model:
         003PM-EQ  Equilibrium chemistry  (pRT3 pre-calculated table)
         003PM-FR  Free chemistry         (log VMR per species)

       Two-night joint retrieval is supported: each night has its own
       topocentric RV parameter (rv_N1, rv_N2); pRT radiative transfer is
       cached and shared between nights.

Changes from 002PM → 003PM
---------------------------
  B  pRT pressure grid updated to (10⁻⁵, 10) bar, matching the new DH Tau B
       PCHIP node range (10^0.7 = 5 bar deepest, 10⁻⁵ shallowest).  All 50
       pRT layers fall within the node range; ~9 layers/dex resolution.

  A  Upper ΔT priors widened to match Piette+2020 Table 1 ([0, 1000] K).
       002PM used [0, 500–600] K for dT_5/6/7 — too tight, favouring
       near-isothermal solutions in the upper atmosphere.

  D  Stale comment "P-T profile (DG, Picos+2025)" corrected to
       "P-T profile (Piette+2020 / Xuan+2024)".

References
----------
Piette, A. A. A. & Madhusudhan, N. (2020). MNRAS 497, 5136.
    §4.2, Table 1 (P-T parameterisation, priors, and pressure grid).
Xuan, J. W. et al. (2024). ApJ 970:71.
    §4.3.4, Table C1 (DH Tau b: anchor at log P = −0.7, 0.2 bar; nodes
    customised from ECF; priors from Table C1).
Rowland, M. J. et al. (2023). On Gaussian smoothing in P-T retrievals.

Usage
-----
As a module (recommended):

    from Guidebook_GAStronomy_Piette_v1.0 import run_retrieval_pipeline

    run_retrieval_pipeline(
        chemistry='equilibrium',          # 'equilibrium' or 'free'
        normalize_method='savgol',        # 'savgol', 'median', 'per_chip_median', or None
        per_chip_scaling=False,
        N_live_points=700,
        evidence_tol=0.5,
        nights={                          # one or two nights
            'N1': {'night': '2022-12-31',
                   'flux_file': 'extracted_spectra_combined_sigmaclipper.npy',
                   'err_file':  'extracted_spectra_combined_err_sigmaclipper.npy'},
            'N2': {'night': '2023-01-01',
                   'flux_file': 'extracted_spectra_combined_sigmaclipper.npy',
                   'err_file':  'extracted_spectra_combined_err_sigmaclipper.npy'},
        },
    )

As a standalone script:

    python Guidebook_GAStronomy_Piette_v1.0.py
"""

# ============================================================
# 0. ENVIRONMENT SETUP (must come before any MPI-using imports)
# ============================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ============================================================
# 1. IMPORTS
# ============================================================
import numpy as np
import pymultinest
import pathlib
import pickle
import pandas as pd
import corner
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for cluster use
import matplotlib.pyplot as plt

import time
import petitRADTRANS as prt
from petitRADTRANS.radtrans import Radtrans

from PyAstronomy.pyasl import fastRotBroad

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from scipy.special import loggamma
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve, cholesky_banded, cho_solve_banded

import sys

# pRT3 configuration
from petitRADTRANS.config import petitradtrans_config_parser
petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')

from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import simplify_species_list
from petitRADTRANS.chemistry.clouds import (    # v3.0: EddySed cloud model
    simple_cdf_free,
    return_cloud_mass_fraction,
)

# ============================================================
# 1b. MODULE-LEVEL SPECIES INFO CACHE
# ============================================================
_SPECIES_INFO_CACHE = None


def _get_species_info():
    """Return species_info.csv as a DataFrame, loaded once per process.

    Eliminates repeated pd.read_csv() calls inside the hot likelihood path
    (free_chemistry, read_species_info, get_species).
    """
    global _SPECIES_INFO_CACHE
    if _SPECIES_INFO_CACHE is None:
        _SPECIES_INFO_CACHE = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'),
            index_col=0,
        )
    return _SPECIES_INFO_CACHE


# ============================================================
# 2. GLOBAL PATHS AND CONSTANTS
# ============================================================
WORKPATH     = '/data2/peng/'
RESULTS_ROOT = pathlib.Path(WORKPATH) / 'retrievals'

# Physical constants for absolute flux scaling (v2.0)
D_DH_TAU_PC = 135.2        # Gaia DR3 distance to DH Tau system (pc)
PC_M         = 3.0857e16   # metres per parsec
R_JUP_M      = 7.149e7     # Jupiter radius in metres
M_JUP_KG     = 1.898e27    # Jupiter mass in kg
G_SI         = 6.674e-11   # Gravitational constant, SI (m³ kg⁻¹ s⁻²)

# Explicit pRT3 line-list name overrides (prevent interactive disambiguation
# prompts that crash MPI jobs when multiple line lists exist for one species).
PRT3_SPECIES_OVERRIDES = {
    '12C-16O':      '12C-16O__HITEMP',       # HITEMP preferred (two files on disk)
    '12C-1H4':      '12C-1H4__MM',           # ExoMol MM; better at high T
    'CO2_main_iso': '12C-16O2__HITEMP',
    'FeH_main_iso': '56Fe-1H',
    'HCN':          '1H-12C-14N',
    'H2S':          '1H2-32S',
    'NH3':          '14N-1H3',
}

# pRT3 species list for equilibrium-chemistry retrievals (K-band K2166)
# Names verified against /net/lem/data2/pRT3_formatted/input_data/opacities/
EQ_SPECIES_PRT3 = [
    '1H2-16O',            # H2O  — POKAZATEL
    '12C-16O__HITEMP',    # ¹²CO — HITEMP (explicit; two files on disk)
    '13C-16O',            # ¹³CO — HITEMP
    '12C-1H4__MM',        # CH4  — MM/ExoMol (high-T optimised)
    '14N-1H3',            # NH3  — CoYuTe
    '1H2-32S',            # H2S  — AYT2
    '1H-12C-14N',         # HCN  — Harris
    '12C-16O2__HITEMP',   # CO2  — HITEMP
    '56Fe-1H',            # FeH  — MoLLIST
    # v4.0: atomic/HF species (Picos+2024 §4.2; Coxon-Hajig/Kurucz line lists)
    '1H-19F',             # HF   — Coxon-Hajig (Wilzewski+2016)
    # '23Na',             # Na   — commented out (log_Na not a free param)
    # '48Ti',             # Ti   — commented out (log_Ti not a free param)
    # '40Ca',             # Ca   — commented out (log_Ca not a free param)
]
EQ_LABEL_NAMES = ['H2O', '12CO', '13CO', 'CH4', 'NH3', 'H2S', 'HCN', 'CO2', 'FeH',
                  'HF']

# v4.0: solar fluorine reference for [F/H] parameterisation (Picos+2024 Eq. 3)
# [F/H] = log10(n_HF/n_H) - log10(n_HF/n_H)_solar
# Solar value from Maiorca et al. (2014); Asplund et al. (2021).
LOG_F_H_SOLAR = -7.6

# v3.0: EddySed cloud species (pRT3 condensate opacity names, verified on disk)
CLOUD_MGSIO3 = 'MgSiO3(s)_crystalline_000'   # enstatite; cloud base ~1600–1900 K at 0.1 bar
CLOUD_FE     = 'Fe(s)_crystalline_000'         # iron; cloud base ~1800–2200 K at 0.1 bar
CLOUD_SPECIES_DEFAULT = [CLOUD_MGSIO3, CLOUD_FE]   # Xuan+2024 §4.3.3 two-species default


# ============================================================
# 3. CLASS: Target
# ============================================================
class Target:
    """Container for one night's observed spectrum.

    Parameters
    ----------
    wl, fl, err : ndarrays
        Wavelength (nm), flux, and error arrays.
        Shape (n_orders, n_dets, n_pixels) for 3D or (n_pixels,) for 1D.
    name : str
        Short identifier, e.g. 'dh_tau_b_N1'.
    """

    def __init__(self, wl, fl, err, name='dh_tau_b'):
        self.name     = name
        self.fullname = 'DH_Tau_B'

        if wl.ndim == 3:
            self.n_orders = wl.shape[0]
            self.n_dets   = wl.shape[1]
        elif wl.ndim == 1:
            self.n_orders = 5   # K2166 default
            self.n_dets   = 3
        else:
            raise ValueError(f'Unrecognised spectrum shape: {wl.shape}')

        self.n_pixels = 2048

        # CRIRES+ K2166 wavelength boundaries [nm] per (order, detector)
        # Rows: orders (index 0 = reddest K-order); cols: detectors 1-3
        self.K2166 = np.array([
            [[1921.318, 1934.583], [1935.543, 1948.213], [1949.097, 1961.128]],
            [[1989.978, 2003.709], [2004.701, 2017.816], [2018.708, 2031.165]],
            [[2063.711, 2077.942], [2078.967, 2092.559], [2093.479, 2106.392]],
            [[2143.087, 2157.855], [2158.914, 2173.020], [2173.983, 2187.386]],
            [[2228.786, 2244.133], [2245.229, 2259.888], [2260.904, 2274.835]],
            [[2321.596, 2337.568], [2338.704, 2353.961], [2355.035, 2369.534]],
            [[2422.415, 2439.061], [2440.243, 2456.145], [2457.275, 2472.388]],
        ])[::-1]   # reverse so index 0 = shallowest / shortest λ order

        # DH Tau B astrometry and observation epoch
        self.ra    = '04h29m41.65s'
        self.dec   = '26d32m56.2s'
        self.JD    = 59944.17637 + 2.4e6   # observation date in JD
        self.color = 'limegreen'

        self.wl  = wl
        self.fl  = fl
        self.err = err
        print(f'[Target {self.name}] wl:{wl.shape}  fl:{fl.shape}  err:{err.shape}')

    def get_mask_isfinite(self):
        """Return boolean mask of finite pixels (True = valid)."""
        if self.wl.ndim == 1:
            self.mask_isfinite = np.isfinite(self.fl)
        else:
            self.mask_isfinite = np.empty(
                (self.n_orders, self.n_dets, self.n_pixels), dtype=bool
            )
            for i in range(self.n_orders):
                for j in range(self.n_dets):
                    self.mask_isfinite[i, j] = np.isfinite(self.fl[i, j])
        return self.mask_isfinite


# ============================================================
# 4. CLASS: Parameters
# ============================================================
class Parameters:
    """Prior specification and prior-transform for MultiNest.

    Parameters
    ----------
    free_params : dict
        {key: (prior_spec, mathtext_label)}
        prior_spec is either:
          - [lo, hi]  → uniform prior
          - {'type': 'gaussian', 'mu': μ, 'sigma': σ}  → Gaussian prior
    constant_params : dict
        Fixed parameter values added directly to self.params.

    Notes
    -----
    For the Piette+2020 P-T parameterisation the prior lower bound of 0 on
    every dT_i is sufficient to enforce monotonicity.  No ordering constraints
    are required inside __call__.
    """

    def __init__(self, free_params, constant_params):
        self.params = {}

        self.param_priors   = {}
        self.param_mathtext = {}
        for key, (prior, mathtext) in free_params.items():
            self.param_priors[key]   = prior
            self.param_mathtext[key] = mathtext

        self.param_keys     = np.array(list(self.param_priors.keys()))
        self.n_params       = len(self.param_keys)
        self.ndim           = self.n_params
        self.free_params    = free_params
        self.constant_params = constant_params
        self.params.update(constant_params)

    @staticmethod
    def uniform_prior(bounds):
        return lambda x: x * (bounds[1] - bounds[0]) + bounds[0]

    @staticmethod
    def gaussian_prior(mu, sigma):
        return lambda x: norm.ppf(x, loc=mu, scale=sigma)

    def __call__(self, cube, ndim=None, nparams=None):
        """MultiNest prior transform: unit hypercube → physical parameter space.

        For the Piette+2020 P-T parameterisation all dT_i ≥ 0 is enforced
        purely by the prior lower bound (= 0), so no explicit constraint code
        is needed here.
        """
        if ndim is None and nparams is None:
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])

        for i, key in enumerate(self.param_keys):
            prior = self.param_priors[key]
            if isinstance(prior, dict) and prior.get('type') == 'gaussian':
                cube[i] = self.gaussian_prior(prior['mu'], prior['sigma'])(cube[i])
            else:
                cube[i] = self.uniform_prior(prior)(cube[i])
            self.params[key] = cube[i]

        return self.cube_copy


# ============================================================
# 5. CLASS: Covariance
# ============================================================
class Covariance:
    """Diagonal covariance matrix built from per-pixel error arrays.

    Only the diagonal is stored (1-D array of variances σ²).
    """

    def __init__(self, err):
        self.err = err
        self.cov_reset()

    def cov_reset(self):
        self.cov = self.err ** 2

    def get_logdet(self):
        self.logdet = np.sum(np.log(self.cov))
        return self.logdet

    def solve(self, b):
        """Return C⁻¹ b for diagonal C."""
        return b / self.cov


# ============================================================
# 5b. CLASS: CovarianceGP
# ============================================================
class CovarianceGP:
    """Correlated noise covariance: block-diagonal diagonal + heteroscedastic GP kernel.

    Implements González Picos et al. (2025) Eq. 5–6 per detector chip:
      Σ₀,ij = δᵢⱼ σᵢ²  +  kᵢⱼ
      kᵢⱼ  = a² × σᵢ σⱼ × exp(−(xᵢ−xⱼ)² / 2l²)

    Block-diagonal structure: the GP length scale l is bounded to ≤ 1 nm
    (log_l prior upper bound = 0).  The minimum K2166 inter-detector gap is
    ≈ 1.07 nm; inter-chip correlation across that gap is < exp(−0.5) ≈ 0.6
    at l=1 nm but is already zero within the banded approximation (B ≤ 100
    pixels ≈ 0.7 nm bandwidth cap).  Splitting at chip_gap_nm=0.5 nm ensures
    each sub-chip fits within one K2166 detector, which is required for the
    per-chip φ scaling (flux_scaling_rolling / _RangeCovProxy) to be correct.

    v4.2: dense N×N Cholesky replaced by banded Cholesky.  Bandwidth
    B = min(ceil(5l/δλ), MAX_BANDWIDTH) pixels; hard cap at MAX_BANDWIDTH=100
    prevents sampling distant uncorrelated pixels regardless of prior bounds.
    Memory per chip: O(N×B) instead of O(N²) — ~18× reduction at B=100,
    N=1800.  Cholesky cost: O(N×B²) instead of O(N³) — ~300× reduction.
    Truncation error: < exp(−12.5) ≈ 3.7×10⁻⁶ (negligible vs. float64).

    Parameters
    ----------
    err : 1-D array
        Per-pixel uncertainties σᵢ (valid/unmasked pixels only).
    wl  : 1-D array
        Corresponding wavelengths in nm (same length as err).
    chip_gap_nm : float
        Wavelength gap (nm) that signals a chip boundary.  Default 0.5 nm —
        below the minimum K2166 inter-detector gap (≈1.07 nm) so each
        K2166 detector becomes its own sub-chip (required for _RangeCovProxy).
        Use 2.0 if you want the old per-order blocking (breaks per-chip φ).

    Usage
    -----
    Call build(a, l) once per likelihood evaluation; then solve() and
    get_logdet() use the stored per-chip banded Cholesky factors.
    """

    MAX_BANDWIDTH = 100   # hard pixel cap on GP kernel width (v4.2)

    def __init__(self, err, wl, chip_gap_nm=0.5):
        self.err = err
        self.wl  = wl
        self.a   = None
        self.l   = None
        self._banded_factors = []
        # Partition pixels into contiguous chip blocks by detecting large gaps.
        breaks = np.where(np.abs(np.diff(wl)) > chip_gap_nm)[0] + 1
        starts = np.concatenate([[0], breaks])
        ends   = np.concatenate([breaks, [len(wl)]])
        self._chip_slices = [slice(int(s), int(e)) for s, e in zip(starts, ends)]
        # Precompute per-chip static metadata (avoids recomputing on every build()).
        self._chip_n = [int(sl.stop - sl.start) for sl in self._chip_slices]
        self._pixel_spacing = []
        for sl in self._chip_slices:
            wl_c = wl[sl]
            self._pixel_spacing.append(
                float(np.median(np.diff(wl_c))) if len(wl_c) > 1 else 1.0
            )

    def build(self, a, l):
        """Build block-diagonal Σ₀ and compute per-chip banded Cholesky factors."""
        self.a = a
        self.l = l
        self._banded_factors = []
        for chip_idx, sl in enumerate(self._chip_slices):
            err_c = self.err[sl]
            wl_c  = self.wl[sl]
            n     = self._chip_n[chip_idx]
            px    = self._pixel_spacing[chip_idx]
            # Bandwidth: 5 length-scales, hard-capped at MAX_BANDWIDTH and chip size.
            B = min(int(np.ceil(5.0 * l / px)), self.MAX_BANDWIDTH, n - 1)
            # Build upper-triangular band storage: ab[k, i] = C[i, i+k].
            ab = np.zeros((B + 1, n))
            ab[0, :] = err_c ** 2 * (1.0 + a**2 + 1e-6)  # noise + GP self-cov + jitter
            for k in range(1, B + 1):
                dlambda     = wl_c[k:] - wl_c[:n - k]          # 1-D, length n-k
                ab[k, :n-k] = (a ** 2
                                * (err_c[:n-k] * err_c[k:])
                                * np.exp(-dlambda ** 2 / (2.0 * l ** 2)))
            # cholesky_banded returns the lower-triangular factor in band format.
            # overwrite_ab=True avoids an internal copy (ab is not reused after this).
            cb = cholesky_banded(ab, lower=True, overwrite_ab=True)
            self._banded_factors.append(cb)

    def solve(self, b):
        """Return Σ₀⁻¹ b using per-chip banded Cholesky factors."""
        result = np.empty_like(b)
        for sl, cb in zip(self._chip_slices, self._banded_factors):
            result[sl] = cho_solve_banded((cb, True), b[sl])
        return result

    def get_logdet(self):
        """Return log|Σ₀| = sum over chips of 2 × Σᵢ log cb[0,i].

        cb[0, :] is the diagonal of the lower-triangular Cholesky factor L,
        so log|Σ₀| = log|LLᵀ| = 2 Σᵢ log(Lᵢᵢ).
        """
        return sum(2.0 * np.sum(np.log(cb[0, :])) for cb in self._banded_factors)

    def solve_for_wl_range(self, b, lo_nm, hi_nm):
        """Apply Σ₀⁻¹ to vector b using only sub-chips in [lo_nm, hi_nm].

        Reuses the banded Cholesky factors already stored by build() — no
        redundant factorisation.  b must contain exactly the valid pixels
        whose wavelengths fall in [lo_nm, hi_nm], in the same order as
        self.wl restricted to that range (i.e. the natural flattened order).

        If precompute_K2166_ranges() has been called, the per-call sub-chip
        scan is replaced by an O(1) dict lookup — eliminates boolean array
        creation and sum() for each of the ~18 sub-chips per K2166 chip.

        Parameters
        ----------
        b      : 1-D array, length = number of valid pixels in [lo_nm, hi_nm]
        lo_nm, hi_nm : float — wavelength window in nm

        Returns
        -------
        result : 1-D array, same shape as b
        """
        result = np.empty_like(b)
        pos = 0
        relevant = getattr(self, '_range_cache', {}).get((lo_nm, hi_nm))
        if relevant is not None:
            for i, n in relevant:
                result[pos:pos + n] = cho_solve_banded(
                    (self._banded_factors[i], True), b[pos:pos + n]
                )
                pos += n
        else:
            for sl, cb in zip(self._chip_slices, self._banded_factors):
                wl_c = self.wl[sl]
                in_range = (wl_c >= lo_nm) & (wl_c <= hi_nm)
                n = int(in_range.sum())
                if n == 0:
                    continue
                result[pos:pos + n] = cho_solve_banded((cb, True), b[pos:pos + n])
                pos += n
        return result

    def precompute_K2166_ranges(self, K2166_bounds):
        """Cache which sub-chips serve each K2166 (order, det) wavelength range.

        Call once after __init__, before the retrieval loop.  Subsequent calls
        to solve_for_wl_range skip the per-call sub-chip scan (eliminates
        ~18 boolean array creations per chip per likelihood call).

        Parameters
        ----------
        K2166_bounds : array, shape (n_orders, n_dets, 2)
            Wavelength boundaries in nm; same as Retrieval.K2166.
        """
        self._range_cache = {}
        for order in range(K2166_bounds.shape[0]):
            for det in range(K2166_bounds.shape[1]):
                lo = float(K2166_bounds[order, det, 0])
                hi = float(K2166_bounds[order, det, 1])
                relevant = []
                for i, sl in enumerate(self._chip_slices):
                    wl_c = self.wl[sl]
                    if len(wl_c) == 0 or wl_c[-1] < lo or wl_c[0] > hi:
                        continue
                    n = int(((wl_c >= lo) & (wl_c <= hi)).sum())
                    if n > 0:
                        relevant.append((i, n))
                self._range_cache[(lo, hi)] = relevant

    def _chip_index_for_mask(self, mask):
        # NOTE: superseded by solve_for_wl_range / _RangeCovProxy (v4.2 fix).
        # This method has an index-space bug: `mask` is on the FULL (uncompressed)
        # pixel array, but _chip_slices are indices into the COMPRESSED (valid-only)
        # array.  Do not call — kept for reference only.
        """Return the chip-slice index whose pixels are selected by mask.

        Finds the chip containing the first selected pixel via O(N_chips)
        linear scan.  Returns None if mask is empty or no chip matches.
        Used by flux_scaling_rolling to reuse pre-computed Cholesky factors.
        """
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        first = int(idx[0])
        for i, sl in enumerate(self._chip_slices):
            if sl.start <= first < sl.stop:
                return i
        return None


# ============================================================
# 5c. CLASS: _ChipCovProxy
# ============================================================
class _ChipCovProxy:
    """Lightweight proxy exposing one chip's banded Cholesky factor.
    Superseded by _RangeCovProxy; retained for reference only.
    """

    __slots__ = ('_cb',)

    def __init__(self, cb):
        self._cb = cb

    def solve(self, b):
        return cho_solve_banded((self._cb, True), b)


# ============================================================
# 5d. CLASS: _RangeCovProxy
# ============================================================
class _RangeCovProxy:
    """Proxy that applies CovarianceGP.solve_for_wl_range for a fixed wavelength window.

    Used by LogLikelihood.flux_scaling_rolling() to compute per-chip φ
    without rebuilding the banded Cholesky factorisation — the factors
    already stored in the full-night CovarianceGP._banded_factors are
    reused directly.

    Parameters
    ----------
    cov    : CovarianceGP — must have build() already called.
    lo_nm  : float — lower wavelength bound of the K2166 chip (nm).
    hi_nm  : float — upper wavelength bound of the K2166 chip (nm).
    """

    __slots__ = ('_cov', '_lo', '_hi')

    def __init__(self, cov, lo_nm, hi_nm):
        self._cov = cov
        self._lo  = lo_nm
        self._hi  = hi_nm

    def solve(self, b):
        return self._cov.solve_for_wl_range(b, self._lo, self._hi)


# ============================================================
# 6. CLASS: LogLikelihood
# ============================================================
class LogLikelihood:
    """Ruffio+2019 log-likelihood with optional linear flux scaling.

    Parameters
    ----------
    d_flux : 1-D array
        Observed flux (full length, including masked pixels).
    d_mask : 1-D bool array
        True where pixel is valid (non-NaN, non-inf).
    scale_flux : True | False | None | 'Single' | 'physical'
        True       → one scaling factor φ per (order, detector) chip.
        'Single'   → one global φ.
        False      → no scaling (φ = 1 everywhere).
        'physical' → no scaling, no marginalized φ degree of freedom (v2.0).
                     Model flux is already absolutely scaled by (R/d)².
    N_params : int
        Number of free retrieval parameters (used in the Ruffio formula).
    retrieval_object : Retrieval | None
        Required when scale_flux is True (for per-chip wavelength boundaries).
    alpha : float
        Ruffio+2019 hyperparameter (default 2).

    Reference
    ---------
    Ruffio, J.-B. et al. (2019). ApJ, 881, 1.  [Eq. A1 – A6]
    """

    def __init__(self, d_flux, d_mask, d_wave, d_err, scale_flux, N_params,
                 retrieval_object=None, alpha=2):
        self.d_flux  = d_flux
        self.d_mask  = d_mask
        self.d_wave  = d_wave
        self.d_err   = d_err
        self.scale_flux = scale_flux
        self.scale_err  = (scale_flux is not False)
        self.N_d     = d_mask.sum()
        self.N_params = N_params
        self.alpha   = alpha
        self.retrieval_object = retrieval_object

        if self.scale_flux is True:
            self.N_phi = retrieval_object.n_orders * retrieval_object.n_dets
        elif self.scale_flux == 'physical':
            self.N_phi = 0   # no marginalized φ; absolute scale is physical
        else:
            self.N_phi = 1

        # Precompute per-chip wavelength masks and data slices (static — they
        # depend only on fixed d_wave/d_mask/K2166 which never change between calls).
        # Eliminates 15 × 3 boolean array operations per PMN_lnL call.
        self._chip_masks  = None
        self._chip_d_cuts = None
        if scale_flux is True and retrieval_object is not None:
            ret = retrieval_object
            masks, d_cuts = [], []
            for order in range(ret.n_orders):
                for det in range(ret.n_dets):
                    mask_od = (
                        (d_wave >= ret.K2166[order, det, 0])
                        & (d_wave <= ret.K2166[order, det, 1])
                        & d_mask
                    )
                    masks.append(mask_od)
                    d_cuts.append(d_flux[mask_od])
            self._chip_masks  = masks
            self._chip_d_cuts = d_cuts

    def __call__(self, m_flux, Cov):
        self.ln_L       = 0.0
        self.chi2_0     = 0.0
        self.m_flux_phi = np.full_like(self.d_flux, np.nan)

        N_d   = self.d_mask.sum()
        m_masked = m_flux[self.d_mask]

        if self.scale_flux is True:
            # Pass the full N_total model array; flux_scaling_rolling applies
            # self.d_mask internally to exclude bad pixels per chip.
            self.m_flux_phi, self.phi = self.flux_scaling_rolling(m_flux, Cov)
        elif self.scale_flux in (None, 'Single'):
            d_masked = self.d_flux[self.d_mask]
            self.m_flux_phi[self.d_mask], self.phi = self.get_flux_scaling(d_masked, m_masked, Cov)
        elif self.scale_flux == 'physical':
            # Absolute flux mode (v2.0): model already scaled by (R/d)².  No φ.
            self.m_flux_phi[self.d_mask] = m_masked
            self.phi = np.ones(1)
        else:
            self.m_flux_phi[self.d_mask] = m_masked
            self.phi = np.ones(1)

        residuals        = self.d_flux - self.m_flux_phi
        inv_cov_residuals = Cov.solve(residuals[self.d_mask])
        chi2_0           = np.dot(residuals[self.d_mask], inv_cov_residuals)

        self.s2 = self.get_err_scaling(chi2_0, N_d) if self.scale_err else 1.0

        logdet_cov = Cov.get_logdet()
        if self.scale_flux in (False, 'physical'):
            # phi=1 fixed — standard multivariate Gaussian (no phi marginalisation).
            # logdet_MT_C_M is absent: it arises only from integrating over phi.
            self.ln_L += (
                -0.5 * N_d * np.log(2 * np.pi)
                - 0.5 * (logdet_cov + chi2_0)
            )
        else:
            # phi marginalised (scale_flux True / 'Single' / None) — Ruffio+2019.
            inv_cov_M     = Cov.solve(m_masked)
            MT_inv_cov_M  = np.dot(m_masked, inv_cov_M)
            logdet_MT_C_M = np.log(MT_inv_cov_M)
            self.ln_L += (
                -0.5 * (N_d - self.N_phi) * np.log(2 * np.pi)
                + loggamma(0.5 * (N_d - self.N_phi + self.alpha - 1))
                - 0.5 * (logdet_cov + logdet_MT_C_M
                         + (N_d - self.N_phi + self.alpha - 1) * np.log(chi2_0))
            )
        self.chi2_0     += chi2_0
        self.chi2_0_red  = self.chi2_0 / self.N_d
        return self.ln_L

    # ---- helpers ----

    def get_flux_scaling(self, d_flux, m_flux, Cov):
        """Analytic optimal linear scaling: φ = (Mᵀ C⁻¹ d) / (Mᵀ C⁻¹ M)."""
        lhs = np.dot(m_flux, Cov.solve(m_flux))
        rhs = np.dot(m_flux, Cov.solve(d_flux))
        phi = rhs / lhs
        return m_flux * phi, phi

    def get_err_scaling(self, chi2, N):
        return np.sqrt(chi2 / N)

    def flux_scaling_rolling(self, m_flux, Cov):
        """Per-chip scaling: one φ per (order, detector) pair.

        Parameters
        ----------
        m_flux : array, shape (N_total,)
            Model flux on the full data grid (all pixels, including bad ones).
        Cov : CovarianceGP or Covariance
            Full-night covariance; only Cov.a and Cov.l are used here to
            build per-chip mini-GPs.

        Returns
        -------
        m_scaled : array, shape (N_total,)
            φ-scaled model; NaN at bad pixels and outside K2166 chip ranges.
        phi_all : array, shape (n_orders × n_dets,)
            Best-fit per-chip scaling factors φ.
        """
        ret = self.retrieval_object
        m_scaled = np.full_like(self.d_flux, np.nan)   # N_total
        phi_all  = np.full(ret.n_orders * ret.n_dets, np.nan)

        chip_i = 0
        for order in range(ret.n_orders):
            for det in range(ret.n_dets):
                # Use precomputed static mask/data-cut (built once in __init__).
                # self.d_wave / self.d_err are night-specific (correct for
                # both LogLike and LogLike2).  self.d_mask excludes bad pixels.
                if self._chip_masks is not None:
                    mask_od = self._chip_masks[chip_i]
                    d_cut   = self._chip_d_cuts[chip_i]
                else:
                    mask_od = (
                        (self.d_wave >= ret.K2166[order, det, 0])
                        & (self.d_wave <= ret.K2166[order, det, 1])
                        & self.d_mask
                    )
                    d_cut = self.d_flux[mask_od]
                chip_i += 1

                if len(d_cut) < 10:
                    continue
                m_cut = m_flux[mask_od]

                if isinstance(Cov, CovarianceGP) and Cov._banded_factors:
                    # Reuse the banded Cholesky factors already computed by
                    # Cov.build() — no redundant factorisation per chip.
                    lo = ret.K2166[order, det, 0]
                    hi = ret.K2166[order, det, 1]
                    Cov_cut = _RangeCovProxy(Cov, lo, hi)
                elif isinstance(Cov, CovarianceGP):
                    # build() not yet called (should not happen in normal flow).
                    Cov_cut = CovarianceGP(
                        err=self.d_err[mask_od],
                        wl=self.d_wave[mask_od],
                    )
                    Cov_cut.build(Cov.a, Cov.l)
                else:
                    Cov_cut = Covariance(err=self.d_err[mask_od])

                m_scaled[mask_od], phi_all[order * ret.n_dets + det] = \
                    self.get_flux_scaling(d_cut, m_cut, Cov_cut)

        # Guard: valid pixels whose wavelengths fall just outside all K2166 chip
        # boundary definitions (detector-edge pixels offset ≤ 0.02 nm from the
        # nominal boundary) remain NaN in m_scaled after the chip loop.  NaN
        # propagates to chi2_0 → ln_L = NaN → every PMN_lnL call returns −1e101,
        # hanging MultiNest.  Treat gap pixels as phi=1 (unscaled model flux) —
        # they are a tiny fraction of edge pixels and the approximation is negligible.
        gap_valid = self.d_mask & np.isnan(m_scaled)
        if gap_valid.any():
            m_scaled[gap_valid] = m_flux[gap_valid]

        return m_scaled, phi_all


# ============================================================
# 7. CLASS: pRT_spectrum
# ============================================================
class pRT_spectrum:
    """Forward model: generates a synthetic spectrum for comparison with data.

    Workflow per likelihood call
    ----------------------------
    1. make_pt()           — build the Piette+2020 P-T profile (PCHIP ΔT nodes)
    2. *_chemistry()       — derive mass fractions (free or equilibrium)
    3. _make_prt_flux()    — run pRT3 radiative transfer (cached)
    4. make_spectrum()     — per-night: RV shift → rotBroad → LSF → interp → norm

    The pRT radiative transfer step is expensive (~seconds).  It is cached
    on the instance via _make_prt_flux() so that a two-night retrieval can
    call make_spectrum() twice (once per night with its own rv_key) without
    repeating the RT calculation.
    """

    def __init__(self, retrieval_object, spectral_resolution=100_000,
                 contribution=False, use_absolute_flux=False):
        self.params             = retrieval_object.parameters.params
        self.data_wave          = retrieval_object.data_wave
        self.target             = retrieval_object.target
        self.atmosphere_objects = retrieval_object.atmosphere_objects
        self.coords             = SkyCoord(
            ra=self.target.ra, dec=self.target.dec, frame='icrs'
        )
        self.species            = retrieval_object.species
        self.spectral_resolution = spectral_resolution
        self.lbl_opacity_sampling = retrieval_object.lbl_opacity_sampling
        self.n_atm_layers       = retrieval_object.n_atm_layers
        self.pressure           = retrieval_object.pressure
        self.contribution       = contribution
        self.normalize_flux     = retrieval_object.normalize_flux
        self.eq_chem            = getattr(retrieval_object, 'eq_chem', None)
        # v3.0: cloud configuration (inherited from retrieval)
        self.cloud_species      = getattr(retrieval_object, 'cloud_species', [])
        self.use_clouds         = bool(self.cloud_species)
        # v2.0: absolute flux mode — multiply pRT surface flux by (R/d)²
        self._use_absolute_flux = use_absolute_flux

        # --- P-T profile (Piette+2020 / Xuan+2024) ---
        self.temperature = self.make_pt()

        # --- gravity: derived from log_M + log_R (absolute flux mode) OR
        #     taken directly from log_g (normalised/Ruffio mode, no M/R needed).
        # G = 6.674e-8 cm³/(g·s²), M_Jup = 1.898e30 g, R_Jup = 7.149e9 cm
        if 'log_M' in self.params and 'log_R' in self.params:
            _M_g  = 10 ** self.params['log_M'] * 1.898e30
            _R_cm = 10 ** self.params['log_R'] * 7.149e9
            self.gravity = 6.674e-8 * _M_g / _R_cm ** 2
        else:
            self.gravity = 10 ** self.params['log_g']

        # --- chemistry ---
        chem = self.params.get('chemistry', 'free')
        if chem == 'free':
            self.mass_fractions, self.CO, self.C_H = \
                self.free_chemistry(self.species, self.params)
        elif chem == 'equilibrium':
            self.mass_fractions, self.CO, self.C_H = \
                self.equilibrium_chemistry(self.params)
        else:
            raise ValueError(f"chemistry must be 'free' or 'equilibrium', got {chem!r}")

        self.MMW = self.mass_fractions['MMW']

        # v3.0: append EddySed cloud mass fractions to self.mass_fractions
        if self.use_clouds:
            self._add_cloud_mass_fractions()

        # Cache slots (set by _make_prt_flux)
        self._prt_wl   = None
        self._prt_flux = None

    # ---- P-T profile (Piette+2020 / Xuan+2024) ----

    def make_pt(self):
        """Piette & Madhusudhan (2020) / Xuan et al. (2024) P-T parameterisation.

        8 pressure nodes customised for DH Tau B following Xuan+2024 Table C1;
        T_anchor at 0.2 bar (log P = −0.7) — inside the K-band photosphere
        where the DH Tau B emission contribution function peaks (~0.1 bar).
        7 ΔT increments (ΔTᵢ ≥ 0 enforced by prior lower bound), each equal
        to T(deeper node) − T(shallower node).
        Monotonic cubic (PCHIP) spline interpolation onto the 50-layer pressure
        grid.  No Gaussian smoothing (Rowland et al. 2023; Xuan+2024 §4.3.4).

        Nodes are dense near the photosphere (4 nodes spanning −0.3 to −1.5 dex
        where 90% of K-band flux originates), and coarser in the deep/upper atm.

        Fixed pressure nodes (log₁₀ bar, deep → shallow):
          index 0: log P = +0.7 (5.0 bar)     — deepest node
          index 1: log P =  0.0 (1.0 bar)     — dT_1 interval
          index 2: log P = −0.3 (0.5 bar)     — dT_2
          index 3: log P = −0.7 (0.2 bar)     — T_anchor (photosphere anchor)
          index 4: log P = −1.0 (0.10 bar)    — dT_4
          index 5: log P = −1.5 (0.032 bar)   — dT_5
          index 6: log P = −3.0 (0.001 bar)   — dT_6
          index 7: log P = −5.0 (10⁻⁵ bar)   — dT_7 (shallowest node)

        dT[j] = T(node j) − T(node j+1), with j ordered deep→shallow.
        dT_3 = T(node 2, 0.5 bar) − T_anchor; dT_4 = T_anchor − T(node 4, 0.1 bar).

        References
        ----------
        Piette & Madhusudhan (2020). MNRAS 497, 5136. §4.2.
        Xuan et al. (2024). ApJ 970:71. §4.3.4, Table C1 (DH Tau b).
        """
        # Fixed pressure nodes log10(bar), deep (index 0) to shallow (index 7)
        # Customised for DH Tau B following Xuan+2024 Table C1: dense sampling
        # in the photospheric 90%-flux region, coarser deep and upper layers.
        LOG_P_NODES = np.array([0.7, 0.0, -0.3, -0.7, -1.0, -1.5, -3.0, -5.0])
        ANCHOR_IDX  = 3   # log P = -0.7 (0.2 bar) — DH Tau B ECF peak ≈ 0.1 bar

        T_anchor = self.params['T_anchor']
        # dT[i] = T(node i) − T(node i+1), deep→shallow ordering (all ≥ 0)
        dT = np.array([self.params[f'dT_{i+1}'] for i in range(7)])

        T_nodes = np.empty(8)
        T_nodes[ANCHOR_IDX] = T_anchor

        # Deeper nodes (j < ANCHOR_IDX): temperature increases with pressure
        for j in range(ANCHOR_IDX - 1, -1, -1):
            T_nodes[j] = T_nodes[j + 1] + dT[j]

        # Shallower nodes (j > ANCHOR_IDX): temperature decreases going up
        for j in range(ANCHOR_IDX + 1, 8):
            T_nodes[j] = T_nodes[j - 1] - dT[j - 1]

        # PCHIP monotonic cubic spline — requires ascending x (shallow→deep)
        log_P_asc = LOG_P_NODES[::-1]   # −5.0 … +0.7
        T_asc     = T_nodes[::-1]

        pchip       = PchipInterpolator(log_P_asc, T_asc)
        log_P_atm   = np.log10(self.pressure)   # ascending, shallowest first
        temperature = pchip(log_P_atm)

        # Safety clamp at the node boundaries.  With the current pressure grid
        # (10⁻⁵–10¹ bar, encompassing all PCHIP nodes from 10⁻⁵ to 10^0.7 bar),
        # these guards are no-ops — every pRT layer falls within the node range.
        # They remain in place as a safeguard if the grid is ever changed.
        temperature = np.where(log_P_atm < log_P_asc[0],  T_asc[0],  temperature)
        temperature = np.where(log_P_atm > log_P_asc[-1], T_asc[-1], temperature)
        temperature = np.clip(temperature, 1.0, 30000.0)   # physical guard

        self.temperature = temperature
        return self.temperature

    # ---- species information ----

    def read_species_info(self, species, info_key):
        df = _get_species_info()
        if info_key == 'COH':
            return list(df.loc[species, ['C', 'O', 'H']])
        return df.loc[species, info_key]

    # ---- free chemistry ----

    def free_chemistry(self, line_species, params):
        """Convert log VMR parameters to pRT mass fractions.

        Computes C/O and [C/H] (relative to solar Asplund+2021) from the
        volume mixing ratios of all C-, O-, and H-bearing species.

        Reference
        ---------
        Picos et al. (2025).  GQ Lup b retrieval. Eq. for [C/H].
        """
        species_info = _get_species_info()
        line_species = set(line_species)
        VMR_He     = 0.15
        VMR_wo_H2  = VMR_He
        mass_frac  = {}
        C = O = H = 0

        for sp in species_info.index:
            prt_name = self.read_species_info(sp, 'pRT_name')
            prt_name = PRT3_SPECIES_OVERRIDES.get(prt_name, prt_name)
            mass     = self.read_species_info(sp, 'mass')
            COH      = self.read_species_info(sp, 'COH')

            if sp in ('H2', 'He'):
                continue
            if prt_name in line_species:
                VMR_i              = 10 ** params[f'log_{sp}'] * np.ones(self.n_atm_layers)
                mass_frac[prt_name] = mass * VMR_i
                VMR_wo_H2          += VMR_i
                C += COH[0] * VMR_i
                O += COH[1] * VMR_i
                H += COH[2] * VMR_i

        mass_frac['He'] = self.read_species_info('He', 'mass') * VMR_He
        mass_frac['H2'] = self.read_species_info('H2', 'mass') * (1 - VMR_wo_H2)
        H              += self.read_species_info('H2', 'H') * (1 - VMR_wo_H2)

        MMW = sum(mass_frac.values()) * np.ones(self.n_atm_layers)
        for k in mass_frac:
            mass_frac[k] /= MMW
        mass_frac['MMW'] = MMW

        CO            = np.nanmean(C / O)
        log_CH_solar  = -3.54   # log10(n_C/n_H)_solar, Asplund+2021
        C_H           = np.nanmean(np.log10(C / H)) - log_CH_solar

        return mass_frac, CO, C_H

    # ---- equilibrium chemistry ----

    def equilibrium_chemistry(self, params):
        """Equilibrium chemistry via pRT3 pre-calculated table.

        The free parameter is [C/H]; [Z/H] is derived internally:
            [Z/H] = [C/H] − log10(C/O / (C/O)_solar)
        where (C/O)_solar = 0.5495 (Lodders 2020).

        ¹³CO is absent from the equilibrium table and is derived from ¹²CO
        via the free parameter log_12CO_13CO (default: solar = log10(70) ≈ 1.845).

        HF, Na, Ca, Ti are absent from the equilibrium table and are injected
        from free params after the main table lookup (v4.0).  HF uses the [F/H]
        parameterisation (Picos+2024 Eq. 3): the free param F_H gives [F/H] and
        VMR_HF is derived as 10^([F/H] + LOG_F_H_SOLAR + log10(2 × VMR_H2)).
        Na, Ca, Ti use direct log VMR free params (log_Na, log_Ca, log_Ti).

        Reference
        ---------
        Lodders, K. (2020). Solar elemental abundances.
        Asplund, M. et al. (2021). ¹²C/¹³C solar ratio = 70.
        Picos et al. (2024). A&A. §4.2, Eq. 3 ([F/H] definition).
        Maiorca et al. (2014). [F/H] solar reference: log10(n_F/n_H)_solar = -7.6.
        """
        C_H = params['C_H']
        CO  = params['C/O']

        CO_SOLAR = 0.5495
        ZH       = C_H - np.log10(CO / CO_SOLAR)

        log_Pquench           = params.get('log_Pquench', None)
        carbon_pressure_quench = 10 ** log_Pquench if log_Pquench is not None else None

        mf_eq, MMW, _ = self.eq_chem.interpolate_mass_fractions(
            co_ratios              = CO * np.ones(self.n_atm_layers),
            log10_metallicities    = ZH * np.ones(self.n_atm_layers),
            temperatures           = self.temperature,
            pressures              = self.pressure,
            carbon_pressure_quench = carbon_pressure_quench,
            full                   = True,
        )

        mass_frac    = {}
        simple_names = simplify_species_list(self.species)
        ratio_12_13  = 10 ** params.get('log_12CO_13CO', np.log10(70.0))

        for prt_name, simple in zip(self.species, simple_names):
            if prt_name == '13C-16O':
                mass_frac['13C-16O'] = mf_eq['CO'] * (29.002355 / 28.009999) / ratio_12_13
            elif simple in mf_eq:
                mass_frac[prt_name] = mf_eq[simple]
            else:
                print(f'WARNING: {prt_name} ({simple}) not in equilibrium table, skipping.')

        mass_frac['H2']  = mf_eq['H2']
        mass_frac['He']  = mf_eq['He']
        mass_frac['MMW'] = MMW

        # v4.0: atomic/HF species — injected from free params after eq-table loop.
        # MMW is not recomputed (atomic abundances are negligible).

        # HF: [F/H] parameterisation (Picos+2024 Eq. 3).
        # [F/H] = log10(n_HF/n_H) - LOG_F_H_SOLAR
        # → log VMR_HF = [F/H] + LOG_F_H_SOLAR + log10(2 × VMR_H2)
        # VMR_H2 derived from the equilibrium mass fraction and MMW.
        if '1H-19F' in self.species and 'F_H' in self.params:
            _mean_MMW = float(np.mean(MMW))
            _vmr_h2   = float(np.mean(mf_eq['H2'])) * _mean_MMW / 2.016
            _log_vmr_hf = (float(self.params['F_H'])
                           + LOG_F_H_SOLAR
                           + np.log10(2.0 * max(_vmr_h2, 1e-10)))
            _vmr_hf = 10 ** _log_vmr_hf * np.ones(self.n_atm_layers)
            mass_frac['1H-19F'] = 20.006243 * _vmr_hf / _mean_MMW

        # Na, Ca, Ti: direct log VMR free parameters.
        _atomic_log = {
            '23Na': ('log_Na', 22.989769),
            '48Ti': ('log_Ti', 47.867),
            '40Ca': ('log_Ca', 40.078),
        }
        for prt_name, (param_key, mol_mass) in _atomic_log.items():
            if prt_name in self.species and param_key in self.params:
                VMR_i = 10 ** float(self.params[param_key]) * np.ones(self.n_atm_layers)
                mass_frac[prt_name] = mol_mass * VMR_i / np.mean(MMW)

        return mass_frac, CO, C_H

    # ---- EddySed cloud mass fractions (v3.0) ----

    def _add_cloud_mass_fractions(self):
        """Add EddySed condensate mass fractions for each cloud species.

        Parameterisation (Xuan+2024 §4.3.3, Ackerman & Marley 2001):
          log_X_<sp> — scaling factor relative to chemical-equilibrium abundance.
            log_X = 0  → equilibrium cloud abundance (reference)
            log_X < 0  → depleted; log_X > 0  → enhanced
          Condensate profile is a step function at the cloud base (T < T_cond(P)).
          The EddySed vertical profile and particle sizes are controlled by fsed,
          log_Kzz, and sigma_lnorm passed to calculate_flux().
        """
        mmw_mean    = float(np.mean(self.MMW))
        metallicity = float(self.C_H)    # [C/H] as proxy for [Fe/H]; same for free & equil chem
        CO_ratio    = float(self.CO)

        for cloud_sp in self.cloud_species:
            sp_base   = cloud_sp.split('(')[0]     # 'MgSiO3(s)_crystalline_000' → 'MgSiO3'
            log_X_key = f'log_X_{sp_base}'         # → 'log_X_MgSiO3'
            if log_X_key not in self.params:
                continue
            mf_eq  = return_cloud_mass_fraction(cloud_sp, metallicity, CO_ratio)
            x_tot  = float(mf_eq) * 10 ** float(self.params[log_X_key])
            x_tot  = max(x_tot, 0.0)
            mf_profile = simple_cdf_free(cloud_sp, self.pressure, self.temperature,
                                         metallicity, x_tot, mmw_mean)
            self.mass_fractions[cloud_sp] = mf_profile

    def _build_cloud_kwargs(self):
        """Build keyword arguments for pRT3 calculate_flux() EddySed cloud model.

        Parameters (Xuan+2024 §4.3.3 / Table 3):
          fsed        — sedimentation efficiency (global; pRT3 cloud_f_sed is scalar)
          log_Kzz     — log10(Kzz / [cm² s⁻¹]), broadcast to all layers
          sigma_lnorm — width of log-normal cloud particle size distribution
        Note: pRT3 calculate_flux() accepts cloud_f_sed as a single float; per-species
        fsed as in Xuan+2024 is not directly supported. A shared fsed is used here.
        """
        p   = self.params
        kzz = np.ones(self.n_atm_layers) * 10 ** float(p['log_Kzz'])
        return {
            'cloud_f_sed'                           : float(p['fsed']),
            'eddy_diffusion_coefficients'           : kzz,
            'cloud_particle_radius_distribution_std': float(p['sigma_lnorm']),
        }

    # ---- radiative transfer (cached) ----

    def _make_prt_flux(self):
        """Run pRT3 radiative transfer and cache the result.

        Returns
        -------
        wl   : 1-D array, nm (rest frame, native non-uniform pRT grid)
        flux : 1-D array, W m⁻² µm⁻¹  (converted from pRT3 native erg s⁻¹ cm⁻² cm⁻¹)
        """
        if self._prt_wl is not None:
            return self._prt_wl, self._prt_flux

        cloud_kwargs = self._build_cloud_kwargs() if self.use_clouds else {}

        wl, flux, _ = self.atmosphere_objects.calculate_flux(
            temperatures               = self.temperature,
            mass_fractions             = self.mass_fractions,
            reference_gravity          = self.gravity,
            mean_molar_masses          = self.MMW,
            return_contribution        = True,
            frequencies_to_wavelengths = True,
            **cloud_kwargs,
        )
        wl   *= 1e7   # cm → nm
        flux *= 1e-7  # erg s⁻¹ cm⁻² cm⁻¹ → W m⁻² µm⁻¹  (pRT3 output unit)

        if self.contribution:
            self.summed_contr = np.nansum(self.atmosphere_objects.contr_em, axis=1)

        self._prt_wl   = wl
        self._prt_flux = flux
        return wl, flux

    # ---- per-night spectrum generation ----

    def make_spectrum(self, data_wave=None, out_res=100_000, rv_key='rv_N1'):
        """Generate a model spectrum for one night.

        Steps
        -----
        1. Retrieve cached pRT flux (rest frame).
        2. Apply RV Doppler shift.
        3. Resample onto uniform grid (required by fastRotBroad).
        4. Rotational broadening (fastRotBroad, epsilon free param [0,1]).
        5. Instrumental LSF broadening (Gaussian, R_in → R_out).
        6. Interpolate onto the night's data wavelength grid.
        7. Continuum normalisation (median or per-chip savgol).

        Parameters
        ----------
        data_wave : 1-D array, nm
        out_res   : instrument resolving power R for this night
        rv_key    : 'rv_N1' or 'rv_N2' — selects the per-night RV param

        Returns
        -------
        flux : 1-D array on data_wave
        """
        if data_wave is None:
            data_wave = self.data_wave

        wl, flux = self._make_prt_flux()

        rv = self.params[rv_key]
        wl_shifted  = wl * (1.0 + rv / const.c.to('km/s').value)

        waves_even  = np.linspace(wl.min(), wl.max(), wl.size)
        flux_even   = np.interp(waves_even, wl_shifted, flux)

        flux_rot    = fastRotBroad(waves_even, flux_even,
                                   self.params['epsilon'], self.params['vsini'])

        in_res      = int(1e6 / self.lbl_opacity_sampling)
        flux_lsf    = self.instr_broadening(waves_even, flux_rot,
                                             out_res=out_res, in_res=in_res)

        flux_interp = np.interp(data_wave.flatten(), waves_even, flux_lsf)

        if self._use_absolute_flux:
            # v2.0: scale pRT surface flux (W/m²/μm) to Earth-observed flux.
            # F_obs = F_surface × (R / d)²
            R_m  = 10 ** self.params['log_R'] * R_JUP_M
            d_m  = D_DH_TAU_PC * PC_M
            flux_interp = flux_interp * (R_m / d_m) ** 2
            return flux_interp   # normalisation disabled in absolute flux mode

        flux_norm = self._apply_normalization(flux_interp, data_wave)
        return flux_norm

    def _apply_normalization(self, flux, data_wave):
        """Apply continuum normalisation to the model spectrum.

        Modes
        -----
        'median' : divide by global median
        'savgol' : per-chip Savitzky-Golay (window=301, polyorder=2)
        'per_chip_median' : divide each (order, detector) chip by its own median
        False/None : no normalisation
        """
        if self.normalize_flux == 'median':
            flux /= np.nanmedian(flux)

        elif self.normalize_flux == 'per_chip_median':
            for order in range(self.target.n_orders):
                for det in range(self.target.n_dets):
                    mask_od = (
                        (data_wave >= self.target.K2166[order, det, 0])
                        & (data_wave <= self.target.K2166[order, det, 1])
                    )
                    finite = mask_od & np.isfinite(flux)
                    if finite.sum() < 10:
                        continue
                    med = np.nanmedian(flux[finite])
                    if med > 0:
                        flux[finite] /= med

        elif self.normalize_flux == 'savgol':
            window_length = 301
            polyorder     = 2
            for order in range(self.target.n_orders):
                for det in range(self.target.n_dets):
                    mask_od = (
                        (data_wave >= self.target.K2166[order, det, 0])
                        & (data_wave <= self.target.K2166[order, det, 1])
                    )
                    finite  = mask_od & np.isfinite(flux)
                    n_fin   = finite.sum()
                    if n_fin <= polyorder:
                        continue
                    win = min(window_length, n_fin)
                    if win % 2 == 0:
                        win -= 1
                    if win <= polyorder:
                        continue
                    smooth = savgol_filter(flux[finite], win, polyorder)
                    scale  = np.nanmedian(np.abs(smooth))
                    if scale == 0:
                        flux[finite] = np.nan
                        continue
                    safe = np.isfinite(smooth) & (np.abs(smooth) > np.finfo(float).eps * scale)
                    idx  = np.where(finite)[0]
                    flux[idx[safe]]  /= smooth[safe]
                    flux[idx[~safe]]  = np.nan

        return flux

    def instr_broadening(self, wave, flux, out_res=100_000, in_res=333_333):
        """Gaussian LSF broadening from in_res to out_res.

        Requires out_res < in_res.
        """
        sigma_LSF = np.sqrt(1 / out_res**2 - 1 / in_res**2) / (2 * np.sqrt(2 * np.log(2)))
        spacing   = np.mean(2 * np.diff(wave) / (wave[1:] + wave[:-1]))
        sigma_px  = sigma_LSF / spacing
        return gaussian_filter(flux, sigma=sigma_px, mode='nearest')


# ============================================================
# 8. CLASS: Retrieval
# ============================================================
class Retrieval:
    """Orchestrates the full atmospheric retrieval pipeline.

    Parameters
    ----------
    parameters        : Parameters instance
    N_live_points     : int — MultiNest live points
    evidence_tolerance : float — MultiNest evidence tolerance (0.5 recommended)
    targets           : Target | list[Target]
                        Single Target (one night) or list of two Targets.
    testing           : bool — if True, restricts to one order/detector
    normalize_flux    : 'savgol' | 'median' | 'per_chip_median' | False | None
    ppid              : int | None — job ID (defaults to parent PID)
    per_chip_scaling  : bool | 'Single' | None
    instrument_res    : int | list[int] — R per night (default 100 000)
    cov_mode          : 'diag' | 'GP'
        'diag' (default) — diagonal covariance (per-pixel variances only).
        'GP'             — correlated noise via heteroscedastic GP kernel
                           (González Picos+2025 §4.2).  Requires log_a and
                           log_l in the free-parameter dictionary.
    """

    def __init__(self, parameters, N_live_points, evidence_tolerance, targets,
                 testing=True, normalize_flux=False, ppid=None,
                 per_chip_scaling=False, instrument_res=100_000,
                 use_absolute_flux=False, cov_mode='diag'):

        self.N_live_points     = int(N_live_points)
        self.evidence_tolerance = float(evidence_tolerance)

        self.targets      = targets if isinstance(targets, list) else [targets]
        self.target       = self.targets[0]
        self.two_night_mode = (len(self.targets) == 2)

        if isinstance(instrument_res, (int, float)):
            self.instrument_res = [int(instrument_res)] * len(self.targets)
        else:
            self.instrument_res = [int(r) for r in instrument_res]
        assert len(self.instrument_res) == len(self.targets), \
            'instrument_res must have one entry per target (night)'
        print(f'Instrument resolving power per night: {self.instrument_res}')

        self.mask_isfinite = self.target.get_mask_isfinite()
        self.K2166         = self.target.K2166
        self.parameters    = parameters
        self.species       = self.get_species(param_dict=parameters.params)

        # Load equilibrium chemistry table once (1.4 GB HDF5)
        chem_mode = parameters.params.get('chemistry', 'free')
        if chem_mode == 'equilibrium':
            print('Loading equilibrium chemistry table (done once)...')
            self.eq_chem = PreCalculatedEquilibriumChemistryTable()
            self.eq_chem.load()
            print('Equilibrium chemistry table loaded.')
        else:
            self.eq_chem = None

        self.testing           = testing
        self.normalize_flux    = normalize_flux
        self.per_chip_scaling  = per_chip_scaling
        self.use_absolute_flux = use_absolute_flux
        self.cov_mode          = cov_mode
        # v3.0: EddySed cloud model — read from constant_params via parameters.params
        self.cloud_species     = list(parameters.params.get('cloud_species', []))
        self.use_clouds        = bool(self.cloud_species)
        self.n_orders         = self.target.n_orders
        self.n_dets           = self.target.n_dets

        if ppid is None:
            ppid = os.getppid()
        self.job_id = ppid

        # --- data arrays ---
        if testing:
            self.order    = self.target.n_orders - 1
            self.detector = 1
            if self.target.wl.ndim == 3:
                self.target.wl  = self.target.wl[self.order, self.detector]
                self.target.fl  = self.target.fl[self.order, self.detector]
                self.target.err = self.target.err[self.order, self.detector]
                self.target.mask_isfinite = self.target.mask_isfinite[self.order, self.detector]
            self.data_wave      = self.target.wl
            self.data_flux      = self.target.fl
            self.data_err       = self.target.err
            self.mask_isfinite  = self.target.mask_isfinite
        else:
            self.data_wave      = self.targets[0].wl
            self.data_flux      = self.targets[0].fl
            self.data_err       = self.targets[0].err
            self.mask_isfinite  = self.targets[0].mask_isfinite
            if self.two_night_mode:
                t2 = self.targets[1]
                t2.get_mask_isfinite()
                self.data_wave2     = t2.wl
                self.data_flux2     = t2.fl
                self.data_err2      = t2.err
                self.mask_isfinite2 = t2.mask_isfinite

        self.n_params  = len(parameters.free_params)
        self.output_dir = (
            RESULTS_ROOT
            / f'{self.job_id}_N{self.N_live_points}_ev{self.evidence_tolerance}'
              f'_Norm{self.normalize_flux}_PerChipScale{self.per_chip_scaling}'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lbl_opacity_sampling = 3
        self.n_atm_layers         = 50
        # Pressure grid matched to DH Tau B PCHIP node range: 10⁻⁵–10¹ bar.
        # Deepest node: 10^+0.7 = 5 bar; shallowest: 10⁻⁵ bar (Xuan+2024 Table C1).
        # Grid extends slightly beyond deepest node to 10 bar for safety margin.
        self.pressure             = np.logspace(-5, 1, self.n_atm_layers)

        # --- covariance + likelihood (night 1) ---
        if self.cov_mode == 'GP':
            self.Cov = CovarianceGP(
                err=self.data_err[self.mask_isfinite],
                wl=self.data_wave[self.mask_isfinite],
            )
            # Precompute sub-chip → K2166 mapping so solve_for_wl_range avoids
            # per-call boolean array creation across all ~18 sub-chips.
            self.Cov.precompute_K2166_ranges(self.K2166)
        else:
            self.Cov = Covariance(err=self.data_err[self.mask_isfinite])
        self.LogLike = LogLikelihood(
            d_flux=self.data_flux, d_mask=self.mask_isfinite,
            d_wave=self.data_wave, d_err=self.data_err,
            scale_flux=self.per_chip_scaling, N_params=self.n_params,
            retrieval_object=self,
        )

        # --- covariance + likelihood (night 2, optional) ---
        if self.two_night_mode:
            if self.cov_mode == 'GP':
                self.Cov2 = CovarianceGP(
                    err=self.data_err2[self.mask_isfinite2],
                    wl=self.data_wave2[self.mask_isfinite2],
                )
                self.Cov2.precompute_K2166_ranges(self.K2166)
            else:
                self.Cov2 = Covariance(err=self.data_err2[self.mask_isfinite2])
            self.LogLike2 = LogLikelihood(
                d_flux=self.data_flux2, d_mask=self.mask_isfinite2,
                d_wave=self.data_wave2, d_err=self.data_err2,
                scale_flux=self.per_chip_scaling, N_params=self.n_params,
                retrieval_object=self,
            )

        self.atmosphere_objects = self.get_atmosphere_objects()
        self.callback_label     = 'live_'
        self.prefix             = 'pmn_'
        self.color              = self.target.color

    # ---- species list ----

    def get_species(self, param_dict):
        """Return list of pRT3 opacity species names.

        Equilibrium chemistry: uses the fixed EQ_SPECIES_PRT3 list.
        Free chemistry: scans for log_* parameters.
        """
        species_info = _get_species_info()

        if param_dict.get('chemistry', 'free') == 'equilibrium':
            self.chem_species = [f'eq_{s}' for s in EQ_LABEL_NAMES]
            return list(EQ_SPECIES_PRT3)

        # Free chemistry (v3.0: exclude cloud fraction and structure params)
        self.chem_species = [
            p for p in param_dict
            if p.startswith('log_')
            and not p.startswith('log_X_')
            and p not in {'log_g', 'log_Pquench', 'log_12CO_13CO', 'log_Kzz'}
        ]
        species = []
        for cs in self.chem_species:
            sp_name = cs[4:]
            if sp_name in species_info.index:
                prt_name = species_info.loc[sp_name, 'pRT_name']
                species.append(PRT3_SPECIES_OVERRIDES.get(prt_name, prt_name))
        return species

    # ---- pRT atmosphere object ----

    def get_atmosphere_objects(self, redo=True):
        """Create or load a cached Radtrans atmosphere object."""
        # v3.0: separate cache for cloudy runs to avoid loading clear-atmosphere object
        cloud_suffix = ('_cloud_' + '_'.join(sorted(self.cloud_species))) if self.cloud_species else ''
        cache_file = RESULTS_ROOT / f'atmosphere_objects_{self.job_id}_N{self.N_live_points}{cloud_suffix}.pickle'
        if cache_file.exists() and not redo:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        print('Creating new atmosphere object...')
        wl_pad = 7   # nm padding before RV shift
        if self.testing:
            wlmin = self.K2166[self.order].min() - wl_pad
            wlmax = self.K2166[self.order].max() + wl_pad
        else:
            all_waves = [self.data_wave]
            if self.two_night_mode:
                all_waves.append(self.data_wave2)
            wlmin = min(w.min() for w in all_waves) - wl_pad
            wlmax = max(w.max() for w in all_waves) + wl_pad

        boundary = np.array([wlmin, wlmax]) * 1e-7 * 1e4   # nm → µm

        atm = Radtrans(
            line_species                  = self.species,
            cloud_species                 = self.cloud_species,   # v3.0: EddySed clouds ([] for clear)
            rayleigh_species              = ['H2', 'He'],
            gas_continuum_contributors    = ['H2--H2', 'H2--He'],
            wavelength_boundaries         = boundary,
            line_opacity_mode             = 'lbl',
            line_by_line_opacity_sampling = self.lbl_opacity_sampling,
            pressures                     = self.pressure,
        )
        with open(cache_file, 'wb') as f:
            pickle.dump(atm, f)
        return atm

    # ---- MultiNest interface ----

    def PMN_lnL(self, cube=None, ndim=None, nparams=None):
        """MultiNest log-likelihood function.

        Creates one pRT_spectrum per call; the expensive pRT RT step is
        cached inside the instance via _make_prt_flux(), so two-night
        retrievals run pRT only once per likelihood evaluation.
        """
        self.model_object = pRT_spectrum(self, use_absolute_flux=self.use_absolute_flux)

        flux1  = self.model_object.make_spectrum(
            data_wave=self.data_wave,
            out_res=self.instrument_res[0],
            rv_key='rv_N1',
        )
        if self.cov_mode == 'GP':
            p = self.parameters.params
            a = 10 ** float(p['log_a'])
            l = 10 ** float(p['log_l'])
            try:
                self.Cov.build(a, l)
            except np.linalg.LinAlgError:
                return -1e101
        try:
            ln_L = self.LogLike(flux1, self.Cov)

            if self.two_night_mode:
                flux2 = self.model_object.make_spectrum(
                    data_wave=self.data_wave2,
                    out_res=self.instrument_res[1],
                    rv_key='rv_N2',
                )
                if self.cov_mode == 'GP':
                    try:
                        self.Cov2.build(a, l)
                    except np.linalg.LinAlgError:
                        return -1e101
                ln_L += self.LogLike2(flux2, self.Cov2)

        except Exception:
            return -1e101

        if not np.isfinite(ln_L):
            return -1e101
        # Cap to prevent Fortran underflow in MultiNest importance weights:
        # with per-chip phi and Ruffio formula, chi2~0 can push ln_L unrealistically
        # large, causing exp(ln_L_i - ln_L_max) to underflow (~1e-309) for all
        # other live points.
        # NOTE: 1e6 was too low for absolute flux mode (N~50k pixels, σ~1.5e-16
        # W/m²/μm), where the log-normalization constant alone is ~1.77e6 — the cap
        # was hit for every sample, producing a flat landscape and instant convergence.
        # 1e10 is safely above any realistic ln_L while still capping runaway values.
        return float(min(ln_L, 1e10))

    def PMN_run(self, N_live_points=None, evidence_tolerance=0.5,
                resume=False, mpi=True):
        """Launch MultiNest sampler."""
        pymultinest.run(
            LogLikelihood         = self.PMN_lnL,
            Prior                 = self.parameters,
            n_dims                = self.parameters.n_params,
            outputfiles_basename  = f'{self.output_dir}/{self.prefix}',
            verbose               = True,
            const_efficiency_mode = True,
            sampling_efficiency   = 0.5,
            n_live_points         = N_live_points,
            resume                = resume,
            evidence_tolerance    = evidence_tolerance,
            dump_callback         = self.PMN_callback,
            n_iter_before_update  = 10,
            use_MPI               = mpi,
        )

    def PMN_callback(self, n_samples, n_live, n_params, live_points,
                     posterior, stats, max_ln_L, ln_Z, ln_Z_err, nullcontext):
        """Live callback: update corner plot during sampling."""
        print(f'[callback] n={n_samples}  max_lnL={max_ln_L:.2f}  '
              f'lnZ={ln_Z:.2f} ± {ln_Z_err:.2f}')
        self.bestfit_params = posterior[np.argmax(posterior[:, -2]), :-2]
        self.posterior      = posterior[:, :-2]
        self.params_dict, self.model_flux, self.model_flux_scaled, self.data_err_scaled = \
            self.get_params_and_spectrum()
        self.cornerplot()

    def PMN_analyse(self):
        """Load MultiNest output and extract equal-weighted posterior."""
        analyzer    = pymultinest.Analyzer(
            n_params             = self.parameters.n_params,
            outputfiles_basename = f'{self.output_dir}/{self.prefix}',
        )
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior()[:, :-1]
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy', self.posterior)
        self.lnZ = stats['nested importance sampling global log-evidence']

    def get_params_and_spectrum(self):
        """Compute posterior-median parameters and best-fit spectrum.

        Returns
        -------
        params_dict      : dict of parameter medians + derived quantities
        model_flux       : 1-D model flux on night-1 data grid
        model_flux_scaled : model flux after applying best-fit φ scaling
        data_err_scaled  : data error array scaled by s2
        """
        self.params_dict = {}
        for i, key in enumerate(self.parameters.param_keys):
            medians = np.percentile(self.posterior[:, i], 50.0)
            self.params_dict[key] = medians

        # Sync median values into parameters.params so that pRT_spectrum.__init__
        # (which reads parameters.params directly and calls make_pt() immediately)
        # has all free-parameter keys populated.  Without this, post-processing on
        # a freshly-constructed Retrieval object fails with KeyError on T_anchor
        # because Parameters.__call__ was never invoked by MultiNest.
        self.parameters.params.update(self.params_dict)

        self.model_object = pRT_spectrum(self, use_absolute_flux=self.use_absolute_flux)
        self.model_flux   = self.model_object.make_spectrum(
            data_wave=self.data_wave,
            out_res=self.instrument_res[0],
            rv_key='rv_N1',
        )
        self.params_dict['[C/H]']        = self.model_object.C_H
        self.params_dict['[C/H]_xsolar'] = 10 ** self.model_object.C_H
        self.params_dict['C/O']          = self.model_object.CO

        if self.two_night_mode:
            self.model_flux2 = self.model_object.make_spectrum(
                data_wave=self.data_wave2,
                out_res=self.instrument_res[1],
                rv_key='rv_N2',
            )

        # Log-likelihood at median params (for chi2, phi, s2)
        if self.cov_mode == 'GP':
            a = 10 ** float(self.params_dict['log_a'])
            l = 10 ** float(self.params_dict['log_l'])
            self.Cov.build(a, l)
        self.log_likelihood = self.LogLike(self.model_flux, self.Cov)
        self.params_dict['phi']  = self.LogLike.phi
        self.params_dict['s2']   = self.LogLike.s2
        self.params_dict['chi2'] = self.LogLike.chi2_0_red
        if self.two_night_mode:
            if self.cov_mode == 'GP':
                self.Cov2.build(a, l)
            ll2 = self.LogLike2(self.model_flux2, self.Cov2)
            self.log_likelihood       += ll2
            self.params_dict['phi_N2']  = self.LogLike2.phi
            self.params_dict['chi2_N2'] = self.LogLike2.chi2_0_red
        if self.callback_label == 'final_':
            self.params_dict['lnZ'] = self.lnZ

        with open(f'{self.output_dir}/{self.callback_label}params_dict.pickle', 'wb') as f:
            pickle.dump(self.params_dict, f)

        # Scaled model flux (night 1)
        if self.LogLike.scale_flux is True:
            self.model_flux_scaled = np.full_like(self.model_flux, np.nan)
            for order in range(self.n_orders):
                for det in range(self.n_dets):
                    mask_od = (
                        (self.data_wave >= self.K2166[order, det, 0])
                        & (self.data_wave <= self.K2166[order, det, 1])
                    )
                    self.model_flux_scaled[mask_od] = (
                        self.model_flux[mask_od]
                        * self.LogLike.phi[order * self.n_dets + det]
                    )
        elif self.LogLike.scale_flux in (None, 'Single'):
            self.model_flux_scaled = self.model_flux * self.LogLike.phi
        else:
            self.model_flux_scaled = self.model_flux

        self.data_err_scaled = self.data_err * self.LogLike.s2

        return self.params_dict, self.model_flux, self.model_flux_scaled, self.data_err_scaled

    def evaluate(self):
        """Post-processing: analyse final posterior, generate corner plot."""
        self.callback_label = 'final_'
        self.PMN_analyse()
        self.params_dict, self.model_flux, self.model_flux_scaled, self.data_err_scaled = \
            self.get_params_and_spectrum()
        self.cornerplot()

    def run_retrieval(self):
        """Full retrieval run: MultiNest + post-processing."""
        print(f'\n------ Nlive: {self.N_live_points}  ev: {self.evidence_tolerance} ------\n')
        self.PMN_run(N_live_points=self.N_live_points,
                     evidence_tolerance=self.evidence_tolerance)
        self.evaluate()
        print('\n----------------- Done -----------------\n')

    def cornerplot(self):
        """Save corner plot of the current posterior to the output directory."""
        labels   = list(self.parameters.param_mathtext.values())
        fontsize = 10
        fig = plt.figure(figsize=(self.n_params, self.n_params), dpi=200)
        corner.corner(
            self.posterior,
            labels         = labels,
            title_kwargs   = {'fontsize': fontsize},
            label_kwargs   = {'fontsize': fontsize},
            color          = self.color,
            linewidths     = 0.5,
            fill_contours  = True,
            quantiles      = [0.16, 0.5, 0.84],
            title_quantiles = [0.16, 0.5, 0.84],
            show_titles    = True,
            fig            = fig,
            quiet          = True,
        )
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(
            f'{self.output_dir}/{self.callback_label}cornerplot.pdf',
            bbox_inches='tight', dpi=200,
        )
        plt.close()

    # ---- diagnostic plots ----

    def save_diagnostic_plots(self):
        """Save residuals, model-vs-data, P-T, and smoothed comparison plots."""
        output_dir = self.output_dir
        K2166      = self.target.K2166
        mask       = np.isfinite(self.data_flux)

        params_dict, model_flux, model_flux_scaled, _ = self.get_params_and_spectrum()

        # Plot 1: residuals
        fig, ax = plt.subplots(3, 5, figsize=(20, 10))
        for order in range(5):
            for det in range(3):
                wb  = K2166[4 - order][det]
                sel = mask & (self.data_wave >= wb.min()) & (self.data_wave <= wb.max())
                res = self.data_flux[sel] - model_flux_scaled[sel]
                std = np.std(res)
                ax[det][order].scatter(self.data_wave[sel], res, s=2, color='royalblue', alpha=0.6)
                ax[det][order].axhline(0, color='k', lw=0.8, ls='--')
                ax[det][order].axhline( 3 * std, color='r', lw=0.8, ls=':')
                ax[det][order].axhline(-3 * std, color='r', lw=0.8, ls=':')
                ax[det][order].set_title(f'Ord {27 - order} Det {det + 1}', fontsize=7)
        plt.tight_layout()
        fig.savefig(output_dir / 'retrieval_data_model_residuals.png', dpi=200)
        plt.close()

        # Plot 2: data vs model
        fig, ax = plt.subplots(3, 5, figsize=(20, 10))
        for order in range(5):
            for det in range(3):
                wb  = K2166[4 - order][det]
                sel = mask & (self.data_wave >= wb.min()) & (self.data_wave <= wb.max())
                ax[det][order].plot(self.data_wave[sel], self.data_flux[sel],
                                    color='darkgray', lw=0.6, alpha=0.7, label='data')
                ax[det][order].plot(self.data_wave[sel], model_flux_scaled[sel],
                                    color='red', lw=1.5, alpha=0.8, label='model')
                ax[det][order].set_ylim(
                    np.nanmin(self.data_flux[sel]) * 0.8,
                    np.nanmedian(self.data_flux[sel]) * 3,
                )
                ax[det][order].set_title(f'Ord {27 - order} Det {det + 1}', fontsize=7)
                ax[det][order].legend(fontsize=5)
        plt.tight_layout()
        fig.savefig(output_dir / 'retrieval_data_model_spectrum.png', dpi=200)
        plt.close()

        # Plot 3: P-T profile
        fig, ax = plt.subplots(figsize=(5, 7))
        ax.plot(self.model_object.temperature, self.model_object.pressure, 'k-')
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel('Pressure (bar)')
        ax.set_title('P-T profile (Piette+2020 PCHIP)')
        fig.savefig(output_dir / 'retrieval_PT_profile.png', dpi=200, bbox_inches='tight')
        plt.close()

        # Plot 4: smoothed comparison
        fig, ax = plt.subplots(3, 5, figsize=(20, 10))
        data_bin  = savgol_filter(self.data_flux[mask], 51, 2)
        model_bin = savgol_filter(model_flux_scaled[mask], 51, 2)
        for det in range(3):
            for order in range(5):
                wb   = K2166[4 - order][det]
                sel2 = (self.data_wave[mask] >= wb.min()) & (self.data_wave[mask] <= wb.max())
                d_m  = np.nanmean(data_bin[sel2])
                m_m  = np.nanmean(model_bin[sel2])
                ax[det][order].scatter(self.data_wave[mask][sel2],
                                       data_bin[sel2] / (d_m or 1),
                                       s=3, color='darkgray', label='data (smooth)')
                ax[det][order].scatter(self.data_wave[mask][sel2],
                                       model_bin[sel2] / (m_m or 1),
                                       s=3, color='red', label='model (smooth)')
                ax[det][order].set_title(f'Ord {27 - order} Det {det + 1}', fontsize=7)
                ax[det][order].legend(fontsize=5)
        plt.tight_layout()
        fig.savefig(output_dir / 'retrieval_data_model_spectrum_binned.png', dpi=200)
        plt.close()

        print(f'Diagnostic plots saved to {output_dir}')


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def estimate_spectral_resolution(wave_3d):
    """Estimate median instrument resolving power R = λ/Δλ from wavelength solution.

    Parameters
    ----------
    wave_3d : ndarray, shape (n_orders, n_dets, n_pixels), nm

    Returns
    -------
    R_median : float
    """
    R_chips = []
    for order in range(wave_3d.shape[0]):
        for det in range(wave_3d.shape[1]):
            w      = wave_3d[order, det]
            finite = np.isfinite(w)
            if finite.sum() < 2:
                continue
            w_v    = w[finite]
            dw     = np.diff(w_v)
            wc     = 0.5 * (w_v[1:] + w_v[:-1])
            R_chips.append(float(np.nanmedian(wc / np.abs(dw))))
    R_median = float(np.median(R_chips))
    print(f'  Estimated R = {R_median:.0f} (median over {len(R_chips)} chips, '
          f'range {min(R_chips):.0f}–{max(R_chips):.0f})')
    return R_median


def _load_night(night, flux_file, err_file, normalize_method, n_orders=5, n_dets=3):
    """Load, normalize, flatten, and mask one night's combined spectrum.

    Parameters
    ----------
    night            : str, e.g. '2022-12-31'
    flux_file        : filename in /data2/peng/{night}/ (no path prefix)
    err_file         : corresponding error filename
    normalize_method : 'savgol' | 'median' | 'per_chip_median' | None
    n_orders, n_dets : spectral format (K2166 default: 5 × 3)

    Returns
    -------
    wave, flux, err : 1-D arrays of valid (non-NaN/inf) pixels
    R_est           : float — estimated instrument resolving power
    """
    spec = np.load(f'/data2/peng/{night}/{flux_file}')   # (3, 5, 2048)
    err  = np.load(f'/data2/peng/{night}/{err_file}')

    wave_file = f'{WORKPATH}{night}/cal/WLEN_K2166_V_DH_Tau_A+B_center.fits'
    wave_hdu  = fits.open(wave_file)
    wave      = np.array(wave_hdu[1].data)[:, :n_orders, :]  # (3, 5, 2048)

    # (dets, orders, pix) → (orders, dets, pix)
    spec = np.transpose(spec, (1, 0, 2))
    err  = np.transpose(err,  (1, 0, 2))
    wave = np.transpose(wave, (1, 0, 2))

    window_length = 301
    polyorder     = 2

    if normalize_method == 'median':
        med  = np.nanmedian(spec)
        spec /= med
        err  /= med
        print(f'  [{night}] Global median normalisation (median = {med:.4e})')

    elif normalize_method == 'savgol':
        for order in range(n_orders):
            for det in range(n_dets):
                fin = np.isfinite(spec[order, det])
                low = np.full(spec[order, det].shape, np.nan)
                low[fin] = savgol_filter(spec[order, det][fin], window_length, polyorder)
                scale = np.nanmedian(np.abs(low[fin]))
                if scale == 0:
                    spec[order, det][fin] = np.nan
                    err[order,  det][fin] = np.nan
                    continue
                floor = np.finfo(float).eps * scale
                safe  = fin & np.isfinite(low) & (np.abs(low) > floor)
                spec[order, det][safe]           /= low[safe]
                err[ order, det][safe]           /= low[safe]
                spec[order, det][fin & ~safe]     = np.nan
                err[ order, det][fin & ~safe]     = np.nan
        print(f'  [{night}] Per-chip Savitzky-Golay normalisation applied.')

    elif normalize_method == 'per_chip_median':
        for order in range(n_orders):
            for det in range(n_dets):
                chip = spec[order, det]
                fin  = np.isfinite(chip)
                if fin.sum() < 10:
                    continue
                med = np.nanmedian(chip[fin])
                if med > 0:
                    spec[order, det] /= med
                    err[order, det]  /= med
        print(f'  [{night}] Per-chip median normalisation applied.')

    else:
        print(f'  [{night}] No normalisation applied.')

    print(f'  [{night}] Estimating resolving power...')
    R_est = estimate_spectral_resolution(wave)

    wave_flat = wave.reshape(-1)
    spec_flat = np.where(spec.reshape(-1) == 0, np.nan, spec.reshape(-1))
    err_flat  = np.where(err.reshape(-1)  == 0, np.nan, err.reshape(-1))
    valid     = np.isfinite(spec_flat) & np.isfinite(err_flat)

    print(f'  [{night}] Valid pixels: {valid.sum()} / {valid.size}')
    return wave_flat[valid], spec_flat[valid], err_flat[valid], R_est


# ============================================================
# 10. DEFAULT PARAMETER SETS
# ============================================================

def make_free_params_equilibrium():
    """Default free parameter dictionary for equilibrium chemistry (004PM-EQ).

    P-T parameterisation: Piette+2020 / Xuan+2024 — T_anchor at 0.2 bar
    (log P = −0.7) plus 7 ΔT increments.  Priors taken directly from
    Xuan+2024 Table C1 (DH Tau b column).

    Node layout (log₁₀ bar, deep → shallow):
      [+0.7, 0.0, -0.3, -0.7*, -1.0, -1.5, -3.0, -5.0]   * = anchor

    v1.1 change: log_g replaced by log_M + log_R (Gaussian priors from
    Xuan+2024 Table 1 for DH Tau b: M = 12 ± 4 M_Jup, R = 2.6 ± 0.6 R_Jup).
    log_g is derived as g = G*M/R² (cgs) — not a free parameter.

    DH Tau B properties (Xuan+2024 Table 1): M = 12±4 MJup, R = 2.6±0.6 RJup.
    Expected log_g ≈ 3.64 (vs. our v1.0 result 4.25 — 6σ too high).

    References
    ----------
    Xuan et al. (2024). ApJ 970:71. §4.3.4, §4.3.5, Table 1, Table C1.
    Piette & Madhusudhan (2020). MNRAS 497, Table 1.
    """
    return {
        'rv_N1':    ({'type': 'gaussian', 'mu': 31.5, 'sigma': 1},  r'$v_{\rm rad,\,N1}$'),
        'rv_N2':    ({'type': 'gaussian', 'mu': 31.5, 'sigma': 1},  r'$v_{\rm rad,\,N2}$'),
        #'rv_N1':     ([-50,50],                                         r'$v_{\rm rad,\,N1}$'),
        #'rv_N2':     ([-50,50],                                         r'$v_{\rm rad,\,N2}$'),
        'vsini':    ([1, 20],                                           r'$v\sin i$'),
        'epsilon':  ([0.0, 1.0],                                        r'$\epsilon_{\rm LD}$'),
        # v2.0: log_M Gaussian from evolutionary models; log_R uniform — data constrains R
        # σ_log_M = 0.434 × σ_M/M = 0.434 × 4/12 ≈ 0.145;  log10(12) = 1.079
        # log_R uniform: data constrains R from absolute flux (R/d)² scaling
        # log_M: Gaussian from Xuan+2024 Table 1 / Figure A1 — M = 12 ± 4 MJup.
        #   σ_log_M = 0.434 × σ_M/M = 0.434 × 4/12 ≈ 0.145;  log10(12) = 1.079.
        # log_R: Gaussian from Xuan+2024 Table 1 / Figure A1 — R = 2.6 ± 0.6 RJup.
        #   σ_log_R = 0.434 × σ_R/R = 0.434 × 0.6/2.6 ≈ 0.100;  log10(2.6) = 0.415.
        'log_M':    ({'type': 'gaussian', 'mu': 1.079, 'sigma': 0.145}, r'$\log M/M_{\rm Jup}$'),
        'log_R':    ({'type': 'gaussian', 'mu': 0.415, 'sigma': 0.100}, r'$\log R/R_{\rm Jup}$'),
        # ---- P-T: Xuan+2024 Table C1 priors for DH Tau b ----
        # T_anchor at log P = -0.7 (0.2 bar); ECF peak for DH Tau B ≈ 0.1 bar.
        'T_anchor': ([1500, 3500], r'$T_{\rm anchor}$'),   # at 0.2 bar; Table C1: U(1800,3000)
        # dT_i = T(node_i) − T(node_{i+1}), deep → shallow
        # Priors match Xuan+2024 Table C1 (DH Tau b) exactly, including
        # NON-ZERO lower bounds that prevent near-isothermal pile-up at the prior floor.
        # Retrieval 2898115 showed dT_3 and dT_6 piling at 0 (32 and 49 K/dex)
        # because lb=0 allowed isothermal solutions in those intervals.
        'dT_1':     ([200, 1000], r'$\Delta T_1$'),  # +0.7 →  0.0 bar (0.7 dex); Table C1: U(200,900)
        'dT_2':     ([  0,  600], r'$\Delta T_2$'),  #  0.0 → -0.3 bar (0.3 dex); Table C1: U(0,450)
        'dT_3':     ([  0,  500], r'$\Delta T_3$'),  # -0.3 → -0.7 bar (0.4 dex); Table C1: U(50,550)
        'dT_4':     ([  0,  500], r'$\Delta T_4$'),  # -0.7 → -1.0 bar (0.3 dex); Table C1: U(0,450)
        'dT_5':     ([  0,  700], r'$\Delta T_5$'),  # -1.0 → -1.5 bar (0.5 dex); Table C1: U(50,650)
        'dT_6':     ([ 50,  800], r'$\Delta T_6$'),  # -1.5 → -3.0 bar (1.5 dex); Table C1: U(200,900)
        'dT_7':     ([ 50,  800], r'$\Delta T_7$'),  # -3.0 → -5.0 bar (2.0 dex); Table C1: U(100,700)
        'C_H':           ([-1.5, 1.5], r'$[{\rm C/H}]$'),
        'C/O':           ([0.1,  1.5], r'C/O'),
        'log_12CO_13CO': ([0.5,  3.0], r'$\log\,^{12}{\rm CO}/^{13}{\rm CO}$'),
        # v4.0 (equil. chem.): atomic species + HF (Picos+2024 §4.2, Table 3)
        # HF uses [F/H] parameterisation: [F/H] = log10(n_HF/n_H) - log10(n_HF/n_H)_solar
        # Solar reference: log10(n_F/n_H)_solar = -7.6 (Maiorca+2014; Asplund+2021).
        # Na, Ca, Ti remain as direct log VMR free parameters.
        'F_H':           ([-3, 3],   r'$[{\rm F/H}]$'),
        #'log_Na':        ([-10, -2], r'$\log$ Na'),
        #'log_Ca':        ([-10, -2], r'$\log$ Ca'),
        #'log_Ti':        ([-10, -2], r'$\log$ Ti'),
    }


def make_free_params_free_chem():
    """Default free parameter dictionary for free chemistry (003PM-FR).

    Identical P-T prior setup as make_free_params_equilibrium() — see that
    function's docstring for the full physical justification.
    Free-chemistry VMR params (log_H2O, log_12CO, log_13CO, log_CH4) unchanged.

    References
    ----------
    Xuan et al. (2024). ApJ 970:71. §4.3.4, Table 3, Figure E1.
    Piette & Madhusudhan (2020). MNRAS 497, Table 1.
    """
    return {
        'rv_N1':    ({'type': 'gaussian', 'mu': 31.5, 'sigma': 1},  r'$v_{\rm rad,\,N1}$'),
        'rv_N2':    ({'type': 'gaussian', 'mu': 31.5, 'sigma': 1},  r'$v_{\rm rad,\,N2}$'),
        'vsini':    ([0, 20],                                           r'$v\sin i$'),
        'epsilon':  ([0.0, 1.0],                                        r'$\epsilon_{\rm LD}$'),
        # log_M / log_R: Gaussian from Xuan+2024 Table 1 / Figure A1 (same as equilibrium mode).
        #   M = 12 ± 4 MJup  → log_M: N(1.079, 0.145)
        #   R = 2.6 ± 0.6 RJup → log_R: N(0.415, 0.100)
        'log_M':    ({'type': 'gaussian', 'mu': 1.079, 'sigma': 0.145}, r'$\log M/M_{\rm Jup}$'),
        'log_R':    ({'type': 'gaussian', 'mu': 0.415, 'sigma': 0.100}, r'$\log R/R_{\rm Jup}$'),
        # ---- P-T: same Xuan+2024 Table C1 priors as equilibrium mode ----
        'T_anchor': ([1500, 3500], r'$T_{\rm anchor}$'),   # at 0.2 bar; Table C1: U(1800,3000)
        'dT_1':     ([200, 1000], r'$\Delta T_1$'),  # +0.7 →  0.0 bar (0.7 dex); Table C1: U(200,900)
        'dT_2':     ([  0,  600], r'$\Delta T_2$'),  #  0.0 → -0.3 bar (0.3 dex); Table C1: U(0,450)
        'dT_3':     ([ 10,  500], r'$\Delta T_3$'),  # -0.3 → -0.7 bar (0.4 dex); Table C1: U(50,550)
        'dT_4':     ([  0,  500], r'$\Delta T_4$'),  # -0.7 → -1.0 bar (0.3 dex); Table C1: U(0,450)
        'dT_5':     ([ 50,  700], r'$\Delta T_5$'),  # -1.0 → -1.5 bar (0.5 dex); Table C1: U(50,650)
        'dT_6':     ([ 50,  800], r'$\Delta T_6$'),  # -1.5 → -3.0 bar (1.5 dex); Table C1: U(200,900)
        'dT_7':     ([ 50,  800], r'$\Delta T_7$'),  # -3.0 → -5.0 bar (2.0 dex); Table C1: U(100,700)
        'log_H2O':  ([-12, -2], r'$\log$ H$_2$O'),
        'log_12CO': ([-12, -2], r'$\log\,^{12}$CO'),
        'log_13CO': ([-12, -2], r'$\log\,^{13}$CO'),
        'log_CH4':  ([-12, -2], r'$\log$ CH$_4$'),
        # v4.0: atomic species + HF (Picos+2024 §4.2, Table 3; prior U(-12, -2))
        'log_HF':   ([-10, -2], r'$\log$ HF'),
        #'log_Na':   ([-10, -2], r'$\log$ Na'),
        #'log_Ca':   ([-10, -2], r'$\log$ Ca'),
        #'log_Ti':   ([-12, -2], r'$\log$ Ti'),
    }


def make_free_params_free_chem_cloudy(cloud_species=None):
    """Free-chemistry free params + EddySed cloud parameters (Xuan+2024 §4.3.3).

    Cloud species defaults to CLOUD_SPECIES_DEFAULT (MgSiO3 + Fe).
    Prior ranges from Xuan+2024 Table 3.

    Per-species cloud parameter (prior range U(-2.3, 1)):
      log_X_<sp>  — cloud mass fraction scaling relative to equilibrium
    Shared cloud structure parameters:
      fsed        — sedimentation efficiency, U(0, 10)
      log_Kzz     — log10(Kzz/cm²s⁻¹), U(5, 13)
      sigma_lnorm — log-normal particle size distribution width, U(1.05, 3)
    """
    if cloud_species is None:
        cloud_species = CLOUD_SPECIES_DEFAULT
    fp = make_free_params_free_chem()
    for cs in cloud_species:
        sp_base = cs.split('(')[0]
        fp[f'log_X_{sp_base}'] = ([-2.3, 1.0], rf'$\log X_{{\rm {sp_base}}}$')
    fp['fsed']        = ([0.0, 10.0],  r'$f_{\rm sed}$')
    fp['log_Kzz']     = ([5.0, 13.0],  r'$\log K_{zz}$')
    fp['sigma_lnorm'] = ([1.05,  3.0], r'$\sigma_{\rm g}$')
    return fp


def make_free_params_equil_chem_cloudy(cloud_species=None):
    """Equilibrium-chemistry free params + EddySed cloud parameters (Xuan+2024 §4.3.3).

    Same cloud priors as make_free_params_free_chem_cloudy().
    """
    if cloud_species is None:
        cloud_species = CLOUD_SPECIES_DEFAULT
    fp = make_free_params_equilibrium()
    for cs in cloud_species:
        sp_base = cs.split('(')[0]
        fp[f'log_X_{sp_base}'] = ([-2.3, 1.0], rf'$\log X_{{\rm {sp_base}}}$')
    fp['fsed']        = ([0.0, 10.0],  r'$f_{\rm sed}$')
    fp['log_Kzz']     = ([5.0, 13.0],  r'$\log K_{zz}$')
    fp['sigma_lnorm'] = ([1.05,  3.0], r'$\sigma_{\rm g}$')
    return fp


# ============================================================
# 11. MAIN PIPELINE ENTRY POINT
# ============================================================

def run_retrieval_pipeline(
    chemistry='equilibrium',
    normalize_method='savgol',
    per_chip_scaling=False,
    N_live_points=700,
    evidence_tol=0.5,
    nights=None,
    free_params=None,
    constant_params=None,
):
    """Run the full DH Tau B retrieval pipeline (003PM — Piette+2020 P-T).

    Parameters
    ----------
    chemistry : 'equilibrium' | 'free'
        Chemistry mode.  Selects default free_params if not provided.
    normalize_method : 'savgol' | 'median' | None
        Continuum normalisation applied to both data and model.
    per_chip_scaling : bool | 'Single'
        Flux scaling mode passed to LogLikelihood.
    N_live_points : int
        MultiNest live points (≥ 500 for production; 200 for quick tests).
    evidence_tol : float
        MultiNest evidence tolerance (0.5 recommended).
    nights : dict | None
        Night specification:
        {
          'N1': {'night': '2022-12-31',
                 'flux_file': 'extracted_spectra_combined_sigmaclipper.npy',
                 'err_file':  'extracted_spectra_combined_err_sigmaclipper.npy'},
          'N2': { ... },   # optional; omit for single-night retrieval
        }
        Defaults to the standard DH Tau B two-night configuration if None.
    free_params : dict | None
        Override the default free parameter dictionary.
    constant_params : dict | None
        Override the default constant parameters.

    Returns
    -------
    retrieval : Retrieval
        The completed Retrieval object (contains posterior, model spectra, etc.)
    """
    # -- defaults --
    if nights is None:
        nights = {
            'N1': {
                'night':     '2022-12-31',
                'flux_file': 'extracted_spectra_combined_sigmaclipper.npy',
                'err_file':  'extracted_spectra_combined_err_sigmaclipper.npy',
            },
            'N2': {
                'night':     '2023-01-01',
                'flux_file': 'extracted_spectra_combined_sigmaclipper.npy',
                'err_file':  'extracted_spectra_combined_err_sigmaclipper.npy',
            },
        }

    if chemistry == 'equilibrium':
        default_free    = make_free_params_equilibrium()
        default_const   = {'chemistry': 'equilibrium'}
    else:
        default_free    = make_free_params_free_chem()
        default_const   = {}

    if free_params is None:
        free_params = default_free
    if constant_params is None:
        constant_params = default_const

    # -- load data --
    night_keys = sorted(nights.keys())   # 'N1' first, then 'N2' if present
    loaded = {}
    for key in night_keys:
        cfg = nights[key]
        wave, flux, err, R = _load_night(
            cfg['night'], cfg['flux_file'], cfg['err_file'], normalize_method,
        )
        loaded[key] = dict(wave=wave, flux=flux, err=err, R=R, night=cfg['night'])
        print(f'{key} ({cfg["night"]}): R = {R:.0f}')

    # -- build Target objects --
    targets = []
    R_list  = []
    for i, key in enumerate(night_keys):
        d = loaded[key]
        t = Target(wl=d['wave'], fl=d['flux'], err=d['err'],
                   name=f'dh_tau_b_{key}')
        targets.append(t)
        R_list.append(d['R'])

    # -- parameters --
    parameters = Parameters(free_params, constant_params)
    cube = np.random.rand(parameters.ndim)
    parameters(cube)

    # -- retrieval --
    retrieval = Retrieval(
        parameters        = parameters,
        N_live_points     = N_live_points,
        evidence_tolerance = evidence_tol,
        targets           = targets if len(targets) > 1 else targets[0],
        testing           = False,
        normalize_flux    = normalize_method,
        per_chip_scaling  = per_chip_scaling,
        instrument_res    = R_list if len(R_list) > 1 else R_list[0],
    )

    retrieval.run_retrieval()

    # -- save outputs --
    params_dict, model_flux, model_flux_scaled, _ = retrieval.get_params_and_spectrum()

    np.place(retrieval.data_flux,  retrieval.data_flux  == 0, np.inf)
    np.place(model_flux,           model_flux           == 0, np.inf)
    np.place(model_flux_scaled,    model_flux_scaled    == 0, np.inf)

    np.save(retrieval.output_dir / 'retrieval_model_flux.npy',        model_flux)
    np.save(retrieval.output_dir / 'retrieval_model_flux_scaled.npy', model_flux_scaled)
    np.save(retrieval.output_dir / 'retrieval_model_wave.npy',        retrieval.data_wave)

    if retrieval.two_night_mode and hasattr(retrieval, 'model_flux2'):
        np.save(retrieval.output_dir / 'retrieval_model_flux_N2.npy',  retrieval.model_flux2)
        np.save(retrieval.output_dir / 'retrieval_model_wave_N2.npy',  retrieval.data_wave2)

    retrieval.save_diagnostic_plots()

    print('+++++++++++ Retrieval and plotting complete +++++++++++')
    print('Arrivederci! ^_^')
    return retrieval


# ============================================================
# 12. STANDALONE SCRIPT ENTRY POINT
# ============================================================

if __name__ == '__main__':
    retrieval = run_retrieval_pipeline(
        chemistry        = 'equilibrium',   # change to 'free' for 003PM-FR
        normalize_method = 'savgol',
        per_chip_scaling = False,
        N_live_points    = 700,
        evidence_tol     = 0.5,
    )

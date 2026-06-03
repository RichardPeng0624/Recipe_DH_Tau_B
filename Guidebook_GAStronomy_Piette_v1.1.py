"""
Guidebook_GAStronomy_Piette_v1.1.py
=====================================
Series 004PM — M+R priors replacing empirical log_g  [updated from v1.0 / 003PM]

Changes from v1.0
-----------------
- log_g is no longer a free parameter.  Instead, companion mass (log_M) and
  radius (log_R) are sampled with Gaussian priors from Xuan+2024 Table 1
  (DH Tau b: M = 12 ± 4 M_Jup, R = 2.6 ± 0.6 R_Jup).
- log_g is derived internally: g = G * M / R² (cgs).
- Motivation: our log_g = 4.25 (retrieval 3078846) is ~6σ above the value
  predicted by evolutionary models (log_g ≈ 3.64 for 12 MJ, 2.6 RJ).
  Xuan+2024 §4.3.5 adopt the same M+R prior approach.
- All other classes, methods, and parameters are unchanged from v1.0.

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
        normalize_method='savgol',        # 'savgol', 'median', or None
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

import sys

# pRT3 configuration
from petitRADTRANS.config import petitradtrans_config_parser
petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')

from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import simplify_species_list

# ============================================================
# 2. GLOBAL PATHS AND CONSTANTS
# ============================================================
WORKPATH     = '/data2/peng/'
RESULTS_ROOT = pathlib.Path(WORKPATH) / 'retrievals'

# Explicit pRT3 line-list name overrides (prevent interactive disambiguation
# prompts that crash MPI jobs when multiple line lists exist for one species).
PRT3_SPECIES_OVERRIDES = {
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
    '12C-16O',            # ¹²CO — HITEMP
    '13C-16O',            # ¹³CO — HITEMP
    '12C-1H4__MM',        # CH4  — MM/ExoMol (high-T optimised)
    '14N-1H3',            # NH3  — CoYuTe
    '1H2-32S',            # H2S  — AYT2
    '1H-12C-14N',         # HCN  — Harris
    '12C-16O2__HITEMP',   # CO2  — HITEMP
    '56Fe-1H',            # FeH  — MoLLIST
]
EQ_LABEL_NAMES = ['H2O', '12CO', '13CO', 'CH4', 'NH3', 'H2S', 'HCN', 'CO2', 'FeH']


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
    scale_flux : True | False | None | 'Single'
        True     → one scaling factor φ per (order, detector) chip.
        'Single' → one global φ.
        False    → no scaling (φ = 1 everywhere).
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

    def __init__(self, d_flux, d_mask, scale_flux, N_params,
                 retrieval_object=None, alpha=2):
        self.d_flux  = d_flux
        self.d_mask  = d_mask
        self.scale_flux = scale_flux
        self.scale_err  = (scale_flux is not False)
        self.N_d     = d_mask.sum()
        self.N_params = N_params
        self.alpha   = alpha
        self.retrieval_object = retrieval_object

        if self.scale_flux is True:
            self.N_phi = retrieval_object.n_orders * retrieval_object.n_dets
        else:
            self.N_phi = 1

    def __call__(self, m_flux, Cov):
        self.ln_L       = 0.0
        self.chi2_0     = 0.0
        self.m_flux_phi = np.full_like(self.d_flux, np.nan)

        N_d   = self.d_mask.sum()
        m_masked = m_flux[self.d_mask]

        if self.scale_flux is True:
            self.m_flux_phi[self.d_mask], self.phi = self.flux_scaling_rolling(m_masked, Cov)
        elif self.scale_flux in (None, 'Single'):
            d_masked = self.d_flux[self.d_mask]
            self.m_flux_phi[self.d_mask], self.phi = self.get_flux_scaling(d_masked, m_masked, Cov)
        else:
            self.m_flux_phi[self.d_mask] = m_masked
            self.phi = np.ones(1)

        residuals        = self.d_flux - self.m_flux_phi
        inv_cov_residuals = Cov.solve(residuals[self.d_mask])
        chi2_0           = np.dot(residuals[self.d_mask], inv_cov_residuals)

        inv_cov_M     = Cov.solve(m_masked)
        MT_inv_cov_M  = np.dot(m_masked, inv_cov_M)
        logdet_MT_C_M = np.log(MT_inv_cov_M)

        self.s2 = self.get_err_scaling(chi2_0, N_d) if self.scale_err else 1.0

        logdet_cov = Cov.get_logdet()
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
        """Per-chip scaling: one φ per (order, detector) pair."""
        ret = self.retrieval_object
        m_scaled  = np.full_like(self.d_flux, np.nan)
        phi_all   = np.full(ret.n_orders * ret.n_dets, np.nan)

        for order in range(ret.n_orders):
            for det in range(ret.n_dets):
                mask_od = (
                    (ret.data_wave >= ret.K2166[order, det, 0])
                    & (ret.data_wave <= ret.K2166[order, det, 1])
                )
                d_cut  = self.d_flux[mask_od]
                m_cut  = m_flux[mask_od]
                Cov_cut = Cov.__class__(err=ret.data_err[mask_od])
                m_scaled[mask_od], phi_all[order * ret.n_dets + det] = \
                    self.get_flux_scaling(d_cut, m_cut, Cov_cut)

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
                 contribution=False):
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

        # --- P-T profile (Piette+2020 / Xuan+2024) ---
        self.temperature = self.make_pt()

        # --- gravity derived from mass + radius (v1.1) ---
        # log_g is no longer a free parameter; it is derived from log_M and log_R.
        # G = 6.674e-8 cm³/(g·s²), M_Jup = 1.898e30 g, R_Jup = 7.149e9 cm
        _M_g  = 10 ** self.params['log_M'] * 1.898e30
        _R_cm = 10 ** self.params['log_R'] * 7.149e9
        self.gravity = 6.674e-8 * _M_g / _R_cm ** 2

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
        df = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'),
            index_col=0,
        )
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
        species_info = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'),
            index_col=0,
        )
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

        Reference
        ---------
        Lodders, K. (2020). Solar elemental abundances.
        Asplund, M. et al. (2021). ¹²C/¹³C solar ratio = 70.
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

        return mass_frac, CO, C_H

    # ---- radiative transfer (cached) ----

    def _make_prt_flux(self):
        """Run pRT3 radiative transfer and cache the result.

        Returns
        -------
        wl   : 1-D array, nm (rest frame, native non-uniform pRT grid)
        flux : 1-D array, W m⁻² µm⁻¹
        """
        if self._prt_wl is not None:
            return self._prt_wl, self._prt_flux

        wl, flux, _ = self.atmosphere_objects.calculate_flux(
            temperatures          = self.temperature,
            mass_fractions        = self.mass_fractions,
            reference_gravity     = self.gravity,
            mean_molar_masses     = self.MMW,
            return_contribution   = True,
            frequencies_to_wavelengths = True,
        )
        wl *= 1e7   # cm → nm

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
        flux_norm   = self._apply_normalization(flux_interp, data_wave)
        return flux_norm

    def _apply_normalization(self, flux, data_wave):
        """Apply continuum normalisation to the model spectrum.

        Modes
        -----
        'median' : divide by global median
        'savgol' : per-chip Savitzky-Golay (window=301, polyorder=2)
        False/None : no normalisation
        """
        if self.normalize_flux == 'median':
            flux /= np.nanmedian(flux)

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
    normalize_flux    : 'savgol' | 'median' | False | None
    ppid              : int | None — job ID (defaults to parent PID)
    per_chip_scaling  : bool | 'Single' | None
    instrument_res    : int | list[int] — R per night (default 100 000)
    """

    def __init__(self, parameters, N_live_points, evidence_tolerance, targets,
                 testing=True, normalize_flux=False, ppid=None,
                 per_chip_scaling=False, instrument_res=100_000):

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

        self.testing          = testing
        self.normalize_flux   = normalize_flux
        self.per_chip_scaling = per_chip_scaling
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
        self.Cov     = Covariance(err=self.data_err[self.mask_isfinite])
        self.LogLike = LogLikelihood(
            d_flux=self.data_flux, d_mask=self.mask_isfinite,
            scale_flux=self.per_chip_scaling, N_params=self.n_params,
            retrieval_object=self,
        )

        # --- covariance + likelihood (night 2, optional) ---
        if self.two_night_mode:
            self.Cov2     = Covariance(err=self.data_err2[self.mask_isfinite2])
            self.LogLike2 = LogLikelihood(
                d_flux=self.data_flux2, d_mask=self.mask_isfinite2,
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
        species_info = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'),
            index_col=0,
        )

        if param_dict.get('chemistry', 'free') == 'equilibrium':
            self.chem_species = [f'eq_{s}' for s in EQ_LABEL_NAMES]
            return list(EQ_SPECIES_PRT3)

        # Free chemistry
        self.chem_species = [
            p for p in param_dict
            if p.startswith('log_')
            and p not in {'log_g', 'log_Pquench', 'log_12CO_13CO'}
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
        cache_file = RESULTS_ROOT / f'atmosphere_objects_{self.job_id}_N{self.N_live_points}.pickle'
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
            line_species                = self.species,
            rayleigh_species            = ['H2', 'He'],
            gas_continuum_contributors  = ['H2--H2', 'H2--He'],
            wavelength_boundaries       = boundary,
            line_opacity_mode           = 'lbl',
            line_by_line_opacity_sampling = self.lbl_opacity_sampling,
            pressures                   = self.pressure,
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
        self.model_object = pRT_spectrum(self)

        flux1  = self.model_object.make_spectrum(
            data_wave=self.data_wave,
            out_res=self.instrument_res[0],
            rv_key='rv_N1',
        )
        ln_L   = self.LogLike(flux1, self.Cov)

        if self.two_night_mode:
            flux2  = self.model_object.make_spectrum(
                data_wave=self.data_wave2,
                out_res=self.instrument_res[1],
                rv_key='rv_N2',
            )
            ln_L  += self.LogLike2(flux2, self.Cov2)

        return ln_L

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

        self.model_object = pRT_spectrum(self)
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
        self.log_likelihood = self.LogLike(self.model_flux, self.Cov)
        self.params_dict['phi']  = self.LogLike.phi
        self.params_dict['s2']   = self.LogLike.s2
        self.params_dict['chi2'] = self.LogLike.chi2_0_red
        if self.two_night_mode:
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
    normalize_method : 'savgol' | 'median' | None
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
        #'rv_N1':    ({'type': 'gaussian', 'mu': 31.5, 'sigma': 1},  r'$v_{\rm rad,\,N1}$'),
        #'rv_N2':    ({'type': 'gaussian', 'mu': 31.5, 'sigma': 1},  r'$v_{\rm rad,\,N2}$'),
        'rv_N1':     ([-50,50],                                         r'$v_{\rm rad,\,N1}$'),
        'rv_N2':     ([-50,50],                                         r'$v_{\rm rad,\,N2}$'),
        'vsini':    ([0, 20],                                           r'$v\sin i$'),
        'epsilon':  ([0.0, 1.0],                                        r'$\epsilon_{\rm LD}$'),
        # v1.1: M+R Gaussian priors from Xuan+2024 Table 1 (DH Tau b)
        # σ_log_M = 0.434 × σ_M/M = 0.434 × 4/12 ≈ 0.145;  log10(12) = 1.079
        # σ_log_R = 0.434 × σ_R/R = 0.434 × 0.6/2.6 ≈ 0.100;  log10(2.6) = 0.415
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
        'dT_2':     ([  0,  500], r'$\Delta T_2$'),  #  0.0 → -0.3 bar (0.3 dex); Table C1: U(0,450)
        'dT_3':     ([ 50,  600], r'$\Delta T_3$'),  # -0.3 → -0.7 bar (0.4 dex); Table C1: U(50,550)
        'dT_4':     ([  0,  500], r'$\Delta T_4$'),  # -0.7 → -1.0 bar (0.3 dex); Table C1: U(0,450)
        'dT_5':     ([ 50,  700], r'$\Delta T_5$'),  # -1.0 → -1.5 bar (0.5 dex); Table C1: U(50,650)
        'dT_6':     ([200, 1000], r'$\Delta T_6$'),  # -1.5 → -3.0 bar (1.5 dex); Table C1: U(200,900)
        'dT_7':     ([100,  800], r'$\Delta T_7$'),  # -3.0 → -5.0 bar (2.0 dex); Table C1: U(100,700)
        'C_H':           ([-1.5, 1.5], r'$[{\rm C/H}]$'),
        'C/O':           ([0.1,  1.5], r'C/O'),
        'log_12CO_13CO': ([0.5,  3.0], r'$\log\,^{12}{\rm CO}/^{13}{\rm CO}$'),
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
        # v1.1: M+R Gaussian priors from Xuan+2024 Table 1 (DH Tau b)
        'log_M':    ({'type': 'gaussian', 'mu': 1.079, 'sigma': 0.145}, r'$\log M/M_{\rm Jup}$'),
        'log_R':    ({'type': 'gaussian', 'mu': 0.415, 'sigma': 0.100}, r'$\log R/R_{\rm Jup}$'),
        # ---- P-T: same Xuan+2024 Table C1 priors as equilibrium mode ----
        'T_anchor': ([1500, 3500], r'$T_{\rm anchor}$'),   # at 0.2 bar; Table C1: U(1800,3000)
        'dT_1':     ([200, 1000], r'$\Delta T_1$'),  # +0.7 →  0.0 bar (0.7 dex); Table C1: U(200,900)
        'dT_2':     ([  0,  500], r'$\Delta T_2$'),  #  0.0 → -0.3 bar (0.3 dex); Table C1: U(0,450)
        'dT_3':     ([ 50,  600], r'$\Delta T_3$'),  # -0.3 → -0.7 bar (0.4 dex); Table C1: U(50,550)
        'dT_4':     ([  0,  500], r'$\Delta T_4$'),  # -0.7 → -1.0 bar (0.3 dex); Table C1: U(0,450)
        'dT_5':     ([ 50,  700], r'$\Delta T_5$'),  # -1.0 → -1.5 bar (0.5 dex); Table C1: U(50,650)
        'dT_6':     ([200, 1000], r'$\Delta T_6$'),  # -1.5 → -3.0 bar (1.5 dex); Table C1: U(200,900)
        'dT_7':     ([100,  800], r'$\Delta T_7$'),  # -3.0 → -5.0 bar (2.0 dex); Table C1: U(100,700)
        'log_H2O':  ([-12, -2], r'$\log$ H$_2$O'),
        'log_12CO': ([-12, -2], r'$\log\,^{12}$CO'),
        'log_13CO': ([-12, -2], r'$\log\,^{13}$CO'),
        'log_CH4':  ([-12, -2], r'$\log$ CH$_4$'),
    }


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

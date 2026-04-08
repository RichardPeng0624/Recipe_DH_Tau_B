from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate


class processor_cross_correlation:
    def __init__(self, wMod, fMod, wlen, cube, nOrder, n_spatial, nDet):
        self.wMod = wMod
        self.fMod = fMod
        self.wlen = wlen
        self.cube = cube
        self.n_spatial = n_spatial
        self.nOrder = nOrder
        self.nDet = nDet

    def xcorr(self, f, g):
        f = np.asarray(f, dtype=float)
        g = np.asarray(g, dtype=float)
        nx = len(f)
        if nx == 0:
            return np.nan
        ones = np.ones(nx)
        f_mean_sub = f - (np.dot(f, ones) / nx)
        g_mean_sub = g - (np.dot(g, ones) / nx)
        corr = np.dot(f_mean_sub, g_mean_sub) / nx
        varf = np.dot(f_mean_sub, f_mean_sub) / nx
        varg = np.dot(g_mean_sub, g_mean_sub) / nx
        denom = np.sqrt(varf * varg)
        if denom == 0 or np.isnan(denom):
            return np.nan
        return corr / denom

    def get_cc_grid(self, rvlag, ncc):
        ccf = np.zeros((self.nDet, self.nOrder, self.n_spatial, ncc))
        coef_spline = interpolate.splrep(self.wMod, self.fMod, s=0.0)

        for irv, rv in enumerate(rvlag):
            beta = rv / 2.998e5
            w_shift = self.wlen * np.sqrt((1.0 - beta) / (1.0 + beta))
            int_mod = interpolate.splev(w_shift, coef_spline, der=0)

            for i_det in range(self.nDet):
                for i_order in range(self.nOrder):
                    for i_obs in range(self.n_spatial):
                        obs = self.cube[i_det, i_order, i_obs, :]
                        model_row = int_mod[i_det, i_order, :]
                        mask = np.isfinite(obs) & np.isfinite(model_row)
                        if np.sum(mask) < 5:
                            ccf[i_det, i_order, i_obs, irv] = np.nan
                        else:
                            ccf[i_det, i_order, i_obs, irv] = self.xcorr(obs[mask], model_row[mask])

        self.rvlag = rvlag
        self.ncc = ncc
        return ccf

    def ccf_tot(
        self,
        rvlag,
        ncc,
        plot=False,
        normalization="median subtracted",
        clean_grids=None,
        v_sys=None,
        central_pix=None,
    ):
        ccf = self.get_cc_grid(rvlag, ncc)
        ccf_sum = np.nansum(ccf, axis=(0, 1))

        if normalization == "median":
            denom = np.nanmedian(ccf_sum)
            if denom != 0:
                ccf_sum = (ccf_sum - denom) / denom
        elif normalization == "max":
            mval = np.nanmax(ccf_sum)
            if mval != 0:
                ccf_sum = ccf_sum / mval
        elif normalization == "median subtracted":
            for i_obs in range(self.n_spatial):
                med = np.nanmedian(ccf_sum[i_obs, :])
                ccf_sum[i_obs, :] -= med

        if clean_grids is None:
            std_ccf = np.nanstd(ccf_sum)
        else:
            g0, g1 = int(clean_grids[0][0]), int(clean_grids[0][1])
            g2, g3 = int(clean_grids[1][0]), int(clean_grids[1][1])
            ccf_clean_map = np.concatenate((ccf_sum[:, g0:g1], ccf_sum[:, g2:g3]), axis=1)
            std_ccf = np.nanstd(ccf_clean_map)

        ccf_snr = np.full_like(ccf_sum, np.nan)
        if std_ccf != 0:
            ccf_snr = ccf_sum / std_ccf

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(rvlag, ccf_sum[0], color="steelblue", lw=1.5)
            axes[0].set_ylabel("CCF")
            axes[0].grid(alpha=0.3)
            if v_sys is not None:
                axes[0].axvline(v_sys, color="gray", ls="--", lw=0.8)
            axes[1].plot(rvlag, ccf_snr[0], color="darkorange", lw=1.5)
            axes[1].set_xlabel("RV lag (km/s)")
            axes[1].set_ylabel("SNR")
            axes[1].grid(alpha=0.3)
            if v_sys is not None:
                axes[1].axvline(v_sys, color="gray", ls="--", lw=0.8)
            if central_pix is not None:
                axes[1].set_title(f"central_pix={central_pix}")
            plt.tight_layout()
            plt.show()

        return ccf_sum, ccf_snr


def _pick_combined_spectrum_path(
    workpath: Path,
    night: str,
    combined_spectrum_name: str,
) -> Path:
    candidates = [
        workpath / night / combined_spectrum_name,
        workpath / combined_spectrum_name,
        workpath / "combined_spectra.npy",
        workpath / "extracted_spectra_combined.npy",
        workpath / "extracted_spectra_combined_flux_cal.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No combined spectra file found in expected locations.")


def run_ccf_workflow(
    workpath: Path | str,
    retrieval_id: str,
    night: str = "2022-12-31",
    combined_spectrum_name: str = "extracted_spectra_combined_flux_cal.npy",
    rvlag: Iterable[float] | None = None,
    clean_grids: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 50), (150, 200)),
    n_shuffle: int = 200,
    random_seed: int = 42,
):
    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    model_wave = np.load(retrieval_dir / "retrieval_model_wave.npy")
    model_flux = np.load(retrieval_dir / "retrieval_model_flux_scaled.npy")

    m = np.isfinite(model_wave) & np.isfinite(model_flux)
    model_wave = model_wave[m].astype(float)
    model_flux = model_flux[m].astype(float)

    sort_idx = np.argsort(model_wave)
    model_wave = model_wave[sort_idx]
    model_flux = model_flux[sort_idx]

    combined_path = _pick_combined_spectrum_path(workpath, night, combined_spectrum_name)
    combined_spectra = np.load(combined_path)

    wave_fits = workpath / night / "cal" / "WLEN_K2166_V_DH_Tau_A+B_center.fits"
    with fits.open(wave_fits) as hdul:
        wave_data = np.array(hdul[1].data)[:, 0:5, :]

    obs_wave = wave_data.reshape(-1).astype(float)
    obs_flux = combined_spectra.reshape(-1).astype(float)
    good_obs = np.isfinite(obs_wave) & np.isfinite(obs_flux)
    obs_wave = obs_wave[good_obs]
    obs_flux = obs_flux[good_obs]

    obs_sort = np.argsort(obs_wave)
    obs_wave = obs_wave[obs_sort]
    obs_flux = obs_flux[obs_sort]

    obs_flux_on_model = np.interp(model_wave, obs_wave, obs_flux, left=np.nan, right=np.nan)
    valid = np.isfinite(obs_flux_on_model) & np.isfinite(model_flux)

    w_model = model_wave[valid]
    f_model = model_flux[valid]
    f_obs = obs_flux_on_model[valid]

    wlen = w_model[None, None, :]
    cube = f_obs[None, None, None, :]

    if rvlag is None:
        rvlag = np.arange(-100, 101, 1)
    rvlag = np.asarray(rvlag, dtype=float)

    ccf_processor = processor_cross_correlation(
        wMod=w_model,
        fMod=f_model,
        wlen=wlen,
        cube=cube,
        nOrder=1,
        n_spatial=1,
        nDet=1,
    )
    ccf_sum, ccf_snr = ccf_processor.ccf_tot(
        rvlag=rvlag,
        ncc=len(rvlag),
        plot=False,
        normalization="median subtracted",
        clean_grids=clean_grids,
    )

    ccf_1d = ccf_sum[0]
    ccf_snr_1d = ccf_snr[0]
    peak_idx = np.nanargmax(ccf_snr_1d)
    peak_rv = float(rvlag[peak_idx])
    peak_snr = float(ccf_snr_1d[peak_idx])

    rng = np.random.default_rng(random_seed)
    null_peaks = np.full(n_shuffle, np.nan)
    for i in range(n_shuffle):
        f_model_shuffle = rng.permutation(f_model)
        ccf_tmp = processor_cross_correlation(
            wMod=w_model,
            fMod=f_model_shuffle,
            wlen=wlen,
            cube=cube,
            nOrder=1,
            n_spatial=1,
            nDet=1,
        )
        _, ccf_snr_tmp = ccf_tmp.ccf_tot(
            rvlag=rvlag,
            ncc=len(rvlag),
            plot=False,
            normalization="median subtracted",
            clean_grids=clean_grids,
        )
        null_peaks[i] = np.nanmax(ccf_snr_tmp[0])

    null_mu = float(np.nanmean(null_peaks))
    null_sigma = float(np.nanstd(null_peaks))
    z_score = (peak_snr - null_mu) / null_sigma if null_sigma > 0 else np.nan

    return {
        "combined_path": combined_path,
        "valid_pixels": int(valid.sum()),
        "rvlag": rvlag,
        "ccf": ccf_1d,
        "ccf_snr": ccf_snr_1d,
        "peak_rv": peak_rv,
        "peak_snr": peak_snr,
        "null_peaks": null_peaks,
        "null_mu": null_mu,
        "null_sigma": null_sigma,
        "z_score": float(z_score),
    }


def summarize_ci(samples: np.ndarray, name: str) -> Dict[str, float | str]:
    q = np.nanpercentile(samples, [0.135, 16, 50, 84, 99.865])
    return {
        "parameter": name,
        "p0.135": q[0],
        "p16": q[1],
        "p50": q[2],
        "p84": q[3],
        "p99.865": q[4],
        "minus_1sigma": q[2] - q[1],
        "plus_1sigma": q[3] - q[2],
        "minus_3sigma": q[2] - q[0],
        "plus_3sigma": q[4] - q[2],
    }


def run_chemistry_workflow(workpath: Path | str, retrieval_id: str):
    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    posterior = np.load(retrieval_dir / "final_posterior.npy")
    with open(retrieval_dir / "final_params_dict.pickle", "rb") as f:
        params_dict = pickle.load(f)

    non_retrieved_keys = {"[Fe/H]", "C/O", "phi", "s2", "chi2", "lnZ"}
    param_keys_in_order = [k for k in params_dict.keys() if k not in non_retrieved_keys]
    param_keys_in_order = param_keys_in_order[: posterior.shape[1]]

    df_post = pd.DataFrame(posterior, columns=param_keys_in_order)

    required = {"log_12CO", "log_13CO", "log_CH4", "log_H2O"}
    if not required.issubset(df_post.columns):
        missing = required.difference(df_post.columns)
        raise KeyError(f"Missing required parameters in posterior columns: {missing}")

    vmr_12co = 10.0 ** df_post["log_12CO"].to_numpy()
    vmr_13co = 10.0 ** df_post["log_13CO"].to_numpy()
    vmr_ch4 = 10.0 ** df_post["log_CH4"].to_numpy()
    vmr_h2o = 10.0 ** df_post["log_H2O"].to_numpy()

    c_total = vmr_12co + vmr_13co + vmr_ch4
    o_total = vmr_12co + vmr_13co + vmr_h2o
    co_ratio_samples = c_total / o_total
    c12_c13_samples = vmr_12co / vmr_13co

    summary = pd.DataFrame(
        [
            summarize_ci(co_ratio_samples, "C/O (from posterior chemistry)"),
            summarize_ci(c12_c13_samples, "12CO/13CO"),
        ]
    )

    return {
        "posterior": posterior,
        "columns": param_keys_in_order,
        "co_ratio_samples": co_ratio_samples,
        "c12_c13_samples": c12_c13_samples,
        "summary": summary,
    }


def run_equil_chemistry_workflow(workpath: Path | str, retrieval_id: str):
    """
    Chemistry diagnostics for an equilibrium-chemistry retrieval.

    The posterior columns are the free parameters in order:
        rv, vsini, log_g, T0–T4, C_H, C/O, log_12CO_13CO
    (plus non-retrieved keys [C/H], phi, s2, chi2, lnZ stored only in params_dict).

    Returns
    -------
    dict with keys:
        posterior, columns,
        co_samples, ch_samples, c12c13_samples,
        summary (pd.DataFrame with median ± 1/3σ CI rows)
    """
    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    posterior = np.load(retrieval_dir / "final_posterior.npy")
    with open(retrieval_dir / "final_params_dict.pickle", "rb") as f:
        params_dict = pickle.load(f)

    non_retrieved_keys = {"[C/H]", "phi", "s2", "chi2", "lnZ"}
    param_keys_in_order = [k for k in params_dict.keys() if k not in non_retrieved_keys]
    param_keys_in_order = param_keys_in_order[: posterior.shape[1]]

    df_post = pd.DataFrame(posterior, columns=param_keys_in_order)

    required = {"C_H", "C/O", "log_12CO_13CO"}
    if not required.issubset(df_post.columns):
        missing = required.difference(df_post.columns)
        raise KeyError(f"Missing required parameters in posterior columns: {missing}")

    co_samples     = df_post["C/O"].to_numpy()
    ch_samples     = df_post["C_H"].to_numpy()
    c12c13_samples = 10.0 ** df_post["log_12CO_13CO"].to_numpy()

    summary = pd.DataFrame(
        [
            summarize_ci(co_samples,     "C/O"),
            summarize_ci(ch_samples,     "[C/H]"),
            summarize_ci(c12c13_samples, "12CO/13CO"),
        ]
    )

    return {
        "posterior": posterior,
        "columns": param_keys_in_order,
        "co_samples": co_samples,
        "ch_samples": ch_samples,
        "c12c13_samples": c12c13_samples,
        "summary": summary,
    }

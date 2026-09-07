"""
analysis.py  –  Reusable analysis, plotting, and workflow wrappers for DH Tau B retrievals.

Designed to be called from notebooks or scripts:

    from analysis import (
        run_species_ccf_validation,   # De Regt+2024 §4.2 per-species CCF
        run_ccf_workflow,             # Pearson CCF with shuffle null test
        run_free_chem_analysis,       # full free-chem analysis suite
        run_equil_chem_analysis,      # full equil-chem analysis suite
        plot_pt_profile_piette,       # Piette+2020 P-T posterior bands
        plot_PT_formation_scenario,   # Sonora Bobcat hot/warm/cold-start PT overlays
    )

Sections
--------
1.  Imports & constants
2.  Pearson CCF (run_ccf_workflow, processor_cross_correlation)
3.  Chemistry workflow helpers (summarize_ci, run_*_workflow)
4.  Plotting: CCF, chemistry, binned model-vs-data, P-T profile
5.  Parameter key constants
6.  De Regt+2024 §4.2 species CCF validation
7.  Top-level convenience functions (run_free_chem_analysis, run_equil_chem_analysis)
8.  Formation scenario: Sonora Bobcat PT overlays
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
from scipy.signal import savgol_filter


# ============================================================
# 2. PEARSON CCF
# ============================================================

class processor_cross_correlation:
    """Pearson cross-correlation of a model spectrum against observed data.

    Used for the global model-vs-data CCF (model detection check).
    For per-species CCF validation use run_species_ccf_validation() instead.
    """

    def __init__(self, wMod, fMod, wlen, cube, nOrder, n_spatial, nDet):
        self.wMod      = wMod
        self.fMod      = fMod
        self.wlen      = wlen
        self.cube      = cube
        self.n_spatial = n_spatial
        self.nOrder    = nOrder
        self.nDet      = nDet

    def xcorr(self, f, g):
        f = np.asarray(f, dtype=float)
        g = np.asarray(g, dtype=float)
        nx = len(f)
        if nx == 0:
            return np.nan
        ones       = np.ones(nx)
        f_ms       = f - np.dot(f, ones) / nx
        g_ms       = g - np.dot(g, ones) / nx
        corr       = np.dot(f_ms, g_ms) / nx
        denom      = np.sqrt(np.dot(f_ms, f_ms) / nx * np.dot(g_ms, g_ms) / nx)
        return np.nan if (denom == 0 or np.isnan(denom)) else corr / denom

    def get_cc_grid(self, rvlag, ncc):
        ccf        = np.zeros((self.nDet, self.nOrder, self.n_spatial, ncc))
        coef_spline = interpolate.splrep(self.wMod, self.fMod, s=0.0)
        for irv, rv in enumerate(rvlag):
            beta    = rv / 2.998e5
            w_shift = self.wlen * np.sqrt((1.0 - beta) / (1.0 + beta))
            int_mod = interpolate.splev(w_shift, coef_spline, der=0)
            for id_ in range(self.nDet):
                for io in range(self.nOrder):
                    for is_ in range(self.n_spatial):
                        obs  = self.cube[id_, io, is_, :]
                        mrow = int_mod[id_, io, :]
                        mask = np.isfinite(obs) & np.isfinite(mrow)
                        ccf[id_, io, is_, irv] = (
                            self.xcorr(obs[mask], mrow[mask]) if mask.sum() >= 5 else np.nan
                        )
        self.rvlag = rvlag
        self.ncc   = ncc
        return ccf

    def ccf_tot(self, rvlag, ncc, plot=False, normalization="median subtracted",
                clean_grids=None, v_sys=None, central_pix=None):
        ccf     = self.get_cc_grid(rvlag, ncc)
        ccf_sum = np.nansum(ccf, axis=(0, 1))

        if normalization == "median":
            denom = np.nanmedian(ccf_sum)
            if denom:
                ccf_sum = (ccf_sum - denom) / denom
        elif normalization == "max":
            mval = np.nanmax(ccf_sum)
            if mval:
                ccf_sum /= mval
        elif normalization == "median subtracted":
            for i in range(self.n_spatial):
                ccf_sum[i] -= np.nanmedian(ccf_sum[i])

        if clean_grids is None:
            std_ccf = np.nanstd(ccf_sum)
        else:
            g = clean_grids
            std_ccf = np.nanstd(
                np.concatenate((ccf_sum[:, int(g[0][0]):int(g[0][1])],
                                ccf_sum[:, int(g[1][0]):int(g[1][1])]), axis=1)
            )
        ccf_snr = ccf_sum / std_ccf if std_ccf else np.full_like(ccf_sum, np.nan)

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(rvlag, ccf_sum[0], color="steelblue", lw=1.5)
            axes[0].set_ylabel("CCF"); axes[0].grid(alpha=0.3)
            if v_sys is not None:
                axes[0].axvline(v_sys, color="gray", ls="--", lw=0.8)
            axes[1].plot(rvlag, ccf_snr[0], color="darkorange", lw=1.5)
            axes[1].set_xlabel("RV lag (km/s)"); axes[1].set_ylabel("SNR")
            axes[1].grid(alpha=0.3)
            if v_sys is not None:
                axes[1].axvline(v_sys, color="gray", ls="--", lw=0.8)
            if central_pix is not None:
                axes[1].set_title(f"central_pix={central_pix}")
            plt.tight_layout(); plt.show()
        return ccf_sum, ccf_snr


def _pick_combined_spectrum_path(workpath: Path, night: str,
                                  combined_spectrum_name: str) -> Path:
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
) -> dict:
    """Pearson CCF of the retrieval model against the observed spectrum.

    Returns peak RV, peak SNR, and a shuffle null-test z-score.
    For per-species CCF validation (De Regt+2024) use run_species_ccf_validation().
    """
    workpath      = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    model_wave = np.load(retrieval_dir / "retrieval_model_wave.npy")
    model_flux = np.load(retrieval_dir / "retrieval_model_flux_scaled.npy")
    m          = np.isfinite(model_wave) & np.isfinite(model_flux)
    model_wave = model_wave[m].astype(float)
    model_flux = model_flux[m].astype(float)
    idx        = np.argsort(model_wave)
    model_wave, model_flux = model_wave[idx], model_flux[idx]

    combined_path    = _pick_combined_spectrum_path(workpath, night, combined_spectrum_name)
    combined_spectra = np.load(combined_path)
    wave_fits        = workpath / night / "cal" / "WLEN_K2166_V_DH_Tau_A+B_center.fits"
    with fits.open(wave_fits) as hdul:
        wave_data = np.array(hdul[1].data)[:, 0:5, :]

    obs_wave = wave_data.reshape(-1).astype(float)
    obs_flux = combined_spectra.reshape(-1).astype(float)
    good     = np.isfinite(obs_wave) & np.isfinite(obs_flux)
    obs_wave, obs_flux = obs_wave[good], obs_flux[good]
    sidx     = np.argsort(obs_wave)
    obs_wave, obs_flux = obs_wave[sidx], obs_flux[sidx]

    obs_on_model = np.interp(model_wave, obs_wave, obs_flux, left=np.nan, right=np.nan)
    valid        = np.isfinite(obs_on_model) & np.isfinite(model_flux)
    w, fm, fo    = model_wave[valid], model_flux[valid], obs_on_model[valid]

    wlen = w[None, None, :]
    cube = fo[None, None, None, :]
    if rvlag is None:
        rvlag = np.arange(-100, 101, 1)
    rvlag = np.asarray(rvlag, dtype=float)

    proc            = processor_cross_correlation(w, fm, wlen, cube, 1, 1, 1)
    ccf_sum, ccf_snr = proc.ccf_tot(rvlag, len(rvlag), plot=False,
                                      normalization="median subtracted",
                                      clean_grids=clean_grids)
    ccf_1d, snr_1d  = ccf_sum[0], ccf_snr[0]
    peak_idx        = int(np.nanargmax(snr_1d))
    peak_rv, peak_snr = float(rvlag[peak_idx]), float(snr_1d[peak_idx])

    rng        = np.random.default_rng(random_seed)
    null_peaks = np.full(n_shuffle, np.nan)
    for i in range(n_shuffle):
        fm_sh = rng.permutation(fm)
        p_sh  = processor_cross_correlation(w, fm_sh, wlen, cube, 1, 1, 1)
        _, snr_sh = p_sh.ccf_tot(rvlag, len(rvlag), plot=False,
                                   normalization="median subtracted",
                                   clean_grids=clean_grids)
        null_peaks[i] = float(np.nanmax(snr_sh[0]))
    null_mu, null_sigma = float(np.nanmean(null_peaks)), float(np.nanstd(null_peaks))
    z_score = (peak_snr - null_mu) / null_sigma if null_sigma > 0 else np.nan

    return dict(combined_path=combined_path, valid_pixels=int(valid.sum()),
                rvlag=rvlag, ccf=ccf_1d, ccf_snr=snr_1d,
                peak_rv=peak_rv, peak_snr=peak_snr,
                null_peaks=null_peaks, null_mu=null_mu, null_sigma=null_sigma,
                z_score=float(z_score))


# ============================================================
# 3. CHEMISTRY WORKFLOW HELPERS
# ============================================================

def summarize_ci(samples: np.ndarray, name: str) -> Dict[str, float | str]:
    """1σ and 3σ confidence intervals from posterior samples."""
    q = np.nanpercentile(samples, [0.135, 16, 50, 84, 99.865])
    return dict(parameter=name,
                p0_135=q[0], p16=q[1], p50=q[2], p84=q[3], p99_865=q[4],
                minus_1sigma=q[2] - q[1], plus_1sigma=q[3] - q[2],
                minus_3sigma=q[2] - q[0], plus_3sigma=q[4] - q[2])


def run_free_chemistry_workflow(workpath: Path | str, retrieval_id: str) -> dict:
    """Compute C/O, [C/H], ¹²C/¹³C posterior samples for a free-chemistry retrieval.

    Assumes free params: log_H2O, log_12CO, log_13CO, log_CH4.
    [C/H] calculated relative to solar A(C) = 8.46 (Asplund+2021).
    """
    workpath      = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id
    posterior     = np.load(retrieval_dir / "final_posterior.npy")
    with open(retrieval_dir / "final_params_dict.pickle", "rb") as f:
        params_dict = pickle.load(f)

    non_ret  = {"[C/H]", "[C/H]_xsolar", "C/O", "phi", "s2", "chi2",
                "phi_N2", "chi2_N2", "lnZ"}
    pk       = [k for k in params_dict if k not in non_ret][:posterior.shape[1]]
    df       = pd.DataFrame(posterior, columns=pk)

    required = {"log_H2O", "log_12CO", "log_13CO", "log_CH4"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing free-chem params: {required - set(df.columns)}")

    VMR_He   = 0.15
    vmr_h2o  = 10.0 ** df["log_H2O"].to_numpy()
    vmr_12co = 10.0 ** df["log_12CO"].to_numpy()
    vmr_13co = 10.0 ** df["log_13CO"].to_numpy()
    vmr_ch4  = 10.0 ** df["log_CH4"].to_numpy()
    vmr_h2   = 1.0 - VMR_He - vmr_h2o - vmr_12co - vmr_13co - vmr_ch4

    C  = vmr_12co + vmr_13co + vmr_ch4
    O  = vmr_h2o  + vmr_12co + vmr_13co
    H  = 2.0 * vmr_h2o + 4.0 * vmr_ch4 + 2.0 * vmr_h2

    co_samples        = C / O
    ch_xsolar_samples = np.log10(C / H) - (8.46 - 12.0)
    c12c13_samples    = vmr_12co / vmr_13co

    summary = pd.DataFrame([
        summarize_ci(co_samples,        "C/O"),
        summarize_ci(ch_xsolar_samples, "[C/H] (dex, ×solar)"),
        summarize_ci(c12c13_samples,    "12CO/13CO"),
    ])
    return dict(posterior=posterior, columns=pk,
                co_samples=co_samples, ch_xsolar_samples=ch_xsolar_samples,
                c12c13_samples=c12c13_samples, summary=summary)


def run_equil_chemistry_workflow(workpath: Path | str, retrieval_id: str) -> dict:
    """Extract C/O, [C/H], ¹²C/¹³C posterior samples for an equilibrium-chemistry retrieval.

    Reads the directly retrieved free params C_H, C/O, log_12CO_13CO.
    """
    workpath      = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id
    posterior     = np.load(retrieval_dir / "final_posterior.npy")
    with open(retrieval_dir / "final_params_dict.pickle", "rb") as f:
        params_dict = pickle.load(f)

    non_ret  = {"[C/H]", "phi", "s2", "chi2", "phi_N2", "chi2_N2", "lnZ"}
    pk       = [k for k in params_dict if k not in non_ret][:posterior.shape[1]]
    df       = pd.DataFrame(posterior, columns=pk)

    required = {"C_H", "C/O", "log_12CO_13CO"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing equil-chem params: {required - set(df.columns)}")

    co_samples     = df["C/O"].to_numpy()
    ch_samples     = df["C_H"].to_numpy()
    c12c13_samples = 10.0 ** df["log_12CO_13CO"].to_numpy()

    summary = pd.DataFrame([
        summarize_ci(co_samples,     "C/O"),
        summarize_ci(ch_samples,     "[C/H] (dex)"),
        summarize_ci(c12c13_samples, "12CO/13CO"),
    ])
    return dict(posterior=posterior, columns=pk,
                co_samples=co_samples, ch_samples=ch_samples,
                c12c13_samples=c12c13_samples, summary=summary)

# ---------------------------------------------------------------------------
# CCF
# ---------------------------------------------------------------------------

def plot_ccf(ccf_out: dict, retrieval_id: str) -> None:
    """Two-panel CCF + SNR plot, plus null-test histogram."""
    rvlag   = ccf_out["rvlag"]
    peak_rv = ccf_out["peak_rv"]
    peak_snr = ccf_out["peak_snr"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(rvlag, ccf_out["ccf"], color="steelblue", lw=1.5,
                 label="Data vs retrieval model CCF")
    axes[0].axvline(0,       color="gray",  ls="--", lw=0.9, label="RV = 0")
    axes[0].axvline(peak_rv, color="tomato", ls="--", lw=1.0,
                    label=f"Peak at {peak_rv:.1f} km/s")
    axes[0].set_ylabel("CCF"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)

    axes[1].plot(rvlag, ccf_out["ccf_snr"], color="darkorange", lw=1.5, label="CCF SNR")
    axes[1].axvline(0,       color="gray",  ls="--", lw=0.9)
    axes[1].axvline(peak_rv, color="tomato", ls="--", lw=1.0,
                    label=f"Peak SNR = {peak_snr:.2f}")
    axes[1].set_xlabel("RV lag (km/s)"); axes[1].set_ylabel("SNR")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)

    plt.suptitle(retrieval_id, fontsize=9)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.hist(ccf_out["null_peaks"], bins=25, alpha=0.7, color="slategray",
             edgecolor="white", label="Null (shuffled model) peak SNR")
    plt.axvline(peak_snr, color="crimson", lw=2,
                label=f"Observed peak SNR = {peak_snr:.2f}")
    plt.xlabel("Peak SNR"); plt.ylabel("Count")
    plt.title("CCF Robustness Null Test")
    plt.legend(fontsize=9); plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Chemistry
# ---------------------------------------------------------------------------

def plot_free_chemistry(chem_out: dict, retrieval_id: str) -> None:
    """
    Three-panel posterior histogram: C/O, C/H (×solar, linear), 12CO/13CO.

    [C/H] is shown on a **linear** ×solar scale (i.e. 10^[C/H]).
    For 12CO/13CO, if the 3σ upper bound exceeds 10× the 1σ upper bound the
    x-axis is clipped at the 90th percentile and the tail fraction is annotated.
    """
    co     = chem_out["co_samples"]
    ch_lin = 10.0 ** chem_out["ch_xsolar_samples"]   # convert dex → linear ×solar
    r      = chem_out["c12c13_samples"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── C/O ─────────────────────────────────────────────────────────────────
    ax = axes[0]
    valid_co = co[np.isfinite(co)]
    ax.hist(valid_co, bins=60, color="teal", alpha=0.75, edgecolor="white")
    q = np.nanpercentile(valid_co, [16, 50, 84])
    ax.axvline(q[1], color="k", ls="-",  lw=1.5, label=f"median = {q[1]:.3g}")
    ax.axvline(q[0], color="k", ls="--", lw=0.9)
    ax.axvline(q[2], color="k", ls="--", lw=0.9,
               label=f"+{q[2]-q[1]:.2g} / −{q[1]-q[0]:.2g}")
    ax.set_xlabel("C/O"); ax.set_ylabel("Count")
    ax.set_title("Posterior C/O", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── C/H (linear ×solar) ─────────────────────────────────────────────────
    ax = axes[1]
    valid_ch = ch_lin[np.isfinite(ch_lin) & (ch_lin > 0)]
    ax.hist(valid_ch, bins=60, color="steelblue", alpha=0.75, edgecolor="white")
    q_ch = np.nanpercentile(valid_ch, [16, 50, 84])
    ax.axvline(q_ch[1], color="k", ls="-",  lw=1.5,
               label=f"median = {q_ch[1]:.3g}")
    ax.axvline(q_ch[0], color="k", ls="--", lw=0.9)
    ax.axvline(q_ch[2], color="k", ls="--", lw=0.9,
               label=f"+{q_ch[2]-q_ch[1]:.2g} / −{q_ch[1]-q_ch[0]:.2g}")
    ax.set_xlabel(r"C/H ($\times$ solar)"); ax.set_ylabel("Count")
    ax.set_title(r"Posterior C/H ($\times$ solar)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── 12CO/13CO — adaptive clipping for unconstrained upper tail ───────────
    ax = axes[2]
    valid_r = r[np.isfinite(r) & (r > 0)]
    q_r     = np.nanpercentile(valid_r, [0.135, 16, 50, 84, 99.865])
    one_sig_upper   = q_r[3]
    three_sig_upper = q_r[4]

    tail_unconstrained = three_sig_upper > 10.0 * one_sig_upper

    if tail_unconstrained:
        clip_val  = np.nanpercentile(valid_r, 90)
        tail_mask = valid_r > clip_val
        tail_frac = tail_mask.sum() / len(valid_r) * 100.0
        plot_data = valid_r[~tail_mask]
        bins_r    = np.linspace(plot_data.min(), clip_val, 60)
        ax.hist(plot_data, bins=bins_r, color="indigo", alpha=0.75, edgecolor="white")
        ax.set_xlim(plot_data.min(), clip_val * 1.02)
        ax.annotate(
            f"{tail_frac:.1f}% of samples\nbeyond x-axis →",
            xy=(1.0, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5,
            color="gray",
        )
    else:
        ax.hist(valid_r, bins=60, color="indigo", alpha=0.75, edgecolor="white")

    ax.axvline(q_r[2], color="k", ls="-",  lw=1.5,
               label=f"median = {q_r[2]:.3g}")
    ax.axvline(q_r[1], color="k", ls="--", lw=0.9)
    if not tail_unconstrained:
        ax.axvline(q_r[3], color="k", ls="--", lw=0.9,
                   label=f"+{q_r[3]-q_r[2]:.2g} / −{q_r[2]-q_r[1]:.2g}")
    else:
        ax.axvline(min(q_r[3], clip_val), color="k", ls="--", lw=0.9,
                   label=f"+{q_r[3]-q_r[2]:.2g} (1σ) / −{q_r[2]-q_r[1]:.2g}")

    ax.set_xlabel(r"$^{12}$CO/$^{13}$CO"); ax.set_ylabel("Count")
    ax.set_title(r"Posterior $^{12}$CO/$^{13}$CO", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.suptitle(f"Free-chemistry retrieval: {retrieval_id}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_equil_chemistry(chem_out: dict, retrieval_id: str) -> None:
    """
    Three-panel posterior histogram: C/O, [C/H] (dex, retrieved directly),
    12CO/13CO — for an equilibrium-chemistry retrieval.

    [C/H] is converted from the directly retrieved dex parameter to linear ×solar
    (i.e. 10^[C/H]) for display, consistent with the free-chemistry plot.
    For 12CO/13CO, if the 3σ upper bound exceeds 10× the 1σ upper bound the
    x-axis is clipped at the 90th percentile and the tail fraction is annotated.
    """
    co     = chem_out["co_samples"]
    ch_lin = 10.0 ** chem_out["ch_samples"]   # convert dex → linear ×solar
    r      = chem_out["c12c13_samples"]        # 10^log_12CO_13CO, linear

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # ── C/O and C/H (linear ×solar) ─────────────────────────────────────────
    for ax, samp, label, color in zip(
        axes[:2],
        [co, ch_lin],
        ["C/O", r"C/H ($\times$ solar)"],
        ["teal", "steelblue"],
    ):
        valid = samp[np.isfinite(samp) & (samp > 0)]
        ax.hist(valid, bins=50, color=color, alpha=0.75, edgecolor="white")
        q = np.nanpercentile(valid, [16, 50, 84])
        ax.axvline(q[1], color="k", ls="-",  lw=1.5, label=f"median = {q[1]:.3g}")
        ax.axvline(q[0], color="k", ls="--", lw=0.9)
        ax.axvline(q[2], color="k", ls="--", lw=0.9,
                   label=f"+{q[2]-q[1]:.2g} / −{q[1]-q[0]:.2g}")
        ax.set_xlabel(label); ax.set_ylabel("Count")
        ax.set_title(f"Posterior {label}", fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # ── 12CO/13CO — adaptive clipping for unconstrained upper tail ───────────
    ax = axes[2]
    valid_r = r[np.isfinite(r) & (r > 0)]
    q_r     = np.nanpercentile(valid_r, [0.135, 16, 50, 84, 99.865])
    one_sig_upper   = q_r[3]
    three_sig_upper = q_r[4]

    tail_unconstrained = three_sig_upper > 10.0 * one_sig_upper

    if tail_unconstrained:
        clip_val  = np.nanpercentile(valid_r, 90)
        tail_mask = valid_r > clip_val
        tail_frac = tail_mask.sum() / len(valid_r) * 100.0
        plot_data = valid_r[~tail_mask]
        bins_r    = np.linspace(plot_data.min(), clip_val, 50)
        ax.hist(plot_data, bins=bins_r, color="indigo", alpha=0.75, edgecolor="white")
        ax.set_xlim(plot_data.min(), clip_val * 1.02)
        ax.annotate(
            f"{tail_frac:.1f}% of samples\nbeyond x-axis →",
            xy=(1.0, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7.5, color="gray",
        )
    else:
        ax.hist(valid_r, bins=50, color="indigo", alpha=0.75, edgecolor="white")

    ax.axvline(q_r[2], color="k", ls="-",  lw=1.5,
               label=f"median = {q_r[2]:.3g}")
    ax.axvline(q_r[1], color="k", ls="--", lw=0.9)
    if not tail_unconstrained:
        ax.axvline(q_r[3], color="k", ls="--", lw=0.9,
                   label=f"+{q_r[3]-q_r[2]:.2g} / −{q_r[2]-q_r[1]:.2g}")
    else:
        ax.axvline(min(q_r[3], clip_val), color="k", ls="--", lw=0.9,
                   label=f"+{q_r[3]-q_r[2]:.2g} (1σ) / −{q_r[2]-q_r[1]:.2g}")
    ax.set_xlabel(r"$^{12}$CO/$^{13}$CO"); ax.set_ylabel("Count")
    ax.set_title(r"Posterior $^{12}$CO/$^{13}$CO", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.suptitle(f"Equilibrium-chemistry retrieval: {retrieval_id}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Binned model vs data (per-order-detector, with telluric overlay)
# ---------------------------------------------------------------------------

def plot_binned_model_vs_data(
    workpath: Path | str,
    retrieval_id: str,
    night: str,
    model_wave_npy: str = "retrieval_model_wave.npy",
    model_flux_npy: str = "retrieval_model_flux_scaled.npy",
    data_npy: str | None = None,
    err_npy: str | None = None,
    win: int | None = None,
    pol: int = 2,
    data_dir: Path | str | None = None,
    method: str = 'savgol',
) -> None:
    """
    Binned data vs best-fit model, 5 orders × 3 detectors, with telluric overlay
    and residual panel.

    Parameters
    ----------
    workpath        Root data directory.
    retrieval_id    Retrieval folder name under workpath/retrievals/.
    night           Observation night string, e.g. '2022-12-31'.
                    Used for the wavelength FITS and telluric FITS paths.
    model_wave_npy  .npy filename inside the retrieval folder (wavelength).
    model_flux_npy  .npy filename inside the retrieval folder (scaled flux).
    data_npy        .npy filename for the combined spectrum.
                    Defaults to 'extracted_spectra_combined_sigmaclipper.npy'.
    err_npy         .npy filename for the combined error.
                    Defaults to 'extracted_spectra_combined_err_sigmaclipper.npy'.
    win             Window size: SavGol window for method='savgol', bin size in
                    pixels for method='downsampling'. Defaults: savgol→31, down→10.
    pol             SavGol polynomial order (ignored for downsampling).
    data_dir        Directory containing data_npy and err_npy.
                    Defaults to workpath/night/ when None.
                    Pass e.g. workpath/'combined_two_nights' to use a combined
                    spectrum while still drawing wavelength/telluric from night.
    method          'savgol' (Savitzky-Golay smoothing, default) or
                    'downsampling' (block-average every win pixels).
    """
    if win is None:
        win = 31 if method == 'savgol' else 10

    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    if data_npy is None:
        data_npy = "extracted_spectra_combined_sigmaclipper.npy"
    if err_npy is None:
        err_npy = "extracted_spectra_combined_err_sigmaclipper.npy"
    _data_dir = Path(data_dir) if data_dir is not None else workpath / night

    # Load model
    _mw_raw = np.load(retrieval_dir / model_wave_npy).flatten().astype(float)
    _mf_raw = np.load(retrieval_dir / model_flux_npy).flatten().astype(float)
    _fin    = np.isfinite(_mw_raw) & np.isfinite(_mf_raw)
    _isort  = np.argsort(_mw_raw[_fin])
    mw_s    = _mw_raw[_fin][_isort]
    mf_s    = _mf_raw[_fin][_isort]

    # Load wavelength grid, data, errors
    wave_fits = workpath / night / "cal" / "WLEN_K2166_V_DH_Tau_A+B_center.fits"
    with fits.open(wave_fits) as hdul:
        wave_cube = np.array(hdul[1].data)[:, 0:5, :].astype(float)

    data_flux = np.load(_data_dir / data_npy)
    err_data  = np.load(_data_dir / err_npy)
    data_cube = data_flux.reshape(wave_cube.shape).astype(float)

    # Load telluric transmission
    telluric_fits = (workpath / night / "chi_tau" / night /
                     "out" / "molecfit" / "TELLURIC_DATA.fits")
    with fits.open(telluric_fits) as hdul:
        t_raw = np.array(hdul[1].data)
    t = np.reshape(t_raw, (7, 3, 2048))
    t = np.array([t[i] for i in range(7)][::-1])
    telluric_cube = np.transpose(t, (1, 0, 2))

    n_det, n_order, n_pix = wave_cube.shape
    order_bounds = []

    fig = plt.figure(figsize=(20, 20))
    gs_outer = fig.add_gridspec(n_order, 1, hspace=0.05)

    for order in range(n_order):
        gs_inner = gs_outer[order].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.)
        ax_top = fig.add_subplot(gs_inner[0])
        ax_res = fig.add_subplot(gs_inner[1], sharex=ax_top)
        ax_tel = ax_top.twinx()

        order_has_data = False
        wl_min_order, wl_max_order = np.inf, -np.inf
        order_resids, order_errs = [], []

        for det in range(n_det):
            w_full = wave_cube[det, order].astype(float)
            f_full = data_cube[det, order].astype(float)
            e_full = err_data[det, order].astype(float)

            valid = (np.isfinite(w_full) & np.isfinite(f_full) &
                     np.isfinite(e_full) & (e_full > 0))
            if valid.sum() < win:
                continue

            f_vld = f_full[valid]; e_vld = e_full[valid]
            w_vld = w_full[valid]

            if method == 'savgol':
                f_bin_vld = savgol_filter(f_vld, win, pol)
                e_bin_vld = np.sqrt(savgol_filter(e_vld**2, win, pol)) / np.sqrt(win / 2)
                m_clip = (mw_s >= w_vld.min()) & (mw_s <= w_vld.max())
                if m_clip.sum() >= win:
                    m_sm     = savgol_filter(mf_s[m_clip], win, pol)
                    m_interp = np.interp(w_vld, mw_s[m_clip], m_sm)
                else:
                    m_interp = np.interp(w_vld, mw_s, mf_s)
            else:  # 'downsampling'
                n_bins = len(f_vld) // win
                if n_bins < 1:
                    continue
                sl = slice(0, n_bins * win)
                _fb = f_vld[sl].reshape(n_bins, win)
                _eb = e_vld[sl].reshape(n_bins, win)
                w_vld     = w_vld[sl].reshape(n_bins, win).mean(axis=1)
                f_bin_vld = _fb.mean(axis=1)
                e_bin_vld = np.sqrt((_eb**2).mean(axis=1)) / np.sqrt(win)
                m_interp  = np.interp(w_vld, mw_s, mf_s)

            wl_lo, wl_hi = w_vld.min(), w_vld.max()
            wl_min_order = min(wl_min_order, wl_lo)
            wl_max_order = max(wl_max_order, wl_hi)

            f_med = np.nanmedian(f_bin_vld) or 1.0
            m_med = np.nanmedian(m_interp)  or 1.0
            f_n_vld = f_bin_vld / f_med
            e_n_vld = e_bin_vld / f_med
            m_n_vld = m_interp  / m_med
            resid_vld = f_n_vld - m_n_vld

            lbl_data  = "Data"  if det == 0 else None
            lbl_model = "Model" if det == 0 else None
            lbl_err   = "±1σ"   if det == 0 else None

            ax_top.errorbar(w_vld, f_n_vld, yerr=e_n_vld,
                            fmt='none', ecolor='gray', elinewidth=0.4,
                            capsize=0, alpha=0.5, label=lbl_err)
            ax_top.plot(w_vld, f_n_vld, color="steelblue", lw=1.0, alpha=0.85,
                        label=lbl_data)
            ax_top.plot(w_vld, m_n_vld, color="tomato",    lw=1.9, alpha=0.95,
                        label=lbl_model, zorder=5)
            ax_res.scatter(w_vld, resid_vld, s=2, color="steelblue",
                           alpha=0.6, linewidths=0)
            order_resids.append(resid_vld)
            order_errs.append(e_n_vld)
            order_has_data = True

            if order < telluric_cube.shape[1]:
                t_w  = telluric_cube[det, order]["lambda"] * 1e3
                t_m  = telluric_cube[det, order]["mtrans"]
                t_ok = np.isfinite(t_w) & np.isfinite(t_m)
                if t_ok.sum() > 10:
                    ax_tel.plot(t_w[t_ok], t_m[t_ok], color="forestgreen", lw=0.8,
                                alpha=0.75,
                                label="Telluric trans." if det == 0 else None)

        order_bounds.append((wl_min_order, wl_max_order))

        if not order_has_data:
            ax_top.set_visible(False); ax_res.set_visible(False); continue

        ax_top.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax_top.set_ylabel("Norm. flux", fontsize=9)
        ax_top.set_title(f"Order {order}", fontsize=10, pad=4)
        ax_top.legend(fontsize=8, loc="upper right", framealpha=0.7)
        ax_top.grid(alpha=0.15, lw=0.5)
        ax_top.tick_params(labelbottom=False, labelsize=8)
        ax_top.set_ylim(0.4, 2.2)

        ax_tel.set_ylim(0, 1)
        ax_tel.set_ylabel("Telluric trans.", fontsize=8, color="forestgreen")
        ax_tel.tick_params(axis="y", labelcolor="forestgreen", labelsize=7)
        ax_tel.legend(fontsize=7, loc="upper left", framealpha=0.6)

        ax_res.axhline(0.0, color="tomato", lw=1.0, ls="--")
        if order_resids:
            _sig3 = 3.0 * np.nanmedian(np.concatenate(order_errs))
            ax_res.axhline( _sig3, color='tomato', lw=0.8, ls=':', alpha=0.7, label='3σ')
            ax_res.axhline(-_sig3, color='tomato', lw=0.8, ls=':', alpha=0.7)
        ax_res.set_xlabel("Wavelength (nm)", fontsize=9)
        ax_res.set_ylabel("Resid.", fontsize=8)
        ax_res.tick_params(labelsize=8)
        ax_res.grid(alpha=0.15, lw=0.5)
        ax_res.set_ylim(-1, 1)
        if np.isfinite(wl_min_order) and np.isfinite(wl_max_order):
            ax_res.set_xlim(wl_min_order, wl_max_order)

    _method_note = (f'SavGol win={win} px' if method == 'savgol'
                    else f'block bin N={win} px')
    plt.suptitle(
        f"Binned data vs best-fit model  |  {retrieval_id}  |  night {night}\n"
        f"{_method_note}  ·  3 detectors per order  ·  per-order-detector normalised",
        fontsize=10, y=1.02,
    )
    plt.show()


def plot_model_vs_data_scatter(
    workpath: Path | str,
    retrieval_id: str,
    night: str,
    model_wave_npy: str = "retrieval_model_wave.npy",
    model_flux_npy: str = "retrieval_model_flux_scaled.npy",
    data_npy: str | None = None,
    err_npy: str | None = None,
    win: int | None = None,
    pol: int = 2,
    data_dir: Path | str | None = None,
    method: str = 'downsampling',
    normalize: bool = True,
) -> float:
    """
    Variant of plot_binned_model_vs_data (2026-07-10, for analysis notebook 4):
    DATA as scattered points, MODEL as a line, RESIDUALS as scattered points
    with the ±3σ(residual) threshold marked, reduced χ² printed.

    Differences from plot_binned_model_vs_data (everything else identical —
    same data loading, telluric overlay, per-order-detector normalisation,
    binning options):
      - data drawn as scatter points with ±1σ error bars (no connecting line);
        residual points carry the same error bars;
      - win=0 (or 1) disables binning entirely: raw valid pixels are plotted;
      - normalize=False skips the per-order-detector median normalisation and
        plots data and model in their native (absolute) flux units, so the
        continuum-level agreement is visible (for absolute-flux retrievals,
        e.g. 007PM-EQ-GP, whose model npys are in W m^-2 um^-1);
      - masked regions inside each chip (telluric-masked / clipped pixels) are
        shaded with gray vertical spans on both panels;
      - residual panel threshold is ±3 × std of the plotted (binned) residuals
        per order, instead of 3 × median propagated error;
      - reduced χ² = Σ((data−model)/σ)²/N is computed per order and overall
        from the UNBINNED valid pixels (binning affects display only), printed
        and annotated on each residual panel; the overall value is returned.
        NOTE: this is a diagonal-error χ² — the retrieval's stored χ² is
        r·C⁻¹·r/N under the GP covariance and is smaller by ≈(1+a²).

    For the two-night combined spectrum (combined_two_nights/, on the Night-2
    pixel grid) pass night='2023-01-01', the *_N2 model npys, and
    data_dir=workpath/'combined_two_nights' with its data/err filenames.
    """
    if win is None:
        win = 31 if method == 'savgol' else 10

    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    if data_npy is None:
        data_npy = "extracted_spectra_combined_sigmaclipper.npy"
    if err_npy is None:
        err_npy = "extracted_spectra_combined_err_sigmaclipper.npy"
    _data_dir = Path(data_dir) if data_dir is not None else workpath / night

    # Load model
    _mw_raw = np.load(retrieval_dir / model_wave_npy).flatten().astype(float)
    _mf_raw = np.load(retrieval_dir / model_flux_npy).flatten().astype(float)
    _fin    = np.isfinite(_mw_raw) & np.isfinite(_mf_raw)
    _isort  = np.argsort(_mw_raw[_fin])
    mw_s    = _mw_raw[_fin][_isort]
    mf_s    = _mf_raw[_fin][_isort]

    # Load wavelength grid, data, errors
    wave_fits = workpath / night / "cal" / "WLEN_K2166_V_DH_Tau_A+B_center.fits"
    with fits.open(wave_fits) as hdul:
        wave_cube = np.array(hdul[1].data)[:, 0:5, :].astype(float)

    data_flux = np.load(_data_dir / data_npy)
    err_data  = np.load(_data_dir / err_npy)
    data_cube = data_flux.reshape(wave_cube.shape).astype(float)

    # Load telluric transmission
    telluric_fits = (workpath / night / "chi_tau" / night /
                     "out" / "molecfit" / "TELLURIC_DATA.fits")
    with fits.open(telluric_fits) as hdul:
        t_raw = np.array(hdul[1].data)
    t = np.reshape(t_raw, (7, 3, 2048))
    t = np.array([t[i] for i in range(7)][::-1])
    telluric_cube = np.transpose(t, (1, 0, 2))

    n_det, n_order, n_pix = wave_cube.shape

    chi2_total, n_total = 0.0, 0

    fig = plt.figure(figsize=(20, 20))
    gs_outer = fig.add_gridspec(n_order, 1, hspace=0.05)

    for order in range(n_order):
        gs_inner = gs_outer[order].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.)
        ax_top = fig.add_subplot(gs_inner[0])
        ax_res = fig.add_subplot(gs_inner[1], sharex=ax_top)
        ax_tel = ax_top.twinx()

        order_has_data = False
        wl_min_order, wl_max_order = np.inf, -np.inf
        order_resids, order_fluxes = [], []
        order_chi2, order_n = 0.0, 0

        for det in range(n_det):
            w_full = wave_cube[det, order].astype(float)
            f_full = data_cube[det, order].astype(float)
            e_full = err_data[det, order].astype(float)

            valid = (np.isfinite(w_full) & np.isfinite(f_full) &
                     np.isfinite(e_full) & (e_full > 0))
            if valid.sum() < max(win, 10):
                continue

            f_vld = f_full[valid]; e_vld = e_full[valid]
            w_vld = w_full[valid]

            # --- gray spans over masked (telluric/clipped) regions inside the chip ---
            _inv = (~valid) & np.isfinite(w_full)
            _inv[:np.argmax(valid)] = False                      # skip leading edge crop
            _inv[len(valid) - np.argmax(valid[::-1]):] = False   # skip trailing edge crop
            if _inv.any():
                _idx    = np.where(_inv)[0]
                _splits = np.where(np.diff(_idx) > 1)[0] + 1
                for _g, _grp in enumerate(np.split(_idx, _splits)):
                    _lbl = ('masked' if (det == 0 and _g == 0) else None)
                    ax_top.axvspan(w_full[_grp[0]], w_full[_grp[-1]],
                                   color='gray', alpha=0.22, lw=0, zorder=0, label=_lbl)
                    ax_res.axvspan(w_full[_grp[0]], w_full[_grp[-1]],
                                   color='gray', alpha=0.22, lw=0, zorder=0)

            # --- chi^2 from UNBINNED pixels (normalisation as in the display) ---
            m_unb  = np.interp(w_vld, mw_s, mf_s)
            if normalize:
                f_med = np.nanmedian(f_vld) or 1.0
                m_med = np.nanmedian(m_unb) or 1.0
            else:
                f_med = m_med = 1.0
            _chip_r = f_vld / f_med - m_unb / m_med
            _chip_c = np.sum((_chip_r / (e_vld / f_med)) ** 2)
            order_chi2 += _chip_c; order_n += len(f_vld)
            chi2_total += _chip_c; n_total += len(f_vld)

            # --- binning for DISPLAY only ---
            if win <= 1:  # no binning: raw valid pixels
                w_bin     = w_vld
                f_bin_vld = f_vld
                e_bin_vld = e_vld
                m_interp  = m_unb
            elif method == 'savgol':
                w_bin     = w_vld
                f_bin_vld = savgol_filter(f_vld, win, pol)
                e_bin_vld = np.sqrt(savgol_filter(e_vld**2, win, pol)) / np.sqrt(win / 2)
                m_clip = (mw_s >= w_vld.min()) & (mw_s <= w_vld.max())
                if m_clip.sum() >= win:
                    m_sm     = savgol_filter(mf_s[m_clip], win, pol)
                    m_interp = np.interp(w_vld, mw_s[m_clip], m_sm)
                else:
                    m_interp = m_unb
            else:  # 'downsampling'
                n_bins = len(f_vld) // win
                if n_bins < 1:
                    continue
                sl = slice(0, n_bins * win)
                _eb       = e_vld[sl].reshape(n_bins, win)
                w_bin     = w_vld[sl].reshape(n_bins, win).mean(axis=1)
                f_bin_vld = f_vld[sl].reshape(n_bins, win).mean(axis=1)
                e_bin_vld = np.sqrt((_eb**2).mean(axis=1)) / np.sqrt(win)
                m_interp  = np.interp(w_bin, mw_s, mf_s)

            wl_min_order = min(wl_min_order, w_vld.min())
            wl_max_order = max(wl_max_order, w_vld.max())

            f_n_vld   = f_bin_vld / f_med
            m_n_vld   = m_interp  / m_med
            resid_vld = f_n_vld - m_n_vld

            lbl_data  = "Data"  if det == 0 else None
            lbl_model = "Model" if det == 0 else None
            lbl_err   = "±1σ"   if det == 0 else None

            e_n_vld = e_bin_vld / f_med
            ax_top.errorbar(w_bin, f_n_vld, yerr=e_n_vld,
                            fmt='none', ecolor='gray', elinewidth=1,
                            capsize=1, alpha=0.45, label=lbl_err, zorder=2)
            ax_top.scatter(w_bin, f_n_vld, s=20, color="steelblue", alpha=0.75,
                           linewidths=1, label=lbl_data, zorder=3)
            ax_top.plot(w_bin, f_n_vld, color="steelblue", lw=0.8, alpha=0.75, zorder=3)
            
            _w_gap, _m_gap = _insert_nan_at_gaps(w_bin, m_n_vld)
            ax_top.plot(_w_gap, _m_gap, color="tomato", lw=2.5, alpha=0.95,
                        label=lbl_model, zorder=5)
            ax_res.errorbar(w_bin, resid_vld, yerr=e_n_vld,
                            fmt='none', ecolor='gray', elinewidth=0.5,
                            capsize=0.5, alpha=0.4, zorder=2)
            ax_res.scatter(w_bin, resid_vld, s=6, color="steelblue",
                           alpha=0.7, linewidths=0, zorder=3)
            order_resids.append(resid_vld)
            order_fluxes.append(f_n_vld)
            order_has_data = True

            if order < telluric_cube.shape[1]:
                t_w  = telluric_cube[det, order]["lambda"] * 1e3
                t_m  = telluric_cube[det, order]["mtrans"]
                t_ok = np.isfinite(t_w) & np.isfinite(t_m)
                if t_ok.sum() > 10:
                    ax_tel.plot(t_w[t_ok], t_m[t_ok], color="forestgreen", lw=0.8,
                                alpha=0.75,
                                label="Telluric trans." if det == 0 else None)

        if not order_has_data:
            ax_top.set_visible(False); ax_res.set_visible(False); continue

        order_chi2_red = order_chi2 / order_n

        ax_top.set_title(f"Order {order}   (reduced χ² = {order_chi2_red:.3f}, unbinned)",
                         fontsize=10, pad=4)
        ax_top.legend(fontsize=8, loc="upper right", framealpha=0.7)
        ax_top.grid(alpha=0.15, lw=0.5)
        if normalize:
            ax_top.axhline(1.0, color="gray", lw=0.5, ls="--", alpha=0.5)
            ax_top.set_ylabel("Norm. flux", fontsize=9)
            ax_top.set_ylim(0.3, 2.2)
        else:
            ax_top.set_ylabel("Flux (W m$^{-2}$ μm$^{-1}$)", fontsize=9)
            _f_all = np.concatenate(order_fluxes)
            _lo, _hi = np.nanpercentile(_f_all, [0.5, 99.5])
            ax_top.set_ylim(max(0.0, _lo - 0.25 * (_hi - _lo)), _hi + 0.35 * (_hi - _lo))

        ax_tel.set_ylim(0, 1)
        ax_tel.set_ylabel("Telluric trans.", fontsize=8, color="forestgreen")
        ax_tel.tick_params(axis="y", labelcolor="forestgreen", labelsize=7)
        ax_tel.legend(fontsize=7, loc="upper left", framealpha=0.6)

        ax_res.axhline(0.0, color="tomato", lw=1.0, ls="--")
        _sig3 = 3.0 * np.nanstd(np.concatenate(order_resids))
        ax_res.axhline( _sig3, color='tomato', lw=0.8, ls=':', alpha=0.8,
                        label='±3σ (resid)')
        ax_res.axhline(-_sig3, color='tomato', lw=0.8, ls=':', alpha=0.8)
        ax_res.legend(fontsize=7, loc="upper right", framealpha=0.6)
        ax_res.set_xlabel("Wavelength (nm)", fontsize=9)
        ax_res.set_ylabel("Resid.", fontsize=8)
        ax_res.tick_params(labelsize=8)
        ax_res.grid(alpha=0.15, lw=0.5)
        _res_ymax = max(2.0 * _sig3, 0.15) if normalize else 2.0 * _sig3
        ax_res.set_ylim(-_res_ymax, _res_ymax)
        if np.isfinite(wl_min_order) and np.isfinite(wl_max_order):
            ax_res.set_xlim(wl_min_order, wl_max_order)

    chi2_red = chi2_total / n_total
    if win <= 1:
        _method_note = 'no binning (raw pixels)'
    else:
        _method_note = (f'SavGol win={win} px' if method == 'savgol'
                        else f'block bin N={win} px (display only)')
    _norm_note = ('per-order-detector normalised' if normalize
                  else 'absolute flux (no normalisation)')
    plt.suptitle(
        f"Data (points) vs best-fit model (line)  |  {retrieval_id}  |  night {night}\n"
        f"{_method_note}  ·  {_norm_note}  ·  "
        f"overall reduced χ² = {chi2_red:.4f} (unbinned, diagonal errors)",
        fontsize=10, y=1.02,
    )
    plt.show()
    print(f"night {night}: reduced chi^2 = {chi2_red:.4f}  ({n_total} unbinned pixels)")
    return chi2_red


# ---------------------------------------------------------------------------
# Direct data vs model for Ruffio (per-chip-median-normalised) retrievals
# ---------------------------------------------------------------------------

def _tel_order_for_chip(telluric_cube, det, wl_lo, wl_hi):
    """Return the telluric-cube order index whose median λ falls in [wl_lo, wl_hi] nm.

    Molecfit output order indices do not always align with K2166 order indices
    (depends on how many echelle orders the standard-star reduction covered).
    We match by wavelength instead of assuming a direct index correspondence.
    """
    for ti in range(telluric_cube.shape[1]):
        tw = telluric_cube[det, ti]['lambda'] * 1e3  # µm → nm
        tok = np.isfinite(tw) & (tw > 500)           # exclude zero-filled slots
        if tok.sum() < 10:
            continue
        if wl_lo <= np.median(tw[tok]) <= wl_hi:
            return ti
    return None


def _insert_nan_at_gaps(w, *arrays, gap_factor=5):
    """Break arrays at large wavelength gaps so matplotlib does not connect them.

    Returns (w_out, arr1_out, arr2_out, ...) with np.nan inserted wherever the
    wavelength step exceeds gap_factor × median step.
    """
    if len(w) < 2:
        return (w,) + arrays
    dw = np.diff(w)
    med = np.median(dw)
    if med <= 0:
        return (w,) + arrays
    gaps = np.where(dw > gap_factor * med)[0] + 1
    if len(gaps) == 0:
        return (w,) + arrays
    result = [np.insert(w.astype(float), gaps, np.nan)]
    for arr in arrays:
        result.append(np.insert(arr.astype(float), gaps, np.nan))
    return tuple(result)


def plot_ruffio_data_vs_model(
    data_wave,
    data_flux,
    data_err,
    model_flux_scaled,
    K2166,
    retrieval_id='',
    win=None,
    pol=2,
    ylim=(0.2, 1.8),
    workpath=None,
    night=None,
    method='downsampling',
) -> None:
    """Plot per-chip-median-normalised data vs φ-scaled model, 5 orders × 3 detectors.

    For Ruffio-style retrievals where the data is already per-chip-median
    normalised (~1.0) and model_flux_scaled is the best-fit φ-scaled model
    on the same wavelength grid — both are already in comparable units, so
    no additional re-normalisation is applied.

    Parameters
    ----------
    data_wave          : 1-D array, wavelengths (nm) on the valid-pixel grid.
    data_flux          : 1-D array, per-chip-median-normalised flux (~1.0).
    data_err           : 1-D array, per-chip-median-normalised errors.
    model_flux_scaled  : 1-D array, φ-scaled model on the same grid.
    K2166              : (5, 3, 2) array of chip [wl_min, wl_max] boundaries (nm).
    retrieval_id       : string used in the plot title.
    win                : Window size — SavGol window for method='savgol', bin size
                        in pixels for method='downsampling'. Defaults: savgol→21, down→10.
    pol                : SavGol polynomial order (ignored for downsampling).
    ylim               : y-axis limits for the flux panel.
    workpath           : root data path; if given together with night, load telluric.
    night              : observation night string, e.g. '2022-12-31'.
    method             : 'savgol' (Savitzky-Golay) or 'downsampling' (block average,
                        default). Downsampling has no smoothing artifacts.
    """
    from scipy.signal import savgol_filter as _savgol
    if win is None:
        win = 21 if method == 'savgol' else 10

    n_orders, n_dets = K2166.shape[:2]
    fig = plt.figure(figsize=(20, 20))
    gs_outer = fig.add_gridspec(n_orders, 1, hspace=0.05)

    # Optional telluric loading
    telluric_cube = None
    if workpath is not None and night is not None:
        _tel_fits = (Path(workpath) / night / 'chi_tau' / night /
                     'out' / 'molecfit' / 'TELLURIC_DATA.fits')
        if _tel_fits.exists():
            with fits.open(_tel_fits) as _hdul:
                _t_raw = np.array(_hdul[1].data)
            _t = np.reshape(_t_raw, (7, 3, 2048))
            _t = np.array([_t[i] for i in range(7)][::-1])
            telluric_cube = np.transpose(_t, (1, 0, 2))

    for order in range(n_orders):
        gs_inner = gs_outer[order].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.)
        ax_top = fig.add_subplot(gs_inner[0])
        ax_res = fig.add_subplot(gs_inner[1], sharex=ax_top)
        ax_tel = ax_top.twinx()  # always create; configured below only if telluric loaded

        order_has_data = False
        wl_min_order, wl_max_order = np.inf, -np.inf
        order_resids, order_errs = [], []
        tel_label_shown = False

        for det in range(n_dets):
            wl_lo, wl_hi = K2166[order, det]
            mask = (
                (data_wave >= wl_lo) & (data_wave <= wl_hi)
                & np.isfinite(data_flux) & np.isfinite(data_err)
                & np.isfinite(model_flux_scaled)
            )
            if mask.sum() < 5:
                continue

            w = data_wave[mask]
            f = data_flux[mask]
            e = data_err[mask]
            m = model_flux_scaled[mask]

            # drop any remaining non-finite pixels
            fin = np.isfinite(f) & np.isfinite(e) & np.isfinite(m)
            w, f, e, m = w[fin], f[fin], e[fin], m[fin]
            if len(w) < 5:
                continue

            if method == 'savgol':
                if win < len(w):
                    f_plot = _savgol(f, win, pol)
                    e_plot = np.sqrt(_savgol(e**2, win, pol)) / np.sqrt(win / 2)
                    m_plot = _savgol(m, win, pol)
                else:
                    f_plot, e_plot, m_plot = f.copy(), e.copy(), m.copy()
            else:  # 'downsampling'
                n_bins = len(w) // win
                if n_bins < 1:
                    continue
                sl = slice(0, n_bins * win)
                _fb = f[sl].reshape(n_bins, win)
                _eb = e[sl].reshape(n_bins, win)
                _mb = m[sl].reshape(n_bins, win)
                w      = w[sl].reshape(n_bins, win).mean(axis=1)
                f_plot = _fb.mean(axis=1)
                e_plot = np.sqrt((_eb**2).mean(axis=1)) / np.sqrt(win)
                m_plot = _mb.mean(axis=1)

            resid = f_plot - m_plot

            # break line plots at masked-pixel gaps so no straight bridge across gaps
            w_p, f_p, m_p = _insert_nan_at_gaps(w, f_plot, m_plot)

            lbl_d = 'Data'  if det == 0 else None
            lbl_m = 'Model' if det == 0 else None
            lbl_e = '±1σ'   if det == 0 else None

            ax_top.errorbar(w, f_plot, yerr=e_plot,
                            fmt='none', ecolor='gray', elinewidth=0.4,
                            capsize=0, alpha=0.5, label=lbl_e)
            ax_top.plot(w_p, f_p, color='steelblue', lw=1.0, alpha=0.85, label=lbl_d)
            ax_top.plot(w_p, m_p, color='tomato',    lw=1.9, alpha=0.95, label=lbl_m, zorder=5)
            ax_res.scatter(w, resid, s=2, color='steelblue', alpha=0.6, linewidths=0)
            order_resids.append(resid)
            order_errs.append(e_plot)

            # telluric: find the order in the cube whose λ matches this chip by wavelength
            if telluric_cube is not None:
                ti = _tel_order_for_chip(telluric_cube, det, wl_lo, wl_hi)
                if ti is not None:
                    _t_w = telluric_cube[det, ti]['lambda'] * 1e3
                    _t_m = telluric_cube[det, ti]['mtrans']
                    _t_ok = np.isfinite(_t_w) & np.isfinite(_t_m) & (_t_w > 500)
                    if _t_ok.sum() > 10:
                        _lbl_tel = 'Telluric trans.' if not tel_label_shown else None
                        ax_tel.plot(_t_w[_t_ok], _t_m[_t_ok], color='forestgreen', lw=0.8,
                                    alpha=0.75, label=_lbl_tel)
                        tel_label_shown = True

            # xlim from actual data wavelengths (mirrors plot_binned_model_vs_data)
            wl_min_order = min(wl_min_order, w.min())
            wl_max_order = max(wl_max_order, w.max())
            order_has_data = True

        if not order_has_data:
            ax_top.set_visible(False)
            ax_res.set_visible(False)
            ax_tel.set_visible(False)
            continue

        ax_top.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.5)
        ax_top.set_ylabel('Norm. flux', fontsize=9)
        ax_top.set_title(f'Order {order}', fontsize=10, pad=4)
        ax_top.legend(fontsize=8, loc='upper right', framealpha=0.7)
        ax_top.grid(alpha=0.15, lw=0.5)
        ax_top.tick_params(labelbottom=False, labelsize=8)
        ax_top.set_ylim(*ylim)

        if telluric_cube is not None:
            ax_tel.set_ylim(0, 1)
            ax_tel.set_ylabel('Telluric trans.', fontsize=8, color='forestgreen')
            ax_tel.tick_params(axis='y', labelcolor='forestgreen', labelsize=7)
            if tel_label_shown:
                ax_tel.legend(fontsize=7, loc='upper left', framealpha=0.6)
        else:
            ax_tel.set_visible(False)

        ax_res.axhline(0.0, color='tomato', lw=1.0, ls='--')
        if order_resids:
            _sig3 = 3.0 * np.nanmedian(np.concatenate(order_errs))
            ax_res.axhline( _sig3, color='tomato', lw=0.8, ls=':', alpha=0.7, label='3σ')
            ax_res.axhline(-_sig3, color='tomato', lw=0.8, ls=':', alpha=0.7)
        ax_res.set_xlabel('Wavelength (nm)', fontsize=9)
        ax_res.set_ylabel('Resid.', fontsize=8)
        ax_res.tick_params(labelsize=8)
        ax_res.grid(alpha=0.15, lw=0.5)
        ax_res.set_ylim(-0.5, 0.5)
        if np.isfinite(wl_min_order) and np.isfinite(wl_max_order):
            ax_res.set_xlim(wl_min_order, wl_max_order)

    _method_note = (f'SavGol win={win} px' if method == 'savgol'
                    else f'block bin N={win} px')
    plt.suptitle(
        f'Data vs best-fit model (Ruffio φ-scaled)  |  {retrieval_id}\n'
        f'{_method_note}  ·  per-chip-median-normalised data  ·  φ-scaled model',
        fontsize=10, y=1.02,
    )
    plt.show()


# ---------------------------------------------------------------------------
# P-T profile posterior
# ---------------------------------------------------------------------------

def plot_pt_profile(
    workpath: Path | str,
    retrieval_id: str,
    param_keys: list[str] | None = None,
    pressure: np.ndarray | None = None,
) -> None:
    """
    Plot 1σ/2σ/3σ P-T posterior bands.

    Works for both free-chemistry and equilibrium-chemistry retrievals as long
    as the nabla/T/P parameters are present in param_keys.

    Parameters
    ----------
    param_keys  Ordered list of retrieved parameter names (matches posterior columns).
                If None, inferred from final_params_dict.pickle using the standard
                non-retrieved exclusion sets.
    pressure    Pressure grid (bar).  Defaults to logspace(-5, 2, 50).
    """
    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    posterior = np.load(retrieval_dir / "final_posterior.npy")

    if param_keys is None:
        import pickle
        with open(retrieval_dir / "final_params_dict.pickle", "rb") as fh:
            params_dict = pickle.load(fh)
        non_retrieved = {"[C/H]", "[C/H]_xsolar", "C/O", "[Fe/H]",
                         "phi", "s2", "chi2", "phi_N2", "chi2_N2", "lnZ"}
        param_keys = [k for k in params_dict.keys() if k not in non_retrieved]
        param_keys = param_keys[: posterior.shape[1]]

    col = {k: i for i, k in enumerate(param_keys)}

    pt_params = {"log_P_RCE", "dlog_P_bot", "dlog_P_top", "T_bottom",
                 "nabla_RCE", "nabla_0", "nabla_1", "nabla_2", "nabla_3", "nabla_4", "nabla_5"}
    missing = pt_params.difference(col.keys())
    if missing:
        raise KeyError(f"param_keys is missing P-T parameters: {missing}")

    if pressure is None:
        pressure = np.logspace(-5, 2, 50)

    n_samples = posterior.shape[0]
    all_temps = np.empty((n_samples, len(pressure)))

    for i, s in enumerate(posterior):
        log_P_RCE  = s[col["log_P_RCE"]]
        dlog_P_bot = s[col["dlog_P_bot"]]
        dlog_P_top = s[col["dlog_P_top"]]
        T_bottom   = s[col["T_bottom"]]

        log_P_nodes = np.array([
            2.0,
            log_P_RCE + 2 * dlog_P_bot,
            log_P_RCE +     dlog_P_bot,
            log_P_RCE,
            log_P_RCE -     dlog_P_top,
            log_P_RCE - 2 * dlog_P_top,
            -5.0,
        ])
        nabla_nodes = np.array([
            s[col["nabla_0"]],
            s[col["nabla_1"]],
            s[col["nabla_2"]],
            s[col["nabla_RCE"]],
            s[col["nabla_3"]],
            s[col["nabla_4"]],
            s[col["nabla_5"]],
        ])

        log_P_atm    = np.log10(pressure)
        nabla_interp = np.interp(log_P_atm, log_P_nodes[::-1], nabla_nodes[::-1])

        temp     = np.empty(len(pressure))
        temp[-1] = T_bottom
        for j in range(len(pressure) - 2, -1, -1):
            temp[j] = temp[j + 1] * (pressure[j] / pressure[j + 1]) ** nabla_interp[j]
        all_temps[i] = temp

    pcts = np.percentile(all_temps, [0.27, 2.28, 15.87, 50.0, 84.13, 97.72, 99.73], axis=0)

    fig, ax = plt.subplots(figsize=(5, 7))
    color = "steelblue"
    ax.fill_betweenx(pressure, pcts[0], pcts[6], color=color, alpha=0.15, linewidth=0,
                     label="3σ (99.7%)")
    ax.fill_betweenx(pressure, pcts[1], pcts[5], color=color, alpha=0.30, linewidth=0,
                     label="2σ (95.4%)")
    ax.fill_betweenx(pressure, pcts[2], pcts[4], color=color, alpha=0.55, linewidth=0,
                     label="1σ (68.3%)")
    ax.plot(pcts[3], pressure, color="navy", lw=1.8, label="Median")

    ax.set_ylim(1e-5, 1e2)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (K)", fontsize=11)
    ax.set_ylabel("Pressure (bar)", fontsize=11)
    ax.set_title(f"P-T profile posterior\n{retrieval_id}", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2, lw=0.5)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.show()

    print(f"Posterior samples used: {n_samples}")
    print(f"Median T range: {pcts[3].min():.0f} – {pcts[3].max():.0f} K")


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------

_FREE_CHEM_PARAMS = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_g",
    "nabla_RCE", "nabla_0", "nabla_1", "nabla_2", "nabla_3", "nabla_4",
    "T_bottom", "log_P_RCE", "dlog_P_bot", "dlog_P_top",
    "log_H2O", "log_12CO", "log_13CO", "log_CH4",
]

_EQUIL_CHEM_PARAMS = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_g",
    "nabla_RCE", "nabla_0", "nabla_1", "nabla_2", "nabla_3", "nabla_4", "nabla_5",
    "T_bottom", "log_P_RCE", "dlog_P_bot", "dlog_P_top",
    "C_H", "C/O", "log_12CO_13CO",
]

# Piette & Madhusudhan (2020) / Xuan et al. (2024) P-T parameterisation.
# T_anchor at 0.2 bar (log P = -0.7) + 7 temperature increments dT_1...dT_7.
# Nodes (log₁₀ bar, deep→shallow): +0.7, 0.0, -0.3, -0.7*, -1.0, -1.5, -3.0, -5.0
# Pressure grid: 10⁻⁵–10¹ bar (50 layers), matching 003PM-EQ series. * = anchor
_PIETTE_EQUIL_PARAMS = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_g",
    "T_anchor", "dT_1", "dT_2", "dT_3", "dT_4", "dT_5", "dT_6", "dT_7",
    "C_H", "C/O", "log_12CO_13CO",
]

# v1.1+ series (004PM-EQ): log_g replaced by log_M + log_R as free parameters.
# log_g is a derived quantity: g = G*M/R² (cgs).
_PIETTE_EQUIL_PARAMS_MR = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_M", "log_R",
    "T_anchor", "dT_1", "dT_2", "dT_3", "dT_4", "dT_5", "dT_6", "dT_7",
    "C_H", "C/O", "log_12CO_13CO",
]


def plot_pt_profile_piette(
    workpath: Path | str,
    retrieval_id: str,
    param_keys: list[str] | None = None,
    pressure: np.ndarray | None = None,
) -> None:
    """Plot 1σ/2σ/3σ P-T posterior bands for the Piette+2020 parameterisation.

    Reconstructs T(P) from T_anchor (at 0.2 bar, log P = -0.7) + dT_1…dT_7
    via PCHIP spline over 8 nodes (Xuan+2024 Table C1, DH Tau b).
    Nodes (log₁₀ bar, deep→shallow): +0.7, 0.0, -0.3, -0.7*, -1.0, -1.5, -3.0, -5.0
    Matches Guidebook_GAStronomy_Piette_v1.0.py make_pt(). * = anchor
    """
    from scipy.interpolate import PchipInterpolator

    workpath = Path(workpath)
    retrieval_dir = workpath / "retrievals" / retrieval_id

    posterior = np.load(retrieval_dir / "final_posterior.npy")

    if param_keys is None:
        param_keys = _PIETTE_EQUIL_PARAMS

    col = {k: i for i, k in enumerate(param_keys)}

    # Fixed log10(P/bar) nodes, deep→shallow — Xuan+2024 Table C1 (DH Tau b)
    LOG_P_NODES = np.array([0.7, 0.0, -0.3, -0.7, -1.0, -1.5, -3.0, -5.0])
    ANCHOR_IDX  = 3   # log P = -0.7 → 0.2 bar

    if pressure is None:
        pressure = np.logspace(-5, 0.7, 200)  # PCHIP node range: 10⁻⁵–5 bar

    n_samples = posterior.shape[0]
    all_temps = np.empty((n_samples, len(pressure)))

    for i, s in enumerate(posterior):
        T_anchor = s[col["T_anchor"]]
        dT = np.array([s[col[f"dT_{j+1}"]] for j in range(7)])

        T_nodes = np.empty(8)
        T_nodes[ANCHOR_IDX] = T_anchor
        for j in range(ANCHOR_IDX - 1, -1, -1):
            T_nodes[j] = T_nodes[j + 1] + dT[j]
        for j in range(ANCHOR_IDX + 1, 8):
            T_nodes[j] = T_nodes[j - 1] - dT[j - 1]

        log_P_asc = LOG_P_NODES[::-1]   # −5.0 … +0.7 (ascending)
        T_asc     = T_nodes[::-1]
        pchip = PchipInterpolator(log_P_asc, T_asc)

        log_P_atm = np.log10(pressure)
        temp = pchip(log_P_atm)
        temp = np.where(log_P_atm < log_P_asc[0],  T_asc[0],  temp)
        temp = np.where(log_P_atm > log_P_asc[-1], T_asc[-1], temp)
        all_temps[i] = np.clip(temp, 1.0, 30000.0)

    pcts = np.percentile(all_temps, [0.27, 2.28, 15.87, 50.0, 84.13, 97.72, 99.73], axis=0)

    fig, ax = plt.subplots(figsize=(5, 7))
    color = "darkorange"
    ax.fill_betweenx(pressure, pcts[0], pcts[6], color=color, alpha=0.15, linewidth=0,
                     label="3σ (99.7%)")
    ax.fill_betweenx(pressure, pcts[1], pcts[5], color=color, alpha=0.30, linewidth=0,
                     label="2σ (95.4%)")
    ax.fill_betweenx(pressure, pcts[2], pcts[4], color=color, alpha=0.55, linewidth=0,
                     label="1σ (68.3%)")
    ax.plot(pcts[3], pressure, color="saddlebrown", lw=1.8, label="Median")

    # Mark PCHIP node pressures on the y-axis so kinks can be attributed
    # to specific nodes rather than appearing as PCHIP artefacts.
    node_pressures = 10 ** LOG_P_NODES   # bar
    node_T_median  = np.interp(np.log10(node_pressures[::-1]),
                               np.log10(pressure), pcts[3])[::-1]
    ax.scatter(node_T_median, node_pressures, s=30, color="k", zorder=5,
               label="PCHIP nodes")
    ax.scatter(node_T_median[ANCHOR_IDX], node_pressures[ANCHOR_IDX],
               s=60, color="red", zorder=6, label="T_anchor")

    ax.set_ylim(1e-5, 10**0.7)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (K)", fontsize=11)
    ax.set_ylabel("Pressure (bar)", fontsize=11)
    ax.set_title(f"P-T profile posterior (Piette+2020)\n{retrieval_id}", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2, lw=0.5)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.show()

    print(f"Posterior samples used: {n_samples}")
    print(f"Median T range: {pcts[3].min():.0f} – {pcts[3].max():.0f} K")


def run_free_chem_analysis(
    workpath: Path | str,
    retrieval_id: str,
    night1: str = "2022-12-31",
    night2: str | None = "2023-01-01",
    rvlag: np.ndarray | None = None,
    clean_grids: tuple = ((-10, -1), (0, 10)),
    combined_spectrum_name: str = "extracted_spectra_combined_sigmaclipper.npy",
    param_keys: list[str] | None = None,
) -> dict:
    """
    Full analysis pipeline for a free-chemistry retrieval.

    Runs (1) CCF workflow, (2) free-chemistry diagnostics,
    (3) night-1 binned model-vs-data, (4) optional night-2 binned model-vs-data,
    (5) P-T profile posterior.

    Returns a dict with 'ccf_out' and 'chem_out'.
    """
    workpath = Path(workpath)
    if rvlag is None:
        rvlag = np.arange(-10, 10, 0.1)

    # 1) CCF
    print("=== CCF workflow (night 1) ===")
    ccf_out = run_ccf_workflow(
        workpath=workpath,
        retrieval_id=retrieval_id,
        night=night1,
        combined_spectrum_name=combined_spectrum_name,
        rvlag=rvlag,
        clean_grids=clean_grids,
        n_shuffle=200,
        random_seed=42,
    )
    print(f"Valid pixels: {ccf_out['valid_pixels']}")
    print(f"Peak SNR = {ccf_out['peak_snr']:.2f} at RV = {ccf_out['peak_rv']:.1f} km/s")
    print(f"Null peak SNR mean±std = {ccf_out['null_mu']:.2f} ± {ccf_out['null_sigma']:.2f}")
    print(f"Detection z-score = {ccf_out['z_score']:.2f}")
    plot_ccf(ccf_out, retrieval_id)

    # 2) Free-chemistry diagnostics
    print("=== Free-chemistry diagnostics ===")
    chem_out = run_free_chemistry_workflow(workpath=workpath, retrieval_id=retrieval_id)
    print("Posterior columns:", chem_out["columns"])
    import pandas as pd; pd.set_option("display.float_format", "{:.4g}".format)
    print(chem_out["summary"].to_string(index=False))
    plot_free_chemistry(chem_out, retrieval_id)

    # 3) Night-1 model vs data
    print(f"=== Night-1 model vs data ({night1}) ===")
    plot_binned_model_vs_data(
        workpath=workpath,
        retrieval_id=retrieval_id,
        night=night1,
        model_wave_npy="retrieval_model_wave.npy",
        model_flux_npy="retrieval_model_flux_scaled.npy",
    )

    # 4) Night-2 model vs data (optional)
    if night2 is not None:
        n2_wave = workpath / "retrievals" / retrieval_id / "retrieval_model_wave_N2.npy"
        if n2_wave.exists():
            print(f"=== Night-2 model vs data ({night2}) ===")
            plot_binned_model_vs_data(
                workpath=workpath,
                retrieval_id=retrieval_id,
                night=night2,
                model_wave_npy="retrieval_model_wave_N2.npy",
                model_flux_npy="retrieval_model_flux_N2.npy",
            )

    # 5) P-T profile
    print("=== P-T profile posterior ===")
    plot_pt_profile(
        workpath=workpath,
        retrieval_id=retrieval_id,
        param_keys=param_keys if param_keys is not None else _FREE_CHEM_PARAMS,
    )

    return {"ccf_out": ccf_out, "chem_out": chem_out}


# v4.0 param lists: equilibrium chem + log_M/log_R + HF/Na/Ca/Ti atomic species
_PIETTE_EQUIL_PARAMS_MR_V4 = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_M", "log_R",
    "T_anchor", "dT_1", "dT_2", "dT_3", "dT_4", "dT_5", "dT_6", "dT_7",
    "C_H", "C/O", "log_12CO_13CO",
    "log_HF", "log_Na", "log_Ca", "log_Ti",
]

# v4.0 + EddySed clouds (MgSiO3+Fe); HF as [F/H] parameter
_PIETTE_EQUIL_PARAMS_MR_V4_CLD = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_M", "log_R",
    "T_anchor", "dT_1", "dT_2", "dT_3", "dT_4", "dT_5", "dT_6", "dT_7",
    "C_H", "C/O", "log_12CO_13CO",
    "F_H", "log_Na", "log_Ca", "log_Ti",
    "log_X_MgSiO3", "log_X_Fe", "fsed", "log_Kzz", "sigma_lnorm",
]

# v3.0: equilibrium chem + clouds, no atomic/HF
_PIETTE_EQUIL_PARAMS_MR_CLD = [
    "rv_N1", "rv_N2", "vsini", "epsilon", "log_M", "log_R",
    "T_anchor", "dT_1", "dT_2", "dT_3", "dT_4", "dT_5", "dT_6", "dT_7",
    "C_H", "C/O", "log_12CO_13CO",
    "log_X_MgSiO3", "log_X_Fe", "fsed", "log_Kzz", "sigma_lnorm",
]


# ---------------------------------------------------------------------------
# De Regt+2024 §4.2 CCF validation
# ---------------------------------------------------------------------------

_C_KMS = 2.99792458e5   # speed of light, km/s


def highpass_filter_chips(flux_3d, sigma_px=300):
    """Per-chip Gaussian high-pass filter (De Regt+2024 §4.2, σ = 300 px).

    For each (detector, order) chip, the Gaussian-smoothed continuum is
    subtracted.  NaN pixels are zeroed before smoothing and restored to 0.
    """
    from scipy.ndimage import gaussian_filter1d
    out = np.zeros_like(flux_3d)
    for d in range(flux_3d.shape[0]):
        for o in range(flux_3d.shape[1]):
            row    = flux_3d[d, o].copy()
            finite = np.isfinite(row)
            if finite.sum() < 10:
                continue
            tmp        = np.where(finite, row, 0.0)
            out[d, o]  = np.where(finite, tmp - gaussian_filter1d(tmp, sigma=sigma_px), 0.0)
    return out


def _hp_chip_single(arr, finite_mask, sigma_px):
    """HP-filter one chip row. Invalid pixels → 0."""
    from scipy.ndimage import gaussian_filter1d
    tmp = np.where(finite_mask, arr, 0.0)
    return np.where(finite_mask, tmp - gaussian_filter1d(tmp, sigma=sigma_px), 0.0)


def run_ccf_acf_deregt(wave_1d, flux_tmpl, obs_wave_3d, R_hp_3d, ivar_3d,
                        rvlag, sigma_px=300):
    """CCF and ACF following De Regt+2024 §4.2.

    CCF(rv) = Σ_chips  HP(M(rv)) × R_hp / σ²
    ACF(rv) = Σ_chips  HP(M(rv)) × HP(M(0)) / σ²

    HP is applied to M(rv) at each RV step so both CCF and ACF share the
    same frequency content as the HP-filtered data residual R_hp.

    Parameters
    ----------
    wave_1d     : 1-D model wavelength array (nm, sorted ascending)
    flux_tmpl   : 1-D spectral difference template (fiducial − no-X flux)
    obs_wave_3d : (nDet, nOrder, nPix) observed wavelength cube (nm)
    R_hp_3d     : (nDet, nOrder, nPix) HP-filtered data residual (obs − noX)
    ivar_3d     : (nDet, nOrder, nPix) inverse variance weights (1/σ²)
    rvlag       : 1-D RV lag array (km/s)
    sigma_px    : HP filter σ in pixels (default 300)
    """
    nDet, nOrder, _ = obs_wave_3d.shape

    chip_wave = [[obs_wave_3d[d, o]  for o in range(nOrder)] for d in range(nDet)]
    chip_fm   = [[np.isfinite(obs_wave_3d[d, o]) for o in range(nOrder)] for d in range(nDet)]
    chip_R    = [[np.where(np.isfinite(R_hp_3d[d, o]), R_hp_3d[d, o], 0.0)
                  for o in range(nOrder)] for d in range(nDet)]
    chip_iv   = [[ivar_3d[d, o] for o in range(nOrder)] for d in range(nDet)]
    chip_M0hp = [[_hp_chip_single(
                      np.interp(chip_wave[d][o], wave_1d, flux_tmpl, left=0., right=0.),
                      chip_fm[d][o], sigma_px)
                  for o in range(nOrder)] for d in range(nDet)]

    nrv = len(rvlag)
    ccf = np.zeros(nrv)
    acf = np.zeros(nrv)
    for i, rv in enumerate(rvlag):
        factor = 1.0 + rv / _C_KMS
        cc = ac = 0.0
        for d in range(nDet):
            for o in range(nOrder):
                w_eval  = chip_wave[d][o] / factor
                m       = np.interp(w_eval, wave_1d, flux_tmpl, left=0., right=0.)
                in_range = (w_eval >= wave_1d[0]) & (w_eval <= wave_1d[-1])
                mhp = _hp_chip_single(m, chip_fm[d][o] & in_range, sigma_px)
                wiv = mhp * chip_iv[d][o]
                cc += float(np.dot(wiv, chip_R[d][o]))
                ac += float(np.dot(wiv, chip_M0hp[d][o]))
        ccf[i] = cc
        acf[i] = ac
    return ccf, acf


def snr_deregt(rvlag, ccf, acf, wing_cutoff=400.0, edge_margin_kms=50.0):
    """SNR = CCF_peak / std(CCF in wings).

    Subtracts the median of each curve first to remove the DC offset that
    accumulates when summing dot-products across chips.  Then finds the actual
    CCF peak (argmax), aligns the ACF amplitude to match at that position, and
    estimates noise from the wings (|rv| > wing_cutoff km/s).

    When the template is negligible (ACF_max/CCF_max < 1e-4, e.g. equilibrium
    CH4 at 2100 K), alpha is set to 0 so the residual reduces to CCF itself.

    edge_margin_kms : exclude rv lags within this many km/s of the rvlag grid's
        own endpoints from the PEAK SEARCH only (added 2026-08-21; std_noise
        and template_fraction are unaffected, computed on the full grid
        exactly as before, so this cannot change any already-reported SNR
        value -- it only guards which lag gets reported as "the peak"). As
        `w_eval = chip_wave/(1+rv/c)` (run_ccf_acf_deregt) approaches the ends
        of a wide rvlag grid, it can shift far enough that template
        coverage/overlap changes non-stationarily chip-by-chip, which can
        produce a several-sigma ramp confined to just the last few RV steps
        -- confirmed for a weak-template species (Ca, job 430784, +/-1000
        km/s grid): CCF/sigma rose monotonically from ~0 to +3.04 over the
        final 24 km/s, immediately after a symmetric dip to -1.5, the
        signature of a finite-support boundary artifact rather than a real
        correlation peak. Set to 0.0 to restore the previous (unguarded)
        behaviour. Falls back to no exclusion (with a warning) if that would
        leave fewer than 10 interior points, so small/custom rvlag grids from
        other callers keep working.

    Returns (snr, peak_rv, peak_val, std_noise, residual, acf_aligned,
             template_fraction, edge_flagged).
      peak_rv      : rvlag at the CCF maximum (search excludes the edge margin).
      peak_val     : CCF value at the maximum.
      std_noise    : std of the CCF in the wings (|rv| > wing_cutoff km/s; full grid, unchanged).
      edge_flagged : True if the unrestricted global argmax fell inside the
                     excluded edge margin -- i.e. the naive peak would have
                     been a boundary artifact, not the reported one.
    """
    interior = (rvlag >= rvlag.min() + edge_margin_kms) & (rvlag <= rvlag.max() - edge_margin_kms)
    if interior.sum() < 10:
        print(f'  [snr_deregt] WARNING: edge_margin_kms={edge_margin_kms} leaves only '
              f'{interior.sum()} interior points on this rvlag grid -- disabling edge exclusion.')
        interior = np.ones_like(rvlag, dtype=bool)

    global_peak_idx = int(np.nanargmax(ccf))
    peak_idx   = int(np.nanargmax(np.where(interior, ccf, -np.inf)))
    edge_flagged = (peak_idx != global_peak_idx)
    peak_rv    = float(rvlag[peak_idx])
    peak_val   = float(ccf[peak_idx])
    acf_at_pk  = float(acf[peak_idx])
    acf_max    = float(np.nanmax(np.abs(acf)))
    ccf_max    = float(np.nanmax(np.abs(ccf)))
    template_fraction = acf_max / ccf_max if ccf_max > 0 else 0.0
    if (np.isfinite(acf_at_pk) and acf_at_pk != 0.0
            and ccf_max > 0 and acf_max / ccf_max > 1e-4):
        alpha = peak_val / acf_at_pk   # align ACF to CCF at the actual peak
    else:
        alpha = 0.0
    acf_aligned = acf * alpha
    residual    = ccf - acf_aligned
    wings       = np.abs(rvlag) > wing_cutoff
    std_noise   = float(np.nanstd(ccf[wings])) if wings.sum() > 20 else float(np.nanstd(ccf))
    snr         = peak_val / std_noise if std_noise > 0 else np.nan
    return snr, peak_rv, peak_val, std_noise, residual, acf_aligned, template_fraction, edge_flagged


def _resolve_mf_key(prt_name, mass_fractions):
    """Return the actual mass_fractions key for prt_name, handling __linelist suffixes.

    pRT3 may store species as '12C-16O__HITEMP' while the user specifies '12C-16O'.
    Tries exact match first, then looks for a key starting with prt_name + '__'.
    Returns None if no match is found.
    """
    if prt_name in mass_fractions:
        return prt_name
    for k in mass_fractions:
        if k.startswith(prt_name + '__'):
            return k
    return None


def make_noX_multi_template(retrieval, pRT_spectrum_class, flux_all, wave,
                             rv_key, species_keys, use_absolute_flux):
    """Return (flux_noX, flux_tmpl) with all species in species_keys zeroed simultaneously.

    species_keys must be the actual mass_fractions keys (already resolved).
    Used for combined templates (e.g. total CO = 12CO + 13CO).
    """
    prt_noX = pRT_spectrum_class(retrieval, spectral_resolution=100_000,
                                  use_absolute_flux=use_absolute_flux)
    for key in species_keys:
        if key not in prt_noX.mass_fractions:
            raise KeyError(f"species '{key}' not in mass_fractions")
        prt_noX.mass_fractions[key] = np.zeros(retrieval.n_atm_layers)
    prt_noX._prt_wl   = None
    prt_noX._prt_flux = None
    flux_noX  = prt_noX.make_spectrum(data_wave=wave, rv_key=rv_key)
    flux_tmpl = flux_all - flux_noX
    return flux_noX, flux_tmpl


def make_noX_template(retrieval, pRT_spectrum_class, flux_all, wave,
                       rv_key, species_key, use_absolute_flux):
    """Return (flux_noX, flux_tmpl_X) with mass fraction of species_key zeroed.

    Uses the spectral difference (fiducial − no-X) as the CCF template to
    avoid the continuum-level artefact that appears when running RT with only
    one species against an H2/He background.

    Parameters
    ----------
    retrieval          : Retrieval instance with best-fit params already set
    pRT_spectrum_class : pRT_spectrum class from the imported Guidebook module
    flux_all           : 1-D full model spectrum (model units)
    wave               : 1-D model wavelength array (nm)
    rv_key             : 'rv_N1' or 'rv_N2'
    species_key        : pRT3 mass-fraction key to zero (e.g. '1H2-16O')
    use_absolute_flux  : bool — passed to pRT_spectrum
    """
    prt_noX = pRT_spectrum_class(retrieval, spectral_resolution=100_000,
                                  use_absolute_flux=use_absolute_flux)
    if species_key not in prt_noX.mass_fractions:
        raise KeyError(f"species '{species_key}' not in mass_fractions")
    prt_noX.mass_fractions[species_key] = np.zeros(retrieval.n_atm_layers)
    prt_noX._prt_wl   = None
    prt_noX._prt_flux = None
    flux_noX  = prt_noX.make_spectrum(data_wave=wave, rv_key=rv_key)
    flux_tmpl = flux_all - flux_noX
    return flux_noX, flux_tmpl


def _plot_ccf_panel_deregt(rvlag, res, retrieval_label, retrieval_dir):
    """Per-species 2×2 CCF panel (De Regt+2024 Fig. 6 style)."""
    label       = res['label']
    ccf         = res['ccf']
    acf_aligned = res['acf_aligned']
    residual    = res['residual']
    snr, peak_rv, std_n = res['snr'], res['peak_rv'], res['std_noise']

    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.plot(rvlag, ccf,         color='steelblue', lw=0.8, label='CCF')
    ax1.plot(rvlag, acf_aligned, color='orange',    lw=0.8, ls='--', alpha=0.8,
             label=f'ACF (aligned at peak, rv={peak_rv:+.1f} km/s)')
    ax1.axvline(peak_rv, color='tomato', ls='-',  lw=1.0, alpha=0.8, label=f'CCF peak ({peak_rv:+.1f} km/s)')
    ax1.axvline(0.0,     color='k',     ls=':', lw=0.8, alpha=0.5, label='rv = 0')
    ax1.axhline(0, color='grey', lw=0.3)
    ax1.set_ylabel('CC'); ax1.legend(fontsize=7)
    ax1.set_title(f'{label} — CCF + ACF (aligned at peak)', fontsize=10)
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.plot(rvlag, residual, color='crimson', lw=0.8)
    ax2.axhline(0, color='grey', lw=0.3)
    ax2.axvline(peak_rv, color='tomato', ls='-',  lw=1.0, alpha=0.8)
    ax2.axvline(0.0,     color='k',     ls=':', lw=0.8, alpha=0.5)
    ax2.set_ylabel('CCF − ACF_aligned'); ax2.set_xlabel('RV lag (km/s)')
    ax2.set_title('CCF − ACF_aligned residuals', fontsize=10)

    ccf_norm = ccf / std_n if (std_n and std_n > 0) else ccf
    ax3.plot(rvlag, ccf_norm, color='steelblue', lw=0.8)
    if snr is not None and not np.isnan(snr):
        ax3.axhline(snr, color='tomato', lw=0.7, ls='--', alpha=0.7,
                    label=f'SNR = {snr:.2f}  (peak)')
    ax3.axvline(peak_rv, color='tomato', ls='-',  lw=1.0, alpha=0.8,
                label=f'CCF peak ({peak_rv:+.1f} km/s)')
    ax3.axvline(0.0,     color='k',     ls=':', lw=0.8, alpha=0.5,
                label='rv = 0')
    ax3.axhline(0, color='grey', lw=0.3)
    ax3.axhline( 3, color='grey', lw=0.6, ls='--', alpha=0.4)
    ax3.axhline(-3, color='grey', lw=0.6, ls='--', alpha=0.4)
    ax3.set_ylabel('CCF / σ_wings'); ax3.set_xlabel('RV lag (km/s)')
    ax3.set_title(f'CCF / σ  (σ = std of CCF wings, |rv|>400 km/s)\nSNR = {snr:.2f} at peak ({peak_rv:+.1f} km/s)', fontsize=10)
    ax3.legend(fontsize=7)

    fig.suptitle(f'{retrieval_label} — {label}  (De Regt+2024 §4.2)', fontsize=12)
    fname = Path(retrieval_dir) / f'validation_deregt_{label}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'  Saved: {fname.name}')


def _plot_snr_summary_deregt(ccf_results, retrieval_label, retrieval_dir,
                              min_template_fraction=1e-4):
    """Bar chart of CCF SNR for detected species only.

    Species whose template_fraction (ACF_max / CCF_max) is below
    min_template_fraction are excluded — their template is too weak to
    produce a meaningful cross-correlation and their CCF peak is spurious.
    """
    valid = {k: r for k, r in ccf_results.items()
             if r.get('template_fraction', 1.0) >= min_template_fraction}
    excluded = {k: r for k, r in ccf_results.items() if k not in valid}
    for k, r in excluded.items():
        print(f'  [{r["label"]}] excluded from SNR plot: '
              f'template_fraction = {r["template_fraction"]:.1e} < {min_template_fraction:.0e}')

    labels_p = [r['label']   for r in valid.values()]
    snr_vals = [r['snr']     for r in valid.values()]
    peak_rvs = [r['peak_rv'] for r in valid.values()]

    palette = ['steelblue', 'tomato', 'darkorange', 'mediumseagreen',
               'slategray', 'mediumpurple', 'peru', 'teal', 'crimson', 'olive']
    colors  = palette[:len(labels_p)]
    x       = np.arange(len(labels_p))

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(labels_p)), 5))
    bars = ax.bar(x, snr_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar_label(bars,
                 labels=[f'{s:.1f}' if not np.isnan(s) else 'NaN' for s in snr_vals],
                 padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels_p, fontsize=11)
    ax.set_ylabel('Detection SNR (De Regt+2024 §4.2)')
    ax.set_title(f'All-Species CCF SNR — {retrieval_label}\n'
                 '(CCF peak / std of CCF wings, |rv|>400 km/s)')
    ax.axhline(3, ls='--', color='grey', lw=0.9, label='SNR = 3')
    ax.axhline(5, ls=':',  color='grey', lw=0.9, label='SNR = 5')
    valid_snr = [s for s in snr_vals if not np.isnan(s)]
    ax.set_ylim(bottom=min(0, min(valid_snr) - 1) if valid_snr else 0)
    for xi, (snr, prv) in enumerate(zip(snr_vals, peak_rvs)):
        ax.text(xi, -0.3, f'{prv:+.1f} km/s', ha='center', va='top',
                fontsize=7, color='navy')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fname = Path(retrieval_dir) / 'validation_deregt_snr_summary.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'  Saved: {fname.name}')


def run_species_ccf_validation(
    retrieval,
    pRT_spectrum_class,
    best_fit_params,
    wave_model,
    obs_flux_N1, err_N1, obs_wave_N1,
    obs_flux_N2, err_N2, obs_wave_N2,
    TRACE_SPECIES,
    retrieval_dir,
    rvlag=None,
    rv_key_N1='rv_N1',
    rv_key_N2='rv_N2',
    use_absolute_flux=True,
    sigma_px=300,
    min_template_fraction=1e-4,
    retrieval_label='',
    vcorr=(0.0, 0.0),
    model_scale_factor=1.0,
):
    """Run De Regt+2024 §4.2 CCF species validation for one retrieval.

    Applies per-chip Gaussian high-pass filtering, generates fiducial and
    no-X templates for each species in TRACE_SPECIES, cross-correlates against
    both nights, and saves PNG panels + SNR summary to retrieval_dir.

    Parameters
    ----------
    retrieval          : Retrieval instance; best-fit params must be set before calling.
    pRT_spectrum_class : pRT_spectrum class from the Guidebook module used for this run.
    best_fit_params    : dict of best-fit parameter values (from final_params_dict.pickle).
    wave_model         : 1-D model wavelength array (nm), e.g. from retrieval_model_wave.npy.
    obs_flux_N1/N2     : (nDet, nOrder, nPix) observed flux cubes (W m⁻² µm⁻¹).
    err_N1/N2          : corresponding error cubes.
    obs_wave_N1/N2     : wavelength cubes (nm) in the same frame as WLEN FITS (instrument frame).
    TRACE_SPECIES      : dict {pRT3_species_key: display_label} of species to validate.
    retrieval_dir      : directory (str or Path) where PNGs are saved.
    rvlag              : 1-D RV lag array (km/s). Default: ±1000 km/s at 1 km/s steps.
    rv_key_N1/N2       : parameter name for per-night RV.
    use_absolute_flux  : passed to pRT_spectrum.
    sigma_px               : Gaussian HP filter σ (pixels).
    min_template_fraction  : species with ACF_max/CCF_max below this are excluded from the
                             SNR summary plot (template too weak for a meaningful CCF).
                             Default 1e-4. Example: equilibrium CH4 at 2100 K gives ~1e-12.
    retrieval_label        : string used in plot titles and filenames.
    vcorr              : (vcorr_N1, vcorr_N2) optional velocity offset (km/s) added to
                         rv_N1/rv_N2 before template generation. Default (0, 0) — no
                         correction needed because the Guidebook forward model and the
                         WLEN FITS chip wavelengths are in the same instrument frame;
                         rv_N1/rv_N2 already encode the correct shift to match the data.
    model_scale_factor : multiply all model / template flux arrays by this factor before
                         computing the CCF residual. Use for Ruffio retrievals where pRT
                         flux (~10^7) and per-chip-median-normalised data (~1.0) are in
                         different units. Compute as
                           np.nanmedian(flux_scaled) / np.nanmedian(flux_raw)
                         from the retrieval output npy files. Default 1.0 (no scaling).

    Returns
    -------
    dict with keys 'templates', 'ccf_results', 'flux_all', 'wave'.
    """
    retrieval_dir = Path(retrieval_dir)
    if rvlag is None:
        rvlag = np.arange(-1000.0, 1001.0, 1.0)

    # Sort wave_model ascending — required so np.interp gives correct results when
    # evaluating the Doppler-shifted template at observed chip wavelengths.
    # retrieval_model_wave.npy is ordered by (order, det, pix) with wavelength
    # decreasing across order boundaries (4 backward jumps), so it must be sorted
    # before being used as the x-axis in np.interp.
    _sort_idx  = np.argsort(wave_model)
    wave_model = wave_model[_sort_idx]
    if not np.all(_sort_idx == np.arange(len(wave_model))):
        print(f'  wave_model sorted ascending ({len(wave_model)} pts, '
              f'[{wave_model[0]:.1f}, {wave_model[-1]:.1f}] nm)')

    # Planet RV in the instrument/chip-wavelength frame (+ optional offset via vcorr)
    rv_N1_val = float(best_fit_params[rv_key_N1]) + vcorr[0]
    rv_N2_val = float(best_fit_params[rv_key_N2]) + vcorr[1]
    print(f'  Planet RV: N1 = {rv_N1_val:+.3f} km/s  N2 = {rv_N2_val:+.3f} km/s')

    # Generate templates at rv=0 (planet rest frame).
    # Dividing chip wavelengths by (1+rv/c) brings them to the same rest frame, so
    # the CCF rv_lag axis is centred on the planet: peak at rv_lag=0 when the template
    # lines align with the data.
    params_rest = dict(best_fit_params)
    params_rest[rv_key_N1] = 0.0
    params_rest[rv_key_N2] = 0.0
    retrieval.parameters.params.update(params_rest)

    # 1. Generate full model spectrum at rv=0 (planet rest frame)
    print('  Generating full model spectrum (Spectrum 1)...')
    prt_full = pRT_spectrum_class(retrieval, spectral_resolution=100_000,
                                   use_absolute_flux=use_absolute_flux)
    flux_all = prt_full.make_spectrum(data_wave=wave_model, rv_key=rv_key_N1)
    print(f'  flux_all: [{np.nanmin(flux_all):.2e}, {np.nanmax(flux_all):.2e}]')

    # Chip wavelengths shifted to planet rest frame
    obs_wave_N1_rest = obs_wave_N1 / (1.0 + rv_N1_val / _C_KMS)
    obs_wave_N2_rest = obs_wave_N2 / (1.0 + rv_N2_val / _C_KMS)

    # Inverse variance per night (zero where data is invalid)
    valid_N1 = np.isfinite(obs_wave_N1_rest)
    valid_N2 = np.isfinite(obs_wave_N2_rest)
    ivar_N1  = np.where(np.isfinite(err_N1) & (err_N1 > 0), 1.0 / err_N1**2, 0.0)
    ivar_N1[~np.isfinite(obs_flux_N1)] = 0.0
    ivar_N2  = np.where(np.isfinite(err_N2) & (err_N2 > 0), 1.0 / err_N2**2, 0.0)
    ivar_N2[~np.isfinite(obs_flux_N2)] = 0.0

    # 2. Generate no-X templates for each species
    print('  Generating no-X templates...')
    templates = {}
    for prt_name, label in TRACE_SPECIES.items():
        resolved = _resolve_mf_key(prt_name, prt_full.mass_fractions)
        if resolved is None:
            print(f'    [{label}] not in mass_fractions — skipping')
            continue
        if resolved != prt_name:
            print(f'    [{label}] key resolved: {prt_name!r} → {resolved!r}')
        try:
            flux_noX, flux_tmpl = make_noX_template(
                retrieval, pRT_spectrum_class, flux_all, wave_model,
                rv_key_N1, resolved, use_absolute_flux)
            if model_scale_factor != 1.0:
                flux_noX  = flux_noX  * model_scale_factor
                flux_tmpl = flux_tmpl * model_scale_factor
            templates[prt_name] = dict(label=label, flux_noX=flux_noX, flux_tmpl=flux_tmpl,
                                        mf_key=resolved)
            print(f'    [{label}] template RMS = {np.nanstd(flux_tmpl):.2e}')
        except Exception as exc:
            print(f'    [{label}] ERROR: {exc}')

    # Add total CO (12CO + 13CO zeroed simultaneously) if both isotopologues are present
    _CO12, _CO13 = '12C-16O', '13C-16O'
    if _CO12 in templates and _CO13 in templates:
        print('    [totalCO] building combined 12CO+13CO template...')
        try:
            co12_key = templates[_CO12]['mf_key']
            co13_key = templates[_CO13]['mf_key']
            flux_noX_co, flux_tmpl_co = make_noX_multi_template(
                retrieval, pRT_spectrum_class, flux_all, wave_model,
                rv_key_N1, [co12_key, co13_key], use_absolute_flux)
            if model_scale_factor != 1.0:
                flux_noX_co  = flux_noX_co  * model_scale_factor
                flux_tmpl_co = flux_tmpl_co * model_scale_factor
            templates['total_CO'] = dict(label='totalCO',
                                         flux_noX=flux_noX_co, flux_tmpl=flux_tmpl_co,
                                         mf_key=None)
            print(f'    [totalCO] template RMS = {np.nanstd(flux_tmpl_co):.2e}')
        except Exception as exc:
            print(f'    [totalCO] ERROR: {exc}')

    # 3. CCF per species over both nights
    print('  Running CCF (±1000 km/s, both nights)...')
    ccf_results = {}
    for prt_name, tmpl in templates.items():
        label, flux_noX, flux_tmpl = tmpl['label'], tmpl['flux_noX'], tmpl['flux_tmpl']
        if np.nanstd(flux_tmpl) < 1e-22:
            print(f'    [{label}] template negligible — skipping')
            continue
        print(f'    [{label}]...', flush=True)

        noX_3d_N1 = np.zeros_like(obs_wave_N1_rest)
        noX_3d_N1[valid_N1] = np.interp(obs_wave_N1_rest[valid_N1], wave_model, flux_noX,
                                          left=0., right=0.)
        R_hp_N1 = highpass_filter_chips(obs_flux_N1 - noX_3d_N1, sigma_px)
        ccf_N1, acf_N1 = run_ccf_acf_deregt(
            wave_model, flux_tmpl, obs_wave_N1_rest, R_hp_N1, ivar_N1, rvlag, sigma_px)

        noX_3d_N2 = np.zeros_like(obs_wave_N2_rest)
        noX_3d_N2[valid_N2] = np.interp(obs_wave_N2_rest[valid_N2], wave_model, flux_noX,
                                          left=0., right=0.)
        R_hp_N2 = highpass_filter_chips(obs_flux_N2 - noX_3d_N2, sigma_px)
        ccf_N2, acf_N2 = run_ccf_acf_deregt(
            wave_model, flux_tmpl, obs_wave_N2_rest, R_hp_N2, ivar_N2, rvlag, sigma_px)

        ccf_raw = ccf_N1 + ccf_N2
        acf = acf_N1 + acf_N2
        ccf = ccf_raw - np.nanmedian(ccf_raw)
        acf = acf    - np.nanmedian(acf)
        snr, peak_rv, peak_val, std_noise, residual, acf_aligned, template_fraction, edge_flagged = \
            snr_deregt(rvlag, ccf, acf)
        ccf_results[prt_name] = dict(
            label=label, ccf=ccf, acf=acf, acf_aligned=acf_aligned,
            residual=residual,
            snr=snr, peak_rv=peak_rv, peak_val=peak_val, std_noise=std_noise,
            template_fraction=template_fraction, edge_flagged=edge_flagged,
            ccf_raw=ccf_raw, ccf_N1=ccf_N1, ccf_N2=ccf_N2,
        )
        flag = '' if template_fraction >= min_template_fraction else '  [template below sensitivity]'
        edge_note = '  [edge-margin artifact excluded]' if edge_flagged else ''
        print(f'      peak CCF = {peak_val:.3e}   peak_rv = {peak_rv:+.1f} km/s   '
              f'SNR = {snr:.2f}   template_fraction = {template_fraction:.1e}{flag}{edge_note}')

    # 4. Per-species plots (skip species whose template is below sensitivity threshold)
    print('  Saving per-species CCF panels...')
    for res in ccf_results.values():
        if res.get('template_fraction', 1.0) < min_template_fraction:
            print(f"    [{res['label']}] template below sensitivity — skipping panel")
            continue
        _plot_ccf_panel_deregt(rvlag, res, retrieval_label, retrieval_dir)

    # 5. SNR summary (species below sensitivity threshold excluded)
    _plot_snr_summary_deregt(ccf_results, retrieval_label, retrieval_dir,
                              min_template_fraction=min_template_fraction)
    print('  CCF validation complete.')
    return {'templates': templates, 'ccf_results': ccf_results,
            'flux_all': flux_all, 'wave': wave_model}


def run_equil_chem_analysis(
    workpath: Path | str,
    retrieval_id: str,
    night1: str = "2022-12-31",
    night2: str | None = "2023-01-01",
    rvlag: np.ndarray | None = None,
    clean_grids: tuple = ((-10, -1), (0, 10)),
    combined_spectrum_name: str = "extracted_spectra_combined_sigmaclipper.npy",
    param_keys: list[str] | None = None,
) -> dict:
    """
    Full analysis pipeline for an equilibrium-chemistry retrieval.

    Runs (1) CCF workflow, (2) equil-chemistry diagnostics,
    (3) night-1 binned model-vs-data, (4) optional night-2 binned model-vs-data,
    (5) P-T profile posterior.

    Returns a dict with 'ccf_out' and 'chem_out'.
    """
    workpath = Path(workpath)
    if rvlag is None:
        rvlag = np.arange(-10, 10, 0.1)

    # 1) CCF
    print("=== CCF workflow (night 1) ===")
    ccf_out = run_ccf_workflow(
        workpath=workpath,
        retrieval_id=retrieval_id,
        night=night1,
        combined_spectrum_name=combined_spectrum_name,
        rvlag=rvlag,
        clean_grids=clean_grids,
        n_shuffle=200,
        random_seed=42,
    )
    print(f"Valid pixels: {ccf_out['valid_pixels']}")
    print(f"Peak SNR = {ccf_out['peak_snr']:.2f} at RV = {ccf_out['peak_rv']:.1f} km/s")
    print(f"Null peak SNR mean±std = {ccf_out['null_mu']:.2f} ± {ccf_out['null_sigma']:.2f}")
    print(f"Detection z-score = {ccf_out['z_score']:.2f}")
    plot_ccf(ccf_out, retrieval_id)

    # 2) Equil-chemistry diagnostics
    print("=== Equilibrium-chemistry diagnostics ===")
    chem_out = run_equil_chemistry_workflow(workpath=workpath, retrieval_id=retrieval_id)
    print("Posterior columns:", chem_out["columns"])
    import pandas as pd; pd.set_option("display.float_format", "{:.4g}".format)
    print(chem_out["summary"].to_string(index=False))
    plot_equil_chemistry(chem_out, retrieval_id)

    # 3) Night-1 model vs data
    print(f"=== Night-1 model vs data ({night1}) ===")
    plot_binned_model_vs_data(
        workpath=workpath,
        retrieval_id=retrieval_id,
        night=night1,
        model_wave_npy="retrieval_model_wave.npy",
        model_flux_npy="retrieval_model_flux_scaled.npy",
    )

    # 4) Night-2 model vs data (optional)
    if night2 is not None:
        n2_wave = workpath / "retrievals" / retrieval_id / "retrieval_model_wave_N2.npy"
        if n2_wave.exists():
            print(f"=== Night-2 model vs data ({night2}) ===")
            plot_binned_model_vs_data(
                workpath=workpath,
                retrieval_id=retrieval_id,
                night=night2,
                model_wave_npy="retrieval_model_wave_N2.npy",
                model_flux_npy="retrieval_model_flux_N2.npy",
            )

    # 5) P-T profile
    print("=== P-T profile posterior ===")
    plot_pt_profile(
        workpath=workpath,
        retrieval_id=retrieval_id,
        param_keys=param_keys if param_keys is not None else _EQUIL_CHEM_PARAMS,
    )

    return {"ccf_out": ccf_out, "chem_out": chem_out}


# ============================================================
# 8. FORMATION SCENARIO: Sonora Bobcat PT overlays
# ============================================================

# Three (Teff, g_MKS) grid points representative of hot/warm/cold formation starts
# for DH Tau B (~11-13 MJup, ~1-2 Myr).  Entropy values follow Marleau & Cumming
# (2014) and Mordasini et al. (2012): hot ≈ 13 kB/baryon, warm ≈ 10.5, cold ≈ 8.5.
# Sonora Bobcat (Marley+2021) gravity in MKS (m s⁻²); log_g (cgs) = log10(g × 100).
_BOBCAT_FORMATION_SCENARIOS = {
    "hot":  {"s_init": 13.0, "teff": 2400, "g_mks": 31,  "label": "Hot start\n$S_\\mathrm{init}\\approx13\\,k_\\mathrm{B}/\\mathrm{baryon}$"},
    "warm": {"s_init": 10.5, "teff": 2000, "g_mks": 56,  "label": "Warm start\n$S_\\mathrm{init}\\approx10.5\\,k_\\mathrm{B}/\\mathrm{baryon}$"},
    "cold": {"s_init":  8.5, "teff": 1600, "g_mks": 178, "label": "Cold start\n$S_\\mathrm{init}\\approx8.5\\,k_\\mathrm{B}/\\mathrm{baryon}$"},
}

_BOBCAT_COLORS = {"hot": "#d62728", "warm": "#ff7f0e", "cold": "#1f77b4"}
_BOBCAT_TARBALL = Path("/data2/peng/sonora_bobcat_pt/structures_m+0.0.tar.gz")


def _bobcat_filename(teff: int, g_mks: int) -> str:
    """Return the filename inside the Sonora Bobcat tarball for given Teff and g (MKS)."""
    # g=10 files have a '+' suffix; all others do not
    suffix = "m+0.0" if g_mks == 10 else "m0.0"
    return f"t{teff}g{g_mks}nc_{suffix}.dat"


def _read_bobcat_pt(teff: int, g_mks: int, tarball: Path = _BOBCAT_TARBALL):
    """Extract and parse a Sonora Bobcat structure file from the tarball.

    Returns (pressure_bar, temperature_K) as 1-D numpy arrays, sorted
    from high pressure (deep) to low pressure (top of atmosphere).
    """
    import tarfile
    fname = _bobcat_filename(teff, g_mks)
    with tarfile.open(tarball, "r:gz") as tf:
        member = tf.getmember(fname)
        fobj   = tf.extractfile(member)
        lines  = fobj.read().decode().splitlines()

    pressure, temperature = [], []
    for line in lines[1:]:   # skip header
        parts = line.split()
        if len(parts) < 3:
            continue
        pressure.append(float(parts[1]))
        temperature.append(float(parts[2]))
    return np.array(pressure), np.array(temperature)


def plot_PT_formation_scenario(
    workpath: Path | str,
    retrieval_id: str,
    param_keys: list[str] | None = None,
    pressure_retrieval: np.ndarray | None = None,
    tarball: Path | str = _BOBCAT_TARBALL,
    scenarios: dict | None = None,
    figsize: tuple = (5.5, 7),
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Overlay Sonora Bobcat hot/warm/cold-start PT profiles on the retrieval posterior.

    The retrieval posterior bands (Piette+2020 parameterisation) are drawn first
    in dark-orange, then three Sonora Bobcat structure profiles are overlaid to
    show where DH Tau B sits relative to expected formation scenarios at ~1-2 Myr.

    Parameters
    ----------
    workpath       : Root directory of the project (contains 'retrievals/').
    retrieval_id   : Sub-directory name of the retrieval run (e.g. '3062330_N600_...').
    param_keys     : Ordered list of posterior parameter names.  Defaults to
                     _PIETTE_EQUIL_PARAMS_MR (log_M + log_R variant).
    pressure_retrieval : Pressure grid for reconstructing the retrieval posterior.
                     Defaults to logspace(-5, 0.7, 200) matching make_pt().
    tarball        : Path to the Sonora Bobcat structures_m+0.0.tar.gz archive.
    scenarios      : Override the default _BOBCAT_FORMATION_SCENARIOS dict.
    figsize        : Figure size in inches.
    save_path      : If given, save the figure to this path (PDF/PNG).

    Returns
    -------
    fig : matplotlib Figure
    """
    from scipy.interpolate import PchipInterpolator

    workpath     = Path(workpath)
    tarball      = Path(tarball)
    retrieval_dir = workpath / "retrievals" / retrieval_id
    scenarios    = scenarios or _BOBCAT_FORMATION_SCENARIOS

    # ---- Load retrieval posterior ----
    posterior = np.load(retrieval_dir / "final_posterior.npy")

    if param_keys is None:
        param_keys = _PIETTE_EQUIL_PARAMS_MR

    col = {k: i for i, k in enumerate(param_keys)}

    # Fixed PCHIP nodes (log10 bar, deep → shallow) — Xuan+2024 Table C1
    LOG_P_NODES = np.array([0.7, 0.0, -0.3, -0.7, -1.0, -1.5, -3.0, -5.0])
    ANCHOR_IDX  = 3   # log P = -0.7 → 0.2 bar

    if pressure_retrieval is None:
        pressure_retrieval = np.logspace(-5, 0.7, 200)

    n_samples  = posterior.shape[0]
    all_temps  = np.empty((n_samples, len(pressure_retrieval)))

    for i, s in enumerate(posterior):
        T_anchor = s[col["T_anchor"]]
        dT       = np.array([s[col[f"dT_{j+1}"]] for j in range(7)])

        T_nodes               = np.empty(8)
        T_nodes[ANCHOR_IDX]  = T_anchor
        for j in range(ANCHOR_IDX - 1, -1, -1):
            T_nodes[j] = T_nodes[j + 1] + dT[j]
        for j in range(ANCHOR_IDX + 1, 8):
            T_nodes[j] = T_nodes[j - 1] - dT[j - 1]

        log_P_asc = LOG_P_NODES[::-1]
        T_asc     = T_nodes[::-1]
        pchip     = PchipInterpolator(log_P_asc, T_asc)

        log_P_atm      = np.log10(pressure_retrieval)
        temp           = pchip(log_P_atm)
        temp           = np.where(log_P_atm < log_P_asc[0],  T_asc[0],  temp)
        temp           = np.where(log_P_atm > log_P_asc[-1], T_asc[-1], temp)
        all_temps[i]   = np.clip(temp, 1.0, 30000.0)

    pcts = np.percentile(all_temps, [0.27, 2.28, 15.87, 50.0, 84.13, 97.72, 99.73], axis=0)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=figsize)

    # Retrieval posterior bands
    ret_color = "darkorange"
    ax.fill_betweenx(pressure_retrieval, pcts[0], pcts[6],
                     color=ret_color, alpha=0.12, linewidth=0)
    ax.fill_betweenx(pressure_retrieval, pcts[1], pcts[5],
                     color=ret_color, alpha=0.28, linewidth=0)
    ax.fill_betweenx(pressure_retrieval, pcts[2], pcts[4],
                     color=ret_color, alpha=0.55, linewidth=0)
    ax.plot(pcts[3], pressure_retrieval,
            color="saddlebrown", lw=2.0, label=f"Retrieval {retrieval_id.split('_')[0]}\n(Piette+2020 median)")

    # PCHIP node markers on the median
    node_pressures  = 10 ** LOG_P_NODES
    mask_range      = (node_pressures >= pressure_retrieval.min()) & (node_pressures <= pressure_retrieval.max())
    node_T_median   = np.interp(np.log10(node_pressures[mask_range]),
                                np.log10(pressure_retrieval), pcts[3])
    ax.scatter(node_T_median, node_pressures[mask_range],
               s=22, color="saddlebrown", zorder=5)

    # Sonora Bobcat formation-scenario profiles
    linestyles = {"hot": "-", "warm": "--", "cold": ":"}
    for key, scen in scenarios.items():
        teff   = scen["teff"]
        g_mks  = scen["g_mks"]
        s_init = scen["s_init"]
        label  = scen.get("label", f"{key} start  $S={s_init}\\,k_B$/bar")
        color  = _BOBCAT_COLORS.get(key, "grey")
        ls     = linestyles.get(key, "-")

        log_g_cgs = np.log10(g_mks * 100)   # MKS → CGS then log10
        pres_b, temp_k = _read_bobcat_pt(teff, g_mks, tarball=tarball)

        full_label = (f"{label}\n"
                      f"$T_{{\\rm eff}}={teff}\\,K$, "
                      f"$\\log g={log_g_cgs:.2f}$")
        ax.plot(temp_k, pres_b, color=color, lw=2.2, ls=ls, label=full_label, zorder=4)

    ax.set_xlim(left=0)
    ax.set_ylim(pressure_retrieval.max(), pressure_retrieval.min())
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)", fontsize=12)
    ax.set_ylabel("Pressure (bar)", fontsize=12)
    ax.set_title("DH Tau B: P-T profile vs. formation scenarios\n(Sonora Bobcat, $[M/H]=0$, no clouds)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.18, lw=0.5)
    ax.tick_params(labelsize=9)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {save_path}")

    plt.show()

    print(f"\nRetrieval posterior: {n_samples} samples")
    print(f"Median T range: {pcts[3].min():.0f}–{pcts[3].max():.0f} K")
    for key, scen in scenarios.items():
        pres_b, temp_k = _read_bobcat_pt(scen["teff"], scen["g_mks"], tarball=tarball)
        log_g_cgs = np.log10(scen["g_mks"] * 100)
        print(f"  {key:4s} start: Teff={scen['teff']} K, log_g={log_g_cgs:.2f}, "
              f"T(1 bar)={np.interp(1.0, pres_b, temp_k):.0f} K, "
              f"T(0.1 bar)={np.interp(0.1, pres_b, temp_k):.0f} K")

    return fig

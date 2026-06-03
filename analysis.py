"""
analysis.py  –  Reusable plotting/workflow wrappers for DH Tau B retrievals.

Designed to be called from notebooks or scripts:

    from analysis import run_free_chem_analysis, run_equil_chem_analysis

Each top-level function runs the full suite (CCF, chemistry, model-vs-data,
P-T profile) for one retrieval and returns a dict of all outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import savgol_filter

from tasting_analysis import (
    run_ccf_workflow,
    run_free_chemistry_workflow,
    run_equil_chemistry_workflow,
)

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
    win: int = None,
    pol: int = 2,
    data_dir: Path | str | None = None,
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
    win, pol        Savitzky-Golay window and polynomial order.
    data_dir        Directory containing data_npy and err_npy.
                    Defaults to workpath/night/ when None.
                    Pass e.g. workpath/'combined_two_nights' to use a combined
                    spectrum while still drawing wavelength/telluric from night.
    """
    #binning window
    win = 31 if win is None else win

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

        for det in range(n_det):
            w_full = wave_cube[det, order].astype(float)
            f_full = data_cube[det, order].astype(float)
            e_full = err_data[det, order].astype(float)

            valid = (np.isfinite(w_full) & np.isfinite(f_full) &
                     np.isfinite(e_full) & (e_full > 0))
            if valid.sum() < win:
                continue

            f_vld     = f_full[valid]; e_vld = e_full[valid]
            f_bin_vld = savgol_filter(f_vld, win, pol)
            e_bin_vld = np.sqrt(savgol_filter(e_vld ** 2, win, pol)) / np.sqrt(win / 2)

            w_vld = w_full[valid]
            wl_lo, wl_hi = w_vld.min(), w_vld.max()
            wl_min_order = min(wl_min_order, wl_lo)
            wl_max_order = max(wl_max_order, wl_hi)

            m_clip = (mw_s >= wl_lo) & (mw_s <= wl_hi)
            if m_clip.sum() >= win:
                m_sm     = savgol_filter(mf_s[m_clip], win, pol)
                m_interp = np.interp(w_vld, mw_s[m_clip], m_sm)
            else:
                m_interp = np.interp(w_vld, mw_s, mf_s)

            f_med = np.nanmedian(f_bin_vld) or 1.0
            m_med = np.nanmedian(m_interp)  or 1.0
            f_n_vld = f_bin_vld / f_med
            e_n_vld = e_bin_vld / f_med
            m_n_vld = m_interp  / m_med

            f_n = np.full(n_pix, np.nan); f_n[valid] = f_n_vld
            e_n = np.full(n_pix, np.nan); e_n[valid] = e_n_vld
            m_n = np.full(n_pix, np.nan); m_n[valid] = m_n_vld
            resid = f_n - m_n

            lbl_data  = "Data"  if det == 0 else None
            lbl_model = "Model" if det == 0 else None
            lbl_err   = "±1σ"   if det == 0 else None

            ax_top.fill_between(w_full, f_n - e_n, f_n + e_n,
                                where=valid, color="gray", alpha=0.35, linewidth=0,
                                label=lbl_err)
            ax_top.plot(w_full, f_n, color="steelblue", lw=1.0, alpha=0.85,
                        label=lbl_data)
            ax_top.plot(w_full, m_n, color="tomato",    lw=1.9, alpha=0.95,
                        label=lbl_model, zorder=5)
            ax_res.fill_between(w_full, -e_n, e_n,
                                where=valid, color="gray", alpha=0.2, linewidth=0)
            ax_res.plot(w_full, resid, color="steelblue", lw=0.8, alpha=0.8)
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
        ax_res.set_xlabel("Wavelength (nm)", fontsize=9)
        ax_res.set_ylabel("Resid.", fontsize=8)
        ax_res.tick_params(labelsize=8)
        ax_res.grid(alpha=0.15, lw=0.5)
        ax_res.set_ylim(-1, 1)
        if np.isfinite(wl_min_order) and np.isfinite(wl_max_order):
            ax_res.set_xlim(wl_min_order, wl_max_order)

    plt.suptitle(
        f"Binned data vs best-fit model  |  {retrieval_id}  |  night {night}\n"
        f"savgol window = {win} px  ·  3 detectors per order  ·  per-order-detector normalised",
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

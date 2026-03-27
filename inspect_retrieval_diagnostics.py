"""
inspect_retrieval_diagnostics.py
=================================
Diagnostic script for investigating the "sketchy retrieval" problem
for DH Tau B companion spectra.

Implements three investigation steps from the approved plan:
  Step 1 — Visual comparison: old (NormFalse) vs savgol model vs data
  Step 2 — Noise budget: empirical scatter vs formal extraction errors
  Step 3 — CCF: cross-correlate spectrum with a CO / best-fit model template

Run from the command line:
    python inspect_retrieval_diagnostics.py

Output figures are saved as PDF files in the Recipe_DH_Tau_B/ directory.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pickle
import pathlib

# ─── paths ────────────────────────────────────────────────────────────────────
WORKPATH   = pathlib.Path('/data2/peng')
NIGHT      = '2022-12-31'
DATA_DIR   = WORKPATH / NIGHT
RECIPE_DIR = WORKPATH / 'Recipe_DH_Tau_B'
RETR_DIR   = WORKPATH / 'retrievals'

RUN_OLD    = '817303_N500_ev0.5_NormFalse_PerChipScaleSingle'
RUN_SAVGOL = '3339582_N500_ev0.5_Normsavgol_PerChipScaleFalse'

# ─── load observed data ────────────────────────────────────────────────────────
def load_data():
    """Load (3,5,2048) arrays and reorder to (5,3,2048)."""
    flux_raw = np.load(DATA_DIR / 'extracted_spectra_combined_flux_cal.npy')   # (3,5,2048)
    err_raw  = np.load(DATA_DIR / 'extracted_spectra_combined_err_flux_cal.npy')

    wave_hdu = fits.open(DATA_DIR / 'cal/WLEN_K2166_V_DH_Tau_A+B_center.fits')
    wave_raw = np.array(wave_hdu[1].data)[:, 0:5, :]   # (3,5,2048)

    # reorder to (orders=5, dets=3, pixels=2048) to match retrieval script
    flux = np.transpose(flux_raw, (1, 0, 2))
    err  = np.transpose(err_raw,  (1, 0, 2))
    wave = np.transpose(wave_raw, (1, 0, 2))

    return wave, flux, err   # (5,3,2048)


def apply_savgol_norm(flux, err, window_length=301, polyorder=2):
    """Apply per-chip Savitzky-Golay continuum normalisation (mirrors retrieval script)."""
    flux_n = flux.copy()
    err_n  = err.copy()
    for order in range(5):
        for det in range(3):
            mask = np.isfinite(flux_n[order, det])
            if mask.sum() < window_length:
                continue
            lf = np.full(flux_n[order, det].shape, np.nan)
            lf[mask] = savgol_filter(flux_n[order, det][mask], window_length, polyorder)
            scale = np.nanmedian(np.abs(lf[mask]))
            if scale == 0:
                flux_n[order, det][mask] = np.nan
                err_n[order, det][mask]  = np.nan
                continue
            floor = np.finfo(float).eps * scale
            safe  = mask & np.isfinite(lf) & (np.abs(lf) > floor)
            flux_n[order, det][safe]            /= lf[safe]
            err_n[order, det][safe]             /= lf[safe]
            flux_n[order, det][mask & ~safe]    = np.nan
            err_n[order, det][mask & ~safe]     = np.nan
    return flux_n, err_n


def load_retrieval_model(run_name):
    """Load flattened model spectrum from a retrieval output directory."""
    d = RETR_DIR / run_name
    wave  = np.load(d / 'retrieval_model_wave.npy')
    flux  = np.load(d / 'retrieval_model_flux.npy')
    fscal = np.load(d / 'retrieval_model_flux_scaled.npy')
    with open(d / 'final_params_dict.pickle', 'rb') as f:
        params = pickle.load(f)
    return wave, flux, fscal, params


# ─── Step 1: visual comparison ────────────────────────────────────────────────
def step1_visual_comparison(wave, flux, err):
    """
    Plot per-chip: raw data + old-run model (NormFalse) in one panel,
    savgol-normalised data + savgol-run model in another panel.
    """
    print('\n=== Step 1: Visual comparison ===')

    mwave_old, _, mfscal_old, params_old = load_retrieval_model(RUN_OLD)
    mwave_sg,  _, mfscal_sg,  params_sg  = load_retrieval_model(RUN_SAVGOL)

    # Extract T-P summary
    T_keys = ['T0', 'T1', 'T2', 'T3', 'T4']
    def _scalar(v):
        return float(np.atleast_1d(v).flat[0]) if v is not None else float('nan')
    T_old  = {k: _scalar(params_old.get(k)) for k in T_keys}
    T_sg   = {k: _scalar(params_sg.get(k))  for k in T_keys}
    s2_old = _scalar(params_old.get('s2'))
    s2_sg  = _scalar(params_sg.get('s2'))

    print(f'  Old run  T0-T4 [K]: {[round(T_old[k]) for k in T_keys]}  s2={s2_old:.1f}')
    print(f'  Savgol run T0-T4: {[round(T_sg[k]) for k in T_keys]}  s2={s2_sg:.1f}')

    flux_sg, err_sg = apply_savgol_norm(flux, err)

    n_orders, n_dets = 5, 3
    fig, axes = plt.subplots(n_orders * 2, n_dets,
                             figsize=(18, n_orders * 2 * 2.4),
                             sharex='col')
    fig.suptitle(
        'Step 1 — Data vs best-fit models\n'
        f'Top: raw flux (old run, T0={round(T_old["T0"])} K, s2={s2_old:.1f}); '
        f'Bottom: savgol-normalised (3339582, T0={round(T_sg["T0"])} K, s2={s2_sg:.1f})',
        fontsize=10)

    for order in range(n_orders):
        for det in range(n_dets):
            w = wave[order, det]
            ax_raw = axes[order * 2,     det]
            ax_sg  = axes[order * 2 + 1, det]

            # ── raw data vs old model ──────────────────────────────────────
            f_raw = flux[order, det]
            e_raw = err[order, det]
            mask_raw = np.isfinite(f_raw) & np.isfinite(e_raw)
            ax_raw.plot(w[mask_raw], f_raw[mask_raw], lw=0.4, color='k', label='data')
            # interpolate model onto chip wavelength grid
            clip = (mwave_old >= w[mask_raw].min()) & (mwave_old <= w[mask_raw].max())
            if clip.sum() > 5:
                itp = interp1d(mwave_old[clip], mfscal_old[clip],
                               bounds_error=False, fill_value=np.nan)
                ax_raw.plot(w[mask_raw], itp(w[mask_raw]),
                            lw=0.8, color='firebrick', label=f'old model T0={round(T_old["T0"])}K')
            ax_raw.set_ylabel(f'Flux\n[W/m²/µm]', fontsize=7)
            ax_raw.tick_params(labelsize=7)
            if order == 0:
                ax_raw.set_title(f'Det {det}', fontsize=8)
            if order == 0 and det == 0:
                ax_raw.legend(fontsize=6, loc='upper right')

            # ── savgol-normalised data vs savgol model ─────────────────────
            f_sg_chip = flux_sg[order, det]
            mask_sg = np.isfinite(f_sg_chip)
            ax_sg.plot(w[mask_sg], f_sg_chip[mask_sg], lw=0.4, color='k', label='data (savgol)')
            clip2 = (mwave_sg >= w[mask_sg].min()) & (mwave_sg <= w[mask_sg].max())
            if clip2.sum() > 5:
                itp2 = interp1d(mwave_sg[clip2], mfscal_sg[clip2],
                                bounds_error=False, fill_value=np.nan)
                ax_sg.plot(w[mask_sg], itp2(w[mask_sg]),
                           lw=0.8, color='steelblue', label=f'savgol model T0={round(T_sg["T0"])}K')
            ax_sg.axhline(1.0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax_sg.set_ylabel(f'Norm. flux', fontsize=7)
            ax_sg.tick_params(labelsize=7)
            if order == 0 and det == 0:
                ax_sg.legend(fontsize=6, loc='upper right')
            if order == n_orders - 1:
                ax_sg.set_xlabel('Wavelength [nm]', fontsize=7)

    plt.tight_layout()
    out = RECIPE_DIR / 'diag_step1_model_comparison.pdf'
    fig.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {out}')


# ─── Step 2: noise budget ──────────────────────────────────────────────────────
def step2_noise_budget(wave, flux, err):
    """
    For each chip, compute:
      - median formal error (from err array)
      - empirical scatter = std of (flux - savgol continuum) in regions
        where the old-run model is close to a smooth baseline (= low model gradient)
      - ratio: scatter / formal_error

    Also compare overall empirical-to-formal error ratio to the reported s2.
    """
    print('\n=== Step 2: Noise budget ===')

    flux_n, err_n = apply_savgol_norm(flux, err)

    ratios = np.full((5, 3), np.nan)

    print(f'  {"Order":>5} {"Det":>3}  {"Formal_err":>12}  {"Empirical_std":>14}  {"Ratio":>7}  {"SNR_formal":>10}')
    for order in range(5):
        for det in range(3):
            f  = flux[order, det]
            fn = flux_n[order, det]
            e  = err[order, det]

            mask = np.isfinite(f) & np.isfinite(e) & (e > 0)
            if mask.sum() < 50:
                continue

            # Empirical scatter: std of savgol-residuals (data / continuum - 1)
            # fn values are already (data / continuum), so residual = fn - 1
            # Use only pixels where fn is valid
            mask_n = np.isfinite(fn)
            if mask_n.sum() < 50:
                continue
            emp_std = np.nanstd(fn[mask_n] - 1.0)

            # Formal error in normalised units = err / continuum ≈ err_n
            # (same division was applied, so err_n is formal error in norm units)
            formal_err_norm = np.nanmedian(err_n[order, det][mask_n & np.isfinite(err_n[order, det])])

            # Ratio
            ratio = emp_std / formal_err_norm if formal_err_norm > 0 else np.nan
            ratios[order, det] = ratio

            # SNR from formal errors (raw spectrum)
            snr = np.nanmedian(f[mask] / e[mask])
            print(f'  {order:>5} {det:>3}  {formal_err_norm:>12.4f}  {emp_std:>14.4f}  {ratio:>7.1f}  {snr:>10.1f}')

    print(f'\n  Median ratio (empirical/formal): {np.nanmedian(ratios):.1f}  '
          f'(compare to sqrt(chi2_red) = s2 reported by retrievals)')

    # Plot ratio map
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(ratios, aspect='auto', cmap='hot_r', vmin=1, vmax=100)
    ax.set_xlabel('Detector')
    ax.set_ylabel('Order')
    ax.set_xticks(range(3)); ax.set_yticks(range(5))
    for o in range(5):
        for d in range(3):
            if np.isfinite(ratios[o, d]):
                ax.text(d, o, f'{ratios[o,d]:.0f}', ha='center', va='center',
                        fontsize=8, color='white' if ratios[o,d] > 30 else 'black')
    plt.colorbar(im, ax=ax, label='Empirical scatter / Formal error')
    ax.set_title('Step 2 — Noise inflation factor per chip\n'
                 '(should be ~1 if errors are correctly estimated)')
    out = RECIPE_DIR / 'diag_step2_noise_budget.pdf'
    fig.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {out}')


# ─── Step 3: CCF ──────────────────────────────────────────────────────────────
def step3_ccf(wave, flux, err):
    """
    Cross-correlate the savgol-normalised observed spectrum with the savgol
    best-fit model template.

    The model stored in retrieval_model_flux_scaled.npy is already shifted to
    the object RV (~32 km/s) and lives on the data wavelength grid. Therefore:
      - CCF peak at v ≈ 0 km/s means the model already matches the data → features detected.
      - If peak is elsewhere, molecular features are either absent or misaligned.

    The CCF SNR = peak / std(CCF in the outer ±[100,150] km/s wings) quantifies
    whether the peak is significant.

    Also run per-order CCFs to identify which orders carry the most signal.
    """
    print('\n=== Step 3: CCF ===')
    print('  NOTE: model template is RV-corrected (RV~32 km/s already applied).')
    print('        CCF peak should be near v=0 if features are detectable.')

    C_LIGHT = 2.99792458e5   # km/s

    # Load savgol best-fit model as template (RV-corrected, in data frame)
    mwave_sg, _, mfscal_sg, params_sg = load_retrieval_model(RUN_SAVGOL)
    rv_model = float(np.atleast_1d(params_sg.get('rv', 32.0)).flat[0])
    print(f'  Model RV used in retrieval: {rv_model:.2f} km/s')

    # Apply savgol normalisation to the observed data
    flux_n, err_n = apply_savgol_norm(flux, err)

    # Velocity grid
    v_min, v_max, dv = -200, 200, 1.0   # km/s
    velocities = np.arange(v_min, v_max + dv, dv)

    ccf_total   = np.zeros(len(velocities))
    ccf_per_ord = np.zeros((5, len(velocities)))

    for iv, v in enumerate(velocities):
        doppler = 1.0 + v / C_LIGHT
        ccf_ord = np.zeros(5)
        for order in range(5):
            for det in range(3):
                w  = wave[order, det]
                fn = flux_n[order, det]
                en = err_n[order, det]
                mask = np.isfinite(fn) & np.isfinite(en) & (en > 0)
                if mask.sum() < 50:
                    continue
                # Shift the model by additional velocity v relative to its stored position
                # (model already at rv_model; we test whether data matches at rv_model + v)
                w_shift = w[mask] * doppler   # shift data wavelength forward to find model match
                clip = (mwave_sg >= w_shift.min()) & (mwave_sg <= w_shift.max())
                if clip.sum() < 5:
                    continue
                itp = interp1d(mwave_sg[clip], mfscal_sg[clip],
                               bounds_error=False, fill_value=np.nan)
                m_interp = itp(w_shift)
                # Residual model (lines only): model - 1 (savgol-normalised model ~1 in continuum)
                m_res = m_interp - 1.0
                f_res = fn[mask] - 1.0
                # Inverse-variance weighted CCF contribution
                weight = 1.0 / (en[mask] ** 2)
                finite = np.isfinite(m_res) & np.isfinite(f_res) & np.isfinite(weight)
                if finite.sum() < 10:
                    continue
                ccf_val = np.sum(weight[finite] * f_res[finite] * m_res[finite])
                ccf_total[iv]  += ccf_val
                ccf_ord[order] += ccf_val
        for order in range(5):
            ccf_per_ord[order, iv] = ccf_ord[order]

    # CCF SNR: peak vs. std of wings
    wing_mask = np.abs(velocities) > 120
    ccf_wing_std = np.std(ccf_total[wing_mask])
    ccf_wing_med = np.median(ccf_total[wing_mask])
    ccf_centered = ccf_total - ccf_wing_med
    peak_idx = np.argmax(ccf_centered)
    peak_v   = velocities[peak_idx]
    ccf_snr  = ccf_centered[peak_idx] / ccf_wing_std if ccf_wing_std > 0 else 0.0

    print(f'  CCF peak at v = {peak_v:.1f} km/s')
    print(f'  CCF peak SNR = {ccf_snr:.2f}  (>3 suggests detection; peak near 0 = features detectable)')

    # Normalise for plotting
    def _norm_ccf(ccf):
        mn, mx = ccf.min(), ccf.max()
        return (ccf - mn) / (mx - mn) if mx > mn else ccf * 0

    ccf_norm = _norm_ccf(ccf_total)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    ax0 = axes[0]
    ax0.plot(velocities, ccf_norm, color='k', lw=1.0)
    ax0.axvline(peak_v, color='firebrick', ls='--',
                label=f'peak @ {peak_v:.1f} km/s  (SNR={ccf_snr:.1f})')
    ax0.axvline(0.0,  color='steelblue', ls=':', label='v=0 (model already RV-corrected)')
    ax0.set_xlabel('Velocity [km/s]')
    ax0.set_ylabel('Normalised CCF')
    ax0.set_title('Total CCF (all orders)')
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.text(0.02, 0.05,
             f'CCF peak SNR = {ccf_snr:.1f}\n(>3 = feature detection)',
             transform=ax0.transAxes, fontsize=8, va='bottom',
             bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    order_labels = [f'Order {o} (~{int(wave[o].mean())} nm)' for o in range(5)]
    for i, order in enumerate(range(5)):
        ax = axes[i + 1]
        ccf_o = ccf_per_ord[order]
        wing_o_std = np.std(ccf_o[wing_mask])
        wing_o_med = np.median(ccf_o[wing_mask])
        ccf_o_c    = ccf_o - wing_o_med
        peak_o_idx = np.argmax(ccf_o_c)
        peak_o     = velocities[peak_o_idx]
        snr_o      = ccf_o_c[peak_o_idx] / wing_o_std if wing_o_std > 0 else 0.0
        ax.plot(velocities, _norm_ccf(ccf_o), lw=0.8, color='steelblue')
        ax.axvline(peak_o, color='firebrick', ls='--', lw=0.8,
                   label=f'peak @ {peak_o:.1f} km/s  SNR={snr_o:.1f}')
        ax.axvline(0.0, color='gray', ls=':', lw=0.8)
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Norm. CCF')
        ax.set_title(order_labels[i])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        print(f'  {order_labels[i]}: peak @ {peak_o:.1f} km/s  SNR={snr_o:.2f}')

    fig.suptitle(
        'Step 3 — CCF: data (savgol-norm.) × savgol model template\n'
        f'Model already at RV={rv_model:.1f} km/s → CCF peak near v=0 means features detected',
        fontsize=10)
    plt.tight_layout()
    out = RECIPE_DIR / 'diag_step3_ccf.pdf'
    fig.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'  → Saved: {out}')

    return velocities, ccf_norm, peak_v, ccf_snr


# ─── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Loading observed data...')
    wave, flux, err = load_data()
    print(f'  Data shape: wave={wave.shape}, flux={flux.shape}, err={err.shape}')

    step1_visual_comparison(wave, flux, err)
    step2_noise_budget(wave, flux, err)
    _, _, peak_v, ccf_snr = step3_ccf(wave, flux, err)

    print('\nAll diagnostics complete.')

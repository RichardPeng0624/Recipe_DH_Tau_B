"""
1. Anchor the broadband atmosphere model to the SAME absolute-flux reference
   already used throughout this project (SINFONI shape + 2MASS Ks photometry,
   Itoh et al. 2005) - i.e. rescale the model by a single constant factor so
   its own synthetic Ks flux matches the observed Ks point exactly, removing
   the retrieval's small residual (R/d)^2 fit offset (~2%) rather than
   trusting the fitted radius alone.
2. Load the actual flux-calibrated CRIRES+ K-band data (both nights, the
   same *_flux_cal.npy products used by the absolute-flux retrieval job
   2027997 - NormNone, no extra normalisation), bin it, and plot it with
   error bars alongside the model/photometry.
3. Redo the blackbody excess fit against the now-anchored model.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import csv

PERM = '/data2/peng/Recipe_DH_Tau_B/Reminisce/sed_fitting'

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.998e8
K_BOLTZ = 1.380649e-23
R_JUP_M = 7.149e7
PC_M = 3.0857e16
D_DH_TAU_PC = 135.2
D_M = D_DH_TAU_PC * PC_M
AU_M = 1.495978707e11

def flambda_to_fnu_Jy(flambda_Wm2um, lam_um):
    return flambda_Wm2um * lam_um**2 * 3.336e11

# ============================================================
# 1. Load broadband atmosphere model (21um-extended) and observed photometry
# ============================================================
wl_nm = np.load(f'{PERM}/sed_model_wl_nm_21um.npy')
flux_Wm2um = np.load(f'{PERM}/sed_model_flux_obs_Wm2um_21um.npy')
wl_um = wl_nm / 1000.0
order = np.argsort(wl_um)
wl_um, flux_Wm2um = wl_um[order], flux_Wm2um[order]
flux_Jy_raw = flambda_to_fnu_Jy(flux_Wm2um, wl_um)

def synth_phot(lam_c, half_width, wl, fl):
    m = (wl > lam_c - half_width) & (wl < lam_c + half_width)
    return np.nanmean(fl[m]) if m.sum() > 3 else np.nan

obs = {}
with open('/data2/peng/paper/data/tables/dh_tau_b_sed_photometry.csv') as f:
    for row in csv.DictReader(f):
        if row['Fnu_Jy'] and row['detection'] == 'detected':
            obs[row['band']] = dict(
                lam=float(row['lambda_eff_um']) if row['lambda_eff_um'] else np.nan,
                fnu=float(row['Fnu_Jy']),
                stat=float(row['Fnu_err_stat_Jy']) if row['Fnu_err_stat_Jy'] else 0.0,
                sys=float(row['Fnu_err_sys_Jy']) if row['Fnu_err_sys_Jy'] else 0.0,
            )

# ---- anchor: rescale the WHOLE model by a single constant so its Ks matches obs Ks ----
ks_model_raw = synth_phot(2.159, 0.131, wl_um, flux_Jy_raw)
anchor_factor = obs['Ks']['fnu'] / ks_model_raw
print(f"Ks anchor: obs={obs['Ks']['fnu']:.5f} Jy, model(raw)={ks_model_raw:.5f} Jy, "
      f"anchor_factor={anchor_factor:.4f} ({(anchor_factor-1)*100:+.1f}%)")
flux_Jy = flux_Jy_raw * anchor_factor

BANDS = [
    ('J', 1.235, 0.081), ('H', 1.662, 0.125), ('Ks', 2.159, 0.131),
    ('[3.6]', 3.550, 0.375), ('[4.5]', 4.493, 0.500),
    ('[5.8]', 5.731, 0.700), ('[8.0]', 7.872, 1.450),
]
model_synth = {name: synth_phot(lam, hw, wl_um, flux_Jy) for name, lam, hw in BANDS}
print("\n=== Anchored model vs observed ===")
for name, lam, hw in BANDS:
    o = obs.get(name)
    if o:
        print(f"{name:6s} obs={o['fnu']:.5f}  model(anchored)={model_synth[name]:.5f}  "
              f"obs/model={o['fnu']/model_synth[name]:.2f}")

PREDICT_BANDS = [("L'", 3.777, 0.350, 251.6), ("M'", 4.680, 0.120, 163.7)]
model_predict = {name: synth_phot(lam, hw, wl_um, flux_Jy) for name, lam, hw, zp in PREDICT_BANDS}

# ============================================================
# 2. Load actual flux-calibrated CRIRES+ K-band data, both nights, bin it
# ============================================================
def load_night_flux_cal(night):
    spec = np.load(f'/data2/peng/{night}/extracted_spectra_combined_flux_cal.npy')       # (3,5,2048) W/m2/um
    err  = np.load(f'/data2/peng/{night}/extracted_spectra_combined_err_flux_cal.npy')
    wave_hdu = fits.open(f'/data2/peng/{night}/cal/WLEN_K2166_V_DH_Tau_A+B_center.fits')
    wave = np.array(wave_hdu[1].data)[:, :5, :]     # (3,5,2048) nm, truncate 7->5 orders (matches _load_night)
    spec = np.transpose(spec, (1, 0, 2))            # (order, det, pix)
    err  = np.transpose(err,  (1, 0, 2))
    wave = np.transpose(wave, (1, 0, 2))
    wave_flat = wave.reshape(-1) / 1000.0            # nm -> um
    spec_flat = np.where(spec.reshape(-1) == 0, np.nan, spec.reshape(-1))
    err_flat  = np.where(err.reshape(-1)  == 0, np.nan, err.reshape(-1))
    valid = np.isfinite(spec_flat) & np.isfinite(err_flat) & (err_flat > 0)
    return wave_flat[valid], spec_flat[valid], err_flat[valid]

w1, f1, e1 = load_night_flux_cal('2022-12-31')
w2, f2, e2 = load_night_flux_cal('2023-01-01')
w_all = np.concatenate([w1, w2])
f_all_Jy = flambda_to_fnu_Jy(np.concatenate([f1, f2]), w_all)
e_all_Jy = flambda_to_fnu_Jy(np.concatenate([e1, e2]), w_all)   # error propagates linearly (same lam^2 factor)
print(f"\nLoaded {len(w_all)} valid CRIRES+ pixels (both nights), "
      f"wave range {w_all.min():.3f}-{w_all.max():.3f} um")

# ---- bin into N_BINS wavelength bins, inverse-variance weighted mean ----
N_BINS = 40
bin_edges = np.linspace(w_all.min(), w_all.max(), N_BINS + 1)
bin_centers, bin_flux, bin_err = [], [], []
for i in range(N_BINS):
    m = (w_all >= bin_edges[i]) & (w_all < bin_edges[i + 1])
    if m.sum() < 5:
        continue
    wts = 1.0 / e_all_Jy[m] ** 2
    fmean = np.sum(f_all_Jy[m] * wts) / np.sum(wts)
    ferr = np.sqrt(1.0 / np.sum(wts))
    bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
    bin_flux.append(fmean)
    bin_err.append(ferr)
bin_centers, bin_flux, bin_err = map(np.array, (bin_centers, bin_flux, bin_err))
print(f"Binned CRIRES+ data into {len(bin_centers)} bins "
      f"(median flux={np.median(bin_flux):.5f} Jy, median err={np.median(bin_err):.5f} Jy)")
print(f"CRIRES+ binned data median vs anchored Ks photometry: "
      f"{np.median(bin_flux)/obs['Ks']['fnu']:.2f}")

np.save(f'{PERM}/crires_bin_centers_um.npy', bin_centers)
np.save(f'{PERM}/crires_bin_flux_Jy.npy', bin_flux)
np.save(f'{PERM}/crires_bin_err_Jy.npy', bin_err)

# ============================================================
# 3. Redo blackbody excess fit against the now-anchored model
# ============================================================
EXCESS_BANDS = ['[3.6]', '[4.5]', '[5.8]', '[8.0]']
lam_ex, fnu_ex, err_ex = [], [], []
for name in EXCESS_BANDS:
    o = obs[name]
    excess = o['fnu'] - model_synth[name]
    err = np.sqrt(o['stat']**2 + o['sys']**2)
    lam_ex.append(o['lam']); fnu_ex.append(excess); err_ex.append(err)
lam_ex, fnu_ex, err_ex = np.array(lam_ex), np.array(fnu_ex), np.array(err_ex)

def bnu_Jy(lam_um, T):
    lam_m = lam_um * 1e-6
    nu = C_LIGHT / lam_m
    x = np.clip(H_PLANCK * nu / (K_BOLTZ * T), 1e-10, 700)
    return (2 * H_PLANCK * nu**3 / C_LIGHT**2) / np.expm1(x) / 1e-26

def fnu_bb_model(lam_um, T, R_bb_Rjup):
    omega = np.pi * (R_bb_Rjup * R_JUP_M / D_M) ** 2
    return omega * bnu_Jy(lam_um, T)

popt, pcov = curve_fit(fnu_bb_model, lam_ex, fnu_ex, p0=[500.0, 1.0], sigma=err_ex,
                        absolute_sigma=True, bounds=([50, 1e-3], [3000, 100]))
perr = np.sqrt(np.diag(pcov))
T_bb, R_bb = popt
T_bb_err, R_bb_err = perr
chi2 = np.sum(((fnu_ex - fnu_bb_model(lam_ex, *popt)) / err_ex) ** 2)
print(f"\n=== Blackbody excess fit (anchored model) ===")
print(f"T_bb = {T_bb:.1f} +- {T_bb_err:.1f} K, R_bb = {R_bb:.3f} +- {R_bb_err:.3f} R_Jup "
      f"({R_bb*R_JUP_M/AU_M:.4f} AU), chi2 = {chi2:.2f} for dof=2")

# ============================================================
# plot
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 6.5))

ax.plot(wl_um, flux_Jy, color='#3b6ba5', lw=1.0, alpha=0.9,
        label=f'Atmosphere model, anchored to Ks ({(anchor_factor-1)*100:+.1f}% vs. raw retrieval R/d scaling)')

wl_bb = np.linspace(0.9, 20, 500)
fnu_bb_curve = fnu_bb_model(wl_bb, T_bb, R_bb)
atm_on_bb = np.interp(wl_bb, wl_um, flux_Jy)
ax.plot(wl_bb, fnu_bb_curve, color='#e07b39', lw=1.2, ls='--',
        label=f'Blackbody excess fit (T={T_bb:.0f}$\\pm${T_bb_err:.0f} K, R={R_bb:.2f} R$_{{Jup}}$)')
ax.plot(wl_bb, atm_on_bb + fnu_bb_curve, color='#2a9d5c', lw=1.2, label='Atmosphere + blackbody (total model)')

sw = np.load(f'{PERM}/sinfoni_wl_um.npy')
sf_Jy = np.load(f'{PERM}/sinfoni_flux_Jy_scaled.npy')
ax.plot(sw, sf_Jy, color='#999999', lw=0.5, alpha=0.6, label="SINFONI J/H/K spectrum (anchored to Ks)")

ax.errorbar(bin_centers, bin_flux, yerr=bin_err, fmt='.', color='#5c4d8a', ms=4, elinewidth=0.6,
            alpha=0.75, zorder=4, label='CRIRES+ K-band data, binned (both nights, flux-calibrated)')

for name, lam, hw in BANDS:
    o = obs.get(name)
    if o is None:
        continue
    yerr = np.sqrt(o['stat']**2 + o['sys']**2)
    ax.errorbar(lam, o['fnu'], yerr=yerr, fmt='o', color='#d1495b', ms=6, zorder=6,
                label='Observed photometry' if name == 'J' else None)

for i, (name, lam, hw, zp) in enumerate(PREDICT_BANDS):
    ax.scatter(lam, model_predict[name], marker='D', facecolors='#f0a03b', edgecolors='#8a5a10', s=60, zorder=7,
               label="Model-predicted L'/M' (no data)" if i == 0 else None)
    ax.annotate(name, (lam, model_predict[name]), textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha='center', color='#8a5a10')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Flux density (Jy)')
ax.set_title('DH Tau B SED: Ks-anchored atmosphere model + CRIRES+ data + blackbody excess')
ax.set_xlim(0.9, 20)
ax.set_ylim(3e-4, 2e-2)
ax.legend(fontsize=7, loc='upper left')
ax.grid(alpha=0.2, which='both')

fig.tight_layout()
fig.savefig(f'{PERM}/dh_tau_b_sed_anchored_crires.png', dpi=150)
fig.savefig('/data2/peng/paper/aa/figures/fig_sed_anchored_crires_draft.png', dpi=150)
print("\nSaved plot.")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import csv

SCRATCH = '/data2/peng/Recipe_DH_Tau_B/Reminisce/sed_fitting'

# ---- load broadband model ----
wl_nm = np.load(f'{SCRATCH}/sed_model_wl_nm.npy')
flux_Wm2um = np.load(f'{SCRATCH}/sed_model_flux_obs_Wm2um.npy')
wl_um = wl_nm / 1000.0
order = np.argsort(wl_um)
wl_um, flux_Wm2um = wl_um[order], flux_Wm2um[order]

def flambda_to_fnu_Jy(flambda_Wm2um, lam_um):
    return flambda_Wm2um * lam_um**2 * 3.336e11

flux_Jy = flambda_to_fnu_Jy(flux_Wm2um, wl_um)

def synth_phot(lam_c, half_width, wl=wl_um, fl=flux_Jy):
    m = (wl > lam_c - half_width) & (wl < lam_c + half_width)
    return np.nanmean(fl[m]) if m.sum() > 3 else np.nan

BANDS = [
    ('J',     1.235, 0.081),
    ('H',     1.662, 0.125),
    ('Ks',    2.159, 0.131),
    ('[3.6]', 3.550, 0.375),
    ('[4.5]', 4.493, 0.500),
    ('[5.8]', 5.731, 0.700),
    ('[8.0]', 7.872, 1.450),
]

# Standard ground-based L'/M' filters (MKO system, Tokunaga & Vacca 2005) -
# no observed photometry exists for DH Tau B at these (Phase 1 literature
# search found none); model-predicted values only, useful for planning
# future AO imaging (NACO/NIRC2-style L'/M' photometry).
PREDICT_BANDS = [
    ("L'",  3.777, 0.350, 251.6),   # lam_eff, half-width, Vega zero point (Jy)
    ("M'",  4.680, 0.120, 163.7),
]

print("=== Synthetic photometry from broadband model ===")
model_synth = {}
for name, lam, hw in BANDS:
    f = synth_phot(lam, hw)
    model_synth[name] = f
    print(f"{name:6s} lam={lam:.3f}um  F_model={f:.5f} Jy")

print("\n=== Model-PREDICTED L'/M' (no observed data exists for DH Tau B) ===")
model_predict = {}
for name, lam, hw, zp in PREDICT_BANDS:
    f = synth_phot(lam, hw)
    mag = -2.5 * np.log10(f / zp) if f > 0 else np.nan
    model_predict[name] = f
    print(f"{name:6s} lam={lam:.3f}um  F_model={f:.5f} Jy  ->  predicted Vega mag = {mag:.2f}")

# ---- load observed photometry table ----
obs = {}
with open('/data2/peng/paper/data/tables/dh_tau_b_sed_photometry.csv') as f:
    for row in csv.DictReader(f):
        if row['Fnu_Jy']:
            obs[row['band']] = dict(
                lam=float(row['lambda_eff_um']) if row['lambda_eff_um'] else np.nan,
                fnu=float(row['Fnu_Jy']),
                stat=float(row['Fnu_err_stat_Jy']) if row['Fnu_err_stat_Jy'] else 0.0,
                sys=float(row['Fnu_err_sys_Jy']) if row['Fnu_err_sys_Jy'] else 0.0,
                detection=row['detection'],
            )

print("\n=== Model vs observed (Jy) ===")
for name, lam, hw in BANDS:
    o = obs.get(name)
    if o:
        ratio = o['fnu'] / model_synth[name]
        print(f"{name:6s} obs={o['fnu']:.5f}+-{o['stat']:.5f}  model={model_synth[name]:.5f}  obs/model={ratio:.2f}")

# ---- SINFONI spectrum, rescaled so its synthetic Ks matches observed Ks ----
d = fits.getdata('/data2/peng/DHTaub_SINFONIspeclib_JHK.fits')
sw, sf = d[:, 0], d[:, 1]
good = sf > 0
sw, sf = sw[good], sf[good]
m = (sw > 2.159 - 0.131) & (sw < 2.159 + 0.131)
sinfoni_ks_raw = np.nanmean(sf[m])
scale = obs['Ks']['fnu'] / sinfoni_ks_raw
sf_Jy = sf * scale * (sw**2) / (2.159**2)   # placeholder, fixed below
# sf is presumably F_lambda-like or arbitrary units; since we only rescale
# by a single multiplicative constant tied to Ks, do NOT apply an extra
# lambda^2 shape correction (unknown native unit) - just linearly rescale.
sf_Jy = sf * scale
print(f"\nSINFONI raw Ks-band mean: {sinfoni_ks_raw:.4e}, scale factor to match obs Ks: {scale:.4e}")

np.save(f'{SCRATCH}/sinfoni_wl_um.npy', sw)
np.save(f'{SCRATCH}/sinfoni_flux_Jy_scaled.npy', sf_Jy)

# ---- plot ----
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(wl_um, flux_Jy, color='#3b6ba5', lw=0.8, alpha=0.85, label='CRIRES+ retrieval (job 2027997) best-fit atmosphere, broadband forward model')

ax.plot(sw, sf_Jy, color='#999999', lw=0.5, alpha=0.7, label="SINFONI J/H/K spectrum (Itoh+2005 shape, rescaled to match Ks photometry)")

for name, lam, hw in BANDS:
    o = obs.get(name)
    if o is None:
        continue
    yerr = np.sqrt(o['stat']**2 + o['sys']**2)
    ax.errorbar(lam, o['fnu'], yerr=yerr, fmt='o', color='#d1495b', ms=6, zorder=5,
                label='Observed photometry (Itoh+2005; Martinez & Kraus 2022)' if name == 'J' else None)
    ax.scatter(lam, model_synth[name], marker='s', facecolors='none', edgecolors='#3b6ba5', s=50, zorder=6,
               label='Model synthetic photometry' if name == 'J' else None)

for i, (name, lam, hw, zp) in enumerate(PREDICT_BANDS):
    ax.scatter(lam, model_predict[name], marker='D', facecolors='#f0a03b', edgecolors='#8a5a10', s=60, zorder=7,
               label="Model-predicted L'/M' (no data - future AO imaging target)" if i == 0 else None)
    ax.annotate(name, (lam, model_predict[name]), textcoords="offset points", xytext=(0, 8),
                fontsize=8, ha='center', color='#8a5a10')

# ALMA points (0.88mm = 873 um)
alma_lam = 873.0
alma_companion = 0.000017   # Jy, non-detection
alma_sigma = 0.000041
alma_3sig = 3 * alma_sigma
ax.errorbar([alma_lam], [alma_3sig], yerr=[alma_3sig*0.4], uplims=True, fmt='v', color='#d1495b', ms=8, zorder=5,
            label='ALMA 0.88mm 3$\\sigma$ upper limit (companion, Wu+2020)')
ax.scatter([alma_lam], [0.0523*1000], marker='*', color='#888888', s=80, zorder=4,
           label='DH Tau A (host star) ALMA 0.88mm, for reference')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Wavelength (µm)')
ax.set_ylabel('Flux density (Jy)')
ax.set_title('DH Tau B multi-wavelength SED')
ax.set_xlim(0.9, 1200)
ax.legend(fontsize=7, loc='lower left')
ax.grid(alpha=0.2, which='both')

fig.tight_layout()
fig.savefig(f'{SCRATCH}/dh_tau_b_sed_phase3.png', dpi=150)
fig.savefig('/data2/peng/paper/aa/figures/fig_sed_phase3_draft.png', dpi=150)
print("\nSaved plot to", f'{SCRATCH}/dh_tau_b_sed_phase3.png', "and paper/aa/figures/fig_sed_phase3_draft.png")

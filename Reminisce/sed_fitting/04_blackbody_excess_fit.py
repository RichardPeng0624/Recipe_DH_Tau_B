"""
Fit a second blackbody component to the mid-IR excess (observed photometry
minus the atmosphere-only forward model) longward of K band, and plot the
SED zoomed to 20um with atmosphere + blackbody + sum overlaid.

Excess convention matches the atmosphere model's own (R/d)^2 scaling:
    F_nu_bb(lambda) = Omega * B_nu(lambda, T_bb),  Omega = pi * (R_bb/d)^2
i.e. same "flux from an emitting sphere of radius R_bb at distance d" picture
already used for the atmosphere component (see 02_run_broadband_model.py).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import csv

PERM = '/data2/peng/Recipe_DH_Tau_B/Reminisce/sed_fitting'

# ---- constants ----
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT = 2.998e8           # m/s
K_BOLTZ = 1.380649e-23      # J/K
R_JUP_M = 7.149e7
PC_M = 3.0857e16
D_DH_TAU_PC = 135.2
D_M = D_DH_TAU_PC * PC_M
AU_M = 1.495978707e11

# ---- load 21um-extended atmosphere model ----
wl_nm = np.load(f'{PERM}/sed_model_wl_nm_21um.npy')
flux_Wm2um = np.load(f'{PERM}/sed_model_flux_obs_Wm2um_21um.npy')
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
model_synth = {name: synth_phot(lam, hw) for name, lam, hw in BANDS}

PREDICT_BANDS = [
    ("L'",  3.777, 0.350, 251.6),
    ("M'",  4.680, 0.120, 163.7),
]
model_predict = {name: synth_phot(lam, hw) for name, lam, hw, zp in PREDICT_BANDS}

# ---- observed photometry ----
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

# ---- excess = observed - atmosphere model, for bands longward of K only ----
EXCESS_BANDS = ['[3.6]', '[4.5]', '[5.8]', '[8.0]']
lam_ex, fnu_ex, err_ex = [], [], []
print("=== Excess (obs - atmosphere model) longward of K ===")
for name in EXCESS_BANDS:
    o = obs[name]
    excess = o['fnu'] - model_synth[name]
    err = np.sqrt(o['stat']**2 + o['sys']**2)
    lam_ex.append(o['lam']); fnu_ex.append(excess); err_ex.append(err)
    print(f"{name:6s} lam={o['lam']:.3f}um  obs={o['fnu']:.5f}  atm_model={model_synth[name]:.5f}  excess={excess:.5f}+-{err:.5f} Jy")
lam_ex, fnu_ex, err_ex = np.array(lam_ex), np.array(fnu_ex), np.array(err_ex)

# ---- blackbody excess model: F_nu = Omega * B_nu(T), Omega = pi*(R_bb/d)^2 ----
def bnu_Jy(lam_um, T):
    """Planck function in Jy/sr, wavelength in micron."""
    lam_m = lam_um * 1e-6
    nu = C_LIGHT / lam_m
    x = H_PLANCK * nu / (K_BOLTZ * T)
    x = np.clip(x, 1e-10, 700)
    bnu_SI = (2 * H_PLANCK * nu**3 / C_LIGHT**2) / (np.expm1(x))   # W/m2/Hz/sr
    return bnu_SI / 1e-26   # -> Jy/sr

def fnu_bb_model(lam_um, T, R_bb_Rjup):
    R_bb_m = R_bb_Rjup * R_JUP_M
    omega = np.pi * (R_bb_m / D_M) ** 2
    return omega * bnu_Jy(lam_um, T)

# initial guess: T~500K, R~1 Rjup (typical warm circumplanetary-disk scale)
p0 = [500.0, 1.0]
try:
    popt, pcov = curve_fit(fnu_bb_model, lam_ex, fnu_ex, p0=p0, sigma=err_ex,
                            absolute_sigma=True, bounds=([50, 1e-3], [3000, 100]))
    perr = np.sqrt(np.diag(pcov))
    T_bb, R_bb = popt
    T_bb_err, R_bb_err = perr
    resid = fnu_ex - fnu_bb_model(lam_ex, *popt)
    chi2 = np.sum((resid / err_ex) ** 2)
    dof = len(lam_ex) - 2
    print(f"\n=== Blackbody excess fit ===")
    print(f"T_bb = {T_bb:.1f} +- {T_bb_err:.1f} K")
    print(f"R_bb = {R_bb:.3f} +- {R_bb_err:.3f} R_Jup = {R_bb*R_JUP_M/AU_M:.4f} AU")
    print(f"chi2 = {chi2:.2f} for dof = {dof}")
    for name, l, f, e, r in zip(EXCESS_BANDS, lam_ex, fnu_ex, err_ex, resid):
        print(f"  {name:6s} excess={f:.5f}+-{e:.5f}  bb_model={f-r:.5f}  resid/err={r/e:+.2f}")
except Exception as e:
    print("Blackbody fit failed:", e)
    T_bb, R_bb = None, None

# ---- plot, zoomed to 20um ----
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(wl_um, flux_Jy, color='#3b6ba5', lw=1.0, alpha=0.9,
        label='Atmosphere model (job 2027997 best-fit)')

if T_bb is not None:
    wl_bb = np.linspace(0.9, 20, 500)
    fnu_bb_curve = fnu_bb_model(wl_bb, T_bb, R_bb)
    ax.plot(wl_bb, fnu_bb_curve, color='#e07b39', lw=1.2, ls='--',
            label=f'Blackbody excess fit (T={T_bb:.0f}$\\pm${T_bb_err:.0f} K, R={R_bb:.2f} R$_{{Jup}}$)')
    # sum: interpolate atmosphere onto the same grid and add
    atm_on_bb = np.interp(wl_bb, wl_um, flux_Jy)
    ax.plot(wl_bb, atm_on_bb + fnu_bb_curve, color='#2a9d5c', lw=1.2,
            label='Atmosphere + blackbody (total model)')

sw = np.load(f'{PERM}/sinfoni_wl_um.npy')
sf_Jy = np.load(f'{PERM}/sinfoni_flux_Jy_scaled.npy')
ax.plot(sw, sf_Jy, color='#999999', lw=0.5, alpha=0.7,
        label="SINFONI J/H/K spectrum (rescaled to Ks photometry)")

for name, lam, hw in BANDS:
    o = obs.get(name)
    if o is None:
        continue
    yerr = np.sqrt(o['stat']**2 + o['sys']**2)
    ax.errorbar(lam, o['fnu'], yerr=yerr, fmt='o', color='#d1495b', ms=6, zorder=5,
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
ax.set_title('DH Tau B SED: atmosphere + blackbody excess fit')
ax.set_xlim(0.9, 20)
ax.set_ylim(3e-4, 2e-2)
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(alpha=0.2, which='both')

fig.tight_layout()
fig.savefig(f'{PERM}/dh_tau_b_sed_bb_excess_20um.png', dpi=150)
fig.savefig('/data2/peng/paper/aa/figures/fig_sed_bb_excess_draft.png', dpi=150)
print("\nSaved plot.")

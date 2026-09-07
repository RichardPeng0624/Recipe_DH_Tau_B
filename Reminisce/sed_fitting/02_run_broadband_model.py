"""
Phase 2 of the SED forward model: use the cached c-k Radtrans object to
compute P-T profile, equilibrium chemistry, and flux at job 2027997's
best-fit parameters, scaled to Earth-observed flux via (R/d)^2.
"""
import sys, os, pickle, importlib.util
import numpy as np
from scipy.interpolate import PchipInterpolator

GB_PATH = '/data2/peng/Recipe_DH_Tau_B/Tasting_guidebook/Guidebook_GAStronomy_Piette_v4.2.5.py'
spec = importlib.util.spec_from_file_location('guidebook_v425', GB_PATH)
GB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GB)

from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import simplify_species_list

SCRATCH = '/var/tmp/peng/claude-3444/-data2-peng/626084bd-877f-42cc-8d70-b597e193433a/scratchpad'
PERM = '/data2/peng/Recipe_DH_Tau_B/Reminisce/sed_fitting'

with open(f'{SCRATCH}/atm_ck_21um.pickle', 'rb') as f:
    atm = pickle.load(f)

with open('/data2/peng/retrievals/2027997_N600_ev0.5_NormNone_PerChipScaleFalse/final_params_dict.pickle', 'rb') as f:
    params = dict(pickle.load(f))

pressure = np.logspace(-5, 1, 50)
n_atm_layers = 50

# ---- P-T profile: replicate pRT_spectrum.make_pt() exactly ----
LOG_P_NODES = np.array([0.7, 0.0, -0.3, -0.7, -1.0, -1.5, -3.0, -5.0])
ANCHOR_IDX = 3
T_anchor = params['T_anchor']
dT = np.array([params[f'dT_{i+1}'] for i in range(7)])
T_nodes = np.empty(8)
T_nodes[ANCHOR_IDX] = T_anchor
for j in range(ANCHOR_IDX - 1, -1, -1):
    T_nodes[j] = T_nodes[j + 1] + dT[j]
for j in range(ANCHOR_IDX + 1, 8):
    T_nodes[j] = T_nodes[j - 1] - dT[j - 1]
log_P_asc = LOG_P_NODES[::-1]
T_asc = T_nodes[::-1]
pchip = PchipInterpolator(log_P_asc, T_asc)
log_P_atm = np.log10(pressure)
temperature = pchip(log_P_atm)
temperature = np.where(log_P_atm < log_P_asc[0], T_asc[0], temperature)
temperature = np.where(log_P_atm > log_P_asc[-1], T_asc[-1], temperature)
temperature = np.clip(temperature, 1.0, 30000.0)
print("T range:", temperature.min(), temperature.max(), "K; T_anchor=", T_anchor)

# ---- equilibrium chemistry ----
eq_chem = PreCalculatedEquilibriumChemistryTable()
C_H = params['C_H']
CO = params['C/O']
CO_SOLAR = 0.5495
ZH = C_H - np.log10(CO / CO_SOLAR)

mf_eq, MMW, _ = eq_chem.interpolate_mass_fractions(
    co_ratios=CO * np.ones(n_atm_layers),
    log10_metallicities=ZH * np.ones(n_atm_layers),
    temperatures=temperature,
    pressures=pressure,
    carbon_pressure_quench=None,
    full=True,
)

BROADBAND_SPECIES = [
    '1H2-16O.R1000', '12C-16O__HITEMP.R1000', '13C-16O.R1000', '12C-1H4__MM.R1000',
    '14N-1H3.R1000', '1H2-32S.R1000', '1H-12C-14N.R1000', '56Fe-1H.R1000',
    '12C-16O2__UCL-4000.R1000',
]
simple_names = simplify_species_list(BROADBAND_SPECIES)
print("Simplified names:", simple_names)

ratio_12_13 = 10 ** params.get('log_12CO_13CO', np.log10(70.0))
mass_frac = {}
for prt_name, simple in zip(BROADBAND_SPECIES, simple_names):
    if 'C-16O2' in prt_name.upper() and False:
        pass
    if prt_name.startswith('13C-16O'):
        mass_frac[prt_name] = mf_eq['CO'] * (29.002355 / 28.009999) / ratio_12_13
    elif simple in mf_eq:
        mass_frac[prt_name] = mf_eq[simple]
    else:
        print(f'WARNING: {prt_name} ({simple}) not in equilibrium table, skipping.')

mass_frac['H2'] = mf_eq['H2']
mass_frac['He'] = mf_eq['He']

print("mass_frac keys:", list(mass_frac.keys()))
for k, v in mass_frac.items():
    print(k, "mean MF =", np.mean(v))

# ---- gravity from log_M, log_R ----
_M_g = 10 ** params['log_M'] * 1.898e30
_R_cm = 10 ** params['log_R'] * 7.149e9
gravity = 6.674e-8 * _M_g / _R_cm ** 2
print("gravity (cgs):", gravity, "log_g:", np.log10(gravity))

# ---- run pRT3 flux calculation ----
wl, flux, _ = atm.calculate_flux(
    temperatures=temperature,
    mass_fractions=mass_frac,
    reference_gravity=gravity,
    mean_molar_masses=MMW,
    return_contribution=True,
    frequencies_to_wavelengths=True,
)
wl_nm = wl * 1e7          # cm -> nm
flux_si = flux * 1e-7     # erg/s/cm2/cm -> W/m2/um

# ---- scale to Earth-observed absolute flux: F_obs = F_surface * (R/d)^2 ----
R_JUP_M = 7.149e7
PC_M = 3.0857e16
D_DH_TAU_PC = 135.2
R_m = 10 ** params['log_R'] * R_JUP_M
d_m = D_DH_TAU_PC * PC_M
flux_obs = flux_si * (R_m / d_m) ** 2

print("wl range (um):", wl_nm.min()/1000, wl_nm.max()/1000)
print("flux_obs range (W/m2/um):", np.nanmin(flux_obs), np.nanmax(flux_obs))

np.save(f'{PERM}/sed_model_wl_nm_21um.npy', wl_nm)
np.save(f'{PERM}/sed_model_flux_obs_Wm2um_21um.npy', flux_obs)
print("Saved model spectrum (21um-extended) to", PERM)

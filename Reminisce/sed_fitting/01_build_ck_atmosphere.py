"""
Build a broadband (0.9-8 um) forward-model spectrum for DH Tau B at the
best-fit parameters of the absolute-flux benchmark retrieval (job 2027997),
using correlated-k opacities (not the retrieval's lbl mode, which is too
slow/heavy over such a wide range). Used for Phase 3 of the SED comparison.
"""
import sys, os, pickle, importlib.util
import numpy as np

GB_PATH = '/data2/peng/Recipe_DH_Tau_B/Tasting_guidebook/Guidebook_GAStronomy_Piette_v4.2.5.py'
spec = importlib.util.spec_from_file_location('guidebook_v425', GB_PATH)
GB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GB)

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable

# ---- load best-fit params from job 2027997 (absolute-flux benchmark) ----
with open('/data2/peng/retrievals/2027997_N600_ev0.5_NormNone_PerChipScaleFalse/final_params_dict.pickle', 'rb') as f:
    p2027997 = pickle.load(f)

params = dict(p2027997)
params['chemistry'] = 'equilibrium'
# rv/vsini/epsilon irrelevant for a broadband SED point-check; keep them but harmless
print("Loaded params:", {k: v for k, v in params.items() if k not in ('phi','phi_N2')})

# ---- build c-k Radtrans atmosphere over 0.85-8.5 um ----
# Main molecular opacity species only (sets broadband continuum/band shape).
# HF, Na, Ca dropped: no correlated-k table exists for HF at all, and the
# project's own 2026-07-21 investigation (job 51459 vs 3497146) established
# that Na/Ca opacity is fully inert to the spectral shape - safe to omit for
# a broadband SED consistency check.
BROADBAND_SPECIES = [
    '1H2-16O', '12C-16O__HITEMP', '13C-16O', '12C-1H4__MM', '14N-1H3',
    '1H2-32S', '1H-12C-14N', '56Fe-1H',
]
species = [s + '.R1000' for s in BROADBAND_SPECIES]
# CO2: only UCL-4000 exists for c-k mode (HITEMP is lbl-only)
species.append('12C-16O2__UCL-4000.R1000')
print("Species requested:", species)

boundary = np.array([0.85, 21.0])  # micron - extended 2026-09-02 for the 20um blackbody-excess check

atm = Radtrans(
    line_species=species,
    rayleigh_species=['H2', 'He'],
    gas_continuum_contributors=['H2--H2', 'H2--He'],
    wavelength_boundaries=boundary,
    line_opacity_mode='c-k',
    pressures=np.logspace(-5, 1, 50),
)
print("Radtrans built OK")

with open('/var/tmp/peng/claude-3444/-data2-peng/626084bd-877f-42cc-8d70-b597e193433a/scratchpad/atm_ck_21um.pickle', 'wb') as f:
    pickle.dump(atm, f)
print("Saved atmosphere object")

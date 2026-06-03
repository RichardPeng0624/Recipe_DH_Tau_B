import getpass
import os
'''
if getpass.getuser() == "peng": # when testing from Jiacheng's work station
    os.environ['pRT_input_data_path'] = "/net/lem/data2/pRT3_formatted/input_data"
'''

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import numpy as np
import pymultinest
import pathlib
import pickle
import pandas as pd
import corner
import matplotlib.pyplot as plt

import time
#from petitRADTRANS import Radtrans --- old pRT2 import ---
import petitRADTRANS as prt
from petitRADTRANS.radtrans import Radtrans

from PyAstronomy.pyasl import fastRotBroad, helcorr

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from scipy.special import loggamma
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.stats import norm

import sys



from petitRADTRANS.config import petitradtrans_config_parser
petitradtrans_config_parser.set_input_data_path('/net/lem/data2/pRT3_formatted')

from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.chemistry.utils import simplify_species_list


#convert to pRT3 format
'''
from petitRADTRANS.__file_conversion import convert_all
from petitRADTRANS.__file_conversion import _correlated_k_opacities_dat2h5_external_species
from petitRADTRANS.chemistry.prt_molmass import get_species_molar_mass

convert_all(clean=True) # convert all line files to pRT3 format
'''
workpath = '/data2/peng/'
results_root = pathlib.Path(workpath) / 'retrievals'


class Target:

    def __init__(self, wl, fl, err, name='dh_tau_b'):
        self.name=name
        self.fullname='DH_Tau_B'
        if len(wl.shape) == 3: # if loading 3D spectra (orders, detectors, pixels)
            self.n_orders= wl.shape[0] # number or orders
            self.n_dets= wl.shape[1] # number of detectors
            
        elif len(wl.shape) ==1: # if loading flattened 1D spectra
            self.n_orders= 5  # hardcoded for now
            self.n_dets= 3    # hardcoded for now

        else:
            print('Error: wavelength/flux/error shape not recognized!')

        self.n_pixels=2048 # number of pixels per detector
        self.K2166=np.array([[[1921.318,1934.583], [1935.543,1948.213], [1949.097,1961.128]],
                            [[1989.978,2003.709], [2004.701,2017.816], [2018.708,2031.165]],
                            [[2063.711,2077.942], [2078.967,2092.559], [2093.479,2106.392]],
                            [[2143.087,2157.855], [2158.914,2173.020], [2173.983,2187.386]],
                            [[2228.786,2244.133], [2245.229,2259.888], [2260.904,2274.835]],
                            [[2321.596,2337.568], [2338.704,2353.961], [2355.035,2369.534]],
                            [[2422.415,2439.061], [2440.243,2456.145], [2457.275,2472.388]]])[::-1]
        
        #set ra and dec for DH Tau B
        self.ra="04h29m41.65s"
        self.dec="26d32m56.2s"
        self.JD=59944.17637 + 2.4e6 # observation date in JD        
        self.color='limegreen' # color of retrieval output
        self.wl = wl
        self.fl = fl
        self.err= err

        print(f'Loaded spectrum for target {self.name} with shape wl:{self.wl.shape}, fl:{self.fl.shape}, err:{self.err.shape}')    

    '''    
    def load_spectrum(self):
        self.cwd = os.getcwd()
        file=pathlib.Path(f'{self.cwd}/{self.name}_spectrum.txt') # should be in same folder
        file=np.genfromtxt(file,skip_header=1,delimiter=' ')
        wl=np.reshape(file[:,0],(self.n_orders,self.n_dets,self.n_pixels)) # wavelength
        fl=np.reshape(file[:,1],(self.n_orders,self.n_dets,self.n_pixels)) # flux
        err=np.reshape(file[:,2],(self.n_orders,self.n_dets,self.n_pixels)) # error
        return wl,fl,err
    '''
 
        
    def get_mask_isfinite(self): # for masking out NaN values

        if len(self.wl.shape) ==1: # if loading flattened 1D spectra
            self.mask_isfinite=np.isfinite(self.fl) # only finite pixels
            
            
        
        elif len(self.wl.shape) ==3: # if loading 3D spectra (orders, detectors, pixels)
            self.mask_isfinite=np.empty((self.n_orders,self.n_dets,self.n_pixels),dtype=bool)
            for i in range(self.n_orders):
                for j in range(self.n_dets):
                    mask_ij = np.isfinite(self.fl[i,j]) # only finite pixels
                    self.mask_isfinite[i,j]=mask_ij
        else:
            print('Error: wavelength/flux/error shape not recognized!')
            
        return self.mask_isfinite
    
class Parameters:

    def __init__(self, free_params, constant_params):

        self.params = {} # all parameters + their values
            
        # Separate the prior range from the mathtext label
        self.param_priors, self.param_mathtext = {}, {}
        for key_i, (prior_i, mathtext_i) in free_params.items():
            self.param_priors[key_i]   = prior_i
            self.param_mathtext[key_i] = mathtext_i

        self.param_keys = np.array(list(self.param_priors.keys())) # keys of free parameters
        self.n_params=len(self.param_keys)
        self.ndim = len(self.param_keys) # number of free parameters
        self.free_params=free_params
        self.constant_params=constant_params
        self.params.update(constant_params) # dictionary with constant parameter values
            
    @staticmethod
    def uniform_prior(bounds):
        return lambda x: x*(bounds[1]-bounds[0])+bounds[0]
    
    @staticmethod
    def gaussian_prior(mu, sigma):
        return lambda x: norm.ppf(x, loc=mu, scale=sigma)
    
    def __call__(self, cube, ndim=None, nparams=None):
        if (ndim is None) and (nparams is None):
            self.cube_copy = cube
        else:
            self.cube_copy = np.array(cube[:ndim])
        
        for i, key_i in enumerate(self.param_keys):
            
            # nabla_0-5 are constrained ≤ nabla_RCE (set first in free_params); skip normal transform
            _nabla_bounded = ['nabla_0','nabla_1','nabla_2','nabla_3','nabla_4','nabla_5']
            if key_i not in _nabla_bounded:
                prior_i = self.param_priors[key_i]

                # Check if prior is a dictionary (Gaussian) or list (uniform)
                if isinstance(prior_i, dict) and prior_i.get('type') == 'gaussian':
                    cube[i] = self.gaussian_prior(prior_i['mu'], prior_i['sigma'])(cube[i])
                else:
                    cube[i] = self.uniform_prior(prior_i)(cube[i]) # cube is vector of length nparams, values [0,1]

            # DG constraint: all gradients must be ≤ nabla_RCE (peak at RCE boundary; Picos+2025 §4.4.2)
            # nabla_RCE must appear before nabla_0-5 in free_params for this to work
            if key_i in ['nabla_0','nabla_1','nabla_2']:  # convective zone: lower bound 0.04
                cube[i] = self.uniform_prior([0.04, self.params['nabla_RCE']])(cube[i])
            if key_i in ['nabla_3','nabla_4','nabla_5']:  # radiative zone: lower bound 0.00
                cube[i] = self.uniform_prior([0.00, self.params['nabla_RCE']])(cube[i])
            
            self.params[key_i] = cube[i] # add free parameter values to parameter dictionary

        return self.cube_copy
    
class Covariance:
     
    def __init__(self, err): 
        self.err = err
        self.cov_reset() # set up covariance matrix
        
    def cov_reset(self): # make diagonal covariance matrix from uncertainties
        self.cov = self.err**2
        print ('Covariance matrix set up with shape:', self.cov.shape)

    def get_logdet(self): # log of determinant
        self.logdet = np.sum(np.log(self.cov)) 
        return self.logdet

    def solve(self, b): # Solve: cov*x = b, for x (x = cov^{-1}*b)
        return 1/self.cov * b # if diagonal matrix, only invert the diagonal
    


class LogLikelihood:

    def __init__(self,retrieval_object, alpha=2):

        self.d_flux = retrieval_object.data_flux
        self.d_mask = retrieval_object.mask_isfinite
        self.scale_flux   = retrieval_object.per_chip_scaling # True | False | None | 'Single'
        self.scale_err    = (retrieval_object.per_chip_scaling is not False) # disable only when False
        self.N_d      = self.d_mask.sum() # number of degrees of freedom / valid datapoints
        self.N_params = retrieval_object.n_params
        self.alpha = alpha # from Ruffio+2019

        self.retrieval_object = retrieval_object
        # N_phi: number of linear scaling parameters marginalised in the log-likelihood
        #   True     -> one phi per (order, detector) pair
        #   None / 'Single' -> one global phi
        #   False    -> no scaling (N_phi = 0, but kept as 1 to avoid log(0) in Ruffio formula)
        if self.scale_flux is True:
            self.N_phi = retrieval_object.n_orders * retrieval_object.n_dets
        else:
            self.N_phi = 1
        
    def __call__(self, m_flux, Cov):

        self.ln_L   = 0.0
        self.chi2_0 = 0.0
        self.m_flux_phi = np.nan*np.ones_like(self.d_flux) # scaled model flux

        N_d = self.d_mask.sum() # Number of (valid) data points
        #d_flux = self.d_flux[self.d_mask] # data flux
        m_flux = m_flux[self.d_mask] # model flux
        
        if self.scale_flux is True:
            # Per-(order, detector) scaling: one phi per chip
            self.m_flux_phi[self.d_mask], self.phi = self.flux_scaling_rolling(m_flux, Cov)
        elif self.scale_flux in (None, 'Single'):
            # Single global scaling factor for the whole flattened spectrum
            d_flux_masked = self.d_flux[self.d_mask]
            self.m_flux_phi[self.d_mask], self.phi = self.get_flux_scaling(d_flux_masked, m_flux, Cov)
        else:  # False: no scaling
            self.m_flux_phi[self.d_mask] = m_flux
            self.phi = np.ones(1)

        residuals_phi = (self.d_flux - self.m_flux_phi) # Residuals wrt scaled model
        inv_cov_0_residuals_phi = Cov.solve(residuals_phi[self.d_mask]) 
        chi2_0 = np.dot(residuals_phi[self.d_mask].T, inv_cov_0_residuals_phi) # Chi-squared for the optimal linear scaling
        logdet_MT_inv_cov_0_M = 0

        #if self.scale_flux:
        inv_cov_0_M    = Cov.solve(m_flux) # Covariance matrix of phi
        MT_inv_cov_0_M = np.dot(m_flux.T, inv_cov_0_M)
        logdet_MT_inv_cov_0_M = np.log(MT_inv_cov_0_M) # (log)-determinant of the phi-covariance matrix

        if self.scale_err: 
            self.s2 = self.get_err_scaling(chi2_0, N_d)
        else:
            self.s2 = 1.0

        logdet_cov_0 = Cov.get_logdet()  # Get log of determinant (log prevents over/under-flow)
        self.ln_L += -1/2*(N_d-self.N_phi) * np.log(2*np.pi)+loggamma(1/2*(N_d-self.N_phi+self.alpha-1)) # see Ruffio+2019
        self.ln_L += -1/2*(logdet_cov_0+logdet_MT_inv_cov_0_M+(N_d-self.N_phi+self.alpha-1)*np.log(chi2_0))
        self.chi2_0 += chi2_0
        self.chi2_0_red = self.chi2_0 / self.N_d # Reduced chi-squared (take degrees of freedom into account)

        return self.ln_L

    def get_flux_scaling(self, d_flux, m_flux, cov): 
        # Solve for linear scaling parameter phi: (M^T * cov^-1 * M) * phi = M^T * cov^-1 * d
        lhs = np.dot(m_flux.T, cov.solve(m_flux)) # Left-hand side
        rhs = np.dot(m_flux.T, cov.solve(d_flux)) # Right-hand side
        phi = rhs / lhs # Optimal linear scaling factor
        return np.dot(m_flux, phi), phi # Return scaled model flux + scaling factors

    def get_err_scaling(self, chi_squared_scaled, N):
        s2 = np.sqrt(1/N * chi_squared_scaled)
        return s2 # uncertainty scaling that maximizes log-likelihood
    
    def flux_scaling_rolling(self, m_flux, Cov):
        # Optimize the flux scaling and error scaling by running through each detector/order separately
        m_flux_scaled = np.nan*np.ones_like(self.d_flux)
        phi_all = np.nan*np.ones((self.retrieval_object.n_orders*self.retrieval_object.n_dets))

        for order in range(self.retrieval_object.n_orders):
            for det in range(self.retrieval_object.n_dets):
                order_det_mask = (self.retrieval_object.data_wave >= self.retrieval_object.K2166[order,det,0]) & (self.retrieval_object.data_wave <= self.retrieval_object.K2166[order,det,1])
                d_flux_cut = self.d_flux[order_det_mask]
                m_flux_cut = m_flux[order_det_mask]
                Cov_cut = Cov.__class__(err=self.retrieval_object.data_err[order_det_mask]) # create new covariance object for cut data

                m_flux_scaled[order_det_mask], phi_all[order*self.retrieval_object.n_dets+det] = self.get_flux_scaling(d_flux_cut, m_flux_cut, Cov_cut)

        return m_flux_scaled, phi_all
    '''
    def error_scaling_rolling(self, chi2_0):
        # Optimize the error scaling by running through each detector/order separately
        s2_all = np.nan*np.ones((self.retrieval_object.n_orders*self.retrieval_object.n_dets))

        for order in range(self.retrieval_object.n_orders):
            for det in range(self.retrieval_object.n_dets):
                order_det_mask = (self.retrieval_object.data_wave >= self.retrieval_object.K2166[order,det,0]) & (self.retrieval_object.data_wave <= self.retrieval_object.K2166[order,det,1])
                N_d_cut = order_det_mask.sum()
                chi2_0_cut = chi2_0[order_det_mask]

                s2_all[order*self.retrieval_object.n_dets+det] = self.get_err_scaling(chi2_0_cut, N_d_cut)

        return s2_all
    '''
    
    

class pRT_spectrum:

    def __init__(self,
                 retrieval_object,
                 spectral_resolution=100_000,
                 contribution=False):  # only for plotting atmosphere.contr_em
        
        self.params=retrieval_object.parameters.params
        self.data_wave=retrieval_object.data_wave
        self.target=retrieval_object.target
        self.atmosphere_objects=retrieval_object.atmosphere_objects
        self.coords = SkyCoord(ra=self.target.ra, dec=self.target.dec, frame='icrs')
        self.species=retrieval_object.species
        self.spectral_resolution=spectral_resolution
        self.lbl_opacity_sampling=retrieval_object.lbl_opacity_sampling

        self.n_atm_layers=retrieval_object.n_atm_layers
        self.pressure = retrieval_object.pressure
        self.temperature = self.make_pt() #P-T profile

        self.gravity = 10**self.params['log_g'] 
        self.contribution=contribution

        self.normalize_flux = retrieval_object.normalize_flux
        self.eq_chem = getattr(retrieval_object, 'eq_chem', None)

        # free chemistry with defined VMRs
        #self.mass_fractions, self.CO, self.FeH = self.free_chemistry(self.species,self.params)
        if self.params.get('chemistry', 'free') == 'free':
            self.mass_fractions, self.CO, self.FeH = self.free_chemistry(
                self.species, self.params
            )
        elif self.params['chemistry'] == 'equilibrium':
            self.mass_fractions, self.CO, self.FeH = self.equilibrium_chemistry(
                self.params
            )
        else:
            raise ValueError("chemistry must be 'free' or 'equilibrium'")
        
        self.MMW = self.mass_fractions['MMW']
    
    def read_species_info(self,species,info_key):
        species_info = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'), index_col=0)
        if info_key == 'pRT_name':
            return species_info.loc[species,info_key]
        if info_key == 'mass':
            return species_info.loc[species,info_key]
        if info_key == 'COH':
            return list(species_info.loc[species,['C','O','H']])
        if info_key in ['C','O','H']:
            return species_info.loc[species,info_key]
        if info_key == 'label':
            return species_info.loc[species,'mathtext_name']
    
    def free_chemistry(self,line_species,params):
        species_info = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'), index_col=0)
        VMR_He = 0.15
        VMR_wo_H2 = 0 + VMR_He  # Total VMR without H2, starting with He
        mass_fractions = {} # Create a dictionary for all used species
        C, O, H = 0, 0, 0

        for species_i in species_info.index:
            line_species_i = self.read_species_info(species_i,'pRT_name')
            mass_i = self.read_species_info(species_i, 'mass')
            COH_i  = self.read_species_info(species_i, 'COH')

            if species_i in ['H2', 'He']:
                continue
            if line_species_i in line_species:
                VMR_i = 10**(params[f'log_{species_i}'])*np.ones(self.n_atm_layers) #  use constant, vertical profile

                # Convert VMR to mass fraction using molecular mass number
                mass_fractions[line_species_i] = mass_i * VMR_i
                VMR_wo_H2 += VMR_i

                # Record C, O, and H bearing species for C/O and metallicity
                C += COH_i[0] * VMR_i
                O += COH_i[1] * VMR_i
                H += COH_i[2] * VMR_i

        # Add the H2 and He abundances
        mass_fractions['He'] = self.read_species_info('He', 'mass')*VMR_He
        mass_fractions['H2'] = self.read_species_info('H2', 'mass')*(1-VMR_wo_H2)
        H += self.read_species_info('H2','H')*(1-VMR_wo_H2) # Add to the H-bearing species
        
        if VMR_wo_H2.any() > 1:
            print('VMR_wo_H2 > 1. Other species are too abundant!')

        MMW = 0 # Compute the mean molecular weight from all species
        for mass_i in mass_fractions.values():
            MMW += mass_i
        MMW *= np.ones(self.n_atm_layers)
        
        for line_species_i in mass_fractions.keys():
            mass_fractions[line_species_i] /= MMW # Turn the molecular masses into mass fractions
        mass_fractions['MMW'] = MMW # pRT requires MMW in mass fractions dictionary
        CO = C/O
        log_CH_solar = 8.46 - 12 # Asplund et al. (2021)
        FeH = np.log10(C/H)-log_CH_solar
        CO = np.nanmean(CO)
        FeH = np.nanmean(FeH)

        if mass_fractions['MMW'].any() < 1.0:
            print('MMW < 1.0! Check mass fractions!')

        return mass_fractions, CO, FeH
    
    def equilibrium_chemistry(self, params):
        """
        Equilibrium chemistry using pRT3 PreCalculatedEquilibriumChemistryTable.
        Table is loaded once in Retrieval.__init__ and passed via retrieval_object.
        Species abundances are a function of T, P, C/O, [Fe/H].
        13CO is absent from the table and is derived from 12CO using the solar
        12C/13C isotope ratio of 70 (Asplund et al. 2021).
        """
        FeH = params['FeH']   # log10 metallicity (solar = 0.0; table range: -2 to +3)
        CO  = params['C/O']   # C/O ratio         (solar ≈ 0.55; table range: 0.1 to 1.6)

        log_Pquench = params.get('log_Pquench', None)
        carbon_pressure_quench = 10**log_Pquench if log_Pquench is not None else None

        mass_fractions_eq, MMW, _ = self.eq_chem.interpolate_mass_fractions(
            co_ratios              = CO  * np.ones(self.n_atm_layers),
            log10_metallicities    = FeH * np.ones(self.n_atm_layers),
            temperatures           = self.temperature,
            pressures              = self.pressure,
            carbon_pressure_quench = carbon_pressure_quench,
            full                   = True,
        )
        # Table keys: 'H2','He','CO','H2O','CH4','HCN','CO2','NH3','H2S','FeH',
        #             'C2H2','PH3','VO','TiO','Na','K','SiO','e-','H-','H'

        mass_fractions = {}
        simple_names = simplify_species_list(self.species)

        for pRT_name, simple_name in zip(self.species, simple_names):
            if pRT_name == '13C-16O':
                # 13CO absent from table; derive from 12CO using the retrieved isotopologue ratio
                # log_12CO_13CO is log10(12CO/13CO); solar value = log10(70) ≈ 1.845
                ratio_12_13 = 10**params.get('log_12CO_13CO', np.log10(70.0))
                mass_fractions['13C-16O'] = (
                    mass_fractions_eq['CO'] * (29.002355 / 28.009999) / ratio_12_13
                )
            elif simple_name in mass_fractions_eq:
                mass_fractions[pRT_name] = mass_fractions_eq[simple_name]
            else:
                print(f'WARNING: {pRT_name} ({simple_name}) not in equilibrium table, skipping.')

        mass_fractions['H2']  = mass_fractions_eq['H2']
        mass_fractions['He']  = mass_fractions_eq['He']
        mass_fractions['MMW'] = MMW

        return mass_fractions, CO, FeH

    
    def make_spectrum(self):

        print(f"Temperature range: {np.min(self.temperature):.1f} - {np.max(self.temperature):.1f} K")
        print(f"Pressure range: {np.min(self.pressure):.2e} - {np.max(self.pressure):.2e} bar")
        print("Mass fractions:")
        for species, mf in self.mass_fractions.items():
            if species != 'MMW':
                print(f"  {species}: {np.mean(mf):.2e}")


        atmosphere=self.atmosphere_objects

        #  Check if atmosphere object is properly set up
        print(f"Atmosphere species: {atmosphere.line_species}")

        '''
        pRT2 codes
        atmosphere.calculate_flux(Temperatures=self.temperature,
                        mass_fractions=self.mass_fractions,
                        reference_gravity=self.gravity,
                        MMW=self.MMW,
                        contribution=self.contribution)
        '''
        # --pRT3 codes--
        wl, flux, _=atmosphere.calculate_flux(temperatures=self.temperature,
                        mass_fractions=self.mass_fractions,
                        reference_gravity=self.gravity,
                        mean_molar_masses=self.MMW, 
                        return_contribution=True,
                        frequencies_to_wavelengths=True) # returns flux in W/m2/um
        wl *= 1e7 # convert wavelengths from cm to nm

        #wl = const.c.to(u.km/u.s).value/atmosphere.frequencies/1e-9 # mircons, pRT2: freq -- pRT3: frequencies
        #flux=atmosphere.flux/np.median(atmosphere.flux) --pRT2 codes--

        print(f"Raw flux range: {np.min(flux):.2e} - {np.max(flux):.2e}")
        print(f'Raw wavelength range: {np.min(wl):.2f} - {np.max(wl):.2f} nm')
        print(f"Flux median: {np.nanmedian(flux):.2e}")

        # RV+bary shifting and rotational broadening
        #v_bary, _ = helcorr(obs_long=-70.40, obs_lat=-24.62, obs_alt=2635, # of Cerro Paranal
                        #ra2000=self.coords.ra.value,dec2000=self.coords.dec.value,jd=self.target.JD) # https://ssd.jpl.nasa.gov/tools/jdc/#/cd
        v_bary = 0 # for testing, set barycentric velocity to 0 to not shift the spectrum and make sure everything else works first
        print (f"Barycentric velocity: {v_bary:.2f} km/s")
        
        wl_shifted= wl*(1.0+(self.params['rv']-v_bary)/const.c.to('km/s').value)
        waves_even = np.linspace(np.min(wl), np.max(wl), wl.size) # wavelength array has to be regularly spaced
        new_spec = np.interp(waves_even, wl_shifted, flux)
        spec = fastRotBroad(waves_even, new_spec, 0.5, self.params['vsini']) # limb-darkening coefficient (0-1)   
        spec = self.convolve_to_resolution(waves_even, spec, self.spectral_resolution)
        self.resolution = int(1e6/self.lbl_opacity_sampling)
        flux=self.instr_broadening(waves_even, spec,out_res=self.resolution,in_res=500000)

        if np.all(flux == flux[0]):
            print("WARNING: Flux is constant!")
            print("This suggests an issue with the atmospheric model")

        # Interpolate/rebin onto the data's wavelength grid
        data_reshape = self.data_wave.reshape(1,-1) if len(self.data_wave.shape) == 3 else self.data_wave
        ref_wave = data_reshape.flatten() # [nm]
        flux = np.interp(ref_wave, waves_even, flux) # pRT wavelengths from cm to nm

        if self.contribution==True:
            contr_em = atmosphere.contr_em # emission contribution
            self.summed_contr = np.nansum(contr_em,axis=1) # sum over all wavelengths

        #combine with telluric template if needed

        # normalize flux if needed
        if self.normalize_flux == 'median':
            # Global median normalisation: one scalar across all orders and detectors
            # so that inter-chip flux ratios (SED shape) are preserved.
            global_median = np.nanmedian(flux)
            flux /= global_median

        elif self.normalize_flux == 'savgol':
            # Normalize each order/detector chunk separately to mirror the data preprocessing.
            window_length = 301
            polyorder = 2

            for order in range(self.target.n_orders):
                for det in range(self.target.n_dets):
                    order_det_mask = (
                        (self.data_wave >= self.target.K2166[order, det, 0])
                        & (self.data_wave <= self.target.K2166[order, det, 1])
                    )
                    finite_mask = order_det_mask & np.isfinite(flux)

                    n_finite = np.sum(finite_mask)
                    if n_finite <= polyorder:
                        continue

                    local_window = min(window_length, n_finite)
                    if local_window % 2 == 0:
                        local_window -= 1
                    if local_window <= polyorder:
                        continue

                    smooth_local = savgol_filter(
                        flux[finite_mask], local_window, polyorder
                    )
                    scale_local = np.nanmedian(np.abs(smooth_local))
                    continuum_floor = np.finfo(float).eps * scale_local if scale_local > 0 else 0.0
                    safe_local = np.isfinite(smooth_local) & (np.abs(smooth_local) > continuum_floor)

                    idx = np.where(finite_mask)[0]
                    flux[idx[safe_local]] = flux[idx[safe_local]] / smooth_local[safe_local]
                    flux[idx[~safe_local]] = np.nan
            
            #flux /= np.nanmedian(flux) # normalize flux -- pRT3 codes--
            print ("Normalized model flux by per-chip Savitzky-Golay filter (window up to 301, polynomial order 2).")
        
        elif self.normalize_flux == False or self.normalize_flux is None:
            print ("No flux normalization applied for the model spectrum.")
            pass  # do nothing  
        
        else:
            print("normalize_flux must be 'median', 'savgol', or False")
      
  


        return flux
            
    # Dynamic Gradients (DG) T-P profile — Picos+2025 §4.4.2, Eqs. 11-12, Table C.1
    def make_pt(self):
        log_P_RCE  = self.params['log_P_RCE']
        dlog_P_bot = self.params['dlog_P_bot']
        dlog_P_top = self.params['dlog_P_top']

        # 7 pressure nodes from deep (10² bar) to shallow (10⁻⁵ bar) — Eq. 11
        log_P_nodes = np.array([
             2.0,                             # P_0 = 10² bar (fixed bottom)
             log_P_RCE + 2*dlog_P_bot,        # below RCE, convective zone
             log_P_RCE +   dlog_P_bot,        # just below RCE
             log_P_RCE,                        # P_RCE (peak gradient)
             log_P_RCE -   dlog_P_top,        # just above RCE, radiative
             log_P_RCE - 2*dlog_P_top,        # radiative zone
            -5.0,                             # P_6 = 10⁻⁵ bar (fixed top)
        ])
        nabla_nodes = np.array([
            self.params['nabla_0'],
            self.params['nabla_1'],
            self.params['nabla_2'],
            self.params['nabla_RCE'],
            self.params['nabla_3'],
            self.params['nabla_4'],
            self.params['nabla_5'],
        ])

        # Linearly interpolate ∇ to all 50 pressure layers (np.interp needs ascending x)
        log_P_atm = np.log10(self.pressure)   # ascending: index 0 = shallowest
        nabla_interp = np.interp(log_P_atm, log_P_nodes[::-1], nabla_nodes[::-1])

        # Compute T bottom-up via Eq. 12: T_j = T_{j-1} * (P_j/P_{j-1})^{∇_j}
        # pressure is ascending → index -1 is the deepest layer (100 bar)
        temp = np.empty(len(self.pressure))
        temp[-1] = self.params['T_bottom']
        for j in range(len(self.pressure) - 2, -1, -1):
            temp[j] = temp[j+1] * (self.pressure[j] / self.pressure[j+1])**nabla_interp[j]

        self.temperature = temp
        return self.temperature
    
    def instr_broadening(self, wave, flux, out_res=1e6, in_res=1e6):
        # Delta lambda of resolution element is FWHM of the LSF's standard deviation
        sigma_LSF = np.sqrt(1/out_res**2-1/in_res**2)/(2*np.sqrt(2*np.log(2)))
        spacing = np.mean(2*np.diff(wave) / (wave[1:] + wave[:-1]))
        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF / spacing
        # Apply gaussian filter to broaden with the spectral resolution
        flux_LSF = gaussian_filter(flux, sigma=sigma_LSF_gauss_filter,mode='nearest')
        return flux_LSF
    
    def convolve_to_resolution(self, in_wlen, in_flux, out_res, in_res=None):
        if isinstance(in_wlen, u.Quantity):
            in_wlen = in_wlen.to(u.nm).value
        if in_res is None:
            in_res = np.mean((in_wlen[:-1]/np.diff(in_wlen)))
        # delta lambda of resolution element is FWHM of the LSF's standard deviation:
        sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))
        spacing = np.mean(2.*np.diff(in_wlen)/(in_wlen[1:]+in_wlen[:-1]))
        # Calculate the sigma to be used in the gauss filter in pixels
        sigma_LSF_gauss_filter = sigma_LSF/spacing
        out_flux = np.tile(np.nan, in_flux.shape)
        nans = np.isnan(in_flux)
        out_flux[~nans] = gaussian_filter(in_flux[~nans], sigma = sigma_LSF_gauss_filter,mode = 'reflect')
        return out_flux
    
class Retrieval:

    def __init__(self,parameters,N_live_points,evidence_tolerance, target, testing=True, normalize_flux=False, ppid=None, per_chip_scaling=False):
        
        self.N_live_points=int(N_live_points) # number of live points
        self.evidence_tolerance=float(evidence_tolerance) # evidence tolerance
        self.target=target
        #self.data_wave,self.data_flux,self.data_err=self.target.load_spectrum()
        self.mask_isfinite=self.target.get_mask_isfinite() # mask nans, shape (orders,detectors)
        self.K2166=self.target.K2166 # wavelength setting
        self.parameters=parameters
        self.species=self.get_species(param_dict=self.parameters.params)

        # Load equilibrium chemistry table once (1.4 GB HDF5; expensive to reload per call)
        chemistry_mode = self.parameters.params.get('chemistry', 'free')
        if chemistry_mode == 'equilibrium':
            print('Loading equilibrium chemistry table (done once)...')
            self.eq_chem = PreCalculatedEquilibriumChemistryTable()
            self.eq_chem.load()
            print('Equilibrium chemistry table loaded.')
        else:
            self.eq_chem = None

        self.testing=testing
        self.normalize_flux=normalize_flux
        self.per_chip_scaling=per_chip_scaling
        
        self.n_orders=self.target.n_orders
        self.n_dets=self.target.n_dets

        self.K2166=self.target.K2166 # wavelength setting
        
        # Add job-specific identifier
        '''
        if job_id is None:
            job_id = os.environ.get('SLURM_JOB_ID', f'job_{int(time.time())}')
        self.job_id = job_id
        '''

        if ppid is None:
            ppid = os.getppid()
        self.job_id = ppid

        if testing == True:
            ############# for now, let's just do one  order/detector ###########
            self.order= target.n_orders -1  # last order
            self.detector= 1  # middle detector

            #set up a clipper to only use the certain part of the spectrum if its 1D
            if len(self.target.wl.shape) ==1: # if loading flattened 1D spectra
                clip_range = self.K2166[self.order,self.detector]
                clip_mask = (self.target.wl >= clip_range[0]) & (self.target.wl <= clip_range[1])
                self.target.wl = self.target.wl[clip_mask]
                self.target.fl = self.target.fl[clip_mask]
                self.target.err = self.target.err[clip_mask]
                self.target.mask_isfinite = self.target.mask_isfinite[clip_mask]
            
            elif len(self.target.wl.shape) ==3: # if loading 3D spectra (orders, detectors, pixels)
                self.target.wl = self.target.wl[self.order,self.detector]
                self.target.fl = self.target.fl[self.order,self.detector]
                self.target.err = self.target.err[self.order,self.detector]
                self.target.mask_isfinite = self.target.mask_isfinite[self.order,self.detector]


            self.data_wave=self.target.wl
            self.data_flux=self.target.fl
            self.data_err=self.target.err
            self.mask_isfinite=self.target.mask_isfinite
            
            if normalize_flux==True:
                self.data_flux /= np.nanmedian(self.data_flux) #normalize input data flux
                print ("Normalized input data flux by its median value.")
        
        else:
            self.data_wave=self.target.wl
            self.data_flux=self.target.fl
            self.data_err=self.target.err
            self.mask_isfinite=self.target.mask_isfinite

            #if normalize_flux==True:
            #self.data_flux /= np.nanmedian(self.data_flux) #normalize input data flux
            # print ("Normalized input data flux by its median value.")
        ####################################################################
        
        self.n_params = len(parameters.free_params)
        self.output_dir = results_root / f'{self.job_id}_N{self.N_live_points}_ev{self.evidence_tolerance}_Norm{self.normalize_flux}_PerChipScale{self.per_chip_scaling}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lbl_opacity_sampling=3
        self.n_atm_layers=50
        self.pressure = np.logspace(-5,2,self.n_atm_layers)  # 10⁻⁵–10² bar; matches DG anchor points (Picos+2025)
        #err_masked = np.where(self.mask_isfinite, self.data_err, np.nan)
        err_masked = self.data_err[self.mask_isfinite]
        self.Cov = Covariance(err=err_masked) # simple diagonal covariance matrix with masked data but only valid pixels
        self.LogLike = LogLikelihood(retrieval_object=self)

        # redo atmosphere objects when adding new species
        self.atmosphere_objects=self.get_atmosphere_objects()
        self.callback_label='live_' # label for plots
        self.prefix='pmn_'
        self.color=self.target.color

    def get_species(self, param_dict):
        """Get list of pRT opacity species names. Handles both free and equilibrium chemistry modes."""
        species_info = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'species_info.csv'), index_col=0)

        if param_dict.get('chemistry', 'free') == 'equilibrium':
            # Use pRT3-valid isotopologue names directly (bypasses species_info.csv,
            # which stores pRT2-style names like CO2_main_iso / FeH_main_iso).
            # simplify_species_list maps these back to equilibrium table keys in
            # equilibrium_chemistry(). Adjust to match your science goals.
            eq_species = [
                '1H2-16O',   # H2O
                '12C-16O',   # 12CO
                '13C-16O',   # 13CO (abundance derived from 12CO * isotopologue ratio)
                '12C-1H4__MM',   # CH4 — MM/ExoMol (2 line lists available; MM chosen over HITRAN2024)
                '14N-1H3',   # NH3
                '1H2-32S',   # H2S
                '56Fe-1H',   # FeH
                # HCN and CO2 excluded: no K-band (1.9-2.5 um) LBL opacity files locally
                # (main absorption bands at ~3 um and ~4.3 um respectively)
            ]
            self.chem_species = eq_species
            return eq_species

        # Free chemistry: scan for log_* params (existing logic, unchanged)
        self.chem_species = []
        for par in param_dict:
            if 'log_' in par and par != 'log_g':
                self.chem_species.append(par)
        species = []
        for chemspec in self.chem_species:
            species.append(species_info.loc[chemspec[4:], 'pRT_name'])
        return species

    def get_atmosphere_objects(self,redo=True):

        file = results_root / f'atmosphere_objects_{self.job_id}_N{self.N_live_points}.pickle'
        if file.exists() and redo==False:
            with open(file,'rb') as file:
                atmosphere_objects=pickle.load(file)
                return atmosphere_objects
        else:
            print ('Creating new atmosphere objects...')

            wl_pad=7 # wavelength padding because spectrum is not wavelength shifted yet
            if self.testing==True:
             
                wlmin=np.min(self.K2166[self.order])-wl_pad
                wlmax=np.max(self.K2166[self.order])+wl_pad
            else:
                wlmin=np.min(self.data_wave)-wl_pad
                wlmax=np.max(self.data_wave)+wl_pad

            wlen_range=np.array([wlmin,wlmax])*1e-7 # nm to cm

            boundary = wlen_range * 1e4  # cm to micron

            print(f'Wavelength range for atmosphere object: {wlen_range*1e7} nm')

            #check up the following parameters for atmosphere object
            print(f'Line species: {self.species}')
            print(f'Pressure levels: {self.pressure}')
            print(f'Wavelength boundaries: {boundary}')
            print(f'Line-by-line opacity sampling: {self.lbl_opacity_sampling}')


            atmosphere_objects = Radtrans(line_species=self.species,
                                rayleigh_species = ['H2', 'He'],
                                gas_continuum_contributors = ['H2--H2', 'H2--He'],
                                wavelength_boundaries=boundary, 
                                line_opacity_mode='lbl',
                                line_by_line_opacity_sampling=self.lbl_opacity_sampling,
                                pressures=self.pressure) # take every nth point (=3 in deRegt+2024)
            

            #atmosphere_objects.setup_opa_structure(self.pressure)
            with open(file,'wb') as file: # save so that they don't need to be created every time
                pickle.dump(atmosphere_objects,file)
            return atmosphere_objects

    def PMN_lnL(self,cube=None,ndim=None,nparams=None):
        self.model_object=pRT_spectrum(self)
        self.model_flux=self.model_object.make_spectrum()
        ln_L = self.LogLike(self.model_flux, self.Cov) # calcuate log-likelihood
        print (f'PMN lnL: {ln_L}')
        return ln_L

    def PMN_run(self,N_live_points=None, evidence_tolerance=0.5, resume=False, mpi=True): # run pymultinest
        pymultinest.run(LogLikelihood=self.PMN_lnL,
                        Prior=self.parameters,
                        n_dims=self.parameters.n_params, 
                        outputfiles_basename=f'{self.output_dir}/{self.prefix}', 
                        verbose=True,const_efficiency_mode=True,sampling_efficiency = 0.5,
                        n_live_points=N_live_points,
                        resume=resume, # resume from prevous unfinished run
                        evidence_tolerance=evidence_tolerance, # recommended is 0.5, high number -> finished earlier
                        dump_callback=self.PMN_callback,
                        n_iter_before_update=10, 
                        use_MPI=mpi) # iterations until calling PMN_callback

    # provides live updates during the retrieval
    def PMN_callback(self,n_samples,n_live,n_params,live_points,posterior, 
                    stats,max_ln_L,ln_Z,ln_Z_err,nullcontext):
        
        print (f'PMN callback at {n_samples} samples, max lnL: {max_ln_L}, lnZ: {ln_Z} +/- {ln_Z_err}')
        self.bestfit_params = posterior[np.argmax(posterior[:,-2]),:-2] # parameters of best-fitting model
        self.posterior = posterior[:,:-2] # remove last 2 columns to get posterior
        self.params_dict,self.model_flux, self.model_flux_scaled, self.data_err_scaled=self.get_params_and_spectrum()
        self.cornerplot() # make cornerplot of live posterior
     
    def PMN_analyse(self):
        # set up PMN analyzer object
        analyzer = pymultinest.Analyzer(n_params=self.parameters.n_params,
                                        outputfiles_basename=f'{self.output_dir}/{self.prefix}')  
        stats = analyzer.get_stats()
        self.posterior = analyzer.get_equal_weighted_posterior() # equally-weighted posterior distribution
        self.posterior = self.posterior[:,:-1] 
        np.save(f'{self.output_dir}/{self.callback_label}posterior.npy',self.posterior)
        self.lnZ = stats['nested importance sampling global log-evidence']

    def get_params_and_spectrum(self): 
        
        # make dictionary of evaluated parameters
        self.params_dict={}
        for i,key in enumerate(self.parameters.param_keys):
            medians = np.array([np.percentile(self.posterior[:,j], [50.0], axis=-1) for j in range(self.posterior.shape[1])])
            self.params_dict[key]=medians[i] # add median of evaluated params

        # create final spectrum
        self.model_object=pRT_spectrum(self)
        self.model_flux=self.model_object.make_spectrum()
        self.params_dict['[Fe/H]']=self.model_object.FeH
        self.params_dict['C/O']=self.model_object.CO

        # get scaling parameters phi and s2 of bestfit model through likelihood
        self.log_likelihood = self.LogLike(self.model_flux, self.Cov)
        self.params_dict['phi']=self.LogLike.phi # flux scaling
        self.params_dict['s2']=self.LogLike.s2 # error scaling
        self.params_dict['chi2']=self.LogLike.chi2_0_red # save reduced chi^2
        if self.callback_label=='final_':
            self.params_dict['lnZ']=self.lnZ # save lnZ
                
        with open(f'{self.output_dir}/{self.callback_label}params_dict.pickle','wb') as file:
            pickle.dump(self.params_dict,file)

        #calculate final scaled model spectrum
        if self.LogLike.scale_flux is True:
            # Per-(order, detector) scaling: phi is an array of length n_orders*n_dets
            self.model_flux_scaled = np.nan*np.ones_like(self.model_flux)
            for order in range(self.n_orders):
                for det in range(self.n_dets):
                    order_det_mask = (self.data_wave >= self.K2166[order,det,0]) & (self.data_wave <= self.K2166[order,det,1])
                    self.model_flux_scaled[order_det_mask] = self.model_flux[order_det_mask] * self.LogLike.phi[order*self.n_dets+det]
        elif self.LogLike.scale_flux in (None, 'Single'):
            # Single global scaling: phi is a scalar
            self.model_flux_scaled = self.model_flux * self.LogLike.phi
        else:
            # scale_flux is False: no scaling
            self.model_flux_scaled = self.model_flux

        #calculate the final error bars after scaling
       
        self.data_err_scaled = np.dot(self.data_err, self.LogLike.s2)
        
        return self.params_dict, self.model_flux, self.model_flux_scaled, self.data_err_scaled

    def evaluate(self):
        self.callback_label='final_'
        self.PMN_analyse() # get/save bestfit params and final posterior
        self.params_dict,self.model_flux, self.model_flux_scaled, self.data_err_scaled=self.get_params_and_spectrum() # all params + scaling
        self.cornerplot()

    def run_retrieval(self): 

        print(f'\n ------ Nlive: {self.N_live_points} - ev: {self.evidence_tolerance} ------ \n')
        # Enable MultiNest MPI parallelization if env var PMN_MPI=="1" (default)
        #mpi_flag = os.environ.get('PMN_MPI', '1') == '1'
        self.PMN_run(N_live_points=self.N_live_points,
                     evidence_tolerance=self.evidence_tolerance)
        self.evaluate() # creates plots and saves self.params_dict
        print('\n ----------------- Done ---------------- \n')
        
    def cornerplot(self):
        labels=list(self.parameters.param_mathtext.values())
        fontsize=10
        fig = plt.figure(figsize=(self.n_params,self.n_params),dpi=200) # fix size to avoid memory issues
        corner.corner(self.posterior, 
                        labels=labels, 
                        title_kwargs={'fontsize':fontsize},
                        label_kwargs={'fontsize':fontsize},
                        color=self.color,
                        linewidths=0.5,
                        fill_contours=True,
                        quantiles=[0.16,0.5,0.84],
                        title_quantiles=[0.16,0.5,0.84],
                        show_titles=True,
                        fig=fig,
                        quiet=True) # supresses errors
        plt.subplots_adjust(wspace=0,hspace=0)
        plt.rc('xtick',labelsize=fontsize)
        plt.rc('ytick',labelsize=fontsize)     

        #plt.show()
        
        fig.savefig(f'{self.output_dir}/{self.callback_label}cornerplot.pdf',bbox_inches="tight",dpi=200)
        
        plt.close() # to avoid memory issues










#------------------main code starts here------------------#




#---------------------------------------------------------------#
#------------------load data and normalization------------------#
#---------------------------------------------------------------#


# Two-night combined spectrum — generated by cooking_combine_spectra_inter_night.ipynb
# Wavelength solution is the night-2022-12-31 barycentric grid (nm).
# Because the data are in the barycentric frame, the retrieved rv is the true
# barycentric stellar RV (~16.3 km/s), NOT the topocentric observed value (~31.7 km/s).
#combined_dir = '/data2/peng/combined_two_nights/'
'''
spectra_AB     = np.load(combined_dir + 'extracted_spectra_combined_two_nights.npy')      # (3, 5, 2048)
spectra_AB_err = np.load(combined_dir + 'extracted_spectra_combined_two_nights_err.npy')  # (3, 5, 2048)
wave           = np.load(combined_dir + 'WLEN_combined_two_nights_bary.npy')               # (3, 5, 2048) nm
# moving axis to (orders, detectors, pixels)
spectra_AB_reordered = np.transpose(spectra_AB, (1, 0, 2))  # (5, 3, 2048)
spectra_AB_err_reordered = np.transpose(spectra_AB_err, (1, 0, 2))  # (5, 3, 2048)
wave_reordered = np.transpose(wave, (1, 0, 2))  # (5, 3, 2048)
'''
#Singlw-night spectrum — generated by cooking_combine_spectra.ipynb for night 2022-12-31
night = '2022-12-31'

spectra_AB = np.load(f'/data2/peng/{night}/extracted_spectra_combined_sigmaclipper.npy') #(3, 5, 2048)
spectra_AB_err = np.load(f'/data2/peng/{night}/extracted_spectra_combined_err_sigmaclipper.npy') #(3, 5, 2048)

path_wl_cal = workpath + night +'/cal/WLEN_K2166_V_DH_Tau_A+B_center.fits'
wave_hdu = fits.open(path_wl_cal)
wave = np.array(wave_hdu[1].data)[:,0:5,]  # (3, 5, 2048)

#Applying barycentric-calibrated wavelength solution
#path_wl_cal = workpath + night + '/barycentric_wavelengths_night1.npy' #(3, 5, 2048)
#wave = np.load(path_wl_cal) #(3, 5, 2048)


# moving axis to (orders, detectors, pixels)
spectra_AB_reordered = np.transpose(spectra_AB, (1, 0, 2))  # (5, 3, 2048)
spectra_AB_err_reordered = np.transpose(spectra_AB_err, (1, 0, 2))  # (5, 3, 2048)
wave_reordered = np.transpose(wave, (1, 0, 2))  # (5, 3, 2048)


Normalize_method = 'savgol' # choose normalization method: 'median' or 'savgol'
scaling_parameter = False  # whether to include scaling parameters in the retrieval (phi and s2)

#Normalize the spectra before flattening


if Normalize_method == 'median':
    # Global median normalisation: divide all chips by one scalar so that
    # inter-chip flux ratios (SED shape) are fully preserved.
    global_median = np.nanmedian(spectra_AB_reordered)
    spectra_AB_reordered     /= global_median
    spectra_AB_err_reordered /= global_median
    print(f'Global median normalisation applied (median = {global_median:.4e})')

elif Normalize_method == 'savgol':
    window_length = 301  # must be odd number
    polyorder = 2

    #create a copy of the specra to hole the continuum proxy (low-frequency feature) for each order and detector
    continuum_proxy = np.full_like(spectra_AB_reordered, np.nan, dtype=float)


    #### extract the low-frequency feature using low-pass savgol filter and divide it out
    for order in range(5):
        for det in range(3):

            mask_finite = np.isfinite(spectra_AB_reordered[order, det])
            low_freq_flux_full = np.full(spectra_AB_reordered[order, det].shape, np.nan)
            low_freq_flux_full[mask_finite] = savgol_filter(
                spectra_AB_reordered[order, det][mask_finite], window_length, polyorder
            )
            continuum_proxy[order, det] = low_freq_flux_full

            # set a floor for the continuum relative to the data scale so that
            # physical-unit fluxes (e.g. ~1e-23 W m⁻² µm⁻¹) are not wiped out
            # by a fixed floor of eps*1.0 ≈ 2.2e-16
            scale = np.nanmedian(np.abs(low_freq_flux_full[mask_finite]))
            if scale == 0:
                spectra_AB_reordered[order, det][mask_finite] = np.nan
                spectra_AB_err_reordered[order, det][mask_finite] = np.nan
                continue
            continuum_floor = np.finfo(float).eps * scale

            safe_mask = mask_finite & np.isfinite(low_freq_flux_full) & (np.abs(low_freq_flux_full) > continuum_floor)

            spectra_AB_reordered[order, det][safe_mask] /= low_freq_flux_full[safe_mask]
            spectra_AB_err_reordered[order, det][safe_mask] /= low_freq_flux_full[safe_mask]
            spectra_AB_reordered[order, det][mask_finite & ~safe_mask] = np.nan
            spectra_AB_err_reordered[order, det][mask_finite & ~safe_mask] = np.nan
else:
    print ("No normalization applied to the spectra before flattening.")
    
#Summarize the setup
print (f'Normalization method: {Normalize_method}')
print (f'Scaling parameters included in likelihood maximization: {scaling_parameter}')

#------------------------------------------------------------------#
#--------------------- flattening and masking ---------------------#
#------------------------------------------------------------------#

#flatten the spectra to (1, 2048*number of orders*number of detectors)

spectra_AB_reshape = spectra_AB_reordered.reshape(1, -1)
spectra_AB_err_reshape = spectra_AB_err_reordered.reshape(1, -1)
wave_reshape = wave_reordered.reshape(1, -1)

spectra_AB_flat = spectra_AB_reshape.flatten()
spectra_AB_err_flat = spectra_AB_err_reshape.flatten()
wave_flat = wave_reshape.flatten()


#check if flattening is correct
print ('shape check after flattening:')
print('wave:', wave_flat.shape)
print('spectra_AB:', spectra_AB_flat.shape)
print('spectra_AB_err:', spectra_AB_err_flat.shape)

#check if the wavelength, spectra, and error are aligned after flattening
for i in range(0, len(wave_flat), 10000):
    print(f'Index {i}: Wavelength = {wave_flat[i]}, Spectrum = {spectra_AB_flat[i]}, Error = {spectra_AB_err_flat[i]}')

    # if not aligned, stop the code
    # track back i to the original indices
    orig_order = i // (3 * 2048)
    orig_det = (i % (3 * 2048)) // 2048
    orig_pixel = i % 2048
    print(f'Original indices - Order: {orig_order}, Detector: {orig_det}, Pixel: {orig_pixel}')

    #set up the boolean check (NaN-safe: treat NaN==NaN as True)
    def _eq(a, b):
        return (np.isnan(a) and np.isnan(b)) or (a == b)

    bool_check = _eq(wave_flat[i],        wave_reordered[orig_order, orig_det, orig_pixel]) and \
                 _eq(spectra_AB_flat[i],   spectra_AB_reordered[orig_order, orig_det, orig_pixel]) and \
                 _eq(spectra_AB_err_flat[i], spectra_AB_err_reordered[orig_order, orig_det, orig_pixel])

    if not bool_check:
        print("Data misalignment detected! Exiting.")
        exit()

#----masking nan and inf values----#

#turn null values to nan and then mask all nans and infs

spectra_AB_flat = np.where(spectra_AB_flat==0, np.nan, spectra_AB_flat)
spectra_AB_err_flat = np.where(spectra_AB_err_flat==0, np.nan, spectra_AB_err_flat)

mask = np.isfinite(spectra_AB_flat) & np.isfinite(spectra_AB_err_flat)


#spectra_AB_reordered = np.where(spectra_AB_reordered==0, np.nan, spectra_AB_reordered)
#spectra_AB_err_reordered = np.where(spectra_AB_err_reordered==0, np.nan, spectra_AB_err_reordered)

#mask = np.isfinite(spectra_AB_reordered) & np.isfinite(spectra_AB_err_reordered)

#turn the nan and inf values to 0 to avoid issues in retrieval
#spectra_AB_reordered = np.where(np.isfinite(spectra_AB_reordered), spectra_AB_reordered, 0)
#spectra_AB_err_reordered = np.where(np.isfinite(spectra_AB_err_reordered), spectra_AB_err_reordered, 0)

#-------------------------------------------------------#
#------------set up retriveal parameters----------------#
#-------------------------------------------------------#

# -------------------------------------------------------#
# --- FREE CHEMISTRY (default) --------------------------#
# -------------------------------------------------------#
'''
constant_params = {#'rv': 32,
                   #'log_g': 3.5,
                   #'T0' : 3000, # bottom of the atmosphere (hotter)
                   #'T1' : 2000,
                   #'T2' : 1200,
                   #'T3' : 800,
                   #'T4' : 400, # top of atmosphere (cooler)
                   }

# free parameters we will retrieve - format: key, prior range, mathtext label (for plotting)
# For uniform prior: 'param': ([min, max], r'label')
# For Gaussian prior: 'param': ({'type': 'gaussian', 'mu': mean, 'sigma': std}, r'label')
free_params = {#'rv': ([27, 37], r'$v_{\rm rad}$'), # km/s - uniform prior
                'rv': ({'type': 'gaussian', 'mu': 32, 'sigma': 0.2}, r'$v_{\rm rad}$'), # km/s - Gaussian prior based on literature value
                'vsini': ([0,20], r'$v$ sin$i$'), # km/s
                #'vsini': ({'type': 'gaussian', 'mu': 9.6, 'sigma': 0.5}, r'$v$ sin$i$'), # km/s
                'log_g':({'type': 'gaussian', 'mu': 3.5, 'sigma': 0.2}, r'log $g$'),
    # --- DG temperature profile (Picos+2025 §4.4.2, Table C.1) ---
    # nabla_RCE MUST be listed before nabla_0-5: its value is used to bound the others in Parameters.__call__
    # Priors informed by GQ Lup B Table C.1 and DH Tau B retrievals 2098448 & 2347145
    'nabla_RCE':  ([0.04, 0.34],  r'$\nabla_{T,\rm RCE}$'),   # peak gradient at RCE boundary
    'nabla_0':    ([0.04, 0.34],  r'$\nabla_{T,0}$'),          # at P_0=10² bar; constrained ≤ nabla_RCE
    'nabla_1':    ([0.04, 0.34],  r'$\nabla_{T,1}$'),          # at P_RCE+2*dlog_P_bot; constrained ≤ nabla_RCE
    'nabla_2':    ([0.04, 0.34],  r'$\nabla_{T,2}$'),          # at P_RCE+dlog_P_bot; constrained ≤ nabla_RCE
    'nabla_3':    ([0.00, 0.34],  r'$\nabla_{T,3}$'),          # at P_RCE-dlog_P_top; constrained ≤ nabla_RCE
    'nabla_4':    ([0.00, 0.34],  r'$\nabla_{T,4}$'),          # at P_RCE-2*dlog_P_top; constrained ≤ nabla_RCE
    'nabla_5':    ([0.00, 0.34],  r'$\nabla_{T,5}$'),          # at P_6=10⁻⁵ bar; constrained ≤ nabla_RCE
    'T_bottom':   ([1500, 5000],  r'$T_{\rm bot}$'),           # K at P_0=10² bar; old retrieval ~2300-2536 K
    'log_P_RCE':  ([-3.0,  1.0],  r'log $P_{\rm RCE}$'),      # log10(bar); photosphere at ~0.1-1 bar
    'dlog_P_bot': ([0.20,  1.60], r'$\Delta\log P_{\rm bot}$'), # spacing of lower nodes (Picos: 0.27)
    'dlog_P_top': ([0.20,  1.60], r'$\Delta\log P_{\rm top}$'), # spacing of upper nodes (Picos: 0.75)
                'log_H2O':([-12,-2], r'log H$_2$O'), # if free chemistry, define VMRs, follow Picos et al. 2024 priors for GQ Lup b
                'log_12CO':([-12,-2], r'log $^{12}$CO'), #the value is log of volume mixing ratio
                'log_13CO':([-12,-2], r'log $^{13}$CO'),
                'log_CH4':([-12,-2], r'log CH$_4$')
                }
'''
# -------------------------------------------------------#
# --- EQUILIBRIUM CHEMISTRY (uncomment to use) ----------#
# -------------------------------------------------------#
# To run equilibrium chemistry:
#   1. Comment out the free_params and constant_params blocks above
#   2. Uncomment the blocks below
#   3. The species list is fixed in get_species() for K-band:
#      H2O, 12CO, 13CO, CH4, NH3, H2S, HCN, CO2, FeH
#      (edit eq_species_names in get_species() to change)
#
constant_params = {
     'chemistry': 'equilibrium',   # REQUIRED: activates equilibrium code path
}
free_params = {
     # rv prior: data are barycentric-corrected, so rv = true barycentric stellar RV.
     # v_bary_star = rv_topocentric + v_bary_corr = 31.7 + (-15.4) ≈ 16.3 km/s
     # (consistent between both nights after bary-correction to 0.12 km/s)
     'rv':    ({'type': 'gaussian', 'mu': 32, 'sigma': 0.5}, r'$v_{\rm rad}$'),
     'vsini': ([0, 20],                                          r'$v$ sin$i$'),
     # log_g: widened from sigma=0.2 to 0.3 to span the N1/N2 spread (3.61–3.83);
     # centre raised to 3.7 from the 1-D marginal product (3.76 ± 0.09)
     'log_g': ({'type': 'gaussian', 'mu': 3.7, 'sigma': 0.3},  r'log $g$'),
     'nabla_RCE':  ([0.04, 0.34],  r'$\nabla_{T,\rm RCE}$'),
     'nabla_0':    ([0.04, 0.34],  r'$\nabla_{T,0}$'),
     'nabla_1':    ([0.04, 0.34],  r'$\nabla_{T,1}$'),
     'nabla_2':    ([0.04, 0.34],  r'$\nabla_{T,2}$'),
     'nabla_3':    ([0.00, 0.34],  r'$\nabla_{T,3}$'),
     'nabla_4':    ([0.00, 0.34],  r'$\nabla_{T,4}$'),
     'nabla_5':    ([0.00, 0.34],  r'$\nabla_{T,5}$'),
     'T_bottom':   ([2000, 5000],  r'$T_{\rm bot}$'),
     'log_P_RCE':  ([-3.0,  1.0],  r'log $P_{\rm RCE}$'),
     'dlog_P_bot': ([0.20,  1.60], r'$\Delta\log P_{\rm bot}$'),
     'dlog_P_top': ([0.20,  1.60], r'$\Delta\log P_{\rm top}$'),
     'FeH':            ([-2.0, 3.0], r'[Fe/H]'),   # log10 metallicity; table: -2 to +3; solar = 0.0
     'C/O':            ([0.1,  1.0], r'C/O'),       # C/O ratio; solar ≈ 0.55
     'log_12CO_13CO':  ([0.0,  6.0], r'log $^{12}$CO/$^{13}$CO'),  # log10 isotopologue ratio; solar log10(70) ≈ 1.85
     # Carbon quench pressure (optional; freeze C-bearing species above this P):
     # 'log_Pquench': ([-6, 2], r'log $P_{\rm quench}$'),
}

N_points = 600
evidence_tol = 0.5


# initialize parameters class object
parameters = Parameters(free_params,constant_params)

# initialize free parameters by randomly drawing from their prior ranges
cube = np.random.rand(parameters.ndim) 
parameters(cube)

species=pd.DataFrame(parameters.param_keys,columns=['species'])
#species.to_csv('species_info.csv',index=False)

# initialize retrieval object
T=Target(wl=wave_flat[mask], fl=spectra_AB_flat[mask], err=spectra_AB_err_flat[mask], name='dh_tau_b')

#order =4 
#det = 1
#mask = np.isfinite(spectra_AB_reordered[order][det]) & np.isfinite(spectra_AB_err_reordered[order][det])
#T=Target(wl=wave_flat, fl=spectra_AB_reordered, err=spectra_AB_err_reordered, name='dh_tau_b')

retrieval=Retrieval(parameters=parameters, 
                    N_live_points=N_points, 
                    evidence_tolerance=evidence_tol, 
                    target=T, 
                    testing=False, 
                    normalize_flux=Normalize_method, per_chip_scaling=scaling_parameter)

#--------------------------------------------------------------------#
#-----------------RUN RETRIEVAL and save the results-----------------#
#--------------------------------------------------------------------#

retrieval.run_retrieval()



retrieval.evaluate()
retrieval.get_params_and_spectrum()

#turn null values into nan
params_dict, model_flux, model_flux_scaled, model_flux_err = retrieval.get_params_and_spectrum()

np.place(retrieval.data_flux, retrieval.data_flux==0, np.inf)
np.place(model_flux, model_flux==0, np.inf)
np.place(model_flux_scaled, model_flux_scaled==0, np.inf)

np.save(retrieval.output_dir / 'retrieval_model_flux.npy', model_flux)
np.save(retrieval.output_dir / 'retrieval_model_flux_scaled.npy', model_flux_scaled)
np.save(retrieval.output_dir / 'retrieval_model_wave.npy', retrieval.data_wave)


#----------------------------------------------------------------------#
#-----------------PLOT THE BEST-FIT MODEL AND RESIDUALS----------------#
#----------------------------------------------------------------------#

mask= np.isfinite(retrieval.data_flux)
#mask_model = np.isfinite(model_flux_scaled)


K2166 = T.K2166

################################---PLOT 1: data and model spectrum for each order and detector---###################################

fig, ax = plt.subplots(3, 5, figsize=(20, 10))

#plot the data and model according to [order, det]:

for order in range(5):
    for det in range(3):
        wave_border = K2166[4-order][det]
        wave_cut_data = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
        wave_cut_model = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]

        data_flux_cut = retrieval.data_flux[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
        model_flux_cut = model_flux_scaled[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]

        residuals = data_flux_cut - model_flux_cut
        std_residuals = np.std(residuals)

        ax[det][order].scatter(wave_cut_data, residuals, label='residuals', color='blue', linewidth=1, alpha=0.7)
        ax[det][order].axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax[det][order].axhline(3*std_residuals, color='red', linestyle=':', linewidth=0.8, label='3 sigma')
        ax[det][order].axhline(-3*std_residuals, color='red', linestyle=':', linewidth=0.8)

                                                       
        ax[det][order].set_title('Order %s - Det %s'%(27-order, det+1))
        ax[det][order].legend() 

plt.tight_layout()

plt.savefig(retrieval.output_dir / 'retrieval_data_model_residuals.png', dpi=300)

################################---PLOT 2: data and model spectrum for each order and detector---###################################

fig, ax = plt.subplots(3, 5, figsize=(20, 10))



for order in range(5):
    for det in range(3):
        wave_border = K2166[4-order][det]

        wave_cut_data = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
        wave_cut_model = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]

        data_flux_cut = retrieval.data_flux[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
        model_flux_cut = model_flux_scaled[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]

        ax[det][order].plot(wave_cut_data, data_flux_cut, label='data', color='darkgray', linewidth=0.6, alpha=0.7)
        ax[det][order].plot(wave_cut_model, model_flux_cut, label='model', color='red', linewidth=2, alpha=0.7)

        ax[det][order].set_ylim(np.nanmin(data_flux_cut)*0.8, np.nanmedian(data_flux_cut)*3)

        ax[det][order].set_title('Order %s - Det %s'%(27-order, det+1))
        ax[det][order].legend() 

'''
for order in range(5):
    for det in range(3):

        wave_border = K2166[4-order][det]

        if retrieval.testing==True:
            mask = retrieval.mask_isfinite
            mask_model = np.isfinite(model_flux_scaled)
            wave_cut_data = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
            wave_cut_model = retrieval.data_wave[mask_model][(retrieval.data_wave[mask_model]>=wave_border.min()) & (retrieval.data_wave[mask_model]<=wave_border.max())]

            data_flux_cut = retrieval.data_flux[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
            model_flux_cut = model_flux_scaled[mask_model][(retrieval.data_wave[mask_model]>=wave_border.min()) & (retrieval.data_wave[mask_model]<=wave_border.max())]

            ax[det][order].plot(wave_cut_data, data_flux_cut, label='data', color='darkgray', linewidth=0.6, alpha=0.7)
            ax[det][order].plot(wave_cut_model, model_flux_cut, label='model', color='red', linewidth=2, alpha=0.7)

        else:

            mask = np.isfinite(retrieval.data_flux[order][det])
            mask_model = np.isfinite(model_flux_scaled[order][det]) 

            ax[det][order].plot(retrieval.data_wave[order][det][mask], retrieval.data_flux[order][det][mask], label='data', color='darkgray', linewidth=0.6, alpha=0.7)
            ax[det][order].plot(retrieval.data_wave[order][det][mask_model], model_flux_scaled[order][det][mask_model], label='model', color='red', linewidth=2, alpha=0.7) 
'''

plt.tight_layout()

#plt.xlim(2245.229,2259.888)

plt.savefig(retrieval.output_dir / 'retrieval_data_model_spectrum.png', dpi=300)
#plt.show()



#################################################################---PLOT 3: P-T profile---#######################################
plt.figure(figsize=(6,8))
plt.plot(retrieval.model_object.temperature, retrieval.model_object.pressure)
plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')
plt.gca().invert_yaxis()

#plt.xlim(1e-6,5)

plt.xlabel('Pressure (bar)')
plt.ylabel('Temperature (K)')


plt.savefig(retrieval.output_dir / 'retrieval_PT_profile.png', dpi=300)
##plt.show()

######################### PLOT 4: data and model spectrum for each order and detector after binning/smoothing the original data to better see the overall trend (noisy data)---###################################
#bin the original data and plot for every order and detector
from scipy.signal import savgol_filter

plt.figure(figsize=(10,6))
fig, ax = plt.subplots(3,5, figsize=(20, 10))

'''
mask = np.isfinite(retrieval.data_flux)
mask_model = np.isfinite(retrieval.model_flux_scaled)
'''

data_flux_bin = savgol_filter(retrieval.data_flux[mask], 51, 2)  # window size 51, polynomial order 2
#data_flux_bin_err = savgol_filter(retrieval.data_err[mask], 51, 2)  # window size 51, polynomial order 2

model_flux_bin = savgol_filter(model_flux_scaled[mask], 51, 2)  # window size 51, polynomial order 2
#model_flux_bin_err = savgol_filter(retrieval.data_err[mask_model], 51, 2)  # window size 51, polynomial order 2



for det in range(3):
    for order in range(5):

        wave_border = K2166[4-order][det]

        
        
        #mask_model = np.isfinite(model_flux)

        wave_cut_data = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
        wave_cut_model = retrieval.data_wave[mask][(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]

        data_flux_cut = data_flux_bin[(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]
        model_flux_cut = model_flux_bin[(retrieval.data_wave[mask]>=wave_border.min()) & (retrieval.data_wave[mask]<=wave_border.max())]

        ax[det][order].scatter(wave_cut_data, data_flux_cut/np.nanmean(data_flux_cut), label='data', s=4, color='darkgray')
        ax[det][order].scatter(wave_cut_model, model_flux_cut/np.nanmean(model_flux_cut), label='model', s=4, color='red')

        ax[det][order].set_title('Order %s - Det %s'%(27-order, det+1))
        ax[det][order].legend()



plt.savefig(retrieval.output_dir / 'retrieval_data_model_spectrum_binned.png', dpi=300)
#plt.show()
print ('+++++++++++ Retrieval and plotting complete +++++++++++')
print ('Arrivederci! ^_^')
 
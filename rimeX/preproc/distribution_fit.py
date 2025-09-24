"""
Fits a non-stationary GEV (dependent on the warming level) to each each timeseries of values for a given indicator and ISIMIP-simulation of a given model and then returns return period values for each position and warming level 
"""
import os
from pathlib import Path
import argparse
import concurrent.futures
import glob
import tqdm
from itertools import groupby, product
import warnings
import math
import numpy as np
import pandas as pd
import xarray as xa
from typing import List

from rimeX.logs import logger, log_parser, setup_logger
from rimeX.config import CONFIG, config_parser
from rimeX.datasets.download_isimip import get_models, get_experiments, get_variables, isimip_parser
from rimeX.datasets.download_isimip import Indicator, _matches
from rimeX.compat import open_dataset, open_mfdataset
from rimeX.tools import dir_is_empty
from rimeX.preproc.warminglevels import get_warming_level_file
from rimeX.preproc.regional_average import get_all_regions

import numpy as np
from scipy.stats import genextreme
from scipy.optimize import minimize
import math
#import dask.array as da
from multiprocessing import Pool, RawArray
from functools import partial
import multiprocessing as mp 
import warnings
warnings.filterwarnings("ignore")

mp.set_start_method('fork')




def get_all_regions():
    """returns regions with masks available

    Returns:
        list: all regions as they are named in the masks
    """    
    return sorted(o.name for o in Path(CONFIG["preprocessing.regional.masks_folder"]).glob("*") if not dir_is_empty(o) and not dir_is_empty(o / "masks"))


def fit_nonstationary_gev_rxNday(values, gmt):
    """Fit GEV distribution dependent on GMT to values and gmt  

    Args:
        values (np.array): GEV distributed values
        gmt (np.array): GMT levels when the values where recorded
    Raises:
        RuntimeError: GMT fit doesn't work

    Returns:
        tuple: parameters of GMT dependent GEV that fit best to the values 
    """    
    # Fit stationary GEV for initial parameters
    c_initial, mu_initial, sigma_initial = genextreme.fit(values)
    xi_initial = -c_initial  # Convert SciPy's c to ξ

    # Initial parameters: [a0, a1, b0, b1, xi]
    initial_params = np.array([
        1, mu_initial, sigma_initial, xi_initial        # a0, a1 (location parameters)
         # b0, b1 (log-scale parameters)
                      # ξ (shape parameter)
    ])

    # Negative log-likelihood function
    def neg_log_likelihood(params):
        """Return neg log likelihood values for current parameters

        Args:
            params (tuple): the four parameters for our non-stationary GEV

        Returns:
            float: neg log likelihood value from trying to reproduce our values with a GEV described by the paras
        """        
        alpha, mu0, sigma0, xi = params
        
        #CAUTION this use of alpha is only guerenteed to not be stupid for precipitation indicators - will have to figure out for other indicators 
        mu = mu0 * np.exp( alpha * gmt/mu0)
        sigma = sigma0 * np.exp( alpha * gmt/mu0)
        
        z = (values - mu) / sigma

        # Check support condition: 1 + ξ*z > 0 for all data points
        if np.any(1 + xi * z <= 0):
            return np.inf  # Return infinity if invalid

        # Compute log-likelihood using SciPy's GEV (c = -ξ)
        log_pdfs = genextreme.logpdf(values, c=-xi, loc=mu, scale=sigma)
        return -np.sum(log_pdfs)

    # Optimize parameters
    result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead', options={'maxfev': 5000, 'maxiter': 5000})  # Increase these values)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    return result.x

def fit_nonstationary_gev_tempBaseMu(values, gmt):
    """Fit GEV distribution dependent on GMT to values and gmt  

    Args:
        values (np.array): GEV distributed values
        gmt (np.array): GMT levels when the values where recorded
    Raises:
        RuntimeError: GMT fit doesn't work

    Returns:
        tuple: parameters of GMT dependent GEV that fit best to the values 
    """    
    # Fit stationary GEV for initial parameters
    c_initial, mu_initial, sigma_initial = genextreme.fit(values)
    xi_initial = -c_initial  # Convert SciPy's c to ξ

    # Initial parameters: [a0, a1, b0, b1, xi]
    initial_params = np.array([
        1, mu_initial, sigma_initial, xi_initial        # a0, a1 (location parameters)
         # b0, b1 (log-scale parameters)
                      # ξ (shape parameter)
    ])

    # Negative log-likelihood function
    def neg_log_likelihood(params):
        """Return neg log likelihood values for current parameters

        Args:
            params (tuple): the four parameters for our non-stationary GEV

        Returns:
            float: neg log likelihood value from trying to reproduce our values with a GEV described by the paras
        """        
        alpha, mu0, sigma0, xi = params
        
        #CAUTION this use of alpha is only guerenteed to not be stupid for precipitation indicators - will have to figure out for other indicators 
        mu = mu0 + alpha * gmt/mu0
        sigma = sigma0 
        
        z = (values - mu) / sigma

        # Check support condition: 1 + ξ*z > 0 for all data points
        if np.any(1 + xi * z <= 0):
            return np.inf  # Return infinity if invalid

        # Compute log-likelihood using SciPy's GEV (c = -ξ)
        log_pdfs = genextreme.logpdf(values, c=-xi, loc=mu, scale=sigma)
        return -np.sum(log_pdfs)

    # Optimize parameters
    result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead', options={'maxfev': 5000, 'maxiter': 5000})  # Increase these values)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    return result.x

def fit_nonstationary_gev_tempBaseMuSigma(values, gmt):
    """Fit GEV distribution dependent on GMT to values and gmt  

    Args:
        values (np.array): GEV distributed values
        gmt (np.array): GMT levels when the values where recorded
    Raises:
        RuntimeError: GMT fit doesn't work

    Returns:
        tuple: parameters of GMT dependent GEV that fit best to the values 
    """    
    # Fit stationary GEV for initial parameters
    c_initial, mu_initial, sigma_initial = genextreme.fit(values)
    xi_initial = -c_initial  # Convert SciPy's c to ξ

    # Initial parameters: [a0, a1, b0, b1, xi]
    initial_params = np.array([
        1, 1, mu_initial, sigma_initial, xi_initial        # a0, a1 (location parameters)
         # b0, b1 (log-scale parameters)
                      # ξ (shape parameter)
    ])

    # Negative log-likelihood function
    def neg_log_likelihood(params):
        """Return neg log likelihood values for current parameters

        Args:
            params (tuple): the four parameters for our non-stationary GEV

        Returns:
            float: neg log likelihood value from trying to reproduce our values with a GEV described by the paras
        """        
        alpha, beta, mu0, sigma0, xi = params
        
        #CAUTION this use of alpha is only guerenteed to not be stupid for precipitation indicators - will have to figure out for other indicators 
        mu = mu0 + alpha * gmt/mu0
        sigma = sigma0 + beta*gmt/sigma0
        
        z = (values - mu) / sigma

        # Check support condition: 1 + ξ*z > 0 for all data points
        if np.any(1 + xi * z <= 0):
            return np.inf  # Return infinity if invalid

        # Compute log-likelihood using SciPy's GEV (c = -ξ)
        log_pdfs = genextreme.logpdf(values, c=-xi, loc=mu, scale=sigma)
        return -np.sum(log_pdfs)

    # Optimize parameters
    result = minimize(neg_log_likelihood, initial_params, method='Nelder-Mead', options={'maxfev': 5000, 'maxiter': 5000})  # Increase these values)

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")
    
    return result.x



def get_return_period( mu0, sigma0, xi,alpha, beta, warming_levels, return_periods, indicator_name):
    """_summary_

    Args:
        alpha (float): GEV param
        mu0 (float): GEV param
        sigma0 (float): GEV param
        xi (float): GEV param
        warming_levels (np.array): warming levels you want to have return periods for
        return_periods (np.array): the return periods you want to calculate for each warming level

    Returns:
        np.array: all return period values for all warming levels from the GEV with the given params 
    """    

    return_periods = 1-1/return_periods
    out = np.zeros([warming_levels.size, return_periods.size])
    xi = -xi
    
    for i,warming_level in enumerate(warming_levels): 
        
        if indicator_name in ['rx1day','rx5day']:
        
            mu = mu0 * np.exp(alpha * warming_level/mu0)
            sigma = sigma0 * np.exp(alpha * warming_level/mu0)
            
        
        elif indicator_name in ['HImax', 'spei_gamma_12_min']: 
            
            mu = mu0 + alpha * warming_level / mu0
            sigma = sigma0
            
        
        elif indicator_name in ['TXx']: 
            
            mu = mu0 + alpha * warming_level / mu0
            sigma = sigma0 + beta * warming_level / sigma0
           
        
        out[i,:] = genextreme.ppf(return_periods, c=xi, loc=mu, scale=sigma)
        

    return out

def process_cell_GEV_fitting_and_evaluation(lat_idx, lon_idx, region_idx):
    """Fits a non-stationary GEV contrained by GMT to all simulated values of the GEV-distributed indicator and their respective GMT values at the region with index region_idx or position (lat_idx, lon_idx)
    and calculates shared_return_periods_out at shared_warming_levels_out

    Args:
        lat_idx (int): index of the lat value or None when region_idx is given
        lon_idx (int): index of the lon value or None when region_idx is given
        region_idx (int): index of the region or None when lat_idx and lon_idx is given

    Returns:
        np.array: numpy array including all return periods; dimension: (warming_levels_out.size, return_periods_out.size) 
    """    
    global shared_values, shared_gmt, n_time, n_lat, n_lon, n_regions, shared_warming_levels_out, shared_return_periods_out, indicator_name
    # Access shared arrays

    if region_idx is not None:
        values_2d = np.frombuffer(shared_values, dtype=np.float64).reshape(n_time, n_regions)
        gmt_1d = np.frombuffer(shared_gmt, dtype=np.float64)
        values_1d = values_2d[:, region_idx]
        gmt_1d = gmt_1d[:]

    else: 
        values_3d = np.frombuffer(shared_values, dtype=np.float64).reshape(n_time, n_lat, n_lon)
        gmt_3d = np.frombuffer(shared_gmt, dtype=np.float64)
        values_1d = values_3d[:, lat_idx, lon_idx]
        gmt_1d = gmt_3d[:]
    
    warming_levels_out = np.frombuffer(shared_warming_levels_out, dtype=np.float64)
    return_periods_out = np.frombuffer(shared_return_periods_out, dtype=np.float64)
    
    
    
    try:
        if indicator_name in ['rx1day','rx5day']:
            params = fit_nonstationary_gev_rxNday(values_1d, gmt_1d)
            alpha, mu0, sigma0, xi = tuple(params)
            beta = 0
            
            return_periods = get_return_period(mu0, sigma0, xi, alpha, beta, warming_levels_out, return_periods_out, indicator_name)
           
        elif indicator_name in ['HImax', 'spei_gamma_12_min']:
            
            if indicator_name == 'spei_gamma_12_min': 
                values_1d = - values_1d
            
            params = fit_nonstationary_gev_tempBaseMu(values_1d, gmt_1d)
            alpha, mu0, sigma0, xi = tuple(params)
            beta = 0
            
     
            return_periods = get_return_period(mu0, sigma0, xi, alpha, beta, warming_levels_out, return_periods_out, indicator_name)
            
            if indicator_name == 'spei_gamma_12_min': 
                return_periods = -return_periods
            
        elif indicator_name in ['TXx']:
            params = fit_nonstationary_gev_tempBaseMuSigma(values_1d, gmt_1d)
            alpha, beta, mu0, sigma0, xi = tuple(params)
            
            return_periods = get_return_period(mu0, sigma0, xi, alpha, beta, warming_levels_out, return_periods_out, indicator_name)

     
        return return_periods 
    
    except Exception as e:
        
        print(e)
   
        return np.full([warming_levels_out.size, return_periods_out.size], np.nan)


def make_return_period_array(
    indicator: Indicator,
    warming_levels_simulations: pd.DataFrame,
    model: str,
    impact_model: str,
    frequency: str,
    warming_levels_output: list,
    return_periods_output: list,
    cpus: int):
    """Fits a non-stationary GEV (dependent on the warming level) to each each timeseries of values for a given indicator and ISIMIP-simulation of a given model 
    and then returns return period values for each position and warming level 

    Args:
        indicator (Indicator): Instance of the indicator class, for now actually only max precip indicators should work
        warming_levels_simulations (pd.DataFrame): a dataframe with the warming levels of all simulations, you get it by runing rime-pre-warming-levels
        model (str): a model that offers simulations for the indicator
        impact_model (str): an impact model that offer simulations for the indicator or None
        frequency (str): the frequerncy of the indicator, for now only annual works
        warming_levels_output (list): warming levels dependent on which the return period values in the outputfile should be calculated 
        return_periods_output (list): return periods calculated for the output
        cpus (int): number of cpus to be used, Warning: script takes about 20h / number of cpus of time 

    Returns:
        xr.dataset: Dataset with dimensions (lat, lon, return_period, warming_level) that includes the values of the return periods modeled in the GEV dependent on the warming level and lat, lon
    """    
    
    # Need to have global variables to be accessable from each paralel process
    global shared_values, shared_gmt, n_time, n_lat, n_lon, shared_warming_levels_out, shared_return_periods_out, n_regions, indicator_name #params, n_params, warming_levels_out, return_periods_out,
    
    n_regions = None
    indicator_name = indicator.name
    simulations = []

    #Filter simulations to only consider simulations from one model (and impact model)
    for simulation in indicator.simulations: 
    
        if impact_model is None: 
            if simulation["climate_forcing"] == model.lower():
                simulations.append(simulation)
        else: 
            if simulation["climate_forcing"] == model.lower() and simulation["model"] == impact_model:
                simulations.append(simulation)

    simulation_paths = []
    warming_levels = []


    # store number of simulated warming levels for each simulation index
    num_warming_levels_simulation = {}
    
    #costruct paths for the simulation
    for i, simulation in enumerate(simulations):
        
        if "ensemble" in simulation.keys():
            ensemble = f'_{simulation["ensemble"]}'
        elif "model" in simulation.keys():
            ensemble = f'_{simulation["model"]}'
        else:
            ensemble = ''
        
        if simulation['climate_scenario'] == 'historical': 
            timeline = f'{CONFIG["isimip.historical_year_min"]+1}_2014'
        else: 
            #sometimes there are model simulations that only go until 2099
            timeline = '2015_2*'

        simulation_paths.append(Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/{simulation["climate_scenario"]}/{simulation["climate_forcing"].lower()}/{simulation["climate_forcing"].lower()}{ensemble}_{simulation["climate_scenario"]}_{indicator.name}_{frequency}_{timeline}.nc'))
        
        if simulation['climate_scenario'] == 'historical':
            warming_levels_simulation = warming_levels_simulations[(warming_levels_simulations.experiment == simulation['climate_scenario']) & (warming_levels_simulations.model == model)].warming_level.values
            warming_levels_simulation_2 = warming_levels_simulations[(warming_levels_simulations.experiment == 'ssp370') & (warming_levels_simulations.model == model) & (warming_levels_simulations.year < 2015)].warming_level.values
            warming_levels_simulation = np.concatenate([warming_levels_simulation,warming_levels_simulation_2])
            num_warming_levels_simulation[str(simulation_paths[-1])] = len(warming_levels_simulation)
        else: 
            warming_levels_simulation = warming_levels_simulations[(warming_levels_simulations.experiment == simulation['climate_scenario']) & (warming_levels_simulations.model == model) & (warming_levels_simulations.year >= 2015)].warming_level.values
            num_warming_levels_simulation[str(simulation_paths[-1])] = len(warming_levels_simulation)
        warming_levels = warming_levels + list(warming_levels_simulation)

    
    

    #fit GEV for all lats and lons
    with xa.open_dataset(simulation_paths[0]) as data:
        lats = list(data.lat.values)
        lons = list(data.lon.values)

    #out = np.zeros([len(lats), len(lons), len(warming_levels_output), len(return_periods_output)])

    #lat_lon = list(product(lats,lons))

    all_simulation_datasets = []

    for i, path in tqdm.tqdm(enumerate(simulation_paths)):

        # have to load file 
        file_pattern = str(path)
        matching_files = sorted(glob.glob(file_pattern))

        
        assert len(matching_files) == 1, f'Aborted processing for {model} because several files found for pattern: {file_pattern}'
        
        simulation_data = xa.open_mfdataset(matching_files)

        if 'historical' in str(path): 
            simulation_data = simulation_data.isel(time = slice(-num_warming_levels_simulation[str(path)],None))
        else:
            simulation_data = simulation_data.isel(time = slice(0, num_warming_levels_simulation[str(path)]))
        
        
        all_simulation_datasets.append(simulation_data)
    
    all_simulations = xa.concat(all_simulation_datasets, dim='time').load()

    warming_levels_all_simulations = xa.Dataset(
    data_vars=dict(
        warming_level=(["time"], np.array(warming_levels)),
    ),
    coords=dict(
        time=np.array(all_simulations.time.values),
    ))
    if 'bnds' in all_simulations.dims.keys():
        all_simulations_dataarray = all_simulations.transpose('time','lat','lon', 'bnds')[indicator.name].values
    else: 
        all_simulations_dataarray = all_simulations.transpose('time','lat','lon')[indicator.name].values
    warming_levels_all_simulations_dataarray = warming_levels_all_simulations['warming_level'].values
    
    
    n_time, n_lat, n_lon = all_simulations_dataarray.shape

    # Create shared memory arrays 
    shared_values = RawArray('d', all_simulations_dataarray.ravel())  # 'd' for double (float64)
    shared_gmt = RawArray('d', warming_levels_all_simulations_dataarray.ravel())
    shared_warming_levels_out = RawArray('d', np.array(warming_levels_output).ravel())
    shared_return_periods_out = RawArray('d', np.array(return_periods_output).ravel())

    ctx = mp.get_context('fork')  
    with ctx.Pool(processes=cpus) as pool:
        results = pool.starmap(process_cell_GEV_fitting_and_evaluation, [(lat, lon, None) for lat in range(n_lat) for lon in range(n_lon)])

    results_array = np.array(results).reshape(n_lat, n_lon, len(warming_levels_output), len(return_periods_output))

    results = xa.Dataset(
    data_vars={
        indicator.name:(["lat", "lon", "warming_level", "return_period"], results_array),
    },
    coords=dict(
        lat=all_simulations.lat.values,
        lon= all_simulations.lon.values,
        warming_level=np.array(warming_levels_output),
        return_period=np.array(return_periods_output),
    ),
    attrs=dict(description=f"Return period value of {indicator.name} at warming level as predicted by a GEV fit on all available simulations within ISIMIP."),
    )


    return results

def make_return_period_array_regional_averages(
    indicator: Indicator,
    warming_levels_simulations: pd.DataFrame,
    regions: list,
    model: str,
    impact_model: str,
    aggregation: str,
    frequency: str,
    warming_levels_output: list,
    return_periods_output: list,
    cpus: int,):
    """Fits a non-stationary GEV (dependent on the warming level) to each each timeseries of values for a given indicator aggregated over a given region 
    and ISIMIP-simulation of a given model. Returns return period values for each region and warming level 

    Args:
        indicator (Indicator): Instance of the indicator class, for now actually only max precip indicators should work
        warming_levels_simulations (pd.DataFrame): a dataframe with the warming levels of all simulations, you get it by runing rime-pre-warming-levels
        regions (list): a list of regions to consider
        model (str): a model that offers simulations for the indicator
        impact_model (str): an impact model that offer simulations for the indicator or None
        aggregation (str): aggregation type of the regional data, e.g. latWeight for area weights
        frequency (str): the frequerncy of the indicator, for now only annual works
        warming_levels_output (list): warming levels dependent on which the return period values in the outputfile should be calculated 
        return_periods_output (list): return periods calculated for the output
        cpus (int): number of cpus to be used, Warning: script takes about 20h / number of cpus of time 

    Returns:
        xr.dataset: Dataset with dimensions (region, return_period, warming_level) that includes the values of the return periods modeled in the GEV dependent on the warming level and region
    """    
    
    # Need to have global variables to be accessable from each paralel process
    global shared_values, shared_gmt, n_time, n_lat, n_lon, n_regions, shared_warming_levels_out, shared_return_periods_out, indicator_name #params, n_params, warming_levels_out, return_periods_out,
    
    indicator_name = indicator.name
     
    simulations = []

    #Filter simulations to only consider simulations from one model (and impact model)
    for simulation in indicator.simulations: 
    
        if impact_model is None: 
            if simulation["climate_forcing"] == model.lower():
                simulations.append(simulation)
        else: 
            if simulation["climate_forcing"] == model.lower() and simulation["model"] == impact_model:
                simulations.append(simulation)

    simulation_paths = {}
    warming_levels = []

    # store number of simulated warming levels for each simulation index
    num_warming_levels_simulation = {}
    
    #costruct paths for the simulation
    for i, simulation in enumerate(simulations):

        simulation_identifier = '_'.join([string for string in simulation.values()])

        simulation_regional_files = {}
        for region in regions: 
        
            if "ensemble" in simulation.keys():
                ensemble = f'_{simulation["ensemble"]}'
            elif "model" in simulation.keys():
                ensemble = f'_{simulation["model"]}'
            else:
                ensemble = ''
            
            if simulation['climate_scenario'] == 'historical': 
                timeline = f'{CONFIG["isimip.historical_year_min"]+1}_2014'
            else: 
                #sometimes there are model simulations that only go until 2099
                timeline = '2015_2*'

            file_pattern = str(Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/{simulation["climate_scenario"]}/{simulation["climate_forcing"].lower()}/{aggregation}/{region}/{simulation["climate_forcing"].lower()}{ensemble}_{simulation["climate_scenario"]}_{indicator.name}_{region.lower()}_{aggregation.lower()}_{frequency}_{timeline}.csv'))
            matching_files = sorted(glob.glob(file_pattern))

        
            assert len(matching_files) == 1, f'Aborted processing for {model} because {len(matching_files)} files found for pattern: {file_pattern} (expected 1)' 

            simulation_regional_files[region] = Path(matching_files[0])

        simulation_paths[simulation_identifier] = simulation_regional_files

        if simulation['climate_scenario'] == 'historical':
            warming_levels_simulation = warming_levels_simulations[(warming_levels_simulations.experiment == simulation['climate_scenario']) & (warming_levels_simulations.model == model)].warming_level.values
            warming_levels_simulation_2 = warming_levels_simulations[(warming_levels_simulations.experiment == 'ssp370') & (warming_levels_simulations.model == model) & (warming_levels_simulations.year < 2015)].warming_level.values
            warming_levels_simulation = np.concatenate([warming_levels_simulation,warming_levels_simulation_2])
            num_warming_levels_simulation[simulation_identifier] = len(warming_levels_simulation)
        else: 
            warming_levels_simulation = warming_levels_simulations[(warming_levels_simulations.experiment == simulation['climate_scenario']) & (warming_levels_simulations.model == model) & (warming_levels_simulations.year >= 2015)].warming_level.values
            num_warming_levels_simulation[simulation_identifier] = len(warming_levels_simulation)
        warming_levels = warming_levels + list(warming_levels_simulation)


    all_simulations = []
    all_simulation_datasets = []

    for simulation_identifier, simulation_regional_files in simulation_paths.items():

        #time_column = pd.read_csv(simulation_regional_files.values()[0])['time']
        all_regional_averages = [pd.read_csv(path).drop(columns=['time']) for path in simulation_regional_files.values()]
        all_regional_data_simulation = pd.concat(all_regional_averages, axis=1)
        #all_regional_data_simulation.insert(0, 'time', time_column)

        if 'historical' in str(simulation_identifier): 
            simulation_data = all_regional_data_simulation.iloc[-num_warming_levels_simulation[simulation_identifier]:]
        else:
            simulation_data = all_regional_data_simulation.iloc[:num_warming_levels_simulation[simulation_identifier]]
        
        
        all_simulation_datasets.append(simulation_data)
    
    all_simulations = pd.concat(all_simulation_datasets, axis = 0)


    
    warming_levels_all_simulations = xa.Dataset(
    data_vars=dict(
        warming_level=(["time"], np.array(warming_levels)),
    ),
    coords=dict(
        time=np.array(list(range(np.array(warming_levels).size))),
    ))

    data_all_simulations = xa.Dataset(
    data_vars={
        indicator.name: ([ "time", "region"], np.array(all_simulations.values)),
    },
    coords=dict(
        time=np.array(list(range(np.array(warming_levels).size))),
        region=np.array(list(all_simulations.columns)),
    ))
    
    all_simulations_dataarray = data_all_simulations.transpose('time','region')[indicator.name].values
    warming_levels_all_simulations_dataarray = warming_levels_all_simulations['warming_level'].values
    
    
    n_time, n_regions = all_simulations_dataarray.shape

    # Create shared memory arrays 
    shared_values = RawArray('d', all_simulations_dataarray.ravel())  # 'd' for double (float64)
    shared_gmt = RawArray('d', warming_levels_all_simulations_dataarray.ravel())
    shared_warming_levels_out = RawArray('d', np.array(warming_levels_output).ravel())
    shared_return_periods_out = RawArray('d', np.array(return_periods_output).ravel())

    ctx = mp.get_context('fork')  
    with ctx.Pool(processes=cpus) as pool:
        results = pool.starmap(process_cell_GEV_fitting_and_evaluation, [(None,None,region_idx) for region_idx in range(n_regions)])

    results_array = np.array(results).reshape(n_regions, len(warming_levels_output), len(return_periods_output))

    results = xa.Dataset(
    data_vars={
        indicator.name:(["region", "warming_level", "return_period"], results_array),
    },
    coords=dict(
        region = np.array(list(all_simulations.columns)),
        warming_level=np.array(warming_levels_output),
        return_period=np.array(return_periods_output),
    ),
    attrs=dict(description=f"Return period value of {indicator.name} at warming level as predicted by a GEV fit on all available simulations within ISIMIP."),
    )


    return results
   

def main():
    ALL_REGIONS = get_all_regions()
    ALL_MODELS = get_models(simulation_round=CONFIG['isimip.simulation_round'])
    DEFAULT_OUTPUT_RETURN_PERIODS = [2,3,4,5,6,7,8,9,10,15,20,25] + [30 + i*10 for i in range(8)] + [150] + [200 + i*100 for i in range(9)]
    DEFAULT_OUTPUT_WARMING_LEVELS = [0.0 + 0.1 * i for i in range(151)]
    all_variables = ['rx5day','rx1day','TXx','HImax', 'spei_gamma_12_min']
    parser = argparse.ArgumentParser(epilog="""""", formatter_class=argparse.RawDescriptionHelpFormatter, parents=[log_parser, config_parser, isimip_parser])
    # parser.add_argument("-v", "--variable", nargs='+', default=[], choices=CONFIG["isimip.variables"])
    parser.add_argument("-i", "--indicator", nargs='+', default=[], choices=all_variables, help="includes additional, secondary indicator with specific monthly statistics")
    parser.add_argument("--overwrite", action='store_true')
    # parser.add_argument("--backends", nargs="+", choices=["csv", "netcdf"], default=["netcdf", "csv"])
    #parser.add_argument("--frequency")
    group = parser.add_argument_group('mask')
    #group.add_argument("--region", nargs='+', default=ALL_REGIONS, choices=ALL_REGIONS)
    #group.add_argument("--weights", nargs='+', default=CONFIG["preprocessing.regional.weights"], choices=CONFIG["preprocessing.regional.weights"])
    group.add_argument("--regional", action='store_true')
    group.add_argument("--regions", nargs='+', help="Choose subset of all regions with available masks. If left empty every region will be considered by default. Only applies when regional flac is active.")
    group.add_argument("--aggregation", default ='latWeight', help="Aggregation type")
    group.add_argument("--cpus", type=int, default = 1)
    group.add_argument("--warming_level_file", type=int)
    group.add_argument("--models", nargs='+', help="All models considered in the processing, they have to have at least one simulation available for the indicator. If left empty every model providing simulations for the indicator will be considered by default.")
    group.add_argument("--impact_models", nargs='+', help="All impact models considered in the processing, they have to have at least one simulation available for the indicator. If left empty every impact model providing simulations for the indicator will be considered by default.")
    group.add_argument("--warming_levels_output", nargs='+', default = DEFAULT_OUTPUT_WARMING_LEVELS, help="Warming levels for which return periods from non-stationary GEV fit should be recorded in output file.")
    group.add_argument("--return_periods_output", nargs='+', default = DEFAULT_OUTPUT_RETURN_PERIODS, help="Return periods from non-stationary GEV fit which should be recorded in output file.")

    o = parser.parse_args()
    setup_logger(o)
    
    for indicator_name in o.indicator:

        indicator = Indicator.from_config(indicator_name)

        if o.warming_level_file is None:
            o.warming_level_file = get_warming_level_file(**{**CONFIG, **vars(o)}) # pd.read_csv('/mnt/PROVIDE/climate_impact_explorer/isimip3/running-21-years/warming_levels.csv')
        
        if o.models is None: 
            
            models_indicator = list(set([simulation["climate_forcing"] for simulation in indicator.simulations]))
            o.models = [model for model in ALL_MODELS if model.lower() in models_indicator]#list(set([simulation["climate_forcing"] for simulation in indicator.simulations]))  # all available models
        
        if 'model' in indicator.simulations[0].keys() and o.impact_models is None:
            o.impact_models = list(set([simulation["model"] for simulation in indicator.simulations]))  # all available impact models
        
        elif 'model' not in indicator.simulations[0].keys(): 
            o.impact_models = None
          
        # all regions with data for the aggregation type available
        if o.regions is None: 
            o.regions = sorted(_.name for _ in Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/historical/{o.models[0].lower()}/{o.aggregation}/').glob("*") if not dir_is_empty(_))

    
        for model in o.models: 

            if o.impact_models is not None: 
            
                for impact_model in o.impact_models:

                    if o.regional: 
                        regional_information = f'_{o.aggregation}_regional'
                    else: 
                        regional_information = ''
                    
                    output_path = Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits/{model}_{impact_model}_{indicator.name}_annual_return_periods_non_stationary_GEV{regional_information}.nc')

                    if not output_path.is_file() or o.overwrite:

                        logger.info(
                            f"Processing GEV fit for {indicator.name} | {model} | {impact_model}"
                        )

                        if o.regional: 
                            results = make_return_period_array_regional_averages(
                            indicator = indicator,
                            warming_levels_simulations = pd.read_csv(o.warming_level_file),
                            regions = o.regions,
                            model = model,
                            impact_model = impact_model,
                            aggregation = o.aggregation,
                            frequency = 'annual',
                            warming_levels_output = o.warming_levels_output,
                            return_periods_output = o.return_periods_output,
                            cpus = o.cpus)
                        
                        else: 
                            results = make_return_period_array(
                            indicator = indicator,
                            warming_levels_simulations = pd.read_csv(o.warming_level_file),
                            model = model,
                            impact_model = impact_model,
                            frequency = 'annual',
                            warming_levels_output = o.warming_levels_output,
                            return_periods_output = o.return_periods_output,
                            cpus = o.cpus)
                        
                        output_dir =  Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits')

                        if not output_dir.is_dir(): 
                            output_dir.mkdir(parents=True, exist_ok=True)
                        
                        results.to_netcdf(output_path)
                    
                    else: 

                        logger.info(
                            f"File for GEV fit for {indicator.name} | {model} | {impact_model} already there. Please activate overwrite flag if you want to recalculate it. "
                        )

            else: 

                if o.regional: 
                    regional_information = f'_{o.aggregation}_regional'
                else: 
                    regional_information = ''

                output_path = Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits/{model}_{indicator.name}_annual_return_periods_non_stationary_GEV{regional_information}.nc')
                  
                if not output_path.is_file() or o.overwrite:

                    logger.info(
                            f"Processing GEV fit for {indicator.name} | {model} | no impact model"
                        )
                    
                    if o.regional:
                        results = make_return_period_array_regional_averages(
                            indicator = indicator,
                            warming_levels_simulations = pd.read_csv(o.warming_level_file),
                            regions = o.regions,
                            model = model,
                            impact_model = None,
                            aggregation = o.aggregation,
                            frequency = 'annual',
                            warming_levels_output = o.warming_levels_output,
                            return_periods_output = o.return_periods_output,
                            cpus = o.cpus)
                    
                    else: 
                        results = make_return_period_array(
                            indicator = indicator,
                            warming_levels_simulations = pd.read_csv(o.warming_level_file),
                            model = model,
                            impact_model = None,
                            frequency = 'annual',
                            warming_levels_output = o.warming_levels_output,
                            return_periods_output = o.return_periods_output,
                            cpus = o.cpus)
                    
                    output_dir =  Path(f'{CONFIG["isimip.climate_impact_explorer"]}/indicators/{indicator.name}/GEV_fits')

                    if not output_dir.is_dir(): 
                        output_dir.mkdir(parents=True, exist_ok=True)

                      
                    results.to_netcdf(output_path)
                
                else: 

                    logger.info(
                        f"File for GEV fit for {indicator.name} | {model} | no impact model already there. Please activate overwrite flag if you want to recalculate it. "
                    )


    

    

if __name__ == "__main__":
    main()
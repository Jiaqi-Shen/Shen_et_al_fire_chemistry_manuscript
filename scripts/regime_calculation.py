import sys
import datetime
import time
from datetime import timedelta, date, datetime
from dateutil.relativedelta import relativedelta
import os
import urllib
import tarfile
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from math import sin, cos, sqrt, atan2, radians

import glob

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import haversine as hs
from shapely.geometry import MultiPoint, Point

from matplotlib.animation import FuncAnimation

from matplotlib.patches import Patch
import regionmask
from tqdm import tqdm

#ROx + ROx, RO2 + NO = RONO2 reactions
ROx = pd.read_excel('../data_input/ROx_GCv12.xlsx', sheet_name='031924_ROx_list', header=None)
ROx = ROx.iloc[:, 0].tolist()
print(f'Total # of ROx species is {len(ROx)}.')
sorted(ROx)

ROx_df = pd.read_csv('../data_output/ROx_EQidx_RadicalChange.csv')
RONO2_df = pd.read_csv('../data_output/RONO2_EQidx_RadicalChange.csv')

#Function to get rxn rates
def get_ROx_rate(root, f, year, month, day, hour):
    with xr.open_dataset(f'{root}/{f}/GEOSChem.RxnRate.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)

    rate_list = []
    for eq in ROx_df['ROx_EQ_idx']:
        rate = data[eq].values
        radical_change = ROx_df[ROx_df['ROx_EQ_idx'] == eq].radical_change.values * (-1)  # convert negative to positive
        rate_dradical = rate * radical_change
        rate_list.append(rate_dradical)
    RO2_rate_sum = 0
    for ratei in rate_list:
        RO2_rate_sum = RO2_rate_sum + ratei
    return RO2_rate_sum

def get_RONO2_rate(root, f, year, month, day, hour):
    with xr.open_dataset(f'{root}/{f}/GEOSChem.RxnRate.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)

    rate_list = []
    for eq in RONO2_df['RONO2_EQ_idx']:
        rate_tot = data[eq].values
        frac = RONO2_df[RONO2_df['RONO2_EQ_idx'] == eq].radical_change.values * (-1)
        rate = rate_tot * frac
        rate_list.append(rate)
    RONO2_rate_sum = 0
    for ratei in rate_list:
        RONO2_rate_sum = RONO2_rate_sum + ratei
    return RONO2_rate_sum

def get_NOx_rate(root, f, year, month, day, hour):
    with xr.open_dataset(f'{root}/{f}/GEOSChem.RxnRate.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
        
    NO2_rate = data['RxnRate_EQ025'].values #NO2 + OH --> HNO3
    RONO2_rate_sum = get_RONO2_rate(root, f, year, month, day, hour)
    RNOx_rate = NO2_rate + RONO2_rate_sum
    return RNOx_rate

def get_HET_rate(root, f, year, month, day, hour):
    with xr.open_dataset(f'{root}/{f}/GEOSChem.RxnRate.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
        
    HET_rate = data['RxnRate_EQ492'].values #HO2 --> O2
    return HET_rate

#masks: CA_mask on, fire_mask off¶
def get_mask(root, f, year, month, day, hour):
    with xr.open_dataset(f'{root}/{f}/GEOSChem.SpeciesConcHourly.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
        LAT = data.lat.values
        LON = data.lon.values
    # CA regionmask
    states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
    ca = states[states.map_keys('California'),states.map_keys('California')]
    ca_mask = ~np.isnan(ca.mask(LON, LAT).values) #convert (value, nan) mask to (True,False)
    # mask = np.logical_and(fire_mask, ca_mask)  
    return ca_mask

#Calculate the rate of RNO2, RHET, RROx, and RTot at each grid for a given time
def rate_cal(root, f, year, month, day, hour, gamma):
    ## RNOx: NO2 + OH, RO2 + NO = RONO2 rate, molec cm-3 dry air s-1
    RNOx = get_NOx_rate(root, f, year, month, day, hour)
    ## RROx: Rate of ROx self reactions (ROx + ROx), molec cm-3 dry air s-1
    RROx = get_ROx_rate(root, f, year, month, day, hour)
    ## RHET: HO2 uptake rate by aerosol, molec cm-3 dry air s-1
    RHET = get_HET_rate(root, f, year, month, day, hour)
    
    RTot = RNOx + RROx + RHET
    rate = np.array([RNOx, RROx, RHET, RTot])
    rate = np.squeeze(rate)
    mask = get_mask(root, f, year, month, day, hour)
    rate = np.where(mask, rate, np.nan)
    return rate

def get_masked_PM(root, f, f_base, f_nofire, year, month, day, hour, gamma):
    with xr.open_dataset(f'{root}/{f_base}/GEOSChem.SpeciesConcHourly.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
        PM25_base = data['PM25'].values

    with xr.open_dataset(f'{root}/{f_nofire}/GEOSChem.SpeciesConcHourly.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
        PM25_nofire = data['PM25'].values
        
    PM25_enh = PM25_base - PM25_nofire
    mask = get_mask(root, f, year, month, day, hour)
    PM25_base = np.where(mask, PM25_base, np.nan)
    PM25_enh = np.where(mask, PM25_enh, np.nan)
    return PM25_base, PM25_enh

def get_masked_spec(root, f, year, month, day, hour, gamma):
    with xr.open_dataset(f'{root}/{f}/GEOSChem.SpeciesConcHourly.{year}{month}{day}_{hour}00z.nc4') as data:
        data = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
        OH = data['SpeciesConc_OH'].values
        HO2 = data['SpeciesConc_HO2'].values
        NO = data['SpeciesConc_NO'].values
        NO2 = data['SpeciesConc_NO2'].values       
    mask = get_mask(root, f, year, month, day, hour)
    OH = np.where(mask, OH, np.nan)
    HO2 = np.where(mask, HO2, np.nan) 
    NO = np.where(mask, NO, np.nan)
    NO2 = np.where(mask, NO2, np.nan) 
    return OH, HO2, NO, NO2

def timeframe_monthly_rate(root, f, year, month, day, hour, gamma):
    if month in {"01","03","05","07","08","10"}: #no simulation data for 1231
        dayi = 31
    elif (month == "02") & (int(year)%4==0):
        dayi = 29
    elif (month == "02") & (int(year)%4!=0):
        dayi = 28
    else:
        dayi = 30
    
    rate_stack = []        
    for D in range(1, dayi+1):
        day=f'{D:02d}'
        rated = rate_cal(root, f, year, month, day, hour, gamma)
        rate_stack.append(rated)
    month_days = pd.date_range(start=f'{year}-{month}-01T{hour}:30', end=f'{year}-{month}-{dayi}T{hour}:30', freq='D')
    time = np.array([f'{day.strftime("%Y-%m-%d")}T{hour}:30' for day in month_days], dtype='datetime64[ns]')
    rate = np.stack(rate_stack, axis=-1)
    rate = rate.transpose(3, 1, 2, 0)
    return rate, time

def timeframe_monthly_other(func, root, f, f_base, f_nofire, year, month, day, hour, gamma):
    if month in {"01","03","05","07","08","10"}: #no simulation data for 1231
        dayi = 31
    elif (month == "02") & (int(year)%4==0):
        dayi = 29
    elif (month == "02") & (int(year)%4!=0):
        dayi = 28
    else:
        dayi = 30

    for D in range(1, dayi+1):
        day=f'{D:02d}'
        if func == 'get_masked_spec':
            result = globals()[func](root, f, year, month, day, hour, gamma)
        elif func == 'get_masked_PM':
            result = globals()[func](root, f, f_base, f_nofire, year, month, day, hour, gamma)
        if D == 1:
            if isinstance(result, tuple):
                var_num = len(result)
            else:
                var_num = 1
            list_of_lists = [[] for _ in range(var_num)]
        for i in range(var_num):        
            list_of_lists[i].append(result[i])
    var = []
    for i in range(var_num):
        vari = np.stack(list_of_lists[i], axis=-1)
        vari = vari.transpose(3, 1, 2, 0).squeeze()
        var.append(vari)
    return var

def regime_cal_data_save(root, f, f_base, f_nofire, year, hour, gamma):
    for month in range(1, 13):
        month = f'{month:02d}'
        rate, time = timeframe_monthly_rate(root, f, year, month, "", hour, gamma)
        spec = timeframe_monthly_other('get_masked_spec', root, f, f_base, f_nofire, year, month, "", hour, gamma)
        PM = timeframe_monthly_other('get_masked_PM', root, f, f_base, f_nofire, year, month, "", hour, gamma)
        OH = spec[0]
        HO2 = spec[1]
        NO = spec[2]
        NO2 = spec[3]
        PM25_base = PM[0]
        PM25_enh = PM[1]
        # func to save data
        def save_data2nc(rate, time, OH, HO2, NO, NO2, PM25_base, PM25_enh, datatype, f):
            with xr.open_dataset('/projectsp/f_xj103_1/jqshen/GEOS_Chem/CA_aerosol_effects_simulations/base/GEOSChem.SpeciesConcHourly.20200901_0000z.nc4') as data:
                    data = data.drop_dims("ilev")
                    existing_da = data.isel(lev=slice(0, 5)).mean(dim="lev", keep_attrs=True)
                    lat = existing_da.lat
                    lon = existing_da.lon
            dims = {'time': len(time), 'latitude': len(lat), 'longitude': len(lon), 'rate_type': 4}
            coords = [time, lat, lon, np.array(['RNOx', 'RROx', 'RHET', 'RTot'])]
            rate = xr.DataArray(rate, coords=coords, dims=['time', 'lat', 'lon','rate_type'])
            OH = xr.DataArray(OH, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
            HO2 = xr.DataArray(HO2, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
            NO = xr.DataArray(NO, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
            NO2 = xr.DataArray(NO2, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
            PM25_base = xr.DataArray(PM25_base, coords=[time, lat, lon], dims=['time', 'lat', 'lon'])
            PM25_enh = xr.DataArray(PM25_enh, coords=[time, lat, lon], dims=['time', 'lat', 'lon']) 
            gamma_da = xr.DataArray(gamma)
            dataset = xr.Dataset({'rate': rate, 'OH': OH, 'HO2': HO2, 'NO': NO, 'NO2': NO2, 
                                  'PM25_base': PM25_base, 'PM25_enh': PM25_enh, 'gamma': gamma_da})
            dataset.to_netcdf(f"../data_output/gamma0.2/Regime_cal_{f}_{datatype}_2020{month}.nc")
        ## func cal monthly average
        def monthlymean(x):
            return np.nanmean(x, axis=0, keepdims=True)
            
        save_data2nc(rate, time, OH, HO2, NO, NO2, PM25_base, PM25_enh, "daily", f)
        save_data2nc(monthlymean(rate), time[[0]], monthlymean(OH), monthlymean(HO2), 
                     monthlymean(NO), monthlymean(NO2), monthlymean(PM25_base), monthlymean(PM25_enh), "monthly", f)

regime_cal_data_save('/projectsp/f_xj103_1/jqshen/GEOS_Chem/CA_aerosol_effects_simulations', 'base', 'base', 'nofire', 
                     "2020", "20", gamma=0.2)
regime_cal_data_save('/projectsp/f_xj103_1/jqshen/GEOS_Chem/CA_aerosol_effects_simulations', 'base_noPhotol', 'base', 'nofire', 
                     "2020", "20", gamma=0.2)
regime_cal_data_save('/projectsp/f_xj103_1/jqshen/GEOS_Chem/CA_aerosol_effects_simulations', 'nofire', 'base', 'nofire', 
                     "2020", "20", gamma=0.2)
regime_cal_data_save('/projectsp/f_xj103_1/jqshen/GEOS_Chem/CA_aerosol_effects_simulations', 'nofire_noPhotol', 'base', 'nofire', 
                     "2020", "20", gamma=0.2)
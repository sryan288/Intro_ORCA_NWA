'''
 Utilities to manipulate data in Northwest Atlantic or load specific datasets
'''
'''
    1) mask_shelf          :Create mask for shelf region only
    2) mask_box            :masks for pre-defined shelf boxes
    3) load_oisst          : load oisst monthly data, option to add sst in pre-defined boxes
    4) deseason            : remove seasonal cycle (output are anomalies), with option to define reference period

'''

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from numpy.polynomial.polynomial import polyval,polyfit
import matplotlib.dates as dates
import xarray as xr
import scipy.stats as stats


#
#------------------------------------------------------------------------
# 1) mask shelf area

def mask_shelf(depth=300):
    """ 
    Input:
    depth (Int)    : shelf break depth that should be used to mask shelf
    
    Output:
    mask (array)   : mask based on tmask, values [0,1]
    
    """
    
    # bounds to cut out NWA
    x_bnds, y_bnds = [809,993], [601,758]   # last value not included
    
    mask = xr.open_dataset("/vortex/clidex/data/ORCA/mesh_files/mask.nc")
    mask = mask.sel(x=slice(*x_bnds),y=slice(*y_bnds)).isel(t=0,z=slice(0,24))
    # mask = mask.where(mask!=0)
    mask3d = mask['tmask'].rename({'z':'deptht'})
    mask3d['nav_lon'] = mask['nav_lon']
    mask3d['nav_lat'] = mask['nav_lat']
   
    # bathymetry
    bathy = xr.open_dataset('/vortex/clidex/data/ORCA/mesh_files/mesh_zgr.nc').sel(x=slice(*x_bnds),y=slice(*y_bnds))
    bathy['hdept'] = bathy['hdept'].where(np.squeeze(mask['tmask'][0,0,::])==1)
    
    # everything shallower or equal the given depth
    mask_shelf = mask3d.where(bathy['hdept']<=depth).where(np.squeeze(mask['tmask'][0,0,::])==1)
    del mask,mask3d
    return mask_shelf


#
#
#-------------------------------------------------------------------------
# 2) mask for pre-defined boxes

def mask_box(mask_shelf):
    """
    No input required
    
    Output:
    mask_dic    : dictionary with masks for different boxes
    
    """
    
    # bounds to cut out NWA
    x_bnds, y_bnds = [810,993], [602,758]
    bathy = xr.open_dataset('/vortex/clidex/data/ORCA/mesh_files/mesh_zgr.nc').sel(x=slice(*x_bnds),y=slice(*y_bnds))
    bathy['hdept'] = bathy['hdept'].where(bathy['hdept']!=0)

    
    # initialize dictionary
    mask_dic = {}

    # separate into north and south of Cape
    mask_dic['shelfN'] = mask_shelf.where((bathy.nav_lon>-70) & (bathy.nav_lon<-66)  &
                                   (bathy.nav_lat>40) & (bathy.nav_lat<46))

    mask_dic['shelfS'] = mask_shelf.where((bathy.nav_lon>-76) & (bathy.nav_lon<-70)  &
                                   (bathy.nav_lat>38) & (bathy.nav_lat<42))
    mask_dic['pioneer'] = mask_shelf.where((bathy.nav_lon>-72) & (bathy.nav_lon<-70)  &
                                   (bathy.nav_lat>39.5) & (bathy.nav_lat<41.5))
    mask_dic['nav_lon'] = mask_shelf['nav_lon']  
    mask_dic['nav_lat'] = mask_shelf['nav_lat'] 
    return mask_dic




#
#
#-------------------------------------------------------------------------
# 3) load oisst monthly data (option to load oisst in pre-defined boxes)
def load_oisst(mask=None):
    """
    Function load oisst monthly mean data over Northwest Atlantic and optionally in pre-defined boxes
    (see mask_box function).
    
    INPUT:
    mask (string)     : OPTIONAL, can be ['shelfN','shelfS']. If provided oisst for specified box is derived 
                        in addition to full dataset.
                        
    OUTPUT:
    oisst (xarray)    : containing data arrays for full domain and if specified for box
    
    """
    
    #load data
#     oisst = xr.open_mfdataset('/vortex/clidex/data/obs/OISST/sst*.nc')
#     oisst = oisst.sel(lat=slice(34,48),lon=slice(282,298))

    # derive monthly mean
#     oisstmm = oisst.resample(time='1M',keep_attrs=True).mean().compute()
    oisstmm = xr.open_dataset('/home/sryan/Data/OISST/oisst_NWA_34N48N_monthly.nc')
    
    if mask:
        # load masks
        masks = mask_box(mask_shelf(300))

        # interpolate masks into oisst grid
        from scipy.interpolate import griddata
        [xm,ym] = np.meshgrid(oisstmm.lon-360,oisstmm.lat,indexing='ij')

        mask_shelfS_oisst = griddata((masks['nav_lon'].values.ravel(),masks['nav_lat'].values.ravel()),
                        masks[mask][0,:,:,0].values.ravel(), (xm, ym),'linear')


        # add to xarray (needed for multiplication below)
        oisstmm[mask] = (('lon','lat'),mask_shelfS_oisst)
        oisstmm[mask] = (oisstmm['sst'].transpose('lon','lat','time'))*oisstmm[mask]
        
    return oisstmm


#
#
#-------------------------------------------------------------------------
# 4) deseason data
def deseason(ds,timevar='time_counter',refperiod=None):
    dummy = timevar + '.month'
    if refperiod:
        if timevar=='time_counter':
            ds = ds.groupby(dummy)-ds.sel(time_counter=slice(*refperiod)).groupby(dummy).mean(timevar)        
        elif timevar=='time':
            ds = ds.groupby(dummy)-ds.sel(time=slice(*refperiod)).groupby(dummy).mean(timevar)
    else:
        ds = ds.groupby(dummy)-ds.groupby(dummy).mean(timevar)
    return ds



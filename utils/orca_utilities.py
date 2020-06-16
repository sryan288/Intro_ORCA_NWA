'''
 Miscellaneous useful tools for ORCA
'''
'''
    1) cut_latlon_box
    2) deseason_month
    3) set_ylim_equal
    4) map_stuff
    5) anomaly
    6) find_common_cmax
    7) mark_2D_percentile
    8) orca_dectrend2D
    9) trend_significance
    10) orca_index                           (derive index for specific lat,lon box, option to remove seasonal cycle)
    11) mask_LC_width                        (loads mask for LC after Furue et al)
    12) load_orca_EIO                        (load all available variables for EIO)  
    13) crosscor                             (cross-correlation using stats pearson nr)
    14) load_mesh                            (loads mesh variables for either EIO or NWA)
'''

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from numpy.polynomial.polynomial import polyval,polyfit
import matplotlib.dates as dates
import xarray as xr
import scipy.stats as stats


#
#------------------------------------------------------------------------
# cut out lat lon box of ORCA
def cut_latlon_box(field,lon,lat,lon_bnds,lat_bnds):
        # ### cut data for box
        index = field.where((lon_bnds[0] < lon) & (lon < lon_bnds[1])
                 & (lat_bnds[0] < lat) & (lat < lat_bnds[1])
                 & (0 < field), drop=True)
        return index
#
#
# areal mean over lat lon box
def mean_latlon_box(field,lon,lat,lon_bnds,lat_bnds):
        # ### cut data for box
        index = field.where((lon_bnds[0] < lon) & (lon < lon_bnds[1])
                 & (lat_bnds[0] < lat) & (lat < lat_bnds[1])
                 & (0 < field), drop=True).mean(dim=('x','y'),skipna=True)
        return index
#
#
#------------------------------------------------------------------------
# deseason xarray
def deseason_month(data):
    data = data.groupby('time_counter.month')-data.groupby('time_counter.month').mean('time_counter')
    data = data.drop('month')
    return data

#
#------------------------------------------------------------------------
# set equal ylimits automatically, relative to data
def set_ylim_equal(mode,ax):
    minimum = np.min(ax.get_ylim())
    maximum = np.max(ax.get_ylim())
    
    if mode=='tight':
        ax.set_ylim(minimum*1.1,maximum*1.1)
    elif mode=='equal':
        ylim=np.max([np.abs(minimum),np.abs(maximum)])
        ax.set_ylim(ylim*(-1.1),ylim*1.1)
#
#
#-------------------------------------------------------------------------
# cartopy map extras
def map_stuff(ax):
    ax.coastlines(resolution='50m',color='k')
    gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=1,color='gray'
                   ,alpha=0.5,linestyle='--')
    gl.xlabels_top, gl.ylabels_right= False, False
    gl.xlocator = mticker.FixedLocator([90,100,110,120,130])
    gl.ylocator = mticker.FixedLocator(np.arange(-35,-10,5))
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    return gl
#
#
#--------------------------------------------------------------------------
# anomaly plot
def anomaly(ax,x,y,offset):
    ax.fill_between(x, 0+offset[0], y, where = y>0+offset[0], facecolor='indianred', interpolate=True)
    ax.fill_between(x, 0-offset[1], y, where = y<0-offset[1], facecolor='blue', interpolate=True)
#
#
#--------------------------------------------------------------------------
# for comparison find common values for vmin and vmax so that colormap is equal
def find_common_cmax(data1,data2):    
    vmin1 = np.round(np.min((data1.fillna(0).values,data1.fillna(0).values)),decimals=2)
    vmax1 = np.round(np.max((data1.fillna(-100000).values,data1.fillna(-100000).values)),decimals=2)
    vmin2 = np.round(np.min((data2.fillna(0).values,data2.fillna(0).values)),decimals=2)
    vmax2 = np.round(np.max((data2.fillna(-100000).values,data2.fillna(-100000).values)),decimals=2)    
    vval = np.max(np.abs((vmin1,vmax1,vmin2,vmax2)))
    return vval
#
#
#--------------------------------------------------------------------------
# create mask for 2D field percentile
def mask_2D_percentile(data,pval,level):
    test =np.nanpercentile(data.sel(time_counter=slice('1958-01-01', '2006-12-31')),pval,axis=0)
    mask= data.values-test
    if level=='above':
        mask[mask>0]=1
        mask[mask<0]=np.nan
    elif level=='below':
        mask[mask>0]=np.nan
        mask[mask<0]=1
    return mask
#
#
#--------------------------------------------------------------------------
# 2D trend filed
def orca_dectrend2D(ds,key):
    # to apply the numpy functions we need to stack our data to derive 1D fields
    stacked = ds[key].fillna(0).stack(loc=('lon','lat'))
    timenum=dates.date2num(stacked.time_counter.to_index().to_pydatetime()) #matlab time
    # apply polyfit to fit 1st order polynomial and polyval to evaluate y-values
    regressions = polyfit(timenum, stacked,1) 
    drift_stacked= polyval(timenum,regressions)
    # feed back into xarray
    foo = xr.DataArray(drift_stacked.T,
                       coords=[stacked['time_counter'], stacked['loc']],
                       dims=['time_counter', 'loc'],name=stacked.name)
    # unstack locations
    trend=foo.unstack('loc')
    # subtract fields
    #ds_detr = ds[key]-trend
    # derive linear trend and convert to trend/decade
    dectrend = ((trend[-1,:,:]-trend[0,:,:])/len(trend))*120
    dectrend = dectrend.where(dectrend!=0)
    return dectrend.transpose(),trend

#
#
#--------------------------------------------------------------
# makes use of scipy linregress function, which returns additional statistical values,
# such as p-value
import scipy.stats as stats
def trend_significance(data,alpha):
    # data has to have time,lon,lat dimensions, i.e. 3D.
    # Won't work with additional depth dimension
    # need to stack data again to have 1D array, which is required by linregress function
    stacked = data.fillna(0).stack(loc=('lon','lat'))
    timenum=dates.date2num(stacked.time_counter.to_index().to_pydatetime()) #matlab time

    # initialize fields
    p_value=[]
    lintrend=[]
    trend = []
    for i in np.arange(0,stacked.shape[1]):
        # do linear regression
        slope,intercept,rval,pval,std_err = stats.linregress(timenum,stacked[:,i])
        p_value.append(pval)
        # derive linear trend
        dummy = slope*timenum+intercept
        trend.append(dummy)
        lintrend.append(dummy[-1]-dummy[0])
    # add values back into xarray & unstack
    foo = xr.DataArray(np.array(trend).transpose(),
                   coords=[stacked['time_counter'], stacked['loc']],
                   dims=['time_counter', 'loc'],name=stacked.name)
    trend = foo.unstack('loc')
    foo = xr.DataArray(np.array(lintrend),
                  coords=[stacked['loc']],
                  dims=['loc'],name=stacked.name)
    lintrend = foo.unstack('loc')
    foo = xr.DataArray(np.array(p_value),
                  coords=[stacked['loc']],
                  dims=['loc'],name=stacked.name)
    pval = foo.unstack('loc')
    # now derive mask for desired significance value alpha
    mask = pval.values-alpha
    mask[mask>0]=np.nan
    mask[mask<0]=1
    
    # trend is full size trend, that can be subtracted from original field
    # lintrend is an integer, which gives trend over whole time period
    return mask,trend,lintrend
#
#
#--------------------------------------------------------------------------
# Derive index over lat, lon box
# need xarray format, can choose whether to subtract seasonal cycle or not

def orca_index(run,var,lon_bnds,lat_bnds,rm_scycle):
    if var in ['temp','MLD','sal']: 
        gridtype = 'T'
        depth = 'deptht'
    elif var in ['U']:
        gridtype = 'U'
        depth = 'depthu'
    elif var in ['V']:
        gridtype = 'V'
        depth = 'depthv'
    
    datapath = '/vortex/clidex/data/ORCA/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2_processed_EIO/'
    ds = xr.open_dataset(datapath + 'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                             run + '_1957_2016_' + var + '_EIO_grid_' + gridtype + '.nc',chunks=23)
    
    #
    ###########################
    # average over box
    index = ds.sel(lon=slice(*lon_bnds),lat=slice(*lat_bnds)).where(ds[list(ds.data_vars)]!=0,
                                                                    drop=True).mean(('lon','lat')) 
    #
    ###########################
    # remove seasonal cycle
    if rm_scycle==True:
         index = index.groupby('time_counter.month')-index.groupby('time_counter.month').mean('time_counter')
         index = index.drop('month')
            
    # Output
    return index
#
#
#--------------------------------------------------------------------------
# Function to subtract LC data and average across longitude. (LC width as defined by Furue et al)
def mean_LC_width(data):
    mask = np.load('../data/LC_mask_Furue.npz')
    LC = data.sel(lon=slice(*mask['lon_bnds']),lat=slice(*mask['lat_bnds']))
    LC = LC*mask['mask_LC']
    # set zero values to nan
    LC=LC.where(LC!=0)
    # average across LC width
    LC = LC.mean('lon')
    
    return LC,mask['LC_w'],mask['LC_e']
#
#
#-----------------------------------------------------------------------------
# load full ORCA data set for the post-processed Eastern Indian Ocean Data
def load_orca_EIO(run):
    datapath = '/vortex/clidex/data/ORCA/ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2_processed_EIO/'
    for var in ['temp','MLD','sal','U','V']:
        if var in ['temp','MLD','sal']: 
            gridtype = 'T'
            if var=='temp':
                ds = xr.open_dataset(datapath + 'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                                 run + '_1957_2016_' + var + '_EIO_grid_' + gridtype + '.nc',
                                chunks={'lat': 30, 'lon': 30, 'time_counter': 236,'deptht':23})
            elif var=='sal':
                ds['vosaline'] = xr.open_dataset(datapath + 'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                                 run + '_1957_2016_' + var + '_EIO_grid_' + gridtype + '.nc',
                                chunks={'lat': 30, 'lon': 30, 'time_counter': 236,'deptht':23})['vosaline']
            elif var=='MLD':
                ds['somxl010'] = xr.open_dataset(datapath + 'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                                 run + '_1957_2016_' + var + '_EIO_grid_' + gridtype + '.nc',
                                chunks={'lat': 30, 'lon': 30, 'time_counter': 236})['somxl010']
            
            
        elif var=='U':
            gridtype = 'U'
            ds2 = xr.open_dataset(datapath + 'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                             run + '_1957_2016_' + var + '_EIO_grid_' + gridtype + '.nc',
                            chunks={'lat': 328, 'lon': 301, 'time_counter': 236,'depthu':23})
        elif var=='V':
            gridtype = 'V'
            ds3 = xr.open_dataset(datapath + 'ORCA025.L46.LIM2vp.CFCSF6.JRA.XIOS2-' + 
                             run + '_1957_2016_' + var + '_EIO_grid_' + gridtype + '.nc',
                            chunks={'lat': 328, 'lon': 301, 'time_counter': 236,'depthv':23})
    del ds['deptht_bounds'],ds['time_counter_bounds'],ds['time_centered_bounds']
    return ds,ds2,ds3
#
#
#----------------------------------------------------------------------------------
# cross correlation using stats.personnr
def crosscor(x,y):
    x1= x.values[~np.isnan(x.values)]
    y1 = y.values[~np.isnan(x.values)]

    return stats.pearsonr(x1,y1)


#
#
#----------------------------------------------------------------------------------
# load mesh files
def load_mesh(region,rename=False):
    '''
    
    INPUT:
    region   : String (either 'EIO' or 'NWA'), otherwise global
    rename   : boolean (True or False) to rename time dimension
    
    OUTPUT:
    meshz,meshh,mask 
    
    '''
    meshpath = '/vortex/clidex/data/ORCA/mesh_files/'
    meshz = xr.open_dataset(meshpath + 'mesh_zgr.nc')
    meshh = xr.open_dataset(meshpath + 'mesh_hgr.nc')
    mask = xr.open_dataset(meshpath + 'mask.nc')

    # cut files depending on region
    if region=='EIO': xbnds=slice(9,310); ybnds=slice(297,625)
    elif region=='NWA': xbnds=slice(809,993); ybnds=slice(601,758)

    def cut_data(ds,rename=False):
        ds = ds.sel(x=xbnds,y=ybnds).isel(z=slice(0,24))
        if rename is True:
            ds = ds.rename({'t':'time_counter'})
        return ds

    meshz = cut_data(meshz,rename)
    meshh = cut_data(meshh,rename)
    mask = cut_data(mask,rename)
    return meshz,meshh,mask
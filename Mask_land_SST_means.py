# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:52:17 2021

@author: nbl370
"""
import netCDF4
import numpy as np
from osgeo import gdal,osr,ogr
import matplotlib.pyplot as plt
import os
import xarray as xr

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def makeMask1(lon,lat):
    source_ds = ogr.Open(os.path.join(__location__,'ne_10m_land.shp'))
    #YOU CAN DOWNLOAD THE LAND MASK HERE https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/ 
    source_layer = source_ds.GetLayer()
 
    # Create high res raster in memory
    mem_ds = gdal.GetDriverByName('MEM').Create('', lon.size, lat.size, gdal.GDT_Byte)
    mem_ds.SetGeoTransform((lon.min(), lon[:][1] - lon[:][0], 0, lat.max(), 0, (lat[:][1] - lat[:][0])))
    band = mem_ds.GetRasterBand(1)
 
    # Rasterize shapefile to grid
    gdal.RasterizeLayer(mem_ds, [1], source_layer, burn_values=[1])
 
    # Get rasterized shapefile as numpy array
    array = band.ReadAsArray()
 
    # Flush memory file
    mem_ds = None
    band = None
    return array
 
for model in ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth3P-HR','HadGEM3-GC31-HM']: 
    data=np.load(os.path.join(__location__,'latlon_background_converted_{}.npy'.format(model)),allow_pickle=True).item()
    new_data={i:[] for i in ['lat','lon']}

    lat=data['lat']
    lon=data['lon']
    
    lons=[]
    for i in range(len(lon)):
        if lon[i]>180.:
            lons.append(lon[i]-360)
        else:
            lons.append(lon[i])
    
    lons=np.array(lons)
    
    lats=lat[::-1]
    
    new_data['lat']=lats
    new_data['lon']=lon
    
    print(lats)
    
    np.save(os.path.join(__location__,'latlon_background_flipped_{}.npy'.format(model)),new_data)
    
    for period in ['PRESENT','FUTURE']:       
        for month in range(1,13):
            data=np.loadtxt(os.path.join(__location__,'Monthly_mean_SST_{}_{}_{}.txt'.format(model,month,period)))
            
            # create the mask -first run this part!
            mask1 = makeMask1(lons,lats) 
            
            mask=np.concatenate([mask1[:,int(len(lons)/2):],mask1[:,:int(len(lons)/2)]],axis=1)
            mask=np.flipud(mask)
            
            masked_variable=np.zeros((len(lat),len(lon)))
            
            for i in range(len(lat)):
                for j in range(len(lon)):
                    if mask[i,j]==1:
                        masked_variable[i,j]=np.nan
                    else:
                        masked_variable[i,j]=data[i,j]

            MSLP=np.loadtxt(os.path.join(__location__,'Monthly_mean_MSLP_{}_{}_{}.txt'.format(model,month,period)))

            SST=np.flipud(masked_variable)
            MSLP=np.flipud(MSLP)

            np.savetxt(os.path.join(__location__,'Monthly_mean_SST_{}_{}_{}_masked_flipped.txt'.format(model,month,period)),SST)
            np.savetxt(os.path.join(__location__,'Monthly_mean_MSLP_{}_{}_{}_flipped.txt'.format(model,month,period)),MSLP)

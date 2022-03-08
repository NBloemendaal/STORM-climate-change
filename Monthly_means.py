# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:34:51 2020

@author: nbl370
"""
import numpy as np
import xarray as xr 
import os
import matplotlib.pyplot as plt
import sys
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

for model,version,grid in zip(['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth3P-HR','HadGEM3-GC31-HM'],['r1i1p1f1','r1i1p1f2','r1i1p2f1','r1i3p1f'],['gn','gr','gr','gn']):
    for variable,variablename in zip(['psl','ts'],['MSLP','SST']):
        variable_all={i:[] for i in ['hist-1950','highres-future']}    
        if model=='CNRM-CM6-1-HR':
            for period,year0list,year1list in zip(['hist-1950','highres-future'],[[1980,1990,2000,2010],[2015,2040,2050]],[[1989,1999,2009,2014],[2039,2049,2050]]): #these go in batches of 10 years                
                variable_all[period]={i:np.zeros((360,720)) for i in range(1,13)}  
                count=0
                for year0,year1 in zip(year0list,year1list):
                        data=xr.open_dataset(os.path.join(__location__,'{}_Amon_{}_{}_{}_{}_{}01-{}12.nc'.format(variable,model,period,version,grid,year0,year1)))
                        VARIABLE=data[variable]
                        #120 "time steps", so 1 per month, 10 years
                        for j in range(0,12): #12 months
                            for i in range(0,int(int(year1)-int(year0)+1)): #number of years
                                variable_all[period][int(j+1)]+=VARIABLE[i*12+j]
                                count+=1
                print(count/12)
                
               
                plt.imshow(np.flipud(VARIABLE[0]))
                plt.title('{},{},{},{},{}'.format(model,period,variable,np.nanmin(VARIABLE[0]),np.nanmax(VARIABLE[0])))
                plt.show()
                lonlat_data={i:[] for i in ['lat','lon']}
                lonlat_data['lon']=data['lon'].values
                lonlat_data['lat']=data['lat'].values 
                
                        
        elif model=='CMCC-CM2-VHR4':
            for period,yearlist in zip(['hist-1950','highres-future'],[np.arange(1979,2015,1),np.arange(2015,2051,1)]): #these go in batches of 10 years       
                variable_all[period]={i:np.zeros((768,1152)) for i in range(1,13)}
                for year in yearlist:
                    for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:                    
                        data=xr.open_dataset(os.path.join(__location__,'{}_Amon_{}_{}_{}_{}_{}{}-{}{}.nc'.format(variable,model,period,version,grid,year,month,year,month)))
                        VARIABLE=data[variable]
                        
                        variable_all[period][int(month)]+=VARIABLE[0]
                                                       
                plt.imshow(np.flipud(VARIABLE[0]))
                plt.title('{},{},{},{},{}'.format(model,period,variable,np.nanmin(VARIABLE[0]),np.nanmax(VARIABLE[0])))
                plt.show()
                lonlat_data={i:[] for i in ['lat','lon']}
                lonlat_data['lon']=data['lon'].values
                lonlat_data['lat']=data['lat'].values 
        
        else: #HadGEM or EC-Earth
            for period,year0,year1 in zip(['hist-1950','highres-future'],[1979,2015],[2015,2051]): #these go in batches of 10 years
                if model=='HadGEM3-GC31-HM':
                    variable_all[period]={i:np.zeros((768,1024)) for i in range(1,13)} 
                else: #EC-Earth3P-HR
                    variable_all[period]={i:np.zeros((512,1024)) for i in range(1,13)} 
                for year in range(year0,year1):             
                    data=xr.open_dataset(os.path.join(__location__,'{}_Amon_{}_{}_{}_{}_{}01-{}12.nc'.format(variable,model,period,version,grid,year,year)))
                    VARIABLE=data[variable].values
                    
                    for month in range(0,12):
                        variable_all[period][int(month)+1]+=VARIABLE[month]
                  
                plt.imshow(np.flipud(VARIABLE[0]))
                plt.title('{},{},{},{},{}'.format(model,period,variable,np.nanmin(VARIABLE[0]),np.nanmax(VARIABLE[0])))
                plt.show()
                lonlat_data={i:[] for i in ['lat','lon']}
                lonlat_data['lon']=data['lon'].values
                lonlat_data['lat']=data['lat'].values               
                    
        var=0
        for period,periodname in zip(['hist-1950','highres-future'],['PRESENT','FUTURE']):
            for month in range(1,13):
                if model=='CNRM-CM6-1-HR' and period=='hist-1950':
                    average=variable_all[period][month]/35.
                else:
                    average=variable_all[period][month]/36.
                
                if variable=='psl' and np.nanmean(average)>3000: #then it's in Pa:
                    average/=100. #convert to hPa
                
                if lonlat_data['lon'][0]<0: #then it's -180-180 projection
                    print('Converted',model,variable,period)
                    plt.imshow(np.flipud(average))
                    plt.show()
                    
                    lon = lonlat_data['lon']
                    lat = lonlat_data['lat']
                    
                    lon_converted=np.concatenate([lonlat_data['lon'][int(len(lon)/2):],lonlat_data['lon'][:int(len(lon)/2)]+360.],axis=0)   
                
                    lonlat_converted={'lat':lat,'lon':lon_converted}
                    
                    (latdim,londim)=average.shape
                    
                    average=np.concatenate([average[:,int(londim/2):],average[:,:int(londim/2)]],axis=1)
                    
                    var=1                
                    
                
                np.savetxt(os.path.join(__location__,'Monthly_mean_{}_{}_{}_{}.txt'.format(variablename,model,month,periodname)),average)
            
        if var==0:
            np.save(os.path.join(__location__,'latlon_background_converted_{}.npy'.format(model)),lonlat_data)
        else:
            np.save(os.path.join(__location__,'latlon_background_converted_{}.npy'.format(model)),lonlat_converted)




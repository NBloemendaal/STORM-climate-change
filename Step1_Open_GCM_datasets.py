
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:12:35 2020

@author: Nadia Bloemendaal, nadia.bloemendaal@vu.nl

This script opens the TC datasets from the global climate models, and extracts the relevant STORM variables from them. 
!!! IMPORTANT !!! Please find more information on the GCM datasets in Roberts et al (2020) Projected Future Changes in Tropical Cyclones using the CMIP6 HighResMIP Multimodel
Ensemble", including information on where to find the GCM datasets. I cannot send you the datasets directly, these need to be downloaded via the Jasmin server so that they have a 
record of who is using their datasets. I will therefore NOT reply to such requests!

This script will generate a substantial amount of files. Some of these are used in the creation of the STORM input dataset, whereas other files can support in any validation
undertakings. I therefore left everything in, but you can alter these output files if deemed necessary.

This script is part of the STORM Climate change research. Please read the corresponding paper before commercing.
Bloemendaal et al (2022) A globally consistent local-scale assessment of future tropical cyclone risk. Paper published in Science Advances.

Copyright (C) 2020 Nadia Bloemendaal. All versions realeased under the GNU General Public License v3.0.
"""

import numpy as np 
import xarray as xr 
import pandas as pd
import datetime
from SELECT_BASIN import Basins_WMO #YOU CAN FIND THIS SCRIPT UNDER THE STORM DIRECTORY ON GITHUB
from scipy import stats
import os
import sys


starttime=datetime.datetime.now()


def Check_EP_formation(lat,lon): 
    """
    Check if formation is in Eastern Pacific (this should be inhibited if basin==NA)
    Parameters
    ----------
    lat : latitude coordinate of genesis.
    lon : longitude coordinate of genesis

    Returns
    -------
    l : 1=yes (formation in EP),0=no (no formation in EP).

    """    
    if lat<=60. and lon<260.:
        l=1
    elif lat<=17.5 and lon<270.:
        l=1
    elif lat<=15. and lon<275.:
        l=1
    elif lat<=10. and lon<276.:
        l=1
    elif lat<=9. and lon<290.:
        l=1
    else:
        l=0
    return l
    
    
def Check_NA_formation(lat,lon): 
    """
    Check if formation is in North Atlantic (this should be inhibited if basin==EP)
    Parameters
    ----------
    lat : latitude coordinate of genesis
    lon : longitude coordinate of genesis

    Returns
    -------
    l : 1=yes (formation in NA) 0=no (no formation in NA).

    """
    if lat<=60. and lat>17.5 and lon>260.:
        l=1
    elif lat<=17.5 and lat>15. and lon>270.:
        l=1
    elif lat<=15. and lat>10 and lon>275.:
        l=1
    elif lat<=10. and lon>276.:
        l=1
    else:
        l=0
    return l

def genesis_basin(lat_genesis,lon_genesis):
    """
    Check if genesis location is inside WMO basin boundaries
    
    Parameters
    ----------
    lat_genesis : latitude genesis
    lon_genesis : longitude genesis (between 0-360)

    Returns
    -------
    basin : basin name. If location is not inside basin, None is returned. 

    """
    var=0
    for basin in ['EP','NA','NI','WP','SP','SI']:
        lat0,lat1,lon0,lon1=Basins_WMO(basin)
        if lat0<=lat_genesis<=lat1 and lon0<=lon_genesis<=lon1:
            if basin=='EP':
                check=Check_EP_formation(lat_genesis,lon_genesis) #1=EP 0=NA
                if check==1:
                    return 'EP'
                else:
                    return 'NA'
            elif basin=='NA':
                check=Check_NA_formation(lat_genesis,lon_genesis) #1=NA 0=EP
                if check==0:
                    return 'EP'
                else:
                    return 'NA'
            else:
                return basin
    if var==0:
        return None
    
 
monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}
 
#'CMCC-CM2-VHR4'
for model in ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth3P-HR','HadGEM3-GC31-HM']:  
    for period in ['PRESENT','FUTURE']:
        __datasets__=os.path.realpath(PLEASE SET YOUR PATH HERE)
        __location__=os.path.join(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))))
                
        months={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        genesis_pressure={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        genesis_location={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        genesis_dpres={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        genesis_pres_var={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        poisson={i:[0] for i in ['EP','NA','NI','SI','SP','WP']}
        poisson_list={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        
        tracks_latitude={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        tracks_longitude={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        tracks_pressure={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        tracks_wind={i:[] for i in ['EP','NA','NI','SI','SP','WP']}   
        tracks_month={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        
        per_year={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        years_list={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        
        track={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        pressure_model={i:[] for i in ['EP','NA','NI','SI','SP','WP']} 
        
        mean_pressure={i:[] for i in ['EP','NA','NI','SI','SP','WP']}
        min_pressure={i:[] for i in ['EP','NA','NI','SI','SP','WP']} 
        max_wind={i:[] for i in ['EP','NA','NI','SI','SP','WP']} 
        
        
        for basin in ['EP','NA','NI','SI','SP','WP']:
            genesis_pressure[basin]={j:[] for j in monthsall[basin]}
            genesis_location[basin]={j:[] for j in monthsall[basin]}
            genesis_dpres[basin]={j:[] for j in monthsall[basin]}
            genesis_pres_var[basin]={j:[] for j in monthsall[basin]}         
                
            tracks_latitude[basin]={i:[] for i in range(2000)}
            tracks_longitude[basin]={i:[] for i in range(2000)}
            tracks_pressure[basin]={i:[] for i in range(2000)}
            tracks_wind[basin]={i:[] for i in range(2000)}   
            tracks_month[basin]={i:[] for i in range(2000)}   
            years_list[basin]={i:[] for i in range(2000)}
    
            if period=='PRESENT':
                per_year[basin]={i:[0] for i in range(1979,2015)}
        
            else:
                per_year[basin]={i:[0] for i in range(2015,2051)}
            
            track[basin]={i:[] for i in range(0,6)}
            pressure_model[basin]={i:[] for i in range(0,6)}    

        count=0
                    
            
        for hemisphere in ['NH','SH']:
            print(model,hemisphere,period,datetime.datetime.now()-starttime)
        
            data=xr.open_dataset(os.path.join(__datasets__,'TC-'+str(hemisphere)+'_TRACK_'+str(model)+'_'+str(period)+'.nc'))
            first_point=data['FIRST_PT'].values #first point of track
            no_point=data['NUM_PTS'].values #number of points in track
            track_id=data['TRACK_ID'].values #track_id (0-5321)
            pressure=data['psl'].values #pressure
            longitude=data['lon'].values #longitude from TRACK algorithm
            latitude=data['lat'].values #latitude from TRACK algorithm
            wind=data['sfcWind'].values #near-surface wind speed (10-meter)
            time=data['time'].values #datetime64    
           
            idx0=[i for i in range(len(time)) if int(str(time[i])[:4])>1978][0]
            idx0=np.argmax(first_point>idx0)-1
            
            #present-climate: (full range) 01/01/1950-30/12/2014
            #we take 01/01/1979 - 30/12/2014 to also have 36 years that most closely correspond
            #to the 38-year time period in IBTrACS.
            #future-climate: 01/01/2015-30/12/2050
     
            for i in range(idx0,len(first_point)):
                start=first_point[i]
                end=first_point[i]+no_point[i]
                
                windlist=wind[start:end]
                idx=np.where(windlist>=0)[0] #no threshold
                if len(idx)>1.:
                    j0=idx[0]   #first index where u>threshold
                    j1=idx[-1]+1  #last index where u>threshold
                
                    lonlist=longitude[start:end][j0:j1]
                    latlist=latitude[start:end][j0:j1]
                    preslist=pressure[start:end][j0:j1]
                    windlist=wind[start:end][j0:j1]
                    
                    #Check if genesis occurs inside a predefined basin
                    basin=genesis_basin(latlist[0],lonlist[0]) 
                    if basin!=None:
                        lat0,lat1,lon0,lon1=Basins_WMO(basin)
                        
                        basinlist=[basin for _ in lonlist]                                 
                    
                        #Explicitly create the datetime-object as every model has its own way of defining time.
                        timelist=[]                        
                        time_dummy=time[start:end][j0:j1]
    
                        for j in range(0,len(time_dummy)):
                            date_time_str=str(time_dummy[j])
                            
                            if model=='HadGEM3-GC31-HM':
                                if j==0:
                                    newyear=datetime.datetime.strptime(date_time_str[:4]+'-01-01 00:00:00', '%Y-%m-%d %H:%M:%S') 
                                    year_baseline=int(date_time_str[:4])
                                    
                                #calculate the number of hours between 1 January [year] and the actual day, in a 30-day world
                                year0=int(date_time_str[:4])
                                month0=int(date_time_str[5:7])
                                day0=int(date_time_str[8:10])
                                hour0=int(date_time_str[11:13])
                                
                                #total number of days
                                total_days=(year0-year_baseline)*360+30*(month0-1)+(day0-1)
                                
                                #convert to hours
                                total_hours=total_days*24+hour0  

                                date_time_obj=newyear+datetime.timedelta(hours=total_hours)

                            elif model =='CNRM-CM6-1-HR' or model=='EC-Earth3P-HR':
                                date_time_obj=datetime.datetime.strptime(date_time_str[:19], '%Y-%m-%dT%H:%M:%S')
                            else:                             
                                date_time_obj=datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                                
                            timelist.append(date_time_obj)                      
    
                        df=pd.DataFrame({'Latitude':latlist,'Longitude':lonlist,'Time':timelist,'Pressure':preslist,'Wind':windlist},index=None)
                        
                        df['Datetime']=pd.to_datetime(df['Time'])
                        df=df.set_index('Datetime')
                        df1=df.resample('3H').interpolate(method='linear')
                        basinlist=[basin for _ in df1['Longitude'].tolist()]
                        df1['Basin']=basinlist
                        df1['Hour']=df1.index.hour
                        df1['Month']=df1.index.month
                        df1['Year']=df1.index.year
                        df1=df1.reset_index()
                             
                        if int(df1['Month'][0]) in monthsall[basin] and df1['Year'][0]>1978:
                            #cut off data outside basin boundaries
                            df1=df1[(df1['Latitude']>=lat0) & (df1['Latitude']<=lat1) & (df1['Longitude']>=lon0) & (df1['Longitude'])<=lon1]
    
                            basin=df1['Basin'][0]
                            month=df1['Month'][0]
                            year=df1['Year'][0]
                            poisson[basin][0]+=1 
                            
                            per_year[basin][int(year)][0]+=1
                            
                            months[basin].append(month)                            
                            years_list[basin][count]=int(year)
                            count+=1
                            mean_pressure[basin].append(np.mean(df1['Pressure']))
                            min_pressure[basin].append(np.min(df1['Pressure']))
                            max_wind[basin].append(np.max(df1['Wind']))
                            
                            tracks_latitude[basin][count]=df1['Latitude'].tolist()
                            tracks_longitude[basin][count]=df1['Longitude'].tolist()
                            tracks_pressure[basin][count]=df1['Pressure'].tolist()
                            tracks_wind[basin][count]=df1['Wind'].tolist()
                            
                            tracks_month[basin][count]=month
                            
                            count+=1
    
                            genesis_pressure[basin][month].append(df1['Pressure'][0])
                            genesis_location[basin][month].append([df1['Latitude'][0],df1['Longitude'][0]])
                            genesis_dpres[basin][month].append(df1['Pressure'][1]-df1['Pressure'][0])
                            
                            for j in range(1,len(df1['Latitude'])-1):
                                track[basin][0].append(df1['Latitude'][j]-df1['Latitude'][j-1])
                                track[basin][1].append(df1['Latitude'][j+1]-df1['Latitude'][j])
                                track[basin][2].append(df1['Longitude'][j]-df1['Longitude'][j-1])
                                track[basin][3].append(df1['Longitude'][j+1]-df1['Longitude'][j])
                                track[basin][4].append(df1['Latitude'][j])
                                track[basin][5].append(df1['Longitude'][j])
                                
                                pressure_model[basin][0].append(df1['Pressure'][j]-df1['Pressure'][j-1])
                                pressure_model[basin][1].append(df1['Pressure'][j+1]-df1['Pressure'][j])
                                pressure_model[basin][2].append(df1['Pressure'][j])
                                pressure_model[basin][3].append(df1['Latitude'][j])
                                pressure_model[basin][4].append(df1['Longitude'][j])
                                pressure_model[basin][5].append(month)
                                      
        dp0_neg,dp0_pos=[],[] 
        for basin in ['EP','NA','NI','SI','SP','WP']:                            
        
            for j in range(len(pressure_model[basin][0])):
                if pressure_model[basin][0][j]<0.:
                    dp0_neg.append(pressure_model[basin][0][j])
                elif pressure_model[basin][0][j]>0:
                    dp0_pos.append(pressure_model[basin][0][j])
        
        
            pneg=np.percentile(dp0_neg,1)
            ppos=np.percentile(dp0_pos,99)  
            
            for month in monthsall[basin]:
                dplist=[v for v in genesis_dpres[basin][month] if np.isnan(v)==False and v>-1000.]
                plist=[v for v in genesis_pressure[basin][month] if np.isnan(v)==False and v>0.]
                
                mudp0,stddp0=stats.norm.fit(dplist)
                mupres,stdpres=stats.norm.fit(plist)
                
                genesis_pres_var[basin][month]=[mupres,stdpres,mudp0,stddp0,pneg,ppos]  
                
            poisson_list[basin]=[poisson[basin][0]/36.]
                                      
        np.save(os.path.join(__location__,'POISSON_GENESIS_PARAMETERS_'+str(period)+'_'+str(model)+'_nothres.npy'),poisson_list)
        np.save(os.path.join(__location__,'TC_TRACK_VARIABLES_'+str(period)+'_'+str(model)+'_nothres.npy'),track)
        np.save(os.path.join(__location__,'TC_PRESSURE_VARIABLES_'+str(period)+'_'+str(model)+'_nothres.npy'),pressure_model)
        np.save(os.path.join(__location__,'DP0_PRES_GENESIS_'+str(period)+'_'+str(model)+'_nothres.npy'),genesis_pres_var)
        
        np.save(os.path.join(__location__,'DP_GEN_'+str(period)+'_'+str(model)+'_nothres.npy'),genesis_dpres)
        np.save(os.path.join(__location__,'PRES_GEN_'+str(period)+'_'+str(model)+'_nothres.npy'),genesis_pressure)
        np.save(os.path.join(__location__,'GEN_LOC_'+str(period)+'_'+str(model)+'_nothres.npy'),genesis_location,fix_imports=True)        
        np.save(os.path.join(__location__,'GENESIS_MONTHS_'+str(period)+'_'+str(model)+'_nothres.npy'),months)
        np.save(os.path.join(__location__,'GENESIS_YEARS_'+str(period)+'_'+str(model)+'_nothres.npy'),years_list)
        
        np.save(os.path.join(__location__,'FREQUENCY_PER_YEAR_'+str(period)+'_'+str(model)+'_nothres.npy'),per_year)
    
        np.save(os.path.join(__location__,'TC_TRACK_LATITUDE_'+str(period)+'_'+str(model)+'_nothres.npy'),tracks_latitude)
        np.save(os.path.join(__location__,'TC_TRACK_LONGITUDE_'+str(period)+'_'+str(model)+'_nothres.npy'),tracks_longitude)
        np.save(os.path.join(__location__,'TC_TRACK_PRESSURE_'+str(period)+'_'+str(model)+'_nothres.npy'),tracks_pressure)
        np.save(os.path.join(__location__,'TC_TRACK_WIND_'+str(period)+'_'+str(model)+'_nothres.npy'),tracks_wind)
        np.save(os.path.join(__location__,'TC_TRACK_MONTH_'+str(period)+'_'+str(model)+'_nothres.npy'),tracks_month)
        
        np.save(os.path.join(__location__,'TC_MEAN_PRESSURE_'+str(period)+'_'+str(model)+'_nothres.npy'),mean_pressure)
        np.save(os.path.join(__location__,'TC_MIN_PRESSURE_'+str(period)+'_'+str(model)+'_nothres.npy'),min_pressure)
        np.save(os.path.join(__location__,'TC_MAX_WIND_'+str(period)+'_'+str(model)+'_nothres.npy'),max_wind)

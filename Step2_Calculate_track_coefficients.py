# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing and derive the coefficients
of the regression formulas. These coefficients will not be used directly in the delta approach (some of the other output variables will). But it's good to keep these coefficients
in case you want to run the GCMs directly through STORM.

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
from scipy.optimize import curve_fit
import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def BOUNDARIES_BASINS(idx):
    if idx=='EP': #Eastern Pacific
        lat0,lat1,lon0,lon1=5,60,180,285
    if idx=='NA': #North Atlantic
        lat0,lat1,lon0,lon1=5,60,255,359
    if idx=='NI': #North Indian
        lat0,lat1,lon0,lon1=5,60,30,100
    if idx=='SI': #South Indian
        lat0,lat1,lon0,lon1=-60,-5,10,135
    if idx=='SP': #South Pacific
        lat0,lat1,lon0,lon1=-60,-5,135,240
    if idx=='WP': #Western Pacific
        lat0,lat1,lon0,lon1=5,60,100,180
    
    return lat0,lat1,lon0,lon1

def LATFUNCTION(X,a,b,c):
    """
    This is the regression function for the latitudinal movement of the TC. This function returns the values
    of the coefficients a,b,c.
    Parameters
    ----------
    X : array of delta-latitude and latitude ([dlat,lat])
    a,b,c : coefficients of regression function for latitude
    """
    dlat,lat=X
    return a+b*dlat+c/lat

def LATEXPECTED(dlat,lat,a,b,c):
    """
    Calculate the expected forward change in latitude.
    
    Parameters
    ----------
    dlat : backward change in latitude (lat[i]-lat[i-1])
    lat : latitude
    a,b,c: coefficient values

    Returns
    -------
    dlat1_list : array of forward change in latitude (lat[i+1]-lat[i])
    """
    dlat1_list=[]
    for k in range(len(dlat)):
        dlat1_list.append(a+b*dlat[k]+c/lat[k])
    return dlat1_list

def LONFUNCTION(x,a,b):
    """
    This is the regression formula for the longitudinal movement of the TC. This function 
    returns optimal fit values of the coefficients a and b. 
    Parameters
    ----------
    x : backward change in longitude (dlon0; lon[i]-lon[i-1]).
    a,b: coefficients

    """
    return a+b*x

def LONEXPECTED(x,a,b):
    """
    Parameters
    ----------
    x : list of backward change in longitude (dlon0; lon[i]-lon[i-1]).
    a,b : coefficients.

    Returns
    -------
    dlon1_list : array of forward change in longitude (lon[i+1]-lon[i])

    """
    dlon1_list=[]
    for k in range(len(x)):
        dlon1_list.append(a+b*x[k])
    return dlon1_list

def track_coefficients(period,model):
    """
    Calculate the track coefficients. This function returns a npy-file with the coefficients
    per 5 deg bin for every basin. 
    """
    step=5.
           
    data=np.load(os.path.join(__location__,'TC_TRACK_VARIABLES_{}_{}_nothres.npy'.format(period,model)),allow_pickle=True).item()
    
    coefficients_list={i:[] for i in range(0,6)} #for every basin
    
    for basin,idx in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):
               
        df=pd.DataFrame({'Latitude':data[basin][4],'Longitude':data[basin][5],'Dlat0':data[basin][0],'Dlat1':data[basin][1],'Dlon0':data[basin][2],'Dlon1':data[basin][3]})
        lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin)
        
        df=df[(df['Latitude']<=lat1) & (df['Latitude']>=lat0) & (df['Longitude']<=lon1) & (df['Longitude']>=lon0)]
            
        latspace=np.linspace(lat0,lat1-5.,int(abs(lat0-lat1+5)/step)+1)
    
        to_bin=lambda x:np.floor(x/step)*step
        df["latbin"]=df.Latitude.map(to_bin)
        Dlat0_group=df.groupby(["latbin"])['Dlat0'].apply(list)
        Dlat1_group=df.groupby(["latbin"])['Dlat1'].apply(list)
        Dlon0_group=df.groupby(["latbin"])['Dlon0'].apply(list)
        Dlon1_group=df.groupby(["latbin"])['Dlon1'].apply(list)
        Lat_group=df.groupby(["latbin"])['Latitude'].apply(list)
        bins_values=np.unique(df["latbin"])   
    
        coeff_array=[[0]]*len(latspace)
        count=0  
        for lat in bins_values:
            i_ind=int((lat-lat0)/step)
            if len(Dlat0_group[lat])>50.:
                
                Lat=Lat_group[lat]
                Dlat0=Dlat0_group[lat]
                Dlat1=Dlat1_group[lat]
                Dlon0=Dlon0_group[lat]
                Dlon1=Dlon1_group[lat]  
                 
                try:
                    opt,l=curve_fit(LONFUNCTION,Dlon0,Dlon1)
                    [a0,a1]=opt
                                        
                    expected=LONEXPECTED(Dlon0,a0,a1)
                    Elon=[]
                    for m in range(len(expected)):
                        Elon.append(expected[m]-Dlon1[m])
                    
                    Elonmu,Elonstd=norm.fit(Elon)
    
                    X=[Dlat0,Lat]
                    opt,l=curve_fit(LATFUNCTION,X,Dlat1)
                    [b0,b1,b2]=opt                
    
                    Dlat1mu,Dlat1std=norm.fit(Dlat0)
                    
                    Dlon1mu,Dlon1std=norm.fit(Dlon0)
    
                    expected=LATEXPECTED(Dlat0,Lat,b0,b1,b2)
                    
                    Elat=[]
                    for m in range(len(expected)):
                        Elat.append(expected[m]-Dlat1[m]) 
       
                    Elatmu,Elatstd=norm.fit(Elat)    
        
                    if abs(Elatmu)<1 and abs(Elonmu)<1: #otherwise: the fit didn't go as planned: large deviation from expected values..
                        coeff_array[i_ind]=[a0,a1,b0,b1,b2,Elatmu,Elatstd,Elonmu,Elonstd,Dlat1mu,Dlat1std,Dlon1mu,Dlon1std]
                        count=count+1
    
                except RuntimeError:
                    print("No fit found") 
         
        if basin=='SI' or basin=='SP': #Southern Hemisphere works the other way around           
            while count<len(latspace):
                for i in reversed(range(len(latspace))):
                    if len(coeff_array[i])==1:
                        if i<len(latspace)-1 and len(coeff_array[i+1])>1:
                            coeff_array[i]=coeff_array[i+1]
                            count=count+1
                        elif i>0 and len(coeff_array[i-1])>1.:
                            coeff_array[i]=coeff_array[i-1]
                            count=count+1
        else:
            while count<len(latspace):
                for i in range(len(latspace)):
                    if len(coeff_array[i])==1:
                        if i>0 and len(coeff_array[i-1])>1:
                            coeff_array[i]=coeff_array[i-1]
                            count=count+1
                        elif i<len(latspace) and len(coeff_array[i+1])>1.:
                            coeff_array[i]=coeff_array[i+1]
                            count=count+1        
        
        coefficients_list[idx]=coeff_array      
  
    np.save(os.path.join(__location__,'JM_LONLATBINS_{}_{}_nothres.npy'.format(period,model)),coefficients_list)                    

for model in ['CMCC-CM2-VHR4','CNRM-CM6-1-HR','EC-Earth3P-HR','HadGEM3-GC31-HM']:  
    for period in ['PRESENT','FUTURE']:
        print(model,period)
        track_coefficients(period,model)

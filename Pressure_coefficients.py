# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing and calculate the environmental
conditions + wind-pressure relationship.

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
import math
import os
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def BOUNDARIES_BASINS(idx):
    if idx=='EP': #Eastern Pacific
        lat0,lat1,lon0,lon1=0,60,180,285
    if idx=='NA': #North Atlantic
        lat0,lat1,lon0,lon1=0,60,255,360
    if idx=='NI': #North Indian
        lat0,lat1,lon0,lon1=0,60,30,100
    if idx=='SI': #South Indian
        lat0,lat1,lon0,lon1=-60,0,10,135
    if idx=='SP': #South Pacific
        lat0,lat1,lon0,lon1=-60,0,135,240
    if idx=='WP': #Western Pacific
        lat0,lat1,lon0,lon1=0,60,100,180
    
    return lat0,lat1,lon0,lon1

def PRESFUNCTION(X,a,b,c,d):
    """
    Fit the data to the pressure function. 
    Parameters
    ----------
    X : array of change in pressure and difference between pressure and mpi ([dp0,p-mpi])
    a,b,c,d : Coefficients

    """
    dp,presmpi=X
    return a+b*dp+c*np.exp(-d*presmpi)

def PRESEXPECTED(dp,presmpi,a,b,c,d):
    """
    Calculate the forward change in pressure (dp1, p[i+1]-p[i])    

    Parameters
    ----------
    dp : backward change in pressure (dp0, p[i]-p[i-1])
    presmpi : difference between pressure and mpi (p-mpi).
    a,b,c,d : coefficients

    Returns
    -------
    dp1_list : array of forward change in pressure (dp1, p[i+1]-p[i])

    """
    dp1_list=[]
    for k in range(len(dp)):
        dp1_list.append(a+b*dp[k]+c*np.exp(-d*presmpi[k]))
    return dp1_list

def MPI_function(T,A,B,C):
    """
    Fit the MPI function to the data. This function returns the optimal coefficients.
    Parameters
    ----------
    T : Sea-surface temperature in Celcius.
    A,B,C : coefficients 

    """
    return A+B*np.exp(C*(T-30.))

def Calculate_P(V,Penv,a,b):
    """
    Convert Vmax to Pressure following the empirical wind-pressure relationship (Harper 2002, Atkinson and Holliday 1977)
    
    Input: 
        Vmax: 10-min mean maximum wind speed in m/s
        Penv: environmental pressure (hPa)
        a,b: coefficients. See Atkinson_Holliday_wind_pressure_relationship.py
    
    Returns:
        Pc: central pressure in the eye
    
    """
    
    Pc=Penv-(V/a)**(1./b)  
    return Pc


def MPI_fields(period,model):
    data=np.load(os.path.join(__location__,'latlon_background_flipped_{}.npy'.format(model)),allow_pickle=True).item()

    lat=data['lat']
    lon=data['lon']
    
    coeflist=np.load(os.path.join(__location__,'COEFFICIENTS_MPI_PRESSURE_DROP_MONTH_new.npy'),allow_pickle=True,encoding='latin1').item()
    
    monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}             
    mpibounds={'EP':[860,880,900,900,880,860],'NA':[920,900,900,900,880,880],'NI':[840,860,880,900,880,860],'SI':[840,880,860,860,840,860],'SP':[840,840,860,860,840,840],'WP':[860,860,860,870,870,860,860]}
    
    basinidx={'EP':0,'NA':1,'NI':2,'SI':3,'SP':4,'WP':5}
    
    MPI_all={i:[] for i in ['EP','NA','NI','SI','SP','WP']} 
    for basin in ['EP','NA','NI','SI','SP','WP']:
        MPI_all[basin]={i:[] for i in monthsall[basin]}
        
        for month,index in zip(monthsall[basin],range(len(monthsall[basin]))): 
            print(model,period,basin,month)
            
            [A,B,C]=coeflist[int(basinidx[basin])][month] 
            
            SST= np.loadtxt(os.path.join(__location__,'Monthly_mean_SST_{}_{}_{}_masked_flipped.txt'.format(model,month,period)))
            MSLP=np.loadtxt(os.path.join(__location__,'Monthly_mean_MSLP_{}_{}_{}_flipped.txt'.format(model,month,period)))
            
            lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin)
            
            lat_0=np.abs(lat-lat0).argmin()
            lat_1=np.abs(lat-lat1).argmin()
            lon_0=np.abs(lon-lon0).argmin()
            lon_1=np.abs(lon-lon1-0.1).argmin()
                        
            SST_field=SST[lat_1:lat_0,lon_0:lon_1]
            MSLP_field=MSLP[lat_1:lat_0,lon_0:lon_1]
                                
            PC_MATRIX=np.zeros((SST_field.shape))
            PC_MATRIX[:]=np.nan
                    
            PRESDROP=MPI_function(SST_field-273.15,A,B,C) #Vmax is given in m/s
            PC_MATRIX=MSLP_field-PRESDROP
            boundary=mpibounds[basin][index]
      
            PC_MATRIX[PC_MATRIX<boundary]=boundary
            PC_MATRIX[PC_MATRIX>1030.]=1030.

            print(np.nanmin(PC_MATRIX),np.nanmax(PC_MATRIX))
            
            MPI_all[basin][month]=PC_MATRIX
            
    np.save(os.path.join(__location__,'MPI_FIELDS_{}_{}.npy'.format(model,period)),MPI_all)


def Pressure_coefficients(period,model):
    data=np.load(os.path.join(__location__,'latlon_background_flipped_{}.npy'.format(model)),allow_pickle=True).item()
    
    lon=data['lon']
    lat=data['lat']
    
    step=5
    pres_variables=np.load(os.path.join(__location__,'TC_PRESSURE_VARIABLES_{}_{}_nothres.npy'.format(period,model)),allow_pickle=True,encoding='latin1').item()
    
    monthsall={'EP':[6,7,8,9,10,11],'NA':[6,7,8,9,10,11],'NI':[4,5,6,9,10,11],'SI':[1,2,3,4,11,12],'SP':[1,2,3,4,11,12],'WP':[5,6,7,8,9,10,11]}             
    
    #basinidx={'EP':0,'NA':1,'NI':2,'SI':3,'SP':4,'WP':5}
    
    Pres_coefficients_all={i:[] for i in ['EP','NA','NI','SI','SP','WP']} 
    
    MPI_DATA=np.load(os.path.join(__location__,'MPI_FIELDS_{}_{}.npy'.format(model,period)),allow_pickle=True,encoding='latin1').item()
          
    for basin in ['EP','NA','NI','SI','SP','WP']:
        Pres_coefficients_all[basin]={i:[] for i in monthsall[basin]}
    
        lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(basin) 
        
        lat_0=np.abs(lat-lat1).argmin()
        lon_0=np.abs(lon-lon0).argmin()
        
        for month in monthsall[basin]:  
            print(model,period,basin,month)
            MPI_MATRIX=MPI_DATA[basin][month]
        
            lat_df,lon_df,mpi_df=[],[],[]
            
            for i in range(len(MPI_MATRIX[:,0])):
                for j in range(len(MPI_MATRIX[0,:])):
                    lat_df.append(lat[i+lat_0])
                    lon_df.append(lon[j+lon_0])
                    mpi_df.append(MPI_MATRIX[i,j])
                               
            df=pd.DataFrame({'Latitude':lat_df,'Longitude':lon_df,'MPI':mpi_df})
            to_bin=lambda x:np.floor(x/step)*step
            df["latbin"]=df["Latitude"].map(to_bin)
            df["lonbin"]=df["Longitude"].map(to_bin)
            MPI=df.groupby(["latbin","lonbin"])['MPI'].apply(list)  
        
            latbins1=np.linspace(lat0,lat1-5,(lat1-5-lat0)//step+1)
            lonbins1=np.linspace(lon0,lon1-5,(lon1-5-lon0)//step+1)
            
            matrix_mpi=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
        
            for latidx in latbins1:
                for lonidx in lonbins1:
                    i_ind=int((latidx-lat0)/5.)
                    j_ind=int((lonidx-lon0)/5.)
                    matrix_mpi[i_ind,j_ind]=np.nanmin(MPI[latidx][lonidx])
            
            if basin=='NA':
                matrix_mpi=np.c_[matrix_mpi,matrix_mpi[:,-1]]
                    
            df_data=pd.DataFrame({'Latitude':pres_variables[basin][3],'Longitude':pres_variables[basin][4],'Pressure':pres_variables[basin][2],'DP0':pres_variables[basin][0],'DP1':pres_variables[basin][1],'Month':pres_variables[basin][5]})
            df_data=df_data[(df_data['Pressure']>0.) & (df_data['DP0']>-10000.) & (df_data['DP1']>-10000.) & (df_data['Longitude']>=lon0) &(df_data['Longitude']<lon1) & (df_data["Latitude"]>=lat0) & (df_data["Latitude"]<lat1)]
            df_data1=df_data[df_data["Month"]==month].copy().reset_index(drop=True)
            
            df_data1["latbin"]=df_data1["Latitude"].map(to_bin)
            df_data1["lonbin"]=df_data1["Longitude"].map(to_bin)    
        
            latbins=np.unique(df_data1["latbin"])
            lonbins=df_data1.groupby("latbin")["lonbin"].apply(list)
            Pressure=df_data1.groupby(["latbin","lonbin"])["Pressure"].apply(list)
            DP1=df_data1.groupby(["latbin","lonbin"])['DP1'].apply(list)
            DP0=df_data1.groupby(["latbin","lonbin"])['DP0'].apply(list) 
                    
            matrix_mean=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_std=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c0=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c1=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c2=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
            matrix_c3=-100*np.ones((int((lat1-lat0)/5),int((lon1-lon0)/5)))
        
            count=0
            lijst=[]
            for latidx in latbins:
                lonlist=np.unique(lonbins[latidx])
                for lonidx in lonlist:
                    lijst.append((latidx,lonidx))
                    
            if len(lijst)==0:
                #skip this run
                print('This run will be skipped - no data!', model,period,basin,month)
            else:                    
                for latidx in latbins:
                    lonlist=np.unique(lonbins[latidx])
                    for lonidx in lonlist:            
                        i_ind=int((latidx-lat0)/5.)
                        j_ind=int((lonidx-lon0)/5.)
                        preslist=[]
                        dp0list=[]
                        dp1list=[]
                        mpi=[]
                        #include all bins from lat-5 to lat+5 and lon-5 to lon+5
                        for lat_sur in [-5,0,5]:
                            for lon_sur in [-5,0,5]:
                                if (int(latidx+lat_sur),int(lonidx+lon_sur)) in lijst:
                                    if np.nanmin(MPI[latidx+lat_sur][lonidx+lon_sur])>0.:
                                        for pr,d0,d1 in zip(Pressure[latidx+lat_sur][lonidx+lon_sur],DP0[latidx+lat_sur][lonidx+lon_sur],DP1[latidx+lat_sur][lonidx+lon_sur]):
                                            preslist.append(pr)
                                            dp0list.append(d0)
                                            dp1list.append(d1)
                                            mpi.append(np.nanmin(MPI[latidx+lat_sur][lonidx+lon_sur]))
                                        
                        if len(preslist)>9.:
                            presmpi_list=[]
                            for y in range(len(preslist)):
                                if preslist[y]<mpi[y]:
                                    presmpi_list.append(0)
                                else:
                                    presmpi_list.append(preslist[y]-mpi[y])
                                    
                            X=[dp0list,presmpi_list]
                            try:
                                opt,l=curve_fit(PRESFUNCTION,X,dp1list,p0=[0,0,0,0],maxfev=5000)
                                [c0,c1,c2,c3]=opt
                                expected=PRESEXPECTED(dp1list,presmpi_list,c0,c1,c2,c3)
                                Epres=[]
                                for ind in range(len(expected)):
                                    Epres.append(expected[ind]-dp0list[ind])
                                    
                                mu,std=norm.fit(Epres)
                                if abs(mu)<1 and c2>0: #otherwise: the fit didn't go as planned: large deviation from expected values..
                                    matrix_mean[i_ind,j_ind]=mu
                                    matrix_std[i_ind,j_ind]=std
                                    matrix_c0[i_ind,j_ind]=c0
                                    matrix_c1[i_ind,j_ind]=c1
                                    matrix_c2[i_ind,j_ind]=c2
                                    matrix_c3[i_ind,j_ind]=c3
                            except RuntimeError:
                                count=count+1
                print (str(count)+' fields out of '+str(len(latbins1)*len(lonbins1))+' bins do not have a fit')
                
                (X,Y)=matrix_mean.shape
                neighbors=lambda x, y : [(x2, y2) for (x2,y2) in [(x,y-1),(x,y+1),(x+1,y),(x-1,y),(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]
                                            if (-1 < x < X and
                                                -1 < y < Y and
                                                (x != x2 or y != y2) and
                                                (0 <= x2 < X) and
                                                (0 <= y2 < Y))]
                var=100
                while var!=0:
                    shadowmatrix=np.zeros((X,Y))
                    zeroeslist=[[i1,j1] for i1,x in enumerate(matrix_mean) for j1,y in enumerate(x) if y==-100]
                    var=len(zeroeslist)
                    for [i,j] in zeroeslist:       
                            lijst=neighbors(i,j)
                            for item in lijst:
                                (i0,j0)=item
                                if matrix_mean[i0,j0]!=-100 and shadowmatrix[i0,j0]==0:
                                    matrix_mean[i,j]=matrix_mean[i0,j0]
                                    matrix_std[i,j]=matrix_std[i0,j0]
                                    matrix_c0[i,j]=matrix_c0[i0,j0]
                                    matrix_c1[i,j]=matrix_c1[i0,j0]
                                    matrix_c2[i,j]=matrix_c2[i0,j0]
                                    matrix_c3[i,j]=matrix_c3[i0,j0]
                                    shadowmatrix[i,j]=1                     
                                    break
                
                print('Filling succeeded')                 
                var=100
                (X,Y)=matrix_mpi.shape
                while var!=0:
                    shadowmatrix=np.zeros((X,Y))
                    zeroeslist=[[i1,j1] for i1,x in enumerate(matrix_mpi) for j1,y in enumerate(x) if math.isnan(y)]
                    var=len(zeroeslist)
                    for [i,j] in zeroeslist:       
                            lijst=neighbors(i,j)
                            for item in lijst:
                                (i0,j0)=item
                                if math.isnan(matrix_mpi[i0,j0])==False and shadowmatrix[i0,j0]==0:
                                    matrix_mpi[i,j]=matrix_mpi[i0,j0]
                                    shadowmatrix[i,j]=1                     
                                    break
             
                for i in range(0,X):
                    for j in range(0,Y):
                        Pres_coefficients_all[basin][month].append([matrix_c0[i,j],matrix_c1[i,j],matrix_c2[i,j],matrix_c3[i,j],matrix_mean[i,j],matrix_std[i,j],matrix_mpi[i,j]])
                
                        if model =='CNRM-CM6-1-HR' and period=='FUTURE' and basin=='NI' and month==5:
                            #use the coefficients to fill the missing data from last run
                            print('The other coefficients have been added')
                            Pres_coefficients_all[basin][month-1].append([matrix_c0[i,j],matrix_c1[i,j],matrix_c2[i,j],matrix_c3[i,j],matrix_mean[i,j],matrix_std[i,j],matrix_mpi[i,j]])
                        
    np.save(os.path.join(__location__,'COEFFICIENTS_JM_PRESSURE_{}_{}.npy'.format(model,period)),Pres_coefficients_all)      

for model in ['CNRM-CM6-1-HR','EC-Earth3P-HR','HadGEM3-GC31-HM','CMCC-CM2-VHR4']:
    for period in ['FUTURE','PRESENT']:
        MPI_fields(period,model)
        Pressure_coefficients(period,model)
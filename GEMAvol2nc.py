# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:52:29 2020

@author: Weather Radar Team
"""
from netCDF4 import Dataset, date2num
import os, gc, warnings
import numpy as np
import wradlib as wrl
from datetime import datetime
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def extractRadarData(radarFile):
    f = wrl.util.get_wradlib_data_file(radarFile)
    raw = wrl.io.read_rainbow(f)
    
    try:
        radarLon=float(raw['volume']['sensorinfo']['lon'])
        radarLat=float(raw['volume']['sensorinfo']['lat'])
        radarAlt=float(raw['volume']['sensorinfo']['alt'])
    except:
        radarLon=float(raw['volume']['radarinfo']['@lon'])
        radarLat=float(raw['volume']['radarinfo']['@lat'])
        radarAlt=float(raw['volume']['radarinfo']['@alt'])
        
    sitecoords=(radarLon,radarLat,radarAlt)
    
    res=250. # resolusi data yang diinginkan dalam meter
    resCoords=res/111229. # resolusi data dalam derajat
    rmax=250000./111229. # range maksimum
    lonMax,lonMin=radarLon+(rmax),radarLon-(rmax) 
    latMax,latMin=radarLat+(rmax),radarLat-(rmax)
    nGrid=int(np.floor((lonMax-lonMin)/resCoords))+1 # jumlah grid
    lonGrid=np.linspace(lonMin,lonMax,nGrid) # grid longitude
    latGrid=np.linspace(latMin,latMax,nGrid) # grid latitude            
    dataContainer = np.zeros((len(lonGrid),len(latGrid))) # penampung data
    
    
    # menentukan waktu (end of observation)
    nSlices=len(raw['volume']['scan']['slice'])
    date=(raw['volume']['scan']['slice'][nSlices-1]['slicedata']['@date'])
    time=(raw['volume']['scan']['slice'][nSlices-1]['slicedata']['@time'])
    try:timeEnd=datetime.strptime('{}{}'.format(date,time),"%Y-%m-%d%H:%M:%S")
    except:timeEnd=datetime.strptime('{}{}'.format(date,time),"%Y-%m-%d%H:%M:%S.%f")
    
    allElevation=[]
    nElevation=len(raw['volume']['scan']['slice']) # jumlah seluruh elevasi
    for i in range(nElevation):
        try:elevation = float(raw['volume']['scan']['slice'][i]['posangle'])
        except:elevation = float(raw['volume']['scan']['slice'][0]['posangle'])
        allElevation.append(elevation)  
        print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(i+1,elevation))
        
        # ekstrak azimuth data
        try:
            azi = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['data']
            azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@depth'])
            azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo']['@rays'])  
        except:
            azi0 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['data']
            azi1 = raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][1]['data']
            azi = (azi0/2) + (azi1/2)
            del azi0, azi1
            azidepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@depth'])
            azirange = float(raw['volume']['scan']['slice'][i]['slicedata']['rayinfo'][0]['@rays'])            
        try:
            azires = float(raw['volume']['scan']['slice'][i]['anglestep'])
        except:
            azires = float(raw['volume']['scan']['slice'][0]['anglestep'])
        azi = (azi * azirange / 2**azidepth) * azires
        
        flag=0
        if np.size(azi) >= 999:
            flag=2
            azi = azi/3
            for ii in range(int(np.floor(np.size(azi)/3))):
                azi[ii] = azi[3*ii]+azi[3*ii+1]+azi[3*ii+2]
            azi = azi[range(int(np.floor(np.size(azi)/3)))]
        elif np.size(azi) >= 500:
            flag=1
            azi = azi/2
            for ii in range(int(np.floor(np.size(azi)/2))):
                azi[ii] = azi[2*ii]+azi[2*ii+1]
            azi = azi[range(int(np.floor(np.size(azi)/2)))]
        
        # esktrak range data
        try:
            stoprange = float(raw['volume']['scan']['slice'][i]['stoprange'])
            rangestep = float(raw['volume']['scan']['slice'][i]['rangestep'])
        except:
            stoprange = float(raw['volume']['scan']['slice'][0]['stoprange'])
            rangestep = float(raw['volume']['scan']['slice'][0]['rangestep'])
        r = np.arange(0, stoprange, rangestep)*1000
        
        data_ = raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['data']
        datadepth = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@depth'])
        datamin = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@min'])
        datamax = float(raw['volume']['scan']['slice'][i]['slicedata']['rawdata']['@max'])
        data_ = datamin + data_ * (datamax - datamin) / 2 ** datadepth
    
        if flag==2:
            data_ = data_/3
            for jj in range(int(np.floor(np.size(data_[:,1])/3))):
                data_[jj,:] = data_[3*jj,:] + data_[3*jj+1,:] + data_[3*jj+2,:]
            data_ = data_[range(int(np.floor(np.size(data_[:,1])/3))),:]
        elif flag==1:
            data_ = data_/2
            for jj in range(int(np.floor(np.size(data_[:,1])/2))):
                data_[jj,:] = data_[2*jj,:] + data_[2*jj+1,:]
            data_ = data_[range(int(np.floor(np.size(data_[:,1])/2))),:]
    
        # If len(azi) == 447 will generate error in wrl.ipol.interpolate_polar
        # "ValueError: operands could not be broadcast together with shapes"
        if len(azi) == 175:
            azi = azi[:-1]
            data_ = data_[:-1,:]
    
        delta = len(r) - len(np.transpose(data_))
        if delta > 0:
            r = r[:-delta]
        data=data_
    
        
        # transformasi dari koordinat bola ke koordinat kartesian
        rangeMesh, azimuthMesh =np.meshgrid(r,azi) # meshgrid azimuth dan range
        lonlatalt = wrl.georef.polar.spherical_to_proj(
            rangeMesh, azimuthMesh, elevation, sitecoords
        ) 
        x, y = lonlatalt[:, :, 0], lonlatalt[:, :, 1]
            
    
        # proses regriding ke data container yang sudah dibuat sebelumnya
        lonMesh, latMesh=np.meshgrid(lonGrid,latGrid)
        gridLatLon = np.vstack((lonMesh.ravel(), latMesh.ravel())).transpose()
        xy=np.concatenate([x.ravel()[:,None],y.ravel()[:,None]], axis=1)
        radius=r[np.size(r)-1]
        center=[x.mean(),y.mean()]
        gridded = wrl.comp.togrid(
            xy, gridLatLon,
            radius, center, data.ravel(),
            wrl.ipol.Linear
        )
        griddedData = np.ma.masked_invalid(gridded).reshape((len(lonGrid), len(latGrid)))
        dataContainer=np.dstack((dataContainer,griddedData))
    
    dataContainer = np.delete(dataContainer,0,2) # menghapus base layer dataContainer
    return lonGrid,latGrid,timeEnd,allElevation,dataContainer

def writeNetcdf(ncpath,site,timeEnd,lonGrid,latGrid,dataContainer,allElevation):
    cmaxData=np.nanmax(dataContainer[:,:,:],axis=2)
    cmaxData[cmaxData<0]=np.nan;cmaxData[cmaxData>100]=np.nan
    
    filename='{}/{}{}.nc'.format(ncpath,site,timeEnd.strftime("%Y%m%d%H%M"))
    print('Writing netcdf file {}'.format(filename))
    ncout = Dataset(filename,'w',format='NETCDF4')
    nlat=len(latGrid)
    nlon=len(lonGrid)
    nelev=len(allElevation)
    
    # create axis size
    ncout.createDimension('time', None)
    ncout.createDimension('lat', nlat)
    ncout.createDimension('lon', nlon)
    ncout.createDimension('lev', nelev)

    # create time axis
    time = ncout.createVariable('time', np.dtype('double').char, ('time',))
    time.long_name = 'time'
    time.units = 'hours since 1990-01-01 00:00:00'
    time.calendar = 'standard'
    time.axis = 'T'
    time[:] = date2num(timeEnd,units=time.units,calendar=time.calendar)

    # create latitude axis
    lat = ncout.createVariable('lat', np.dtype('double').char, ('lat'))
    lat.standard_name = 'latitude'
    lat.long_name = 'latitude'
    lat.units = 'degrees_north'
    lat.axis = 'Y'   
    lat[:] = sorted(latGrid[:])
    
    # create longitude axis
    lon = ncout.createVariable('lon', np.dtype('double').char, ('lon'))
    lon.standard_name = 'longitude'
    lon.long_name = 'longitude'
    lon.units = 'degrees_east'
    lon.axis = 'X'
    lon[:] = sorted(lonGrid[:])
    
    # create altitude axis
    lev = ncout.createVariable('lev', np.dtype('double').char, ('lev'))
    lev.standard_name = 'elevation'
    lev.long_name = 'elevation'
    lev.units = 'degrees_angle'
    lev.axis = 'Z'
    lev[:] = allElevation[:]
    
    # create variable cmax
    voutCMAX = ncout.createVariable('max_dbz', np.dtype('double').char, ('lon', 'lat'))
    voutCMAX.long_name = 'max dBZ'
    voutCMAX.units = 'dBZ'
    voutCMAX[:] = cmaxData[:].transpose()
    
    # create variable ppi
    voutPPI = ncout.createVariable('ppi_dbz', np.dtype('double').char, ('lon', 'lat','lev'))
    voutPPI.long_name = 'ppi dBZ'
    voutPPI.units = 'dBZ'
    for i in range(len(allElevation)):
        ppiData=dataContainer[:,:,i]
        ppiData[ppiData<0]=np.nan
        ppiData[ppiData>100]=np.nan
        voutPPI[:,:,i]=ppiData[:].transpose()
        del ppiData
    
    ncout.close()
    gc.collect();del gc.garbage[:] 
    print('Complete writing netcdf file')

def main():
    site='SBY'
    radarFile='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/2020062106000600dBZ.vol'
    ncpath='D:/project_webprogramming/wxradarexplore/radarDataConversion/nc'
    try:os.makedirs(ncpath)
    except:pass
    lonGrid,latGrid,timeEnd,allElevation,dataContainer=extractRadarData(radarFile)
    writeNetcdf(ncpath,site,timeEnd,lonGrid,latGrid,dataContainer,allElevation)

main()
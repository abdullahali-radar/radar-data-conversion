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
    data, metadata = wrl.io.read_gamic_hdf5(f)
    
    radarLon=float(metadata['VOL']['Longitude'])
    radarLat=float(metadata['VOL']['Latitude'])
    radarAlt=float(metadata['VOL']['Height'])
    sitecoords=(radarLon,radarLat,radarAlt)
    
    res=250. # resolusi data yang diinginkan dalam meter
    resCoords=res/111229. # resolusi data dalam derajat
    rmax=250000./111229. # range maksimum
    lonMax,lonMin=radarLon+(rmax),radarLon-(rmax) 
    latMax,latMin=radarLat+(rmax),radarLat-(rmax)
    nGrid=int(np.floor((lonMax-lonMin)/resCoords))+1 # jumlah grid
    lonGrid=np.linspace(lonMax,lonMin,nGrid) # grid longitude
    latGrid=np.linspace(latMax,latMin,nGrid) # grid latitude            
    dataContainer = np.zeros((len(lonGrid),len(latGrid))) # penampung data
    
    allElevation=[]
    nElevation=len(data)
    for i in range(nElevation):
        sweep='SCAN'+str(i)
        timeEnd=datetime.strptime(str(metadata[sweep]['Time']),"b'%Y-%m-%dT%H:%M:%S.%fZ'")
        elevation=float('{0:.1f}'.format(metadata[sweep]['elevation'])) # ekstrak data elevasi
        allElevation.append(elevation)   
        print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(i+1,elevation))
        
        azi=metadata[sweep]['az'] # mengekstrak data azimuth disetiap elevasi
        r=metadata[sweep]['r'] # mengekstrak data range disetiap elevasi
        sweep_data=data[sweep]['Z']['data'] # mengekstrak data radar
        
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
            radius, center, sweep_data.ravel(),
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
    voutCMAX[:] = cmaxData[:]
    
    # create variable ppi
    voutPPI = ncout.createVariable('ppi_dbz', np.dtype('double').char, ('lon', 'lat','lev'))
    voutPPI.long_name = 'ppi dBZ'
    voutPPI.units = 'dBZ'
    for i in range(len(allElevation)):
        ppiData=dataContainer[:,:,i]
        ppiData[ppiData<0]=np.nan
        ppiData[ppiData>100]=np.nan
        voutPPI[:,:,i]=ppiData
        del ppiData
    
    ncout.close()
    gc.collect();del gc.garbage[:] 
    print('Complete writing netcdf file')

def main():
    site='YOG'
    radarFile='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/YOG201711271250.mvol'
    ncpath='D:/project_webprogramming/wxradarexplore/radarDataConversion/nc'
    try:os.makedirs(ncpath)
    except:pass
    lonGrid,latGrid,timeEnd,allElevation,dataContainer=extractRadarData(radarFile)
    writeNetcdf(ncpath,site,timeEnd,lonGrid,latGrid,dataContainer,allElevation)

main()
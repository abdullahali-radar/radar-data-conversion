# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:14:59 2020

@author: Weather Radar Team
"""

from datetime import datetime
import wradlib as wrl
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, math, geojsoncontour, geojson, gdal, subprocess, json
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

def getUniqueID(i):
    switcher = {
        5:1,
        10:2,
        15:3,
        20:4,
        25:5,
        30:6,
        35:7,
        40:8,
        45:9,
        50:10,
        55:11,
        60:12,
        65:13
    }
    return switcher.get(i,np.nan)

def extractGEMA(radarFile):
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
    
    return dataContainer,lonMesh,latMesh,timeEnd

def writeGeoJSON(dataContainer,lonMesh,latMesh,fileGeoJSON):
    print('\nWriting geojson : {}'.format(fileGeoJSON))
    cmaxData=np.nanmax(dataContainer[:,:,:],axis=2)
    cmaxData[cmaxData<0]=np.nan;cmaxData[cmaxData>100]=np.nan
    clevsZ = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
    colors=['#07FEF6','#0096FF','#0002FE','#01FE03','#00C703','#009902','#FFFE00','#FFC801','#FF7707','#FB0103','#C90002','#980001','#FF00FF','#9800FE']
    contourf=plt.contourf(lonMesh,latMesh,cmaxData,clevsZ,colors=colors,alpha=0.5)
    geojson=geojsoncontour.contourf_to_geojson(
                    contourf=contourf,
                    min_angle_deg=3.0,
                    ndigits=3,
                    stroke_width=0.0,
                    fill_opacity=0.1)
    
    # Add attribute to JSON file
    dataDict=json.loads(geojson)
    for i in range(len(dataDict['features'])):
        properties=dataDict['features'][i]['properties']
        valueInt=int(float(properties['title'][0:4]))
        properties['value']=valueInt
        properties['id']=getUniqueID(valueInt)
        
    with open(fileGeoJSON,'w') as fp:
        json.dump(dataDict,fp)
    plt.close()
    print('Finished writing geojson file')
    
def writeShapefile(fileGeoJSON,fileShapefile):
    print('\nWriting shapefile : {}'.format(fileShapefile))
    f=open(fileGeoJSON)
    data=json.load(f)
    f.close()
    with open('data.geojson', 'w') as f:
        geojson.dump(data, f)
        
    args = ['ogr2ogr', '-f', 'ESRI Shapefile', fileShapefile, 'data.geojson']
    subprocess.Popen(args)
    print('Finished writing shapefile')

def main():
    radarFile='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/2020062106000600dBZ.vol'
    site='SBY'
    dataContainer,lonMesh,latMesh,timeEnd=extractGEMA(radarFile)
    path='D:/project_webprogramming/wxradarexplore/radarDataConversion'
    fileGeoJSON='{}/{}{}.json'.format(path,site,timeEnd.strftime("%Y%m%d%H%M"))
    fileShapefile='{}/{}{}.shp'.format(path,site,timeEnd.strftime("%Y%m%d%H%M"))
    writeGeoJSON(dataContainer,lonMesh,latMesh,fileGeoJSON)
    writeShapefile(fileGeoJSON,fileShapefile)

main()
    


    
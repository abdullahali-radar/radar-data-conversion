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

def extractEEC(radarFile):
    f = wrl.util.get_wradlib_data_file(radarFile)
    raw = wrl.io.read_generic_netcdf(f)
    
    radarLon = float(raw['variables']['longitude']['data'])
    radarLat = float(raw['variables']['latitude']['data'])
    radarAlt = float(raw['variables']['altitude']['data'])
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
      
    # ekstrak waktu radar (end of observation)
    try:timeEnd=datetime.strptime(str(raw['variables']['time_coverage_end']['data']),"%Y-%m-%dT%H:%M:%SZ")
    except:timeEnd=datetime.strptime(str(raw['variables']['time_coverage_end']['data']),"%Y-%m-%dT%H:%M:%S.%fZ") 
    
    # define flag option untuk melihat apakah gates vary atau tidak
    sweep_start_idx = raw['variables']['sweep_start_ray_index']['data']
    sweep_end_idx = raw['variables']['sweep_end_ray_index']['data']
    try:
        if raw['gates_vary']=='true':
            ray_n_gates=raw['variables']['ray_n_gates']['data']
            ray_start_index=raw['variables']['ray_start_index']['data']
            flag='true'
        elif raw['gates_vary']=='false':
            flag='false'
    except :
        if raw['n_gates_vary']=='true':
            ray_n_gates=raw['variables']['ray_n_gates']['data']
            ray_start_index=raw['variables']['ray_start_index']['data']
            flag='true'
        elif raw['n_gates_vary']=='false':
            flag='false'
    
    allElevation=[]
    nElevation=np.size(raw['variables']['fixed_angle']['data'])
    for i in range(nElevation):
        elevation=float('{0:.1f}'.format(raw['variables']['fixed_angle']['data'][i]))
        allElevation.append(elevation) 
        print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(i+1,elevation))
        
        # ekstrak azimuth data
        azi = raw['variables']['azimuth']['data'][sweep_start_idx[i]:sweep_end_idx[i]]   
        
        # ekstrak range data dan radar (dBZ) data berdasarkan nilai flag
        r_all = raw['variables']['range']['data'] 
        if flag == 'false':
            data = raw['variables']['DBZH']['data'][sweep_start_idx[i]:sweep_end_idx[i], :]
            r = r_all    
        else:              
            data = np.array([])
            n_azi = sweep_end_idx[i]-sweep_start_idx[i]        
            try:
                for ll in range(sweep_start_idx[i],sweep_end_idx[i]):
                    data = np.append(data,raw['variables']['DBZH']['data'][ray_start_index[ll]:ray_start_index[ll+1]])
                data = data.reshape((n_azi,ray_n_gates[sweep_start_idx[i]]))
            except:
                pass
            r = r_all[0:ray_n_gates[sweep_start_idx[i]]]
        
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
    radarFile='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/1013SOR-20200620-122001-PPIVol.nc'
    site='SOR'
    dataContainer,lonMesh,latMesh,timeEnd=extractEEC(radarFile)
    path='D:/project_webprogramming/wxradarexplore/radarDataConversion'
    fileGeoJSON='{}/{}{}.json'.format(path,site,timeEnd.strftime("%Y%m%d%H%M"))
    fileShapefile='{}/{}{}.shp'.format(path,site,timeEnd.strftime("%Y%m%d%H%M"))
    writeGeoJSON(dataContainer,lonMesh,latMesh,fileGeoJSON)
    writeShapefile(fileGeoJSON,fileShapefile)

main()
    


    
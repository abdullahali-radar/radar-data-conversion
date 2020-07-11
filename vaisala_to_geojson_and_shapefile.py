# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:14:59 2020

@author: Weather Radar Team
"""

from datetime import datetime
import wradlib as wrl
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, math, geojsoncontour, geojson, gdal, subprocess, json, glob
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

def searchFile(path,time,scanName):
    searchTime=time.strftime("%Y%m%d%H%M")[2:-1] 
    results = glob.glob('{}/*{}*'.format(path,searchTime))
    radarFiles=[]
    sweepNumbers=[]
    for file in results:
        f = wrl.util.get_wradlib_data_file(file)
        raw=wrl.io.read_iris(f)
        ppiVolType=raw['product_hdr']['product_configuration']['product_name']
        if str(ppiVolType)==scanName:
            sweepNumber=raw['product_hdr']['product_configuration']['product_specific_info']['sweep_number']
            radarFiles.append(file)
            sweepNumbers.append(sweepNumber)
    return radarFiles,sweepNumbers

def extractVAISALA(radarFiles,sweepNumbers,time,scanName,moment):
    f = wrl.util.get_wradlib_data_file(radarFiles[0])
    raw=wrl.io.read_iris(f)
    
    # ekstrak lokasi radar
    radarLon=float(raw['product_hdr']['product_end']['longitude'])
    radarLat=float(raw['product_hdr']['product_end']['latitude'])
    radarAlt=float(raw['product_hdr']['product_end']['ground_height'])
    radarLat -= 360 # radarLat=radarLat-360
    sitecoords=(radarLon,radarLat,radarAlt)
    
    # mempersiapkan container data
    res=250. # resolusi data yang diinginkan dalam meter
    resCoords=res/111229. # resolusi data dalam derajat
    rmax=250000./111229. # range maksimum
    lonMax,lonMin=radarLon+(rmax),radarLon-(rmax) 
    latMax,latMin=radarLat+(rmax),radarLat-(rmax)
    nGrid=int(np.floor((lonMax-lonMin)/resCoords)) # jumlah grid
    lonGrid=np.linspace(lonMin,lonMax,nGrid) # grid longitude
    latGrid=np.linspace(latMin,latMax,nGrid) # grid latitude            
    dataContainer = np.zeros((len(lonGrid),len(latGrid))) # penampung data
    allElevation=[]
    
    for file,sweep in zip(radarFiles,sweepNumbers):
        f = wrl.util.get_wradlib_data_file(file)
        raw=wrl.io.read_iris(f)
        timeEnd=raw['product_hdr']['product_end']['ingest_time']
        
        # ekstrak azimuth data
        missing_ray = None
        x = raw['data'][sweep]['sweep_data'][moment]
        az_start = x['azi_start'].copy()
        az_stop = x['azi_stop'].copy()
        ixmissing = np.array([], dtype="i4")
        if missing_ray is not None:
            ismissing1 = (az_start == missing_ray)
            ismissing2 = (az_stop == missing_ray)
            ismissing = (ismissing1 & ismissing2)
            ixmissing = np.where(ismissing)[0]
        if len(ixmissing) > 0:
            # beamwidth = data["ingest_header"]["task_configuration"]
            # ["task_misc_info"]["horizontal_beam_width"]
            nrays = raw["nrays"]
            # Interpolate az_start
            f = interpolate.interp1d(np.arange(nrays)[~ismissing],
                                 az_start[~ismissing])
            az_start[ixmissing] = f(np.arange(nrays)[ismissing])
            # Interpolate az_start
            f = interpolate.interp1d(np.arange(nrays)[~ismissing],
                                 az_stop[~ismissing])
            az_stop[ixmissing] = f(np.arange(nrays)[ismissing])
        az_stop[az_stop < az_start] += 360.
        az = (az_start + az_stop) / 2.
        az[az > 360.] -= 360.
        az_start[az_start > 360.] -= 360.
        az_stop[az_stop > 360.] -= 360.
        rollby = -np.argmin(az)
        az = np.roll(az, rollby, axis=0)
        az_start = np.roll(az_start, rollby, axis=0)
        az_stop = np.roll(az_stop, rollby, axis=0)
        assert np.all(np.diff(az) > 0), "List of azimuth angles " \
                                    "is not strictly increasing."
    
        # ekstrak range data
        range_info = raw['ingest_header']['task_configuration']['task_range_info']
        first_bin = range_info['range_first_bin']
        # last_bin = range_info['range_last_bin']
        range_step = range_info['step_output_bins']
        # nbins = data["nbins"]
        # We assume that range_info['range_first_bin'] specifies
        #    the midpoint of the first bin
        # If, however, range_info['range_first_bin'] is zero,
        #    we have to treat it differently
        # ATTENTION: The resulting ranges are not fully consistent
        #    with range_info['range_last_bin']
        if first_bin > 0.:
            r = np.arange(raw["nbins"]) * range_step + first_bin
        else:
            r = np.arange(raw["nbins"]) * range_step + range_step/2.
        # divide by 1e2 to get from cm to m according to spec
        r = r / 1e2
        
        # ekstrak elevation data
        elevation=float('{0:.1f}'.format(raw['data'][sweep]['ingest_data_hdrs']['DB_DBZ']["fixed_angle"]))
        allElevation.append(elevation)
        print('Extracting radar data : SWEEP-{0} at Elevation Angle {1:.1f} deg ...'.format(sweep,elevation))
        
        # ekstrak radar data
        data = raw['data'][sweep]['sweep_data'][moment]['data']
        
        # transformasi dari koordinat bola ke koordinat kartesian
        rangeMesh, azimuthMesh =np.meshgrid(r,az) # meshgrid azimuth dan range
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
    site='AMQ'
    pathVAISALA='D:/project_webprogramming/wxradarexplore/radarDataExtraction/data/AMQ'
    time=datetime(2020,6,20,12,10)
    scanName='RAW_PPIVOLA '
    moment='DB_DBZ'
    radarFiles,sweepNumbers=searchFile(pathVAISALA,time,scanName)
    dataContainer,lonMesh,latMesh,timeEnd=extractVAISALA(radarFiles,sweepNumbers,time,scanName,moment)
    
    path='D:/project_webprogramming/wxradarexplore/radarDataConversion'
    fileGeoJSON='{}/{}{}.json'.format(path,site,timeEnd.strftime("%Y%m%d%H%M"))
    fileShapefile='{}/{}{}.shp'.format(path,site,timeEnd.strftime("%Y%m%d%H%M"))
    writeGeoJSON(dataContainer,lonMesh,latMesh,fileGeoJSON)
    writeShapefile(fileGeoJSON,fileShapefile)

main()
    


    
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:47:02 2020

@author: Weather Radar Team
"""

from datetime import datetime
import os, sys

eecFolder='D:/project_webprogramming/wxradarexplore/radarDataConversion/20200620'
gemaFolder='D:/Rainbow5.40/rainbow/rawdata'
rb5bin='D:/Rainbow5.40/rainbow/bin'
sesnsorID='Ambon'
siteID='AMQ'
scanning='RAW_PPIVOL_A.vol'
for file in os.listdir(eecFolder):
    print(file)
    time=datetime(int(file[5:9]),int(file[9:11]),int(file[11:13]),int(file[14:16]),int(file[16:18]))
    infile='{}/{}'.format(eecFolder,file)
    outpath='{}/{}/{}/{}'.format(gemaFolder,siteID,scanning,time.strftime("%Y-%m-%d"))
    try:os.makedirs(outpath)
    except:pass
    outfiledbz='{}/{}'.format(outpath,time.strftime("%Y%m%d%H%M0000dBZ.vol"))
    outfilev='{}/{}'.format(outpath,time.strftime("%Y%m%d%H%M0000V.vol"))
    outfilew='{}/{}'.format(outpath,time.strftime("%Y%m%d%H%M0000W.vol"))
    
    if sys.platform=='linux':
        command1='cd {};./RainH5ToRb5 --infile={} --type=volume'.format(rb5bin,infile)
    elif sys.platform=='win32':
        command1='{}/RainH5ToRb5 --infile={} --type=volume'.format(rb5bin,infile)
    
    commanddbz='--rbdatatype=dBZ --outfile={} --sensorid={} --sensorname={}'.format(outfiledbz,siteID,sesnsorID)
    commandv  ='--rbdatatype=dBZ --outfile={} --sensorid={} --sensorname={}'.format(outfilev,siteID,sesnsorID)
    commandw  ='--rbdatatype=dBZ --outfile={} --sensorid={} --sensorname={}'.format(outfilew,siteID,sesnsorID)
    
    fullcommanddbz='{} {}'.format(command1,commanddbz)
    fullcommandv='{} {}'.format(command1,commandv)
    fullcommandw='{} {}'.format(command1,commandw)
    
    p = os.popen(fullcommanddbz)
    print(p.read())
    p = os.popen(fullcommandv)
    print(p.read())
    p = os.popen(fullcommandw)
    print(p.read())
    
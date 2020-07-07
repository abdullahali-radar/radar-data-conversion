# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:42:30 2020

@author: Weather Radar Team
"""

import h5py, os, glob
import wradlib as wrl
from datetime import datetime
import numpy as np


pathBARONraw='D:/project_webprogramming/wxradarexplore/radarDataConversion/YOG'
pathBARONodim='D:/project_webprogramming/wxradarexplore/radarDataConversion/YOG_odim'
patternOdom='odim.{}'

for file in os.listdir(pathBARONraw):
    fileBARONraw='{}/{}'.format(pathBARONraw,file)
    f = wrl.util.get_wradlib_data_file(fileBARONraw)
    data, metadata = wrl.io.read_gamic_hdf5(f)
    timeStart=datetime.strptime(str(metadata['SCAN0']['Time']),"b'%Y-%m-%dT%H:%M:%S.%fZ'")
    results = glob.glob('{}/odim.{}*'.format(pathBARONodim,timeStart.strftime("%Y%m%d_%H%M")))  
    if results!=[]:
        fileBARONodim=results[0]
        print(fileBARONodim)
        f=h5py.File(fileBARONodim,'r+')
        nElevation=len(data)
        for i in range(nElevation):
            sweep='SCAN'+str(i)
            bin_count=metadata[sweep]['bin_count']
            bin_range=metadata[sweep]['bin_range']
            sweep_data=data[sweep]['Z']['data']
            
            h5group='dataset'+str(i+1)
            dbzdataodim=f[h5group]['data1']['data']
            
            where=f[h5group]['where']
            rscale=where.attrs['rscale']
            nbins=where.attrs['nbins']
            rmax=bin_count*bin_range
            rscale_new=rmax/nbins
            where.attrs.modify('rscale',float(rscale_new));
            where.attrs.__setitem__('rscale',np.float64(rscale_new));
            
            
            rscaleNew=where.attrs['rscale']
            nbinsNew=where.attrs['nbins']
            print('rmax       : {}'.format(rmax))
            print('odim       : rscale {} nbins {}'.format(rscale, nbins))
            print('odim new   : rscale {} nbins {}'.format(rscaleNew, nbinsNew))
            print('odim data  : {}'.format(dbzdataodim.shape))
            print('gamic      : rscale {} nbins {}'.format(bin_range, bin_count))
            print('gamic data : {}\n'.format(sweep_data.shape))
            
            
        f.close()
    else:
        print('Not found file')
    


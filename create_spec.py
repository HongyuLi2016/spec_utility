#!/usr/bin/env python
import numpy as np
import pyfits
from optparse import OptionParser
#import matplotlib.pyplot as plt
from read_DRP import read_DRP
import os
#from redshift import redshift
import spec_utility as su

if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='fname',default=None,help='filename')
  parser.add_option('-z', action='store',type='float' ,dest='z',default=0.0,help='redshift')
  (options, args) = parser.parse_args()
  lhy=read_DRP(options.fname)
  folder = '%s_spec'%lhy.PLATEIFU
  os.system('mkdir -p %s'%folder)
  lhy.deRedden()
  lhy.deRedshift(z=options.z)
  ff = open('%s/speclist.dat'%folder,'w')
  for i in range(lhy.good_spaxel.shape[0]):
    for j in range(lhy.good_spaxel.shape[1]):
      if lhy.good_spaxel[i,j]:
        wave = lhy.wave
        flux = lhy.flux[:,i,j]
        error = np.sqrt(1.0/lhy.ivar[:,i,j])
        good = (lhy.mask[:,i,j] == 0) * (flux > 0.0) * (error > 0.0)
        mask = np.zeros(flux.shape,dtype='int')
        error[~good] = 1.0
        mask[~good] = 2
        flag = mask
        su.out_put_spec(wave, flux, error, flag,  i, j, folder = folder)
        print >> ff, 'spec_{0:0>3d}_{1:0>3d}.dat'.format(i,j)
        pass
  ff.close()

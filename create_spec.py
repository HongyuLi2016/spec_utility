#!/usr/bin/env python
import numpy as np
import pyfits
from optparse import OptionParser
#import matplotlib.pyplot as plt
from read_DRP import read_DRP
import os
from redshift import redshift

def out_put_spec(wave,flux,error,flag,i,j,folder='spec'):
  with open('{0}/spec_{1:0>3d}_{2:0>3d}.dat'.format(folder,i,j),'w') as ff:
    #print >>ff, '# wavelenght    flux    error    flag'
    for i in range(len(wave)):
      print >>ff, '{0:>e}    {1:>e}    {2:>e}    {3:>d}'.format(wave[i],flux[i],error[i],flag[i])
  

if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='fname',default=None,help='filename')
  (options, args) = parser.parse_args()
  lhy=read_DRP(options.fname)
  folder = '%s_spec'%lhy.PLATEIFU
  os.system('mkdir -p %s'%folder)
  nflux = lhy.deredden()
  ff = open('%s/speclist.dat'%folder,'w')
  for i in range(lhy.good_spaxel.shape[0]):
    for j in range(lhy.good_spaxel.shape[1]):
      if lhy.good_spaxel[i,j]:
        wave = lhy.wave
        flux = nflux[:,i,j]
        error = np.sqrt(1.0/lhy.ivar)[:,i,j]
        good = (lhy.mask[:,i,j] == 0) * (flux > 0.0) * (error > 0.0)
        mask = np.zeros(flux.shape,dtype='int')
        mask[~good] = 2
        flag = mask 
        out_put_spec(wave, flux, error, flag,  i, j, folder = folder)
        print >> ff, 'spec_{0:0>3d}_{1:0>3d}.dat'.format(i,j)
        pass
  ff.close()

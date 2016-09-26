#!/usr/bin/env python
import numpy as np
import pyfits
from optparse import OptionParser
import matplotlib.pyplot as plt
from PyAstronomy.pyasl import unred
import spec_utils as su
'''
read MaNGA DRP datacube, file format is discriped in https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/datamodel (6.5.3, 6.5.4)
'''

class read_DRP:
  def __init__(self,fname):
    hdulist=pyfits.open(fname)
    self.header = hdulist[0].header
    self.header1 = hdulist[1].header
    self.header2 = hdulist[5].header
    self.flux = hdulist[1].data
    self.ivar = hdulist[2].data
    self.mask = hdulist[3].data
    self.wave = hdulist[4].data
    self.specres = hdulist[5].data
    self.specresd = hdulist[6].data
    self.bsinfo = hdulist[7].data
    self.gimg = hdulist[8].data
    self.rimg = hdulist[9].data
    self.iimg = hdulist[10].data
    self.zimg = hdulist[11].data
    self.gpsf = hdulist[12].data
    self.rpsf = hdulist[13].data
    self.ipsf = hdulist[14].data
    self.zpsf = hdulist[15].data        
    #self.gcorrel = hdulist[16].data
    #self.rcorrel = hdulist[17].data
    #self.icorrel = hdulist[18].data
    #self.zcorrel = hdulist[19].data

    #create position information
    self.IFURA = self.header['IFURA']
    self.IFUDEC = self.header['IFUDEC']
    self.OBJRA = self.header['OBJRA']
    self.OBJDEC = self.header['OBJDEC']
    self.PLATEIFU = self.header['PLATEIFU']
    self.ebvgal = self.header['EBVGAL']
    shape = self.rimg.shape
    xtem = np.arange(shape[0],dtype=float)
    ytem = np.arange(shape[1],dtype=float) 

    xposition_tem, yposition_tem = np.meshgrid( xtem, ytem )
    xposition_tem -= self.header1['CRPIX1']
    yposition_tem -= self.header1['CRPIX2']
    self.xposition = self.header1['CD1_1'] * xposition_tem * 3600.
    self.yposition = self.header1['CD2_2'] * yposition_tem * 3600.

    good = self.mask == 0
    #print good
    #self.mask[~good] = 2
    self.good_spaxel = good.sum(axis=0) > 0 
        
    #print self.mask

  def deRedden(self, R_v=3.1):
    new_flux = np.zeros_like(self.flux)
    for i in range(self.flux.shape[1]):
      for j in range(self.flux.shape[2]):
        new_flux[:,i,j] = unred(self.wave, self.flux[:,i,j], self.ebvgal, R_V=R_v)
    self.flux = new_flux

  def deRedshift(self, z=0.03):
    new_flux = np.zeros_like(self.flux)
    new_error = np.zeros_like(self.flux)
    for i in range(self.flux.shape[1]):
      for j in range(self.flux.shape[2]):
        new_flux[:,i,j],new_error[:,i,j] = su.resample(self.wave/(1+z),self.wave,self.flux[:,i,j],error=self.ivar[:,i,j])
    self.flux = new_flux
    self.ivar = new_error


if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='fname',default='data/manga-8319-1902-LINCUBE.fits',help='filename')
  (options, args) = parser.parse_args()
  lhy=read_DRP(options.fname)
  #print lhy.mask[:,0,0]
  #plt.plot(lhy.wave, lhy.flux[:,18,18],'k')
  #lhy.deRedden()  
  #plt.plot(lhy.wave, lhy.flux[:,18,18],'r')
  #lhy.deRedshift(z=0.03)
  #plt.plot(lhy.wave, lhy.flux[:,18,18],'b')
  #plt.imshow((lhy.rimg))
  #plt.show()
  #print lhy.header

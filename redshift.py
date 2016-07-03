#!/usr/bin/env python
import numpy as np
import sys
import pyfits
from PySpectrograph import Spectrum
from PySpectrograph.Utilities.fit import interfit
import matplotlib.pyplot as plt

def ncor(x, y):
    """Calculate the normalized correlation of two arrays"""
    d=np.correlate(x,x)*np.correlate(y,y)
    if d<=0: return 0
    return np.correlate(x,y)/d**0.5

def xcor_redshift(spectra, template, sub=False, z1=0, z2=1, zstep=0.001):
    """Meaure the redshift of a spectra by cross correlating it 
       with a template

       returns an array of correlation values
    """
    zvalue=np.arange(z1,z2,zstep)
    cc_arr=np.zeros(len(zvalue))
    sflux=continuum_subtract(spectra)
    tflux=continuum_subtract(template)
    for i,z in enumerate(zvalue):
      nflux=np.interp(spectra.wavelength, template.wavelength*(1+z), tflux)
      cc_arr[i]=ncor(abs(sflux), abs(nflux))
    return  zvalue, cc_arr

def continuum_subtract(spec, function='polynomial', order=7):
    """Fit a function to a spectra and subtract the continuum"""
    wc=interfit(spec.wavelength, spec.flux, function=function, order=order)
    wc.interfit()
 
    return spec.flux-wc(spec.wavelength)


class redshift:
  def __init__(self, wave, flux, mask=None, z1=0.0, z2=0.5, zstep=0.0001, plot=False,savepath='./'):
    template_base_dir='/Users/lhy/workspace/zSALT/zsalt/template/'
    template_list=['spDR2-023.fit','spDR2-024.fit','spDR2-025.fit','spDR2-026.fit','spDR2-027.fit','spDR2-028.fit']
    self.z_all = np.zeros(len(template_list))
    #template_list=['spDR2-024.fit']
    if plot:
      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
    for i in range(len(template_list)):
      hdu=pyfits.open('%s/%s'%(template_base_dir, template_list[i]))
      farr=hdu[0].data[0]
      xarr=np.arange(len(farr))
      warr=10**(hdu[0].header['CRVAL1']+hdu[0].header['CD1_1']*(xarr+1))
      template=Spectrum.Spectrum(warr, farr, stype='continuum')
      

      spec=Spectrum.Spectrum(wave, flux, stype='continuum')
      #plt.plot(continuum_subtract(spec))
      #plt.plot(spec.flux)
      #plt.plot(continuum_subtract(template))
      #plt.plot(hdu[0].data[1])
      z_arr, cc_arr=xcor_redshift(spec, template, z1=z1, z2=z2, zstep=0.0001)
      i_max = cc_arr == np.nanmax(cc_arr)
      self.z_all[i] = z_arr[i_max]
      if plot:
        ax.plot(z_arr,cc_arr)
 
    z_final = np.nanpercentile(self.z_all,[20,50,80])
    self.z = z_final[1]
    self.z_err = (z_final[2]-z_final[0])*0.5
    
    if plot:
      ax.set_xlim([z1,z2])
      fig.savefig('%s/xcor.png'%savepath)    
   

if __name__=='__main__':
  wave = np.loadtxt('data/sepc_013_022.dat',usecols=[0])
  flux = np.loadtxt('data/sepc_013_022.dat',usecols=[1])
  #plt.plot(wave,flux)
  #plt.show()
  lhy=redshift(wave,flux,plot=True,z1=0.0,z2=0.2,savepath='data/')
  print lhy.z_all
  print 'z=%.3f z_err=%.4f'%(lhy.z,lhy.z_err)

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import re

def findline(ff,line,imax=10000,p=False):
  i = 0
  while True:
    junk = ff.readline()
    if p:
      print junk
    i+=1
    if junk == line:
      break      
    if i >  imax:
      #print 'Warnning - Maxmum lines read!'
      break


class read:
  '''
  This is a class used for reading starlight output files
  Parameters:
    chi2dof: chi2 per degree of freedom
    adev: absolute deviation of the synthesis spectrum
    sum_of_x: summition of the weight of all the templates
    Flux_tot: Total flux in input spectrum unit
    Mini_tot: used to calculate initial total mass
    Mcor_tot: used to calculate total mass
    v0_min: best-fitting velocity
    vd_min: best-fitting velocity dispersion
    AV_min: best-fiting dust extinction
    ID: template spectrum ID
    x_j: luminosity fraction of the jth template
    Mini_j: initial mass fraction of the jth template
    Mcor_j: current mass fraction of the jth template
    Age_j: Age of the jth template
    Z_j: metallicity of the jth tempalte
    LM_j: Tempalte spectrum value at the nomolazition wavelength
    wave: wavelenght array
    obs: observed spectrum
    syn: synthesis spectrum
    weight: weight for each pixel used in the fitting
    good: good pixels used in the fitting
  '''
  def __init__(self,fname):
    with open(fname,'r') as ff:
      findline(ff,'## Normalization info\n')
      findline(ff,'## S/N\n')

      # read in synthesis results, line by line
      findline(ff,'## Synthesis Results - Best model ##\n')
      junk = ff.readline()
      tem = ff.readline()
      pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
      self.chi2pdof = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([+-]?[0-9.]+)')
      self.adev = float(pat.search(tem).group(1))      
      junk = ff.readline()
      tem = ff.readline()
      pat = re.compile(r'([+-]?[0-9.]+)')
      self.sum_of_x = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
      self.Flux_tot = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
      self.Mini_tot = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
      self.Mcor_tot = float(pat.search(tem).group(1))
      junk = ff.readline()
      tem = ff.readline()
      pat = re.compile(r'([+-]?[0-9.]+)')
      self.v0_min = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([+-]?[0-9.]+)')
      self.vd_min = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([+-]?[0-9.]+)')
      self.AV_min = float(pat.search(tem).group(1))
      tem = ff.readline()
      pat = re.compile(r'([+-]?[0-9.]+)')
      self.YAV_min = float(pat.search(tem).group(1))
      # read in star formation history 
      findline(ff,'# j     x_j(%)      Mini_j(%)     Mcor_j(%)     age_j(yr)     Z_j      (L/M)_j    YAV?  Mstars   component_j        a/Fe...       SSP_chi2r SSP_adev(%)   SSP_AV   SSP_x(%)\n')
      ID = []
      x_j = []
      Mini_j = []
      Mcor_j = []
      Age_j = []
      Z_j = []
      LM_j = []
      YAV = []
      Mstars = []
      component_j = []
      alpha_Fe = []
      SSP_chi2r = []
      SSP_adec = []
      SSP_AV = []
      SSP_x = []
      while True:
        tem = ff.readline()
        if tem == '' or tem == '\n':
          break
        pat = re.compile(r'(\d+) +([+-]?[0-9.]+) +([-+]?\d+\.\d*?[eE][-+]?\d+) +([-+]?\d+\.\d*?[eE][-+]?\d+) +([-+]?\d+\.\d*?[eE][-+]?\d+) +([+-]?[0-9.]+) +([-+]?\d+\.\d*?[eE][-+]?\d+) +(\d+) +([+-]?[0-9.]+) +(\S+) +([+-]?[0-9.]+) +([-+]?\d+\.\d*?[eE][-+]?\d+) +([+-]?[0-9.]+) +([+-]?[0-9.]+) +([+-]?[0-9.]+)')
        ID.append(int(pat.search(tem).group(1)))
        x_j.append(float(pat.search(tem).group(2)))
        Mini_j.append(float(pat.search(tem).group(3)))
        Mcor_j.append(float(pat.search(tem).group(4)))
        Age_j.append(float(pat.search(tem).group(5)))
        Z_j.append(float(pat.search(tem).group(6)))
        LM_j.append(float(pat.search(tem).group(7)))
        YAV.append(int(pat.search(tem).group(8)))
        Mstars.append(float(pat.search(tem).group(9)))
        component_j.append((pat.search(tem).group(10)))
        alpha_Fe.append(float(pat.search(tem).group(11)))
        SSP_chi2r.append(float(pat.search(tem).group(12)))
        SSP_adec.append(float(pat.search(tem).group(13)))
        SSP_AV.append(float(pat.search(tem).group(14)))
        SSP_x.append(float(pat.search(tem).group(15)))

      self.ID = np.array(ID)
      self.x_j = np.array(x_j)
      self.Mini_j = np.array(Mini_j)
      self.Mcor_j = np.array(Mcor_j)
      self.Age_j = np.array(Age_j)
      self.Z_j = np.array(Z_j)
      self.LM_j = np.array(LM_j)
      self.YAV = np.array(YAV)
      self.Mstars = np.array(Mstars)
      self.component_j = np.array(component_j)
      self.alpha_Fe = np.array(alpha_Fe)
      self.SSP_chi2r = np.array(SSP_chi2r)
      self.SSP_adec = np.array(SSP_adec)
      self.SSP_AV = np.array(SSP_AV)
      self.SSP_x = np.array(SSP_x)
              
      # read in synthetic spectrum 
      findline(ff,'## Synthetic spectrum (Best Model) ##l_obs f_obs f_syn wei\n')
      junk = ff.readline()
      wave = []
      obs = []
      syn = []
      weight = []
      pat = re.compile(r'([+-]?[0-9.]+) +([+-]?[0-9.]+) +([+-]?[0-9.]+) +([+-]?[0-9.]+)')
      while True:
        tem = ff.readline()
        if tem == '' or tem == '\n':
          break
        wave.append(float(pat.search(tem).group(1)))
        obs.append(float(pat.search(tem).group(2)))
        syn.append(float(pat.search(tem).group(3)))
        weight.append(float(pat.search(tem).group(4)))
    self.wave = np.array(wave)
    self.obs = np.array(obs)
    self.syn = np.array(syn)
    self.weight = np.array(weight)
    self.good = self.weight > 0.0
    #plt.plot(self.wave,self.obs,'r')
    #plt.plot(self.wave,self.syn,'k')
    #plt.show()
 
if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='fname',default=None,help='file name')
  (options, args) = parser.parse_args()
  lhy=read(options.fname)

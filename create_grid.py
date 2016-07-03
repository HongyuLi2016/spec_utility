#!/usr/bin/env python
import numpy as np
from optparse import OptionParser
import os
from read_rc import read_rc
if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='folder',default=None,help='folder name')
  #parser.add_option('-b', action='store_true',dest='base',default=None,help='use 45 base file')
  #parser.add_option('-o', action='store',type='string' ,dest='opath',default=None,help='output dir')
  #parser.add_option('-i', action='store',type='string' ,dest='ipath',default=None,help='input dir')
  base_dir = '/Users/lhy/workspace/starlight/STARLIGHTv04/BasesDir/'
  mask_dir = '/Users/lhy/workspace/starlight/STARLIGHTv04/'
  config_file = 'StCv04.C11.config'
  base_file = 'Base.BC03.N'
  mask_file = 'Masks.EmLines.SDSS.gm'
  extinction = 'CCM' # CCM OR CAL
  v0_start = 0.0
  vd_start = 150.0
  
  (options, args) = parser.parse_args()
  flist=read_rc('%s/speclist.dat'%options.folder)
  os.system('mkdir -p %s/out'%options.folder)
  Nspec = len(flist)
  with open('%s/grid_%s.in'%(options.folder,options.folder),'w') as ff:
    print >>ff, '{0:<70d}    [Number of fits to run]'.format(Nspec)
    print >>ff, '{0:<70s}    [base_dir]'.format(base_dir)
    print >>ff, '{0:<70s}    [obs_dir]'.format(options.folder+'/')
    print >>ff, '{0:<70s}    [mask_dir]'.format(mask_dir)
    print >>ff, '{0:<70s}    [out_dir]'.format(options.folder+'/out/')
    print >>ff, '{0:<70d}    [your phone number]'.format(-2007200)
    print >>ff, '{0:<70.1f}    [llow_SN]   lower-lambda of S/N window'.format(4730.0)
    print >>ff, '{0:<70.1f}    [lupp_SN]   upper-lambda of S/N window'.format(4780.0)
    print >>ff, '{0:<70.1f}    [Olsyn_ini]   lower-lambda for fit'.format(3400.0)
    print >>ff, '{0:<70.1f}    [Olsyn_fni]   upper-lambda for fit'.format(9000.0)
    print >>ff, '{0:<70d}    [Odlsyn]    delta-lambda for fit'.format(1)   
    print >>ff, '{0:<70d}    [fscale_chi2] fudge-factor for chi2'.format(1)
    print >>ff, '{0:<70s}    [FIT/FXK] Fit or Fix kinematics'.format('FIT')
    print >>ff, '{0:<70d}    [IsErrSpecAvailable]  1/0 = Yes/No'.format(1)
    print >>ff, '{0:<70d}    [IsFlagSpecAvailable] 1/0 = Yes/No'.format(1)
    for i in range(Nspec):
      outname ='{}_out.dat'.format(flist[i][0:-4])
      print >>ff, '{0}    {1}    {2}    {3}    {4}    {5:.1f}    {6:.1f}    {7}'.format(flist[i],\
              config_file,base_file,mask_file,extinction,v0_start,vd_start,outname)

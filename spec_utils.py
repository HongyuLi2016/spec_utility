#!/usr/bin/evn python
import numpy as np
from scipy.interpolate import interp1d

def resample(wave,newWave,flux,error=None,interpKind=None,fill_value=0.0):
  if interpKind is None:
    interpKind = 'linear'
  fSpec = interp1d(wave,flux,kind = interpKind,fill_value=fill_value,bounds_error=False)
  newFlux = fSpec(newWave)
  if error is not None:
    fError = interp1d(wave,error,kind = interpKind,fill_value=fill_value,bounds_error=False)
    newError = fError(newWave)
    return newFlux,newError
  else:
    return newFlux

def bit_mask(mask_array,bit):
  '''
  For dap MPL5 velocity map files
  30: DONOTUSE        Do not use this spaxel for science
  19: MULTICOMP       Multi-component velocity features present
  9: NOCORRECTION    Appropriate correction not available
  8: NEARBOUND       Fitted value is too near an imposed boundary; see TRM
  7: FITFAILED       Attempted fit for property failed
  6: MATHERROR       Mathematical error in computing value
  5: UNRELIABLE      Value is deemed unreliable; see TRM for definition
  4: NOVALUE         Spaxel was not fit because it did not meet selection criteria
  3: FORESTAR        Foreground star
  2: DEADFIBER       Major contributing fiber is dead
  1: LOWCOV          Low coverage in this spaxel
  0: NOCOV           No coverage in this spaxel
  '''
  return mask_array & 2**bit > 0

def out_put_spec(wave,flux,error,flag,i,j,folder='spec'):
  with open('{0}/spec_{1:0>3d}_{2:0>3d}.dat'.format(folder,i,j),'w') as ff:
    #print >>ff, '# wavelenght    flux    error    flag'
    for i in range(len(wave)):
      print >>ff, '{0:>e}    {1:>e}    {2:>e}    {3:>d}'.format(wave[i],flux[i],error[i],flag[i])

def grid(flist,oname,path,config_file = 'StCv04.C11.config',base_file = 'Base.BC03.N',\
         mask_file ='Masks.EmLines.SDSS.gm',extinction = 'CAL',llow_SN = 4730.0,
         lupp_SN = 4780.0, Olsyn_ini=3400.0, Olsyn_fni=9000.0,v0_start=0.0,\
         vd_start = 150.0, base_dir = '/home/lhy/STARLIGHTv04/BasesDir/'):
  mask_dir = '/home/lhy/STARLIGHTv04/'

  Nspec = len(flist)
  with open('%s/%s'%(path,oname),'w') as ff:
    print >>ff, '{0:<70d}    [Number of fits to run]'.format(Nspec)
    print >>ff, '{0:<70s}    [base_dir]'.format(base_dir)
    print >>ff, '{0:<70s}    [obs_dir]'.format(path+'/')
    print >>ff, '{0:<70s}    [mask_dir]'.format(mask_dir)
    print >>ff, '{0:<70s}    [out_dir]'.format(path+'/out/')
    print >>ff, '{0:<70d}    [your phone number]'.format(-2007200)
    print >>ff, '{0:<70.1f}    [llow_SN]   lower-lambda of S/N window'.format(llow_SN)
    print >>ff, '{0:<70.1f}    [lupp_SN]   upper-lambda of S/N window'.format(lupp_SN)
    print >>ff, '{0:<70.1f}    [Olsyn_ini]   lower-lambda for fit'.format(Olsyn_ini)
    print >>ff, '{0:<70.1f}    [Olsyn_fni]   upper-lambda for fit'.format(Olsyn_fni)
    print >>ff, '{0:<70d}    [Odlsyn]    delta-lambda for fit'.format(1)
    print >>ff, '{0:<70d}    [fscale_chi2] fudge-factor for chi2'.format(1)
    print >>ff, '{0:<70s}    [FIT/FXK] Fit or Fix kinematics'.format('FIT')
    print >>ff, '{0:<70d}    [IsErrSpecAvailable]  1/0 = Yes/No'.format(1)
    print >>ff, '{0:<70d}    [IsFlagSpecAvailable] 1/0 = Yes/No'.format(1)
    for i in range(Nspec):
      outname ='{}_out.dat'.format(flist[i][0:-4])
      print >>ff, '{0}    {1}    {2}    {3}    {4}    {5:.1f}    {6:.1f}    {7}'.format(flist[i],\
              config_file,base_file,mask_file,extinction,v0_start,vd_start,outname)


#!/usr/bin/env python
import numpy as np
from optparse import OptionParser
import os
from read_rc import read_rc
import spec_utils as su
if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='folder',default=None,help='folder name')
  parser.add_option('-s', action='store_false',dest='separate',default=True,help='Separate')
  (options, args) = parser.parse_args()
  outfolder = '%s/out'%options.folder
  gridfolder = '%s/grid'%options.folder
  specfolder = '%s/spec'%options.folder
  os.system('mkdir -p %s'%outfolder)
  os.system('mkdir -p %s'%gridfolder)
  flist = read_rc('%s/list.cat'%specfolder)
  if options.separate:
    for i in range(len(flist)):
      #su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4], gridfolder) #Chab
      #su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4], gridfolder, base_file='BaseSalp', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalp/') #Salp
      #su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4], gridfolder, base_file='BaseSalpGe', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalpGe/') #SalpGe
      #su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4], gridfolder, base_file='BaseMilesJunqiang', base_dir = '/home/lhy/STARLIGHTv04/BaseDirMiles/',
      #        Olsyn_ini=3622.0, Olsyn_fni=7409.0, mask_file='Masks.nomask') #
      #        MILES do not mask emls, use Junqiang template
      su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4], gridfolder, base_file='BaseMilesDefault', base_dir = '/home/lhy/STARLIGHTv04/BaseDirMiles/',
              Olsyn_ini=3622.0, Olsyn_fni=7409.0, mask_file='Masks.nomask') # MILES
  else:
    #su.grid(flist,'grid_spec_all.in'%options.folder, gridfolder) #Chab
    #su.grid(flist,'grid_spec_all.in', gridfolder, base_file='BaseSalp', base_dir='/home/lhy/STARLIGHTv04/BasesDirSalp/') #Salp
    #su.grid(flist,'grid_spec_all.in'%options.folder,gridfolder,base_file='BaseSalpGe', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalpGe/') #SalpGe
    su.grid([flist[i]],'grid_spec_all.in'%flist[i][0:-4],gridfolder,base_file='BaseMilesDefault', base_dir = '/home/lhy/STARLIGHTv04/BaseDirMiles/',
            Olsyn_ini=3622.0, Olsyn_fni=7409.0) # MILES

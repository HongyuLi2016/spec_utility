#!/usr/bin/env python
import numpy as np
from optparse import OptionParser
import os
import spec_utils as su
if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-s', action='store_false',dest='separate',default=True,help='Separate')
  (options, args) = parser.parse_args()
  if len(args) == 0:
      print 'Please provide a folder name'
      exit()
  outfolder = '%s/out'%args[0]
  gridfolder = '%s/grid'%args[0]
  specfolder = '%s/spec'%args[0]
  os.system('mkdir -p %s'%outfolder)
  os.system('mkdir -p %s'%gridfolder)
  flist = np.genfromtxt('%s/list.cat'%specfolder, dtype='40S')
  if options.separate:
    for i in range(len(flist)):
      su.grid([flist[i]], 'grid_%s.in' % flist[i][0:-4], gridfolder)  # Chab
      # su.grid([flist[i]], 'grid_%s.in' % flist[i][0:-4], gridfolder,
      #         base_file='BaseSalp',
      #         base_dir='/home/lhy/STARLIGHTv04/BasesDirSalp/',
      #         Olsyn_ini=3622.0, Olsyn_fni=9000.0)  # Salp
      # su.grid([flist[i]], 'grid_%s.in'%flist[i][0:-4], gridfolder,
      #         base_file='BaseSalpGe',
      #         base_dir='/home/lhy/STARLIGHTv04/BasesDirSalpGe/')  # SalpGe
      # su.grid([flist[i]], 'grid_%s.in'%flist[i][0:-4], gridfolder,
      #         base_file='BaseMilesDefault',
      #         base_dir='/home/lhy/STARLIGHTv04/BaseDirMiles/',
      #         Olsyn_ini=3622.0, Olsyn_fni=7409.0)  # MILES
  else:
    su.grid(flist, 'grid_spec_all.in', gridfolder)  # Chab
    # su.grid(flist, 'grid_spec_all.in', gridfolder, base_file='BaseSalp',
    #         base_dir='/home/lhy/STARLIGHTv04/BasesDirSalp/',
    #         Olsyn_ini=3622.0, Olsyn_fni=9000.0, extinction='CCM')  # Salp
    # su.grid(flist, 'grid_spec_all.in', gridfolder, base_file='BaseSalpGe',
    #         base_dir='/home/lhy/STARLIGHTv04/BasesDirSalpGe/')  # SalpGe
    # su.grid(flist, 'grid_spec_all.in', gridfolder, base_file='BaseMilesDefault',
    #         base_dir='/home/lhy/STARLIGHTv04/BaseDirMiles/',
    #         Olsyn_ini=3622.0, Olsyn_fni=7409.0)  # MILES

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
  os.system('mkdir -p %s/out'%options.folder)
  flist = read_rc('%s/speclist.dat'%options.folder)
  if options.separate:
    for i in range(len(flist)):
      su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4],options.folder) #Chab
      #su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4],options.folder,base_file = 'BaseSalp', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalp/') #Salp
      su.grid([flist[i]],'grid_%s.in'%flist[i][0:-4],options.folder,base_file = 'BaseSalpGe', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalpGe/') #SalpGe
  else:
    su.grid(flist,'grid_%s.in'%options.folder,options.folder) #Chab
    #su.grid(flist,'grid_%s.in'%options.folder,options.folder,base_file = 'BaseSalp', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalp/') #Salp
    su.grid(flist,'grid_%s.in'%options.folder,options.folder,base_file = 'BaseSalpGe', base_dir = '/home/lhy/STARLIGHTv04/BasesDirSalpGe/') #SalpGe

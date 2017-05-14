#!/usr/bin/env python
'''
Convert starlgiht output file to python pickle format
'''
from sps_starlight_out import read
from optparse import OptionParser
import numpy as np
import os
PPXFPATH = os.environ.get('PPXFPATH')
if PPXFPATH is None:
    raise RuntimeError('Enviroment variable PPXFPAHT must be set')

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', action='store_true', dest='plot',
                      default=False, help='plot')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - folder name must be probided!'
        exit(1)
    flist = np.genfromtxt('{}/spec/list.cat'.format(args[0]), dtype='20S')
    for i in range(len(flist)):
        fname = flist[i][0:-4]+'_out.dat'
        try:
            lhy = read('{}/out/{}'.format(args[0], fname))
            if options.plot:
                lhy.plot(shape=(25, 6), fname='sps_{:04d}.png'.format(i),
                         filterPath='{}/data/SDSS_r_filter'.format(PPXFPATH),
                         outfolder='{}/figs_star'.format(args[0]))
            lhy.dump(shape=(25, 6), fname='sps_{:04d}.dat'.format(i),
                     filterPath='{}/data/SDSS_r_filter'.format(PPXFPATH),
                     outfolder='{}/dump_star'.format(args[0]))
        except:
            print 'Warning - dump {} fiald!'.format(fname)

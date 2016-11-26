#!/usr/bin/env python
from starlight_out import read
from optparse import OptionParser
import numpy as np


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', action='store_true', dest='plot', default=False, help='plot')
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
                         filterPath='/home/lhy/python/ppxf/data/SDSS_r_filter',
		         mlPath='/home/lhy/python/ppxf/'
                                'data/Base.BC03.SALP.MLR_r.S150',
 		         outfolder='{}/figs_star'.format(args[0]))
	    lhy.dump(shape=(25, 6), fname='sps_{:04d}.dat'.format(i),
		     filterPath='/home/lhy/python/ppxf/data/SDSS_r_filter',
                         mlPath='/home/lhy/python/ppxf/'
                                'data/Base.BC03.SALP.MLR_r.S150',
                         outfolder='{}/dump_star'.format(args[0]))
	except:
	    print 'Warning - dump {} fiald!'.format(fname)

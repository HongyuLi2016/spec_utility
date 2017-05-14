#!/usr/bin/env python
import numpy as np
import pyfits
import spec_utils as su
from optparse import OptionParser


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', action='store', type='string', dest='folder',
                      default='dump', help='folder')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - folder name must be probided!'
        exit(1)
    path = options.folder
    su.ssp_rebin_map('{}/{}/{}_sspMaps.fits'.format(args[0], path, args[0]),
                     binname='{}/spec/bin.dat'.format(args[0]),
                     outname='{}/{}/{}_ssp_rebinMaps.fits'.format(args[0], path, args[0]))

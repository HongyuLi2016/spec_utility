#!/usr/bin/env python
import numpy as np
import pyfits
import spec_utils as su
from optparse import OptionParser
from astropy.cosmology import Planck13
from astropy import units as u


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', action='store', type='string', dest='folder',
                      default='dump', help='folder')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - folder name must be probided!'
        exit(1)
    z = pyfits.open('{}/information.fits'.format(args[0]))[1].data.redshift[0]
    Ldist = Planck13.luminosity_distance(z)
    Ldist_cm = Ldist.to(u.cm).value
    ID, xNode, yNode = su.load_bin_file('{}/spec/node.dat'.format(args[0]))
    su.ssp_write_map(ID, xNode, yNode, outname='{}_sspMaps.fits'.format(args[0]),
                     path='{}/{}'.format(args[0], options.folder), Ldist_cm=Ldist_cm)

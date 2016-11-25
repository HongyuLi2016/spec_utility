#!/usr/bin/env python
# import numpy as np
# import spec_utils as su
from read_DRP import read_DRP
from optparse import OptionParser
import pyfits


if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - folder name must be probided!'
        exit(1)
    z = pyfits.open('{}/information.fits'
                    .format(args[0]))[1].data['redshift'][0]
    ID = args[0].split('_')[1]
    lhy = read_DRP('{}/manga-{}-LINCUBE.fits.gz'.format(args[0], ID))
    lhy.deRedden()
    lhy.deRedshift(z=z)
    sn = lhy.SNR()
    lhy.voronoiBin(outFolder='{}/spec'.format(args[0]))

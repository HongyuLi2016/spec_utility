#!/usr/bin/env python
'''
read DRP linecube, Voronoi bin spectum and output to a text file
'''
from sps_read_linDRP import read_DRP
from optparse import OptionParser
import pyfits


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', action='store', type='string', dest='path',
		      default=None, help='path to cubes')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - folder name must be probided!'
        exit(1)
    z = pyfits.open('{}/information.fits'
                    .format(args[0]))[1].data['redshift'][0]
    ID = args[0].split('_')[1]
    if options.path is None:
	path = args[0]
    else:
        path = options.path
    lhy = read_DRP('{}/manga-{}-LINCUBE.fits.gz'.format(path, ID))
    lhy.deRedden()
    lhy.deRedshift(z=z)
    sn = lhy.SNR()
    lhy.voronoiBin(outFolder='{}/spec'.format(args[0]), targetSN=30)

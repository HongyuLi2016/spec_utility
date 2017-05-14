#!/usr/bin/env python
import numpy as np
import spec_utils as su
from ppxf_sps import ppxf_sps
from optparse import OptionParser
import os
PPXFPATH = os.environ.get('PPXFPATH')
if PPXFPATH is None:
    raise RuntimeError('Enviroment variable PPXFPAHT must be set')
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', action='store_true', dest='plot', default=False, help='plot')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - folder name must be probided!'
        exit(1)
    flist = np.genfromtxt('{}/spec/list.cat'.format(args[0]), dtype='20S')
    wave, flux, error, goodbins =\
        su.load_txt_spec('{}/spec/{}'.format(args[0], flist[0]))
    vscale = np.log(wave[-1] / wave[0]) / len(wave) * su.c
    lhy = ppxf_sps(vscale)
    lhy.load_miles_all(2.51, path='{}/MILES_Padova00_un_1.30_fits'
                       .format(PPXFPATH))
    #lhy.select_templates(range(24, 50), range(1, 7))
    lhy.select_templates(list(range(1,50,2)), list(range(1, 7)))
    print 'Star running pPXF'
    print '--------------------------------------------------'
    for i in range(len(flist)):
        print 'Fitting spectrum {}'.format(flist[i])
        wave, flux, error, goodbins =\
            su.load_txt_spec('{}/spec/{}'.format(args[0], flist[i]))
        lhy.load_galaxy(wave, flux, error=error, good=goodbins)
        lhy.run(moments=2, mdegree=0)
        if options.plot:
            lhy.plot(fname='sps_{:04d}.png'.format(i),
                     filterPath='{}/data/SDSS_r_filter'.format(PPXFPATH),
                     outfolder='{}/figs'.format(args[0]))
        lhy.dump(fname='sps_{:04d}.dat'.format(i),
                 filterPath='{}/data/SDSS_r_filter'.format(PPXFPATH),
                 outfolder='{}/dump'.format(args[0]))

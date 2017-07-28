#!/usr/bin/env python
import numpy as np
import spec_utils as su
from sps_ppxf import ppxf_sps
from optparse import OptionParser
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
        print('Error - folder name must be probided!')
        exit(1)
    flist = np.genfromtxt('{}/spec/list.cat'.format(args[0]), dtype='20S')
    wave, flux, error, goodbins =\
        su.load_txt_spec('{}/spec/{}'.format(args[0], flist[0]))
    vscale = np.log(wave[-1] / wave[0]) / len(wave) * su.c
    lhy = ppxf_sps(vscale)
    lhy.load_miles_all(2.51, path='{}/MILES_Padova00_un_1.30_fits'
                       .format(PPXFPATH))
    # lhy.select_templates(range(24, 50), range(1, 7))
    lhy.select_templates(list(range(1, 50, 2)), list(range(1, 7)))
    # lhy.load_bc03(2.8, path='{}/BC03_Salp'.format(PPXFPATH))
    age = [20, 45, 55, 61, 67, 70, 78, 90, 104, 110, 116, 120, 125,
           130, 135, 138, 139, 150, 158, 166, 171, 181, 193, 201, 213]
    # age = [113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133,
    #        135, 137, 139, 143, 147, 152, 155, 159, 163, 169, 177,
    #        186, 197, 210]
    for i in range(len(age)):
        age[i] -= 1
    # lhy.select_templates(age, range(0, 6))
    print('Star running pPXF')
    print('--------------------------------------------------')
    for i in range(len(flist)):
        print('Fitting spectrum {}'.format(flist[i]))
        wave, flux, error, goodbins =\
            su.load_txt_spec('{}/spec/{}'.format(args[0], flist[i]))
        lhy.load_galaxy(wave, flux, error=error, good=goodbins,
                        # fit_range=[3322.0, 9000.0])
                        fit_range=[3541.0, 7409.0])
        lhy.run(moments=2, mdegree=0, clip=False, regul=1./0.004)
        if options.plot:
            lhy.plot(fname='sps_{:04d}.png'.format(i),
                     filterPath='{}/data/SDSS_r_filter'.format(PPXFPATH),
                     outfolder='{}/figs'.format(args[0]))
        lhy.dump(fname='sps_{:04d}.dat'.format(i),
                 filterPath='{}/data/SDSS_r_filter'.format(PPXFPATH),
                 outfolder='{}/dump'.format(args[0]))

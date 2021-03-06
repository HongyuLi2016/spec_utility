#!/usr/bin/env python
import numpy as np
import spec_utils as su
from optparse import OptionParser
import os
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-o', action='store', type='string', dest='outfolder',
                      default='figs', help='output folder')
    parser.add_option('-s', action='store_true', dest='save',
                      default=False, help='save as a png fig')
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print 'Error - file name must be probided!'
        exit(1)

    figname = args[0][0:-4]+'.png'
    data = su.ssp_load_data(args[0])
    wave = data['wave']
    obs = data['obs']
    syn = data['syn']
    goodpixels = data['goodpixels']
    Lweights = data['Lweights']
    Mweights = data['Mweights']
    ml_obs_r = data['ml_obs_r']
    ml_int_r = data['ml_int_r']
    Mage = data['Mage']
    MageLog = data['MageLog']
    Lage = data['Lage']
    LageLog = data['LageLog']
    MZ = data['MZ']
    MZLog = data['MZLog']
    LZ = data['LZ']
    LZLog = data['LZLog']
    ebv = data['ebv']
    logAge_grid = data['logAge_grid']
    metal_grid = data['metal_grid']
    gas = data.pop('gas', None)

    parameters = {'M*/L_int_r': ml_int_r, 'M*/L_obs_r': ml_obs_r,
                  'M<logAge>': MageLog, 'L<logAge>': LageLog,
                  'M<[Z/H]>': MZLog, '<L[Z/H]>': LZLog, 'E(b-v)': ebv}

    fig = su.plot_sps(wave, obs, syn, goodpixels, Mweights, Lweights,
                      np.unique(logAge_grid), np.unique(metal_grid),
                      parameters=parameters, gas=gas)
    if options.save:
        os.system('mkdir -p {}'.format(options.outfolder))
        fig.savefig('{}/{}'.format(options.outfolder, figname), dpi=200)
    else:
        plt.show()

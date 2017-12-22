#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : /Users/lhy/astro_workspace/astro-packages/bin/sps-plot_maps.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 13.09.2017
# Last Modified Date: 13.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
import numpy as np
from JAM.utils import util_fig
from JAM.utils import velocity_plot
import matplotlib.pyplot as plt
from optparse import OptionParser
from astropy import units as u
import pyfits
import pickle
import os

parser = OptionParser()
(options, args) = parser.parse_args()
if len(args) != 1:
    print 'Error - please provide a folder name'
    exit(1)

# read information
with open('{}/info.dat'.format(args[0])) as f:
    info = pickle.load(f)
Re_arcsec = info['Re']
a = info['a']
b = info['b']
pa = info['pa']
theta = np.radians(90.0-pa)
ba = info['ba']
Ldist = info['Ldist'] * u.Mpc
Ldist_cm = Ldist.to(u.cm).value

#read rebin maps
rebin_data = pyfits.open('{}/{}_ssp_rebinMaps.fits'.format(args[0], args[0]))

bin_id = rebin_data['bin_id'].data
rebinxbin = rebin_data['x0'].data
rebinybin = rebin_data['y0'].data
rebinml_r = rebin_data['ml_obs_r'].data
rebinMnogas = np.log10(rebin_data['Mnogas'].data)
rebinL = np.log10(rebin_data['Mnogas'].data/rebinml_r)
print np.log10(np.sum(rebin_data['Mnogas'].data))
rebinLlogAge = rebin_data['LageLog'].data
rebinMlogAge = rebin_data['MageLog'].data
rebinLZ = rebin_data['LZlog'].data
rebinMZ = rebin_data['MZlog'].data
ebv = rebin_data['ebv'].data
maps = {'ml_r': rebinml_r, 'LlogAge': rebinLlogAge, 'MlogAge': rebinMlogAge,
        'LZ': rebinLZ, 'MZ': rebinMZ, 'ebv': ebv, 'Mnogas': rebinMnogas,
        'L': rebinL}
barlabels = {'ml_r': '$\mathbf{M^*/L[M_{\odot}/L_{\odot}]}$',
             'LlogAge': '$\mathbf{L\ logAge[Gyrs]}$',
             'MlogAge': '$\mathbf{M\ logAge[Gyrs]}$',
             'LZ': '$\mathbf{L\ [Z/H]}$', 'MZ': '$\mathbf{M\ [Z/H]}$',
             'ebv': '$\mathbf{E(B-V)}$', 'Mnogas': '$M^*\ [M_{\odot}]$',
             'L': '$L\ [L_{\odot}]$'}
bad = bin_id != -1

# read information
with open('{}/info.dat'.format(args[0])) as f:
    info = pickle.load(f)
Re_arcsec = info['Re']
a = info['a']
b = info['b']
pa = info['pa']
theta = np.radians(90.0-pa)
ba = info['ba']
Ldist = info['Ldist']

os.system('mkdir -p {}/properties'.format(args[0]))
phi = np.linspace(0, 2*np.pi, 100)
xx_tem = a * np.cos(phi)
yy_tem = b * np.sin(phi)
xx = np.cos(-theta)*xx_tem+np.sin(-theta)*yy_tem
yy = -np.sin(-theta)*xx_tem+np.cos(-theta)*yy_tem
for key in maps.keys():
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(111)
    velocity_plot(rebinxbin[bad], rebinybin[bad], maps[key][bad],
                  barlabel=barlabels[key], vmap='map', markersize=0.5)
    ax.plot(xx*0.125, yy*0.125, 'c', alpha=0.6, lw=1)
    ax.plot(xx*1.0, yy*1.0, 'c', alpha=0.6, lw=1)
    # ax.plot(xx*1.5, yy*1.5, 'c', alpha=0.6, lw=1)
    fig.savefig('{}/properties/map-{}.png'.format(args[0], key), dpi=150)

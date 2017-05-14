#!/usr/bin/env python
import numpy as np
from JAM.utils import util_fig
from JAM.utils import velocity_plot
import matplotlib.pyplot as plt
from optparse import OptionParser
import pyfits
import pickle
import os
util_fig.ticks_font.set_size(10)

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

#read rebin maps
rebin_data = pyfits.open('{}/{}_ssp_rebinMaps.fits'.format(args[0], args[0]))

bin_id = rebin_data['bin_id'].data
rebinxbin = rebin_data['x0'].data
rebinybin = rebin_data['y0'].data
rebinml_r_obs = rebin_data['ml_obs_r'].data
rebinml_r_int = rebin_data['ml_int_r'].data
rebinMnogas = rebin_data['Mnogas'].data
rebinL_obs = rebinMnogas / rebinml_r_obs
rebinL_int = rebinMnogas / rebinml_r_int
rebinLlogAge = rebin_data['LageLog'].data
rebinMlogAge = rebin_data['MageLog'].data
rebinLZ = rebin_data['LZlog'].data
rebinMZ = rebin_data['MZlog'].data
ebv = rebin_data['ebv'].data
maps = {'ml_r_obs': rebinml_r_obs, 'ml_r_int': rebinml_r_int,
        'LlogAge': rebinLlogAge, 'MlogAge': rebinMlogAge,
        'LZ': rebinLZ, 'MZ': rebinMZ, 'ebv': ebv, 'Mnogas': rebinMnogas,
        'L_obs': rebinL_obs, 'L_int': rebinL_int}
ylabels = {'ml_r_obs': '$\mathbf{M^*/L[M_{\odot}/L_{\odot}]}$',
           'ml_r_int': '$\mathbf{M^*/L[M_{\odot}/L_{\odot}]}$',
           'LlogAge': '$\mathbf{L\ logAge[Gyrs]}$',
           'MlogAge': '$\mathbf{M\ logAge[Gyrs]}$',
           'LZ': '$\mathbf{L\ [Z/H]}$', 'MZ': '$\mathbf{M\ [Z/H]}$',
           'ebv': '$\mathbf{E(B-V)}$', 'Mnogas': 'Mnogas',
           'L_obs': 'L', 'L_int': 'L'}
bad = bin_id != -1

os.system('mkdir -p {}/properties'.format(args[0]))
x0 = np.cos(theta)*rebinxbin + np.sin(theta)*rebinybin
y0 = -np.sin(theta)*rebinxbin + np.cos(theta)*rebinybin
good = bin_id != -1
r_ell = ((x0/a)**2 + (y0/b)**2)**0.5
nre = 2.5
inRe = (r_ell < nre) * good
inRe_r_ell =r_ell[inRe]
bin_id = np.floor(r_ell[inRe] / 0.1).astype(int)
r = np.arange(0.05, nre, 0.1)
profile = np.zeros([3, len(r)])
rst = {'r': r, 'inRe_r_ell': inRe_r_ell}
for key in maps.keys():
    inReMap = maps[key][inRe]
    for i in range(len(r)):
        inBin = bin_id == i
        if inBin.sum() > 0:
            profile[:, i] = np.percentile(inReMap[inBin], [16, 50, 84])
        else:
            profile[:, i] = np.nan
    profile[0, :] -= profile[1, :]
    profile[0, :] = -profile[0, :]
    profile[2, :] -= profile[1, :]
    rst['profile_{}'.format(key)] = profile.copy()
    rst['inReMap_{}'.format(key)] = inReMap.copy()
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(111)
    ax.set_title(key)
    fig.subplots_adjust(left=0.18, bottom=0.130, right=0.965,
                        top=0.92, wspace=0., hspace=0.)
    ax.plot(inRe_r_ell, inReMap, '.')
    ax.errorbar(r, profile[1, :], yerr=profile[[0, 2], :],
                fmt='o', color='k')
    ax.set_xlabel(r'$\rm Radius\ [R_e]$')
    ax.set_ylabel(ylabels[key])
    util_fig.set_labels(ax)
    # plt.tight_layout()
    fig.savefig('{}/properties/profile-{}.png'.format(args[0], key), dpi=150)
with open('{}/properties/profiles.dat'.format(args[0]), 'w') as f:
    pickle.dump(rst, f)

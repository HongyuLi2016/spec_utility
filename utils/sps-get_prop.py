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

def get_slope(r, y, lim=[0.5, 1.5]):
    good = (r<lim[1]) * (r>lim[0]) * (~np.isinf(y[1, :]))
    z = np.polyfit(r[good], y[1, good], 1)
    p = np.poly1d(z)
    fit_x = r[good]
    fit_y = p(fit_x)
    return z[0], fit_x, fit_y

parser = OptionParser()
(options, args) = parser.parse_args()
if len(args) != 1:
    print 'Error - please provide a folder name'
    exit(1)

with open('{}/properties/profiles.dat'.format(args[0])) as f:
    data = pickle.load(f)
# print data.keys()

nre = [0.5, 1.0, 1.5]
glim = [0.5, 1.5]
# logAge
LlogAge = np.zeros(len(nre))
MlogAge = np.zeros(len(nre))
# [Z/H]
LZ = np.zeros(len(nre))
MZ = np.zeros(len(nre))
# M*/L
ml_obs = np.zeros(len(nre))
ml_int = np.zeros(len(nre))
# Extinction
Ar = np.zeros(len(nre))
# Luminority
Lobs = np.zeros(len(nre))
Lint = np.zeros(len(nre))
for i in range(len(nre)):
    ii = data['inRe_r_ell'] < nre[i]
    Mnogas_in = data['inReMap_Mnogas'][ii]
    Lobs_in = data['inReMap_L_obs'][ii]
    Lint_in = data['inReMap_L_int'][ii]
    MlogAge_in = data['inReMap_MlogAge'][ii]
    LlogAge_in = data['inReMap_LlogAge'][ii]
    LZ_in = data['inReMap_LZ'][ii]
    MZ_in = data['inReMap_MZ'][ii]
    MlogAge[i] = np.average(MlogAge_in, weights=Mnogas_in)
    LlogAge[i] = np.average(LlogAge_in, weights=Lobs_in)
    MZ[i] = np.average(MZ_in, weights=Mnogas_in)
    LZ[i] = np.average(LZ_in, weights=Lobs_in)
    ml_obs[i] = Mnogas_in.sum()/Lobs_in.sum()
    ml_int[i] = Mnogas_in.sum()/Lint_in.sum()
    Ar[i] = 2.5 * np.log10(Lint_in.sum()/Lobs_in.sum())
    Lobs[i] = Lobs_in.sum()
    Lint[i] = Lint_in.sum()
gMlogAge = get_slope(data['r'], data['profile_MlogAge'], lim=glim)
gLlogAge = get_slope(data['r'], data['profile_LlogAge'], lim=glim)
gMZ = get_slope(data['r'], data['profile_MZ'], lim=glim)
gLZ = get_slope(data['r'], data['profile_LZ'], lim=glim)
gml_obs = get_slope(data['r'], data['profile_ml_r_obs'], lim=glim)
gml_int = get_slope(data['r'], data['profile_ml_r_int'], lim=glim)

inRe_maps = {}
inRe_maps['MlogAge'] = data['inReMap_MlogAge']
inRe_maps['LlogAge'] = data['inReMap_LlogAge']
inRe_maps['MZ'] = data['inReMap_MZ']
inRe_maps['LZ'] = data['inReMap_LZ']
inRe_maps['ml_r_obs'] = data['inReMap_ml_r_obs']
inRe_maps['ml_r_int'] = data['inReMap_ml_r_int']
profile = {}
profile['MlogAge'] = data['profile_MlogAge']
profile['LlogAge'] = data['profile_LlogAge']
profile['MZ'] = data['profile_MZ']
profile['LZ'] = data['profile_LZ']
profile['ml_r_obs'] = data['profile_ml_r_obs']
profile['ml_r_int'] = data['profile_ml_r_int']
gradient = {}

gradient['MlogAge'] = gMlogAge
gradient['LlogAge'] = gLlogAge
gradient['MZ'] = gMZ
gradient['LZ'] = gLZ
gradient['ml_r_obs'] = gml_obs
gradient['ml_r_int'] = gml_int

ylabels = {'ml_r_obs': '$\mathbf{M^*/L[M_{\odot}/L_{\odot}]}$',
           'ml_r_int': '$\mathbf{M^*/L[M_{\odot}/L_{\odot}]}$',
           'LlogAge': '$\mathbf{L\ logAge[Gyrs]}$',
           'MlogAge': '$\mathbf{M\ logAge[Gyrs]}$',
           'LZ': '$\mathbf{L\ [Z/H]}$', 'MZ': '$\mathbf{M\ [Z/H]}$',
           'ebv': '$\mathbf{E(B-V)}$', 'Mnogas': 'Mnogas',
           'L_obs': 'L', 'L_int': 'L'}

inRe_r_ell = data['inRe_r_ell']
r = data['r']
for key in inRe_maps.keys():
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(111)
    ax.set_title(key)
    fig.subplots_adjust(left=0.18, bottom=0.130, right=0.965,
                        top=0.92, wspace=0., hspace=0.)
    ax.plot(inRe_r_ell, inRe_maps[key], '.')
    ax.plot(gradient[key][1], gradient[key][2], 'r', lw=2)
    ax.text(0.7, 0.9, '$\Delta: {:5.3f}$'.format(gradient[key][0]),
            color='r', transform=ax.transAxes)
    ax.errorbar(r, profile[key][1, :], yerr=profile[key][[0, 2], :],
                fmt='o', color='k')
    ax.set_xlabel(r'$\rm Radius\ [R_e]$')
    ax.set_ylabel(ylabels[key])
    util_fig.set_labels(ax)
    fig.savefig('{}/properties/profile-{}.png'.format(args[0], key), dpi=150)
rst = {}
rst['nre'] = nre
rst['glim'] = glim
rst['MlogAge'] = MlogAge
rst['LlogAge'] = LlogAge
rst['MZ'] = MZ
rst['LZ'] = LZ
rst['ml_obs'] = ml_obs
rst['ml_int'] = ml_int
rst['gMlogAge'] = gMlogAge[0]
rst['gLlogAge'] = gLlogAge[0]
rst['gMZ'] = gMZ[0]
rst['gLZ'] = gLZ[0]
rst['gml_obs'] = gml_obs[0]
rst['gml_int'] = gml_int[0]
with open('{}/properties/prop.dat'.format(args[0]), 'w') as f:
    pickle.dump(rst, f)

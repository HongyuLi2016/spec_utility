#!/usr/bin/evn python
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib import colors, rc
from matplotlib.patches import Rectangle
import pyfits
import pickle
rc('mathtext', fontset='stix')

_cdict = {'red': ((0.000,   0.01,   0.01),
                  (0.170,   0.0,    0.0),
                  (0.336,   0.4,    0.4),
                  (0.414,   0.5,    0.5),
                  (0.463,   0.3,    0.3),
                  (0.502,   0.0,    0.0),
                  (0.541,   0.7,    0.7),
                  (0.590,   1.0,    1.0),
                  (0.668,   1.0,    1.0),
                  (0.834,   1.0,    1.0),
                  (1.000,   0.9,    0.9)),
          'green': ((0.000,   0.01,   0.01),
                    (0.170,   0.0,    0.0),
                    (0.336,   0.85,   0.85),
                    (0.414,   1.0,    1.0),
                    (0.463,   1.0,    1.0),
                    (0.502,   0.9,    0.9),
                    (0.541,   1.0,    1.0),
                    (0.590,   1.0,    1.0),
                    (0.668,   0.85,   0.85),
                    (0.834,   0.0,    0.0),
                    (1.000,   0.9,    0.9)),
          'blue': ((0.000,   0.01,   0.01),
                   (0.170,   1.0,    1.0),
                   (0.336,   1.0,    1.0),
                   (0.414,   1.0,    1.0),
                   (0.463,   0.7,    0.7),
                   (0.502,   0.0,    0.0),
                   (0.541,   0.0,    0.0),
                   (0.590,   0.0,    0.0),
                   (0.668,   0.0,    0.0),
                   (0.834,   0.0,    0.0),
                   (1.000,   0.9,    0.9))
          }

sauron = colors.LinearSegmentedColormap('sauron', _cdict)

ticks_font =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=10,
                                           weight='bold', stretch='normal')
text_font =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=15,
                                           weight='bold', stretch='normal')
ticks_font1 =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=8,
                                           weight='bold', stretch='normal')
label_font =\
    matplotlib.font_manager.FontProperties(family='times new roman',
                                           style='normal', size=15,
                                           weight='bold', stretch='normal')

c = 299792.458


def resample(wave, newWave, flux, error=None,
             interpKind=None, fill_value=0.0):
    if interpKind is None:
        interpKind = 'linear'
    fSpec = interp1d(wave, flux, kind=interpKind, fill_value=fill_value,
                     bounds_error=False)
    newFlux = fSpec(newWave)
    if error is not None:
        fError = interp1d(wave, error, kind=interpKind,
                          fill_value=fill_value, bounds_error=False)
        newError = fError(newWave)
        return newFlux, newError
    else:
        return newFlux


def bit_mask(mask_array, bit):
    '''
    For dap MPL5 velocity map files
    30: DONOTUSE        Do not use this spaxel for science
    19: MULTICOMP       Multi-component velocity features present
    9: NOCORRECTION    Appropriate correction not available
    8: NEARBOUND       Fitted value is too near an imposed boundary; see TRM
    7: FITFAILED       Attempted fit for property failed
    6: MATHERROR       Mathematical error in computing value
    5: UNRELIABLE      Value is deemed unreliable; see TRM for definition
    4: NOVALUE         Spaxel was not fit because it did not meet
                       selection criteria
    3: FORESTAR        Foreground star
    2: DEADFIBER       Major contributing fiber is dead
    1: LOWCOV          Low coverage in this spaxel
    0: NOCOV           No coverage in this spaxel
    '''
    return mask_array & 2**bit > 0


def out_put_spec(wave, flux, error, flag, folder='spec',
                 specname='spec_0000.dat'):
    with open('{0}/{1}'
              .format(folder, specname), 'w') as ff:
        # print >>ff, '# wavelenght    flux    error    flag'
        for i in range(len(wave)):
            print >>ff, '{0:>e}    {1:>e}    {2:>e}    {3:>d}'\
                .format(wave[i], flux[i], error[i], flag[i])


def grid(flist, oname, path, config_file='StClhy_medium.config',
         base_file='Base.BC03.S', mask_file='Masks.EmLines.SDSS.gm',
         extinction='CAL', llow_SN=4730.0, lupp_SN=4780.0, Olsyn_ini=3400.0,
         Olsyn_fni=9000.0, v0_start=0.0, vd_start=150.0,
         base_dir='/home/lhy/STARLIGHTv04/BasesDir/'):
    mask_dir = '/home/lhy/STARLIGHTv04/'

    Nspec = len(flist)
    with open('{}/{}'.format(path, oname), 'w') as ff:
        print >>ff, '{0:<70d}    [Number of fits to run]'.format(Nspec)
        print >>ff, '{0:<70s}    [base_dir]'.format(base_dir)
        print >>ff, '{0:<70s}    [obs_dir]'.format('spec/')
        print >>ff, '{0:<70s}    [mask_dir]'.format(mask_dir)
        print >>ff, '{0:<70s}    [out_dir]'.format('out/')
        print >>ff, '{0:<70d}    [your phone number]'.format(-2007200)
        print >>ff, '{0:<70.1f}    [llow_SN]   lower-lambda of S/N window'\
            .format(llow_SN)
        print >>ff, '{0:<70.1f}    [lupp_SN]   upper-lambda of S/N window'\
            .format(lupp_SN)
        print >>ff, '{0:<70.1f}    [Olsyn_ini]   lower-lambda for fit'\
            .format(Olsyn_ini)
        print >>ff, '{0:<70.1f}    [Olsyn_fni]   upper-lambda for fit'\
            .format(Olsyn_fni)
        print >>ff, '{0:<70d}    [Odlsyn]    delta-lambda for fit'.format(1)
        print >>ff, '{0:<70d}    [fscale_chi2] fudge-factor for chi2'\
            .format(1)
        print >>ff, '{0:<70s}    [FIT/FXK] Fit or Fix kinematics'.format('FIT')
        print >>ff, '{0:<70d}    [IsErrSpecAvailable]  1/0 = Yes/No'.format(1)
        print >>ff, '{0:<70d}    [IsFlagSpecAvailable] 1/0 = Yes/No'.format(1)
        for i in range(Nspec):
            outname = '{}_out.dat'.format(flist[i][0:-4])
            print >>ff, '{0}    {1}    {2}    {3}    {4}'\
                '    {5:.1f}    {6:.1f}    {7}'.format(flist[i], config_file,
                                                       base_file, mask_file,
                                                       extinction, v0_start,
                                                       vd_start, outname)


def sdss_r_filter(path='./'):
    return np.loadtxt(path)


def plot_sps(wave, obs, syn, good, Mweights, Lweights, Ages, Z,
             parameters={}):
    # print Ages, Z
    plt.clf()
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(3, 3)
    gs.update(left=0.09, right=0.98, top=0.98, bottom=0.07,
              hspace=0.3, wspace=0.3)
    ax1 = plt.subplot(gs[0, :])
    ll, rr = np.min(wave), np.max(wave)
    mn, mx = np.min(syn[good]), np.max(syn[good])
    resid = mn + obs - syn
    mn1 = np.min(resid[good])
    ax1.set_xlim([ll, rr] + np.array([-0.02, 0.02])*(rr - ll))
    ax1.set_ylim([mn1, mx] + np.array([-0.05, 0.05])*(mx - mn1))
    ax1.plot(wave, obs, 'k', linewidth=0.5)
    ax1.plot(wave, syn, 'r', linewidth=1)
    ax1.plot(wave[good], resid[good], 'd', color='LimeGreen',
             mec='LimeGreen', ms=4)
    ax1.plot(wave[good], good*0 + mn, '.k', ms=1)
    w = np.flatnonzero(np.diff(good) > 1)
    if w.size > 0:
        for wj in w:
            j = slice(good[wj], good[wj+1] + 1)
            ax1.plot(wave[j], resid[j], 'b')
        w = np.hstack([0, w, w + 1, -1])  # Add first and last point
    else:
        w = [0, -1]
    for gj in good[w]:
        ax1.plot(wave[[gj, gj]], [mn, syn[gj]], 'LimeGreen')
    set_labels(ax1)
    ax2 = plt.subplot(gs[1, 0:2])
    plot_2d(Ages, Z, Mweights, log=False, ax=ax2, fig=fig,
            xlabel='log Age', ylabel='[Z/H]', clabel='M_weights')
    ax3 = plt.subplot(gs[2, 0:2])
    plot_2d(Ages, Z, Lweights, log=False, ax=ax3, fig=fig,
            xlabel='log Age', ylabel='[Z/H]', clabel='L_weights')
    ax4 = plt.subplot(gs[1:3, 2])
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    outText = ''
    for i in range(len(parameters)):
        outText += '{:12s} {:<10.3f}\n'.format(parameters.keys()[i]+':',
                                               parameters.values()[i])
    ax4.text(0.1, 0.1, outText, transform=ax4.transAxes,
             fontproperties=text_font)
    return fig


def set_labels(ax, rotate=False, font=ticks_font):
    for l in ax.get_xticklabels():
        if rotate:
            l.set_rotation(60)
        l.set_fontproperties(font)
    for l in ax.get_yticklabels():
        # if rotate:
        #    l.set_rotation(0)
        l.set_fontproperties(font)


def get_Rectangle(x):
    '''
    pos_x: left vetex of the rectangle
    size_x: size of the rectangle
    '''
    pos_x = np.zeros_like(x)
    size_x = np.zeros_like(x)
    for i in range(1, len(pos_x)):
        pos_x[i] = (x[i]+x[i-1])*0.5
    pos_x[0] = x[0] - (pos_x[1] - x[0])
    for i in range(0, len(pos_x)-1):
        size_x[i] = pos_x[i+1] - pos_x[i]
    size_x[-1] = (x[-1]-pos_x[-1])*2.0
    return pos_x, size_x


def plot_2d(x, y, z, ax=None, log=True, c_lim=None, fig=None,
            clabel=None, cmap='BuPu', xlabel='', ylabel=''):
    if (len(x) < 2) or (len(y) < 2):
        print 'Error - axis must be greater than 2 (spec_utils.plot_2d)'
        exit(1)
    if (len(x) != z.shape[0]) or (len(y) != z.shape[1]):
        print 'Error - len(x) {} len(y) {} do not match z shape {}*{}'\
            .format(len(x), len(y), z.shape[0], z.shape[1])
        exit(1)
    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = plt.gcf()
    if c_lim is not None:
        z = z.clip(10**c_lim[0], 10**c_lim[1])
    if log:
        z = np.log10(z)
    if cmap == 'sauron':
        cmap = sauron
    else:
        cmap = plt.get_cmap(cmap)
    set_labels(ax, rotate=True)
    pos_x, size_x = get_Rectangle(x)
    pos_y, size_y = get_Rectangle(y)
    norm = colors.Normalize(vmin=z.min()-abs(z.min()*0.02),
                            vmax=z.max()+abs(z.min()*0.02))
    for i in range(len(pos_x)):
        for j in range(len(pos_y)):
            clr = z[i, j]
            squre = Rectangle(xy=(pos_x[i], pos_y[j]), fc=cmap(norm(clr)),
                              width=size_x[i], height=size_y[j],
                              ec='gray', zorder=1, lw=1)
            ax.add_artist(squre)
            # ax.plot(pos_x[i],pos_y[j],'g.')
    xexcess = (pos_x[-1]+size_x[-1] - pos_x[0]) * 0.01
    yexcess = (pos_y[-1]+size_y[-1] - pos_y[0]) * 0.02
    xlim = [pos_x[0]-xexcess, pos_x[-1]+size_x[-1]+xexcess]
    ylim = [pos_y[0]-yexcess, pos_y[-1]+size_y[-1]+yexcess]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_yticks(y)
    pos = ax.get_position()
    px = pos.x1
    py = pos.y0
    ph = pos.y1-pos.y0
    axc = fig.add_axes([px, py, 0.01, ph])
    matplotlib.colorbar.ColorbarBase(axc, orientation='vertical',
                                     norm=norm, cmap=cmap)
    if clabel is not None:
        axc.set_ylabel(clabel, fontproperties=label_font)
    set_labels(axc, font=ticks_font1)
    ax.set_ylabel(ylabel, fontproperties=label_font)
    ax.set_xlabel(xlabel, fontproperties=label_font)


def ssp_dump_data(ml_obs_r, ml_int_r, ebv, Mnogas, MageLog, MZlog, Mage, MZ,
                  LageLog, LZLog, Lage, LZ, wave, obs, syn, goodpixels,
                  Mweights, Lweights, logAge_grid, metal_grid,
                  fname='spsout.dat', outfolder='.'):
    '''
    dump sps results into a binary file (dictionary)
    '''
    data = {'ml_obs_r': ml_obs_r, 'ml_int_r': ml_int_r, 'ebv': ebv,
            'Mnogas': Mnogas, 'MageLog': MageLog,
            'MZLog': MZlog, 'Mage': Mage, 'MZ': MZ, 'LageLog': LageLog,
            'LZLog': LZLog, 'Lage': Lage, 'LZ': LZ, 'wave': wave, 'obs': obs,
            'syn': syn, 'goodpixels': goodpixels, 'Lweights': Lweights,
            'Mweights': Mweights, 'logAge_grid': logAge_grid,
            'metal_grid': metal_grid}
    with open('{}/{}'.format(outfolder, fname), 'wb') as f:
        pickle.dump(data, f)


def convertDump2txt(fname):
    data = ssp_load_data(fname)
    with open('{}.txt'.format(fname[0:-4]), 'w') as f:
        f.write('ml_obs: {:.3f}\n'.format(data['ml_obs_r']))
        f.write('ml_int: {:.3f}\n'.format(data['ml_int_r']))
        f.write('ebv: {:.3f}\n'.format(data['ebv']))
        f.write('M<logAge>: {:.3f}\n'.format(data['MageLog']))
        f.write('M<[Z/H]>: {:.3f}\n'.format(data['MZLog']))
        f.write('L<logAge>: {:.3f}\n'.format(data['LageLog']))
        f.write('L<[Z/H]>: {:.3f}\n'.format(data['LZLog']))
        print data['logAge_grid'].shape
        for i in range(data['logAge_grid'].shape[1]):
            for j in range(data['logAge_grid'].shape[0]):
                f.write('{:7.3f} {:7.3f} {:7.3f} {:7.3f}\n'
                        .format(data['logAge_grid'][j, i],
                                data['metal_grid'][j, i],
                                data['Mweights'][j, i],
                                data['Lweights'][j, i]))


def ssp_load_data(fname):
    '''
    load sps binary file, return a dictionary
    '''
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def ssp_write_map(ID, xbin, ybin, outname='sspMap.fits', path='.'):
    ml_obs_r = np.zeros(len(ID))
    ml_int_r = np.zeros(len(ID))
    ebv = np.zeros(len(ID))
    Mnogas = np.zeros(len(ID))
    MageLog = np.zeros(len(ID))
    MZlog = np.zeros(len(ID))
    Mage = np.zeros(len(ID))
    MZ = np.zeros(len(ID))
    LageLog = np.zeros(len(ID))
    LZLog = np.zeros(len(ID))
    Lage = np.zeros(len(ID))
    LZ = np.zeros(len(ID))
    for i in range(len(ID)):
        try:
            fname = 'sps_{:04d}.dat'.format(ID[i])
            data = ssp_load_data('{}/{}'.format(path, fname))
            ml_obs_r[i] = data['ml_obs_r']
            ml_int_r[i] = data['ml_int_r']
            ebv[i] = data['ebv']
            Mnogas[i] = data['Mnogas']
            MageLog[i] = data['MageLog']
            MZlog[i] = data['MZLog']
            Mage[i] = data['Mage']
            MZ[i] = data['MZ']
            LageLog[i] = data['LageLog']
            LZLog[i] = data['LZLog']
            Lage[i] = data['Lage']
            LZ[i] = data['LZ']
        except:
            print 'Warning - load bin {:04d} faild!'.format(ID[i])
            ml_obs_r[i] = np.nan
            ml_int_r[i] = np.nan
            ebv[i] = np.nan
            Mnogas[i] = np.nan
            MageLog[i] = np.nan
            MZlog[i] = np.nan
            Mage[i] = np.nan
            MZ[i] = np.nan
            LageLog[i] = np.nan
            LZLog[i] = np.nan
            Lage[i] = np.nan
            LZ[i] = np.nan
    c1 = pyfits.Column(name='ID', format='J', array=ID)
    c2 = pyfits.Column(name='xbin', format='D', array=xbin)
    c3 = pyfits.Column(name='ybin', format='D', array=ybin)
    c4 = pyfits.Column(name='ml_obs_r', format='D', array=ml_obs_r)
    c5 = pyfits.Column(name='ml_int_r', format='D', array=ml_int_r)
    c6 = pyfits.Column(name='ebv', format='D', array=ebv)
    c7 = pyfits.Column(name='Mnogas', format='D', array=Mnogas)
    c8 = pyfits.Column(name='MageLog', format='D', array=MageLog)
    c9 = pyfits.Column(name='MZlog', format='D', array=MZlog)
    c10 = pyfits.Column(name='Mage', format='D', array=Mage)
    c11 = pyfits.Column(name='MZ', format='D', array=MZ)
    c12 = pyfits.Column(name='LageLog', format='D', array=LageLog)
    c13 = pyfits.Column(name='LZLog', format='D', array=LZLog)
    c14 = pyfits.Column(name='Lage', format='D', array=Lage)
    c15 = pyfits.Column(name='LZ', format='D', array=LZ)
    coldefs = pyfits.ColDefs([c1, c2, c3, c4, c5, c6, c7, c8, c9,
                              c10, c11, c12, c13, c14, c15])
    hdu = pyfits.PrimaryHDU()
    tbhdu = pyfits.BinTableHDU.from_columns(coldefs)
    hdulist = pyfits.HDUList([hdu, tbhdu])
    hdulist.writeto('{}/{}'.format(path, outname), clobber=True)


def load_txt_spec(fname):
    '''
    load txt spectrum in starlight format
    '''
    data = np.genfromtxt(fname, dtype=[('wave', 'f8'), ('flux', 'f8'),
                                       ('err', 'f8'), ('flag', 'i8')])
    return data['wave'], data['flux'], data['err'], data['flag'] == 1


def load_bin_file(fname, Bar=False):
    '''
    load txt node.dat file for voronoi bin information
    '''
    data = np.genfromtxt(fname, dtype=[('ID', 'i8'), ('xNode', 'f8'),
                         ('yNode', 'f8'), ('xBar', 'f8'), ('yBar', 'f8')])
    if Bar:
        return data['ID'], data['xNode'], data['yNode'],\
               data['xBar'], data['ybar']
    else:
        return data['ID'], data['xNode'], data['yNode']


def load_spaxel_file(fname):
    pass

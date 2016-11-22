#!/usr/bin/env python
import numpy as np
import pyfits
import matplotlib.pyplot as plt
from scipy import ndimage
from ppxf import ppxf
import ppxf_util as util


def convert_mask(lam, good, new_lam):
    '''
    convert linear maksed pixels to logrebined pixels
    '''
    new_good = np.ones_like(new_lam, dtype=bool)
    for i in range(len(new_lam)):
        ii = abs(lam - new_lam[i]) == abs(lam - new_lam[i]).min()
        if not good[ii]:
            new_good[i] = False
    return new_good


class ppxf_sps():
    '''
    A warper for calling ppxf to do stellar population
    '''
    def __init__(self, velscale):
        self.velscale = velscale

    def load_miles_defalt(self, FWHM_gal, path='miles_models'):
        '''
        load miles stellar library, convolve with observed resolusion
        (FWHM_gal), and log rebin using velscale
        '''
        velscale = self.velscale
        FWHM_tem = 2.51
        data = np.loadtxt('{}/information_default.dat'.format(path),
                          dtype=[('name', '45S'), ('Z', 'f8'), ('age', 'f8'),
                                 ('ml_u', 'f8'), ('ml_g', 'f8'), ('ml_r', 'f8'),
                                 ('ml_i', 'f8'), ('mnogas', 'f8')])
        nAges = 26
        nMetal = 6
        fnames = data['name'].reshape(nMetal, nAges).T
        self.logAge_grid = np.log10(data['age'].reshape(nMetal, nAges).T)
        self.metal_grid = data['Z'].reshape(nMetal, nAges).T
        self.ml_u_grid = data['ml_u'].reshape(nMetal, nAges).T
        self.ml_g_grid = data['ml_g'].reshape(nMetal, nAges).T
        self.ml_r_grid = data['ml_r'].reshape(nMetal, nAges).T
        self.ml_i_grid = data['ml_i'].reshape(nMetal, nAges).T
        self.mnogas_grid = data['mnogas'].reshape(nMetal, nAges).T
        try:
            hdu = pyfits.open(path+'/'+fnames[0, 0])
        except:
            print 'Error - Can not open templates file'
            exit(0)
        ssp = hdu[0].data
        h2 = hdu[0].header
        FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
        sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
        lamRange_temp = h2['CRVAL1'] + \
            np.array([0., h2['CDELT1']*(h2['NAXIS1']-1)])
        sspNew, logLam2, velscale = util.log_rebin(lamRange_temp,
                                                   ssp, velscale=velscale)
        self.logLamT = logLam2
        self.lamRange_temp = lamRange_temp
        self.templates = np.empty((sspNew.size, nAges, nMetal))
        for i in range(nAges):
            for j in range(nMetal):
                # print fnames[i, j], self.metal_grid[i, j],\
                #    self.logAge_grid[i, j]
                hdu = pyfits.open(path + '/' + fnames[i, j])
                ssp = hdu[0].data
                ssp = ndimage.gaussian_filter1d(ssp, sigma)
                sspNew, logLam2, velscale = \
                    util.log_rebin(lamRange_temp, ssp, velscale=velscale)
                # Templates are *not* normalized here
                self.templates[:, i, j] = sspNew

    def load_miles_all(self, FWHM_gal, path='MILES_Padova00_un_1.30_fits'):
        velscale = self.velscale
        FWHM_tem = 2.51
        data = np.loadtxt('{}/information.dat'.format(path),
                          dtype=[('name', '45S'), ('Z', 'f8'), ('age', 'f8'),
                                 ('ml_u', 'f8'), ('ml_g', 'f8'), ('ml_r', 'f8'),
                                 ('ml_i', 'f8'), ('mnogas', 'f8')])
        nAges = 50
        nMetal = 7
        fnames = data['name'].reshape(nMetal, nAges).T
        self.logAge_grid = np.log10(data['age'].reshape(nMetal, nAges).T)
        self.metal_grid = data['Z'].reshape(nMetal, nAges).T
        self.ml_u_grid = data['ml_u'].reshape(nMetal, nAges).T
        self.ml_g_grid = data['ml_g'].reshape(nMetal, nAges).T
        self.ml_r_grid = data['ml_r'].reshape(nMetal, nAges).T
        self.ml_i_grid = data['ml_i'].reshape(nMetal, nAges).T
        self.mnogas_grid = data['mnogas'].reshape(nMetal, nAges).T
        try:
            hdu = pyfits.open(path+'/'+fnames[0, 0])
        except:
            print 'Error - Can not open templates file'
            exit(0)
        ssp = hdu[0].data
        h2 = hdu[0].header
        FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
        sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
        lamRange_temp = h2['CRVAL1'] + \
            np.array([0., h2['CDELT1']*(h2['NAXIS1']-1)])
        sspNew, logLam2, velscale = util.log_rebin(lamRange_temp,
                                                   ssp, velscale=velscale)
        self.logLamT = logLam2
        self.lamRange_temp = lamRange_temp
        self.templates = np.empty((sspNew.size, nAges, nMetal))
        for i in range(nAges):
            for j in range(nMetal):
                hdu = pyfits.open(path + '/' + fnames[i, j])
                ssp = hdu[0].data
                ssp = ndimage.gaussian_filter1d(ssp, sigma)
                sspNew, logLam2, velscale = \
                    util.log_rebin(lamRange_temp, ssp, velscale=velscale)
                # Templates are *not* normalized here
                self.templates[:, i, j] = sspNew

    def select_templates(self, age, z):
        '''
        select a subset templates to use
        '''
        self.templates = self.templates[np.ix_(np.arange(
            self.templates.shape[0]), age, z)]
        self.logAge_grid = self.logAge_grid[np.ix_(age, z)]
        self.metal_grid = self.metal_grid[np.ix_(age, z)]
        self.ml_u_grid = self.ml_u_grid[np.ix_(age, z)]
        self.ml_g_grid = self.ml_g_grid[np.ix_(age, z)]
        self.ml_r_grid = self.ml_r_grid[np.ix_(age, z)]
        self.ml_i_grid = self.ml_i_grid[np.ix_(age, z)]
        self.mnogas_grid = self.mnogas_grid[np.ix_(age, z)]

    def load_galaxy(self, wave, flux, fit_range=None,
                    error=None, good=None):
        '''
        read galaxy spectrum, specify fitting range, log rebin the observed
        galaxy spectrum
        '''
        if error is None:
            error = np.ones_like(wave)
        if good is None:
            good = np.ones_like(wave, dtype=bool)
        if fit_range is None:
            self.fit_range = self.lamRange_temp.copy()
        else:
            self.fit_range = fit_range
        if (self.fit_range[0] < self.lamRange_temp[0]) or \
                (self.fit_range[1] > self.lamRange_temp[1]):
            print 'Error - fit range {:.1f} - {:.1f} is larger than'\
                ' template range {:.1f} - {:.1f}'\
                .format(self.fit_range[0], self.fit_range[1],
                        self.lamRange_temp[0], self.lamRange_temp[1])
            exit(0)
        velscale = self.velscale
        self.obs_wave = wave
        self.obs_flux = flux
        ii = (wave > self.fit_range[0]) * (wave < self.fit_range[1])
        self.obs_norm = np.nanmedian(flux[ii])
        fit_wave = wave[ii]
        fit_flux = flux[ii] / self.obs_norm
        fit_error = error[ii] / self.obs_norm
        fit_good = good[ii]
        lamRange = [fit_wave[0], fit_wave[-1]]

        galaxy, logWave, velscale = util.log_rebin(lamRange, fit_flux,
                                                   velscale=velscale)
        noise, logWave, velscale = util.log_rebin(lamRange, fit_error,
                                                  velscale=velscale)
        new_good = convert_mask(fit_wave, fit_good, np.exp(logWave))
        self.galaxy = galaxy
        self.noise = noise
        self.wave = np.exp(logWave)
        self.good = new_good

    def run(self, *args, **kwargs):
        '''
        recommanded parameters
        moments = 4 (defalt: 2)
        mdegree = 0 for CAL extinction (defalt: 0)
                if mdegree is 0, CAL law is used automaticlly
        regul = 0.0 (defalt: 0)
        good = mask_array, 1 for good, 0 for mask
        '''
        mdegree = kwargs.pop("mdegree", 0)

        if mdegree == 0:
            lam = self.wave
            reddening = 0.0
        else:
            lam = None
            reddening = None
        c = 299792.458
        dv = c * np.log(self.lamRange_temp[0]/self.wave[0])
        start = [0.0, 3.0 * self.velscale]
        pp = ppxf(self.templates, self.galaxy, self.noise, self.velscale,
                  start, mask=self.good, quiet=False, degree=-1,
                  vsyst=dv, clean=False, lam=lam, reddening=reddening,
                  *args, **kwargs)
        self.pp = pp

    def ebv(self):
        return self.pp.reddening

    def weights(self):
        return self.pp.weights.reshape((self.templates.shape[1],
                                       self.templates.shape[2]))

    def nogas_weights(self):
        return self.weights() * self.mnogas_grid

    def luminosity_weights(self):
        return self.nogas_weights() / self.ml_r_grid

    def ml_u(self):
        return self.nogas_weights().sum() /\
            (self.nogas_weights() / self.ml_u_grid).sum()

    def ml_g(self):
        return self.nogas_weights().sum() /\
            (self.nogas_weights() / self.ml_g_grid).sum()

    def ml_r(self):
        return self.nogas_weights().sum() /\
            (self.nogas_weights() / self.ml_r_grid).sum()

    def ml_i(self):
        return self.nogas_weights().sum() /\
            (self.nogas_weights() / self.ml_i_grid).sum()

if __name__ == '__main__':
    data = np.loadtxt('vazdekis_ml_test.txt')
    wave = data[:, 0]
    c = 299792.458
    vscale = np.log(wave[-1] / wave[0]) / len(wave) * c
    flux = data[:, 1]
    lhy = ppxf_sps(vscale)
    # lhy.load_miles_defalt(2.8, path='/home/lhy/python/ppxf/miles_models')
    lhy.load_miles_all(2.8, path='/home/lhy/python/ppxf/'
                                 'MILES_Padova00_un_1.30_fits')
    # lhy.select_templates(range(24, 50), range(1, 7))
    lhy.load_galaxy(wave, flux, error=flux * 0.0 + 1.0, fit_range=None)
    lhy.run(moments=4, mdegree=0)
    # print lhy.weights()
    # print lhy.nogas_weights() - lhy.weights()
    # print lhy.ml_r()
    plt.clf()
    plt.subplot(211)
    lhy.pp.plot()   # produce pPXF plot
    plt.subplot(212)
    s = lhy.templates.shape
    weights = lhy.pp.weights.reshape(s[1:])/lhy.pp.weights.sum()
    plt.imshow(np.rot90(weights), interpolation='nearest',
               cmap='gist_heat', aspect='auto', origin='upper',
               extent=[np.log10(1), np.log10(17.7828), -1.9, 0.45])
    plt.colorbar()
    plt.title("Mass Fraction")
    plt.xlabel("log$_{10}$ Age (Gyr)")
    plt.ylabel("[M/H]")
    plt.tight_layout()
    plt.show()
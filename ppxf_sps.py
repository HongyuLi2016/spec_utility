#!/usr/bin/env python
import numpy as np
import pyfits
import glob
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

    def load_miles(self, FWHM_gal, path='miles_models'):
        '''
        load miles stellar library, convolve with observed resolusion
        (FWHM_gal), and log rebin using velscale
        '''
        velscale = self.velscale
        vazdekis = glob.glob('{}/Mun1.30*.fits'.format(path))
        vazdekis.sort()
        FWHM_tem = 2.51
        try:
            hdu = pyfits.open(vazdekis[0])
        except:
            print 'Error - Can not open templates file'
            exit(0)
        ssp = hdu[0].data
        h2 = hdu[0].header
        lamRange_temp = h2['CRVAL1'] + \
            np.array([0., h2['CDELT1']*(h2['NAXIS1']-1)])
        sspNew, logLam2, velscale = util.log_rebin(lamRange_temp,
                                                   ssp, velscale=velscale)
        nAges = 26
        nMetal = 6
        templates = np.empty((sspNew.size, nAges, nMetal))
        FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
        sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels
        logAge_grid = np.empty((nAges, nMetal))
        metal_grid = np.empty((nAges, nMetal))
        logAge = np.linspace(np.log10(1), np.log10(17.7828), nAges)
        metal = [-1.71, -1.31, -0.71, -0.40, 0.00, 0.22]
        metal_str = ['m1.71', 'm1.31', 'm0.71', 'm0.40', 'p0.00', 'p0.22']
        for k, mh in enumerate(metal_str):
            files = [s for s in vazdekis if mh in s]
            for j, filename in enumerate(files):
                hdu = pyfits.open(filename)
                ssp = hdu[0].data
                ssp = ndimage.gaussian_filter1d(ssp, sigma)
                sspNew, logLam2, velscale = \
                    util.log_rebin(lamRange_temp, ssp, velscale=velscale)
                # Templates are *not* normalized here
                templates[:, j, k] = sspNew
                logAge_grid[j, k] = logAge[j]
                metal_grid[j, k] = metal[k]
        self.logAge_grid = logAge_grid
        self.metal_grid = metal_grid
        self.logLamT = logLam2
        self.templates = templates
        self.lamRange_temp = lamRange_temp
        # fit range used in ppxf

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
            print 'Error - fit range {:.1f} - {:.1f} is larger than template range {:.1f} - {:.1f}'.format(self.fit_range[0],
                    self.fit_range[1], self.lamRange_temp[0], self.lamRange_temp[1])
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

if __name__ == '__main__':
    data = np.loadtxt('vazdekis_ml_test.txt')
    wave = data[:, 0]
    c = 299792.458
    vscale = np.log(wave[-1] / wave[0]) / len(wave) * c
    flux = data[:, 1]
    lhy = ppxf_sps(vscale)
    good = flux > 1.0
    lhy.load_miles(2.8, path='/home/lhy/python/ppxf/miles_models')
    lhy.load_galaxy(wave, flux, error=flux/1000.0, fit_range=None)
    lhy.run(moments=4, mdegree=0)
    # plt.plot(lhy.logLamT, lhy.templates[:, 0, 0])
    plt.clf()
    plt.subplot(211)
    lhy.pp.plot()   # produce pPXF plot
    print lhy.pp.sol
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

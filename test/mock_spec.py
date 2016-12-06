#!/usr/bin/env
import spec_utils as su
from ppxf import reddening_curve  # Calzetti (2000)
import numpy as np
from ppxf_sps import ppxf_sps


def linexp_sfh(t, A=1.0, ti=1.0, tau=1.0):
    return (A * (t-ti) * np.exp(-(t-ti)/tau)).clip(0.0)


def whiteNoise(wave, A):
    size = len(wave)
    noise = np.random.normal(scale=A, size=size)
    return noise


def redden(wave, flux, ebv):
    fratio = reddening_curve(wave, ebv)
    return flux * fratio


def stack(w, flux):
    flux = np.atleast_2d(flux)
    if len(w) != flux.shape[1]:
        print 'Error - length of weight must be equal to the'\
            ' number of templates'
        exit()
    if len(flux.shape) != 2:
        raise ValueError('Error - flux must be a 2d array')
    return (w*flux).sum(axis=1)


class miles:
    def __init__(self, age, metal, vscale=70.0, linear=True, FWHM=2.51):
        lhy = ppxf_sps(vscale)
        lhy.load_miles_all(FWHM, path='/home/lhy/python/ppxf/'
                                      'MILES_Padova00_un_1.30_fits')
        lhy.select_templates(age, metal)
        self.templates = lhy.templates.reshape(lhy.templates.shape[0], -1)
        self.logAge = lhy.logAge_grid.reshape(-1)
        self.metal = lhy.metal_grid.reshape(-1)
        self.ml = lhy.ml_r_grid.reshape(-1)
        self.nogasM = lhy.mnogas_grid.reshape(-1)
        self.wave = np.exp(lhy.logLamT)
        if linear:
            nwave = np.arange(np.ceil(self.wave[0]),
                              np.floor(self.wave[-1]), 1.0)
            ntemplates = np.zeros((len(nwave), self.templates.shape[1]))
            for i in range(len(self.logAge)):
                ntemplates[:, i] = su.resample(self.wave, nwave,
                                               self.templates[:, i])
            self.wave = nwave
            self.templates = ntemplates

    def stack(self, w):
        w = np.asarray(w)
        return stack(w, self.templates)

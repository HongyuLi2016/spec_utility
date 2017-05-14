#!/usr/bin/env python
import numpy as np
# import matplotlib.pyplot as plt
from optparse import OptionParser
from scipy.integrate import quad
from scipy import interpolate
import re
import spec_utils as su
import warnings
import os

warnings.simplefilter("ignore")


def findline(ff, line, imax=10000, p=False):
    i = 0
    while True:
        junk = ff.readline()
        if p:
            print junk
        i += 1
        if junk == line:
            break
        if i > imax:
            # print 'Warnning - Maxmum lines read!'
            break


L_sun = 3.826e33
L_sun_r = 2.607506e+32
L_ratio_r = L_sun / L_sun_r


class read:
    '''
    This is a class used for reading starlight output files
    Parameters:
      fobs_norm: normlisation factor
      chi2dof: chi2 per degree of freedom
      adev: absolute deviation of the synthesis spectrum
      sum_of_x: summition of the weight of all the templates
      Flux_tot: Total flux in input spectrum unit
      Mini_tot: used to calculate initial total mass
      Mcor_tot: used to calculate total mass
      v0_min: best-fitting velocity
      vd_min: best-fitting velocity dispersion
      AV_min: best-fiting dust extinction
      ID: template spectrum ID
      x_j: luminosity fraction of the jth template
      Mini_j: initial mass fraction of the jth template
      Mcor_j: current mass fraction of the jth template
      Age_j: Age of the jth template
      Z_j: metallicity of the jth tempalte
      LM_j: Tempalte spectrum value at the nomolazition wavelength
      wave: wavelenght array
      obs: observed spectrum
      syn: synthesis spectrum
      weight: weight for each pixel used in the fitting
      good: good pixels used in the fitting
    '''
    def __init__(self, fname):
        with open(fname, 'r') as ff:
            findline(ff, '## Normalization info\n')
            ff.readline()
            ff.readline()
            ff.readline()
            tem = ff.readline()
            pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
            self.fobs_norm = float(pat.search(tem).group(1))

            findline(ff, '## S/N\n')

            # read in synthesis results, line by line
            findline(ff, '## Synthesis Results - Best model ##\n')
            ff.readline()
            tem = ff.readline()
            pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
            self.chi2pdof = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([+-]?[0-9.]+)')
            self.adev = float(pat.search(tem).group(1))
            ff.readline()
            tem = ff.readline()
            pat = re.compile(r'([+-]?[0-9.]+)')
            self.sum_of_x = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
            self.Flux_tot = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
            self.Mini_tot = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([-+]?\d+\.\d*?[eE][-+]?\d+)')
            self.Mcor_tot = float(pat.search(tem).group(1))
            ff.readline()
            tem = ff.readline()
            pat = re.compile(r'([+-]?[0-9.]+)')
            self.v0_min = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([+-]?[0-9.]+)')
            self.vd_min = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([+-]?[0-9.]+)')
            self.AV_min = float(pat.search(tem).group(1))
            tem = ff.readline()
            pat = re.compile(r'([+-]?[0-9.]+)')
            self.YAV_min = float(pat.search(tem).group(1))
            # read in star formation history
            findline(ff, '# j     x_j(%)      Mini_j(%)     Mcor_j(%)  '
                     '   age_j(yr)     Z_j      (L/M)_j    YAV?  Mstars '
                     '  component_j        a/Fe...    '
                     '   SSP_chi2r SSP_adev(%)   SSP_AV   SSP_x(%)\n')
            ID = []
            x_j = []
            Mini_j = []
            Mcor_j = []
            Age_j = []
            Z_j = []
            LM_j = []
            YAV = []
            Mstars = []
            component_j = []
            alpha_Fe = []
            SSP_chi2r = []
            SSP_adec = []
            SSP_AV = []
            SSP_x = []
            while True:
                tem = ff.readline()
                if tem == '' or tem == '\n':
                    break
                pat = re.compile(r'(\d+) +([+-]?[0-9.]+) '
                                 '+([-+]?\d+\.\d*?[eE][-+]?\d+) '
                                 '+([-+]?\d+\.\d*?[eE][-+]?\d+) '
                                 '+([-+]?\d+\.\d*?[eE][-+]?\d+) '
                                 '+([+-]?[0-9.]+) '
                                 '+([-+]?\d+\.\d*?[eE][-+]?\d+) '
                                 '+(\d+) +([+-]?[0-9.]+) +(\S+) '
                                 '+([+-]?[0-9.]+) '
                                 '+([-+]?\d+\.\d*?[eE][-+]?\d+) '
                                 '+([+-]?[0-9.]+) +([+-]?[0-9.]+) '
                                 '+([+-]?[0-9.]+)')
                ID.append(int(pat.search(tem).group(1)))
                x_j.append(float(pat.search(tem).group(2)))
                Mini_j.append(float(pat.search(tem).group(3)))
                Mcor_j.append(float(pat.search(tem).group(4)))
                Age_j.append(float(pat.search(tem).group(5)))
                Z_j.append(float(pat.search(tem).group(6)))
                LM_j.append(float(pat.search(tem).group(7)))
                YAV.append(int(pat.search(tem).group(8)))
                Mstars.append(float(pat.search(tem).group(9)))
                component_j.append((pat.search(tem).group(10)))
                alpha_Fe.append(float(pat.search(tem).group(11)))
                SSP_chi2r.append(float(pat.search(tem).group(12)))
                SSP_adec.append(float(pat.search(tem).group(13)))
                SSP_AV.append(float(pat.search(tem).group(14)))
                SSP_x.append(float(pat.search(tem).group(15)))

            self.ID = np.array(ID)
            self.x_j = np.array(x_j)
            self.Mini_j = np.array(Mini_j)
            self.Mcor_j = np.array(Mcor_j)
            self.Age_j = np.array(Age_j)
            self.Z_j = np.array(Z_j)
            self.LM_j = np.array(LM_j)
            self.YAV = np.array(YAV)
            self.Mstars = np.array(Mstars)
            self.component_j = np.array(component_j)
            self.alpha_Fe = np.array(alpha_Fe)
            self.ml_r = self.alpha_Fe.copy()
            self.SSP_chi2r = np.array(SSP_chi2r)
            self.SSP_adec = np.array(SSP_adec)
            self.SSP_AV = np.array(SSP_AV)
            self.SSP_x = np.array(SSP_x)

            # read in synthetic spectrum
            findline(ff, '## Synthetic spectrum (Best Model)'
                     ' ##l_obs f_obs f_syn wei\n')
            ff.readline()
            wave = []
            obs = []
            syn = []
            weight = []
            pat = re.compile(r'([+-]?[0-9.]+) +([+-]?[0-9.]+)'
                             ' +([+-]?[0-9.]+) +([+-]?[0-9.]+)')
            while True:
                tem = ff.readline()
                if tem == '' or tem == '\n':
                    break
                wave.append(float(pat.search(tem).group(1)))
                obs.append(float(pat.search(tem).group(2)))
                syn.append(float(pat.search(tem).group(3)))
                weight.append(float(pat.search(tem).group(4)))
        self.wave = np.array(wave)
        self.obs = np.array(obs) * self.fobs_norm
        self.syn = np.array(syn) * self.fobs_norm
        self.weight = np.array(weight)
        self.good = self.weight > 0.0
        # plt.plot(self.wave,self.obs,'r')
        # plt.plot(self.wave,self.syn,'k')
        # plt.show()

    def weights(self, shape=(25, 6)):
        return self.Mini_j.reshape(shape[1], shape[0]).T

    def nogas_weights(self, shape=(25, 6)):
        return self.Mcor_j.reshape(shape[1], shape[0]).T

    def luminosity_weights(self, shape=(25, 6)):
        return self.x_j.reshape(shape[1], shape[0]).T

    def logAge_grid(self, shape=(25, 6)):
        return np.log10(self.Age_j.reshape(shape[1], shape[0]).T)

    def metal_grid(self, shape=(25, 6)):
        return np.log10(self.Z_j.reshape(shape[1], shape[0]).T/0.02)

    def get_ml_int(self):
        '''
        calculate the intrisic M*/L
        '''
        if self.ml_r.sum() == 0:
            # print 'Warning - M*/L of sps must be included in Basefile!'
            pass
        ml = np.sum(self.Mcor_j) / np.sum(self.Mcor_j / self.ml_r)
        return ml

    def get_ml_obs(self, Filter=None):
        '''
        calculate the observed M*/L
        filter should be n*2 array
        '''
        if Filter is None:
            print 'Error - Filter must be provided!'
            exit()
        filter_wave = Filter[:, 0]
        filter_flux = Filter[:, 1]
        f_filter = interpolate.interp1d(filter_wave, filter_flux,
                                        kind='linear', bounds_error=False,
                                        fill_value=0.0)
        ff = interpolate.interp1d(self.wave, self.obs, kind='linear',
                                  bounds_error=False,
                                  fill_value=0.0)

        def function(wave):
            return ff(wave) * f_filter(wave)

        I = quad(function, 5000.0, 8000.0)
        ml = self.Mcor_tot / I[0] / L_ratio_r
        return ml

    def plot(self, shape=(25, 6), fname='sps.png', parameters=None,
             filterPath='/home/lhy/python/ppxf/data/SDSS_r_filter',
             outfolder='figs_star'):
        os.system('mkdir -p {}'.format(outfolder))
        w_light = self.luminosity_weights(shape=shape)
        w_inimass = self.weights(shape=shape)
        logAge_grid = self.logAge_grid(shape=shape)
        metal_grid = self.metal_grid(shape=shape)

        if parameters is None:
            parameters = {'M<logAge>': self.MageLog(),
                          'L<logAge>': self.LageLog(),
                          'M<[Z/H]>': self.MZLog(),
                          'L<[Z/H]>': self.LZLog(), 'E(b-v)': self.ebv()}
            try:
                Filter = su.sdss_r_filter(filterPath)
                ml_obs = self.get_ml_obs(Filter)
                parameters['M*/L_obs_r'] = ml_obs
            except:
                print 'Warnning - Calculate obs ml faild!'
            try:
                ml_int = self.get_ml_int()
                parameters['M*/L_int_r'] = ml_int
            except:
                print 'Warnning - Calculate int ml faild!'

        good = np.where(self.weight > 0)[0]

        fig =\
            su.plot_sps(self.wave, self.obs, self.syn, good,
                        w_inimass/w_inimass.sum(), w_light/w_light.sum(),
                        np.unique(logAge_grid), np.unique(metal_grid),
                        parameters=parameters)
        if fname is not None:
            fig.savefig('{}/{}'.format(outfolder, fname), dpi=400)

    def MageLog(self):
        rst = np.average(np.log10(self.Age_j), weights=self.Mini_j)
        return rst

    def MZLog(self):
        rst = np.average(np.log10(self.Z_j / 0.02), weights=self.Mini_j)
        return rst

    def Mage(self):
        rst = np.average(self.Age_j, weights=self.Mini_j)
        return np.log10(rst)

    def MZ(self):
        rst = np.average(self.Z_j / 0.02, weights=self.Mini_j)
        return np.log10(rst)

    def LageLog(self):
        rst = np.average(np.log10(self.Age_j), weights=self.x_j)
        return rst

    def LZLog(self):
        rst = np.average(np.log10(self.Z_j / 0.02), weights=self.x_j)
        return rst

    def Lage(self):
        rst = np.average(self.Age_j, weights=self.x_j)
        return np.log10(rst)

    def LZ(self):
        rst = np.average(self.Z_j / 0.02, weights=self.x_j)
        return np.log10(rst)

    def ebv(self, Rv=4.05):
        return self.AV_min / Rv

    def dump(self, fname='spsout.dat', outfolder='dump_star',
             filterPath='/home/lhy/python/ppxf/data/SDSS_r_filter',
             shape=(25, 6)):
        os.system('mkdir -p {}'.format(outfolder))
        try:
            Filter = su.sdss_r_filter(filterPath)
            ml_obs = self.get_ml_obs(Filter)
        except:
            print 'Warnning - Calculate obs ml faild!'
            ml_obs = np.nan
        try:
            ml_int = self.get_ml_int()
        except:
            print 'Warnning - Calculate int ml faild!'
            ml_int = np.nan
        logAge_grid = self.logAge_grid(shape=shape)
        metal_grid = self.metal_grid(shape=shape)
        good = np.where(self.weight > 0)[0]
        su.ssp_dump_data(ml_obs, ml_int, self.ebv(), self.Mcor_tot,
                         self.MageLog(), self.MZLog(), self.Mage(), self.MZ(),
                         self.LageLog(), self.LZLog(), self.Lage(), self.LZ(),
                         self.wave, self.obs, self.syn, good,
                         self.weights(shape=shape) /
                         self.weights(shape=shape).sum(),
                         self.luminosity_weights(shape=shape) /
                         self.luminosity_weights(shape=shape).sum(),
                         logAge_grid, metal_grid,
                         fname=fname, outfolder=outfolder)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', action='store', type='string', dest='fname',
                      default=None, help='file name')
    (options, args) = parser.parse_args()
    lhy = read(options.fname)
    # lhy.plot(shape=(26, 6))
    # lhy.dump(shape=(26, 6))
    print lhy.Age_j
    # print lhy.SSP_x
    # print lhy.wave
    # print lhy.x_j
    # print lhy.Mini_j
    # print lhy.Mage()
    # print lhy.MZ()

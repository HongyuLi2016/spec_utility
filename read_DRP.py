#!/usr/bin/env python
import numpy as np
import pyfits
from optparse import OptionParser
import matplotlib.pyplot as plt
from PyAstronomy.pyasl import unred
import spec_utils as su
from voronoi_2d_binning import voronoi_2d_binning
import os
import warnings

warnings.simplefilter("ignore")

'''
read MaNGA DRP linecube, file format is discriped in
https://trac.sdss.org/wiki/MANGA/TRM/TRM_MPL-5/datamodel (6.5.3, 6.5.4)
'''


class read_DRP:
    def __init__(self, fname):
        hdulist = pyfits.open(fname)
        self.header = hdulist[0].header
        self.header1 = hdulist[1].header
        self.header2 = hdulist[5].header
        self.flux = hdulist[1].data
        self.ivar = hdulist[2].data
        self.mask = hdulist[3].data
        self.wave = hdulist[4].data
        self.specres = hdulist[5].data
        self.specresd = hdulist[6].data
        self.bsinfo = hdulist[7].data
        self.gimg = hdulist[8].data
        self.rimg = hdulist[9].data
        self.iimg = hdulist[10].data
        self.zimg = hdulist[11].data
        self.gpsf = hdulist[12].data
        self.rpsf = hdulist[13].data
        self.ipsf = hdulist[14].data
        self.zpsf = hdulist[15].data
        # self.gcorrel = hdulist[16].data
        # self.rcorrel = hdulist[17].data
        # self.icorrel = hdulist[18].data
        # self.zcorrel = hdulist[19].data

        # create position information
        self.IFURA = self.header['IFURA']
        self.IFUDEC = self.header['IFUDEC']
        self.OBJRA = self.header['OBJRA']
        self.OBJDEC = self.header['OBJDEC']
        self.PLATEIFU = self.header['PLATEIFU']
        self.ebvgal = self.header['EBVGAL']
        shape = self.rimg.shape
        xtem = np.arange(shape[0], dtype=float)
        ytem = np.arange(shape[1], dtype=float)

        xposition_tem, yposition_tem = np.meshgrid(xtem, ytem)
        xposition_tem -= self.header1['CRPIX1']
        yposition_tem -= self.header1['CRPIX2']
        self.xposition = self.header1['CD1_1'] * xposition_tem * 3600.
        self.yposition = self.header1['CD2_2'] * yposition_tem * 3600.

        good = self.mask == 0
        # print good
        # self.mask[~good] = 2
        self.good_spaxel = good.sum(axis=0)/float(len(self.mask)) > 0.8

        # print self.mask

    def deRedden(self, R_v=3.1):
        new_flux = np.zeros_like(self.flux)
        for i in range(self.flux.shape[1]):
            for j in range(self.flux.shape[2]):
                new_flux[:, i, j] = unred(self.wave, self.flux[:, i, j],
                                          self.ebvgal, R_V=R_v)
        self.flux = new_flux

    def deRedshift(self, z=0.03):
        new_flux = np.zeros_like(self.flux)
        new_error = np.zeros_like(self.flux)
        for i in range(self.flux.shape[1]):
            for j in range(self.flux.shape[2]):
                new_flux[:, i, j], new_error[:, i, j] = \
                    su.resample(self.wave/(1+z), self.wave,
                                self.flux[:, i, j], error=self.ivar[:, i, j])
        self.flux = new_flux
        self.ivar = new_error

    def SNR(self, window=[4730.0, 4780.0]):
        sn = np.empty(self.gimg.shape)
        in_window = (self.wave > window[0]) * (self.wave < window[1])
        for i in range(sn.shape[0]):
            for j in range(sn.shape[1]):
                if self.good_spaxel[i, j]:
                    sn[i, j] = np.nanmedian(self.flux[:, i, j][in_window]) /\
                        np.nanstd(self.flux[:, i, j][in_window])
                else:
                    sn[i, j] = np.nan
        self.sn = sn
        return sn

    def voronoiBin(self, targetSN=30, save=True, outFolder='spec',
                   binName='bin.dat', nodeName='node.dat', *args, **kwargs):
        os.system('mkdir -p {}'.format(outFolder))
        good = self.good_spaxel.reshape(-1)
        xbin = self.xposition.reshape(-1)[good]
        ybin = self.yposition.reshape(-1)[good]
        signal = self.sn.reshape(-1)[good]
        noise = np.ones_like(signal)
        binID = np.empty(self.gimg.shape, dtype=int)
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = \
            voronoi_2d_binning(xbin, ybin, signal, noise, targetSN,
                               plot=0, quiet=1, *args, **kwargs)
        binID.reshape(-1)[~good] = -1
        binID.reshape(-1)[good] = binNum
        self.binID = binID
        self.xNode = xNode
        self.yNode = yNode
        self.xBar = xBar
        self.yBar = yBar
        if save:
            f = open('{}/{}'.format(outFolder, binName), 'w')
            for i in range(binID.shape[0]):
                for j in range(binID.shape[1]):
                    print >>f, '{:5} {:5} {:5} {:6.2f} {:6.2f}'\
                        .format(i, j, binID[i, j], self.xposition[i, j],
                                self.yposition[i, j])
            f.close()

            f = open('{}/{}'.format(outFolder, nodeName), 'w')
            for i in range(len(xNode)):
                print >>f, '{:5} {:6.2f} {:6.2f} {:6.2f} {:6.2f}'\
                    .format(i, xNode[i], yNode[i], xBar[i], yBar[i])
            f.close()

        flist = open('{}/list.cat'.format(outFolder), 'w')
        for i in range(len(xNode)):
            print >>flist, 'spec_{:04d}.dat'.format(i)
            iBin = binID == i
            stackedSpec = self.flux[:, iBin].sum(axis=1)
            stackedErr = np.sqrt((1.0 / self.ivar[:, iBin]).sum(axis=1))
            good = (stackedErr > 0.0) * (stackedSpec > 0.0)
            stackedErr[~good] = 1.0
            f = open('{}/spec_{:04d}.dat'.format(outFolder, i), 'w')
            for j in range(len(stackedSpec)):
                flag = 1 if good[j] else 2
                print >>f, '{:12.4e}    {:12.4e}    {:12.4e}    {:4d}'\
                    .format(self.wave[j], stackedSpec[j], stackedErr[j], flag)
            f.close()
        flist.close()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', action='store', type='string', dest='fname',
                      default='/home/lhy/python/data/'
                      'manga-8319-1902-LINCUBE.fits',
                      help='filename')
    (options, args) = parser.parse_args()
    lhy = read_DRP(options.fname)
    lhy.SNR()
    lhy.voronoiBin()
    # plt.imshow(lhy.SNR())
    # print lhy.mask[:,0,0]
    # plt.plot(lhy.wave, lhy.flux[:,18,18],'k')
    # lhy.deRedden()
    # plt.plot(lhy.wave, lhy.flux[:,18,18],'r')
    # lhy.deRedshift(z=0.03)
    # plt.plot(lhy.wave, lhy.flux[:,18,18],'b')
    # plt.imshow((lhy.rimg))
    plt.show()
    # print lhy.header

#!/usr/bin/env python
import numpy as np
import pickle
flist = np.genfromtxt('lhy', dtype='S30')
# basic parameters
Re_arcsec = np.zeros(len(flist))
a = np.zeros(len(flist))
b = np.zeros(len(flist))
ba = np.zeros(len(flist))
pa = np.zeros(len(flist))
Ldist = np.zeros(len(flist))
Adist = np.zeros(len(flist))
z = np.zeros(len(flist))
# average values
ml_obs = np.zeros([len(flist), 3])
ml_int = np.zeros([len(flist), 3])
MlogAge = np.zeros([len(flist), 3])
LlogAge = np.zeros([len(flist), 3])
MZ = np.zeros([len(flist), 3])
LZ = np.zeros([len(flist), 3])
# gradient values
gml_obs = np.zeros(len(flist))
gml_int = np.zeros(len(flist))
gMlogAge = np.zeros(len(flist))
gLlogAge = np.zeros(len(flist))
gMZ = np.zeros(len(flist))
gLZ = np.zeros(len(flist))

for i in range(len(flist)):
    try:
        with open('{}/info.dat'.format(flist[i])) as f:
            info = pickle.load(f)
        with open('{}/properties/prop.dat'.format(flist[i])) as f:
            prop = pickle.load(f)
        Re_arcsec[i] = info['Re']
        a[i] = info['a']
        b[i] = info['b']
        ba[i] = info['ba']
        pa[i] = info['pa']
        Ldist[i] = info['Ldist']
        Adist[i] = info['Adist']
        z[i] = info['z']
        ml_obs[i, :] = prop['ml_obs']
        ml_int[i, :] = prop['ml_int']
        MlogAge[i, :] = prop['MlogAge']
        LlogAge[i, :] = prop['LlogAge']
        MZ[i, :] = prop['MZ']
        LZ[i, :] = prop['LZ']
        gml_obs[i] = prop['gml_obs']
        gml_int[i] = prop['gml_int']
        gMlogAge[i] = prop['gMlogAge']
        gLlogAge[i] = prop['gLlogAge']
        gMZ[i] = prop['gMZ']
        gLZ[i] = prop['gLZ']
        print flist[i]
    except:
        # print flist[i]
        Re_arcsec[i] = np.nan
        a[i] = np.nan
        b[i] = np.nan
        ba[i] = np.nan
        pa[i] = np.nan
        Ldist[i] = np.nan
        Adist[i] = np.nan
        z[i] = np.nan
        ml_obs[i, :] = np.nan
        ml_int[i, :] = np.nan
        MlogAge[i, :] = np.nan
        LlogAge[i, :] = np.nan
        MZ[i, :] = np.nan
        LZ[i, :] = np.nan
        gml_obs[i] = np.nan
        gml_int[i] = np.nan
        gMlogAge[i] = np.nan
        gLlogAge[i] = np.nan
        gMZ[i] = np.nan
        gLZ[i] = np.nan

zip_paras = {}
zip_paras['name'] = flist
zip_paras['Re'] = Re_arcsec
zip_paras['a'] = a
zip_paras['b'] = b
zip_paras['ba'] = ba
zip_paras['pa'] = pa
zip_paras['Ldist'] = Ldist
zip_paras['Adist'] = Adist
zip_paras['z'] = z
zip_paras['ml_obs'] = ml_obs
zip_paras['ml_int'] = ml_int
zip_paras['MlogAge'] = MlogAge
zip_paras['LlogAge'] = LlogAge
zip_paras['MZ'] = MZ
zip_paras['LZ'] = LZ
zip_paras['gml_obs'] = gml_obs
zip_paras['gml_int'] = gml_int
zip_paras['gMlogAge'] = gMlogAge
zip_paras['gLlogAge'] = gLlogAge
zip_paras['gMZ'] = gMZ
zip_paras['gLZ'] = gLZ
# zip_paras[''] =
with open('zip_paras.dat', 'wb') as f:
    pickle.dump(zip_paras, f)
    print zip_paras['ml_obs'][:, 1]

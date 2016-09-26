#!/usr/bin/evn python
import numpy as np
from scipy.interpolate import interp1d

def resample(wave,newWave,flux,error=None,interpKind=None,fill_value=0.0):
  if interpKind is None:
    interpKind = 'linear'
  fSpec = interp1d(wave,flux,kind = interpKind,fill_value=fill_value,bounds_error=False)
  newFlux = fSpec(newWave)
  if error is not None:
    fError = interp1d(wave,error,kind = interpKind,fill_value=fill_value,bounds_error=False)
    newError = fError(newWave)
    return newFlux,newError
  else:
    return newFlux

def bit_mask(mask_array,bit):
  '''
  For dap MPL5 velocity map files
  30: DONOTUSE        Do not use this spaxel for science
  19: MULTICOMP       Multi-component velocity features present
  9: NOCORRECTION    Appropriate correction not available
  8: NEARBOUND       Fitted value is too near an imposed boundary; see TRM
  7: FITFAILED       Attempted fit for property failed
  6: MATHERROR       Mathematical error in computing value
  5: UNRELIABLE      Value is deemed unreliable; see TRM for definition
  4: NOVALUE         Spaxel was not fit because it did not meet selection criteria
  3: FORESTAR        Foreground star
  2: DEADFIBER       Major contributing fiber is dead
  1: LOWCOV          Low coverage in this spaxel
  0: NOCOV           No coverage in this spaxel
  '''
  return mask_array & 2**bit > 0


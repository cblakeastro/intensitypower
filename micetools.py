import numpy as np
import healpy as hp
from astropy.io import fits
from sklearn.decomposition import FastICA

########################################################################
# Read in MICE intensity map.                                          #
########################################################################

def readmiceintmap(nzbin,nside,dzbin,zmin,zmax,dofg):
  stem = '/Users/cblake/Data/hipower/stevesim/'
  print '\nReading in MICE intensity mapping data...'
  skyfile = stem + 'skymaskMICEcc-nside128.npy'
  print skyfile
  insky = np.load(skyfile)
  npix = hp.nside2npix(nside)
  ipixlst = np.arange(npix)[insky]
  npix = len(ipixlst)
  iz1,iz2 = int(np.rint((zmin-0.2)/dzbin)),int(np.rint((zmax-0.2)/dzbin))
  if (dofg):
    print 'Including FGs...'
    hifile = stem + 'dT_obs-MICE_withcc.npy'
  else:
    print 'No FGs...'
    hifile = stem + 'dT_HI-MICE240zbins_withrsd_withcc.npy'
  print hifile
  data = np.load(hifile)
  denshpz = data[iz1:iz2,insky]
  zbin = np.linspace(zmin+0.5*dzbin,zmax-0.5*dzbin,nzbin)
  tmeanz = 0.0559 + 0.2324*zbin - 0.024*(zbin**2) # in mK
  winhpz = np.empty((nzbin,npix))
  for iz in range(nzbin):
    winhpz[iz,:] = np.repeat(tmeanz[iz],npix)
  return denshpz,winhpz,ipixlst,npix

########################################################################
# Read in MICE galaxy catalogue.                                       #
########################################################################

def readmicegalcat(zmin,zmax):
  infile = '/Users/cblake/Data/hipower/stevesim/MICEv2-ELGs_withcc.fits'
  print '\nReading in MICE ELG data...'
  print infile
  hdulist = fits.open(infile)
  table = hdulist[1].data
  ras = table.field('ra')
  dec = table.field('dec')
  red = table.field('z')
  hdulist.close()
  ngal = len(ras)
  print ngal,'galaxies read in'
  cut = (red > zmin) & (red < zmax)
  ras,dec,red = ras[cut],dec[cut],red[cut]
  ngal = len(ras)
  print ngal,'galaxies with',zmin,'< z <',zmax
  return ras,dec,red,ngal

########################################################################
# FASTICA foreground subtraction code from Steve Cunnington.           #
########################################################################

def cleanFG(dathpz,N_IC):
  print '\nCleaning Foregrounds...'
  dathpz_clean = FASTICAclean(dathpz, N_IC)
  return dathpz_clean

def FASTICAclean(Input, N_IC):
  Input = np.swapaxes(Input,0,1) #Put in [npix, nz] form which is req'd for FASTICA
  ica = FastICA(n_components=N_IC, whiten=True)
  S_ = ica.fit_transform(Input) # Reconstruct signals
  A_ = ica.mixing_ # Get estimated mixing matrix
  Recon_FG = np.dot(S_, A_.T) + ica.mean_ #Reconstruct foreground
  Residual = Input - Recon_FG #Residual of fastICA is HI plus any Noise
  CleanFullSky = Residual #rebuild full sky array
  CleanFullSky = np.swapaxes(CleanFullSky,0,1) #return to [nz, npix] form
  return CleanFullSky

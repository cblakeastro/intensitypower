########################################################################
# Code to build mock galaxy and intensity mapping datasets from a      #
# dark matter simulation, and measure and model their auto- and cross- #
# power spectrum multipoles.                                           #
#                                                                      #
# The mock intensity mapping dataset includes:                         #
# - angular pixelization using healpix                                 #
# - binning in redshift channels                                       #
# - smoothing by a Gaussian telescope beam                             #
# - addition of a Gaussian noise in each cell                          #
#                                                                      #
# The mock galaxy and intensity mapping datasets are transferred to    #
# an FFT cuboid.                                                       #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import boxtools
import sphertools
import hppixtogrid
import measpk

def main():

########################################################################
# Parameters.                                                          #
########################################################################

# Parameters for survey cone
  rmin = 165.         # Minimum R.A. of survey [deg]
  rmax = 195.         # Maximum R.A. of survey [deg]
  dmin = -15.         # Minimum Dec. of survey [deg]
  dmax = 15.          # Maximum Dec. of survey [deg]
  zmin = 0.3          # Minimum redshift of survey
  zmax = 0.7          # Maximum redshift of survey
# Parameters applied to intensity map
  nside = 128         # healpix resolution for angular pixelization
  nzbin = 160         # number of redshift bins
  sigdeg = 0.25       # Standard deviation of telescope beam [deg]
  signoise = 1.       # Gaussian noise per cell
  ngrid = 128         # Grid size of FFT
# Parameters for power spectrum estimation
  kmin = 0.           # Minimum wavenumber [h/Mpc]
  kmax = 0.3          # Maximum wavenumber [h/Mpc]
  nkbin = 15          # Number of Fourier bins
# Parameters for cosmological model
  om = 0.273          # Matter density
  bgal = 1.           # Bias of galaxy sample
  bdens = 1.          # Bias of intensity sample
  betagal = om**0.55  # RSD distortion parameter of galaxy sample
  betadens = om**0.55 # RSD distortion parameter of intensity sample
  sigv = 400.         # Pairwise velocity dispersion
  pkmodfile = 'pkcambhalofit_zeq0_gigglez.dat' # Model power spectrum

########################################################################
# Initializations.                                                     #
########################################################################

  cosmo = FlatLambdaCDM(H0=100.,Om0=om)
  data = np.loadtxt(pkmodfile)
  kmod,pkmod = data[:,0],data[:,1]
  nx,ny,nz = ngrid,ngrid,ngrid
  dobound = True
  dzbin = (zmax-zmin)/float(nzbin)
  zlims = np.linspace(zmin,zmax,nzbin+1)

########################################################################
# Get list of healpix pixels falling within survey boundaries.         #
########################################################################

  ipixlst,npix = sphertools.getipixlst(nside,dobound,rmin,rmax,dmin,dmax)

########################################################################
# Get angular healpix window function.                                 #
########################################################################

  data = np.loadtxt('pixwin_nside128.dat')
  lwin,pixwin = data[:,0],data[:,1]

########################################################################
# Read in dark matter simulation data.                                 #
########################################################################

  dxpos,dypos,dzpos,dxvel,dyvel,dzvel,ngal = boxtools.readgiggquick()
  lgrid = 1000.

########################################################################
# Dimensions of FFT cuboid for embedding the survey cone.              #
########################################################################

  lx,ly,lz,x0,y0,z0 = boxtools.boxsize(zmin,zmax,dobound,rmin,rmax,dmin,dmax,cosmo)

########################################################################
# Apply redshift-space distortions to the dark matter catalogue.       #
########################################################################

  dxpos,dypos,dzpos = boxtools.applyrsd(dxpos,dypos,dzpos,dxvel,dyvel,dzvel,lgrid,lgrid,lgrid,x0,y0,z0)

########################################################################
# Split the data into two samples (1st = density, 2nd = galaxy).       #
########################################################################

  dxpos,dypos,dzpos,ngal,dxpos2,dypos2,dzpos2,ngal2 = boxtools.makecrosssample(dxpos,dypos,dzpos,ngal)

########################################################################
# Cut dataset to embedding cuboid.                                     #
########################################################################

  cut = (dxpos < lx) & (dypos < ly) & (dzpos < lz)
  dxpos,dypos,dzpos = dxpos[cut],dypos[cut],dzpos[cut]
  ngal = len(dxpos)
  cut = (dxpos2 < lx) & (dypos2 < ly) & (dzpos2 < lz)
  dxpos2,dypos2,dzpos2 = dxpos2[cut],dypos2[cut],dzpos2[cut]
  ngal2 = len(dxpos2)
  vol,nc = lx*ly*lz,nx*ny*nz
  vpix,ndens = vol/nc,float(ngal)/vol
  sigshotnoise = 1./np.sqrt(ndens*vpix)

########################################################################
# Convert galaxy and intensity particles to (R.A.,Dec.,redshift).      #
########################################################################

  dras,ddec,dred = boxtools.getradecred(dobound,rmin,rmax,dmin,dmax,dxpos,dypos,dzpos,x0,y0,z0,cosmo)
  dras2,ddec2,dred2 = boxtools.getradecred(dobound,rmin,rmax,dmin,dmax,dxpos2,dypos2,dzpos2,x0,y0,z0,cosmo)

########################################################################
# Cut datasets to survey cone.                                         #
########################################################################

  cut = (dras > rmin) & (dras < rmax) & (ddec > dmin) & (ddec < dmax) & (dred > zmin) & (dred < zmax)
  dxpos,dypos,dzpos = dxpos[cut],dypos[cut],dzpos[cut]
  dras,ddec,dred = dras[cut],ddec[cut],dred[cut]
  ngal = len(dras)
  print 'Cut to',ngal,'objects'
  cut = (dras2 > rmin) & (dras2 < rmax) & (ddec2 > dmin) & (ddec2 < dmax) & (dred2 > zmin) & (dred2 < zmax)
  dxpos2,dypos2,dzpos2 = dxpos2[cut],dypos2[cut],dzpos2[cut]
  ngal2 = len(dxpos2)
  print 'Cut to',ngal2,'objects'

########################################################################
# Create galaxy number distribution on FFT grid.                       #
########################################################################

  galgrid = boxtools.discret(dxpos2,dypos2,dzpos2,nx,ny,nz,lx,ly,lz,0.,0.,0.)
        
########################################################################
# Determine noise power spectrum, including both the shot noise        #
# component and the applied noise.                                     #
########################################################################

  signoisez = sphertools.getsignoisez(nside,nzbin,zlims,sigshotnoise,vpix,cosmo)
  pknoise = sphertools.calcpknoisehpz(nside,nzbin,zlims,sigshotnoise,True,signoisez,npix,cosmo)
  signoisez = sphertools.getsignoisez(nside,nzbin,zlims,signoise,vpix,cosmo)
  pknoise += sphertools.calcpknoisehpz(nside,nzbin,zlims,signoise,True,signoisez,npix,cosmo)

########################################################################
# Window functions in the FFT grid and (redshift,healpix) grid.        #
########################################################################

  wingridgal = boxtools.getmodwingrid(nx,ny,nz,lx,ly,lz,x0,y0,z0,dobound,rmin,rmax,dmin,dmax,zmin,zmax,cosmo)
  wingriddens = np.copy(wingridgal)
  winhpz = sphertools.getmodwinhpz(nside,nzbin,zlims,npix,ndens,cosmo)

########################################################################
# Create overdensity field in (redshift,healpix) grid.                 #
########################################################################

  denshpz = sphertools.gethpz(dras,ddec,dred,nside,ipixlst,nzbin,zlims)
  denshpz = (np.sum(winhpz)/np.sum(denshpz))*(denshpz/winhpz) - 1.

########################################################################
# Add noise to the field.                                              #
########################################################################

  denshpz += sphertools.getnoisehpz(nzbin,ipixlst,signoise,True,signoisez)

########################################################################
# Convolve the field with the telescope beam.                          #
########################################################################

  denshpz = sphertools.convhpzbeam(nside,ipixlst,nzbin,denshpz,sigdeg)

########################################################################
# Convert the field from the (redshift,healpix) map to the FFT grid.   #
########################################################################

  densgrid,wingriddens = hppixtogrid.hppixtogrid(nzbin,nside,denshpz,ipixlst,zlims,dobound,rmin,rmax,dmin,dmax,nx,ny,nz,lx,ly,lz,x0,y0,z0,cosmo)

########################################################################
# Measure the auto- and cross-power spectra.                           #
########################################################################

  measpk.measpk(galgrid,wingridgal,densgrid,wingriddens,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,kmod,pkmod,betagal,betadens,sigv,bgal,bdens,pknoise,sigdeg,lwin,pixwin,dzbin,cosmo)
  return

if __name__ == '__main__':
  main()

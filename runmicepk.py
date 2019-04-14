########################################################################
# Code to build mock galaxy and intensity mapping datasets from the    #
# MICE simulation, and measure and model their auto- and cross-        #
# power spectrum multipoles.                                           #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
from astropy.cosmology import FlatLambdaCDM
import boxtools
import micetools
import sphertools
import hppixtogrid
import measpk
import pktools

def main():

########################################################################
# Parameters.                                                          #
########################################################################

# Parameters for survey cone
  rmin = 0.          # Minimum R.A. of survey [deg]
  rmax = 90.         # Maximum R.A. of survey [deg]
  dmin = 30.         # Minimum Dec. of survey [deg]
  dmax = 90.         # Maximum Dec. of survey [deg]
  zmin = 0.2         # Minimum redshift of survey
  zmax = 0.6         # Maximum redshift of survey
# Parameters applied to intensity map
  nside = 128        # Healpix resolution for angular pixelization
  dzbin = 0.005      # Width of redshift bins
  ngrid = 256        # Grid size of FFT
  dobeam = True      # Convolve with telescope beam
  sigdeg = 1.        # Standard deviation of telescope beam [deg]
  donoise = True     # Include noise in temperature map
  signoise = 0.3     # Gaussian noise per cell
  dofg = False       # Read in data with foregrounds, and apply subtraction
# Parameters applied to galaxy map
  winfile = 'winx_MICEv2-ELGs.dat' # Window function
# Parameters for power spectrum estimation
  doconv = False     # Determine full convolution with window
  kmin = 0.          # Minimum wavenumber [h/Mpc]
  kmax = 0.3         # Maximum wavenumber [h/Mpc]
  nkbin = 30         # Number of Fourier bins
  binopt = 1         # 1) power spectrum multipoles
                     # 2) power spectrum wedges P(k,mu)
                     # 3) 2D power spectrum P(k_perp,k_par)
  nwedge = 4         # Number of wedges if binopt=2
  kmin2 = 0.         # Minimum wavenumber for 2D power if binopt=3
  kmax2 = 0.18       # Maximum wavenumber for 2D power if binopt=3
  nk2d = 9           # Number of bins for 2D power if binopt=3
# Parameters for cosmological model
  om = 0.25          # Matter density
  bgal = 1.17        # Bias of galaxy sample
  bdens = 1.09       # Bias of intensity sample
# RSD distortion parameters
  cosmo = FlatLambdaCDM(H0=100.,Om0=om)
  zeff = 0.5*(zmin+zmax)
  ffid = cosmo.Om(zeff)**0.55
  betagal = ffid/bgal  # RSD distortion parameter of galaxy sample
  betadens = ffid/bdens# RSD distortion parameter of intensity sample
  sigvgal = 260.       # Pairwise velocity dispersion of galaxies [km/s]
  sigvdens = 440.      # Pairwise velocity dispersion of HI [km/s]
  pkmodfile = 'pkcambhalofit_zeq0pt4_mice.dat' # Model power spectrum

########################################################################
# Initializations.                                                     #
########################################################################

  data = np.loadtxt(pkmodfile)
  kmod,pkmod = data[:,0],data[:,1]
  nx,ny,nz = ngrid,ngrid,ngrid
  dobound = False
  nzbin = int(np.rint((zmax-zmin)/dzbin))
  zlims = np.linspace(zmin,zmax,nzbin+1)

########################################################################
# Get angular healpix window function.                                 #
########################################################################

  data = np.loadtxt('pixwin_nside128.dat')
  lwin,pixwin = data[:,0],data[:,1]

########################################################################
# Dimensions of FFT cuboid for embedding the survey cone.              #
########################################################################

  lx,ly,lz,x0,y0,z0 = boxtools.boxsize(zmin,zmax,dobound,rmin,rmax,dmin,dmax,cosmo)
  vol,nc = lx*ly*lz,nx*ny*nz
  vpix = vol/nc

########################################################################
# Read in MICE intensity mapping data.                                 #
########################################################################

  denshpz,winhpz,ipixlst,npix = micetools.readmiceintmap(nzbin,nside,dzbin,zmin,zmax,dofg)

########################################################################
# Determine noise power spectrum.                                      #
########################################################################

  if (donoise):
    signoisez = sphertools.getsignoisez(nside,nzbin,zlims,signoise,vpix,cosmo)
    pknoise = sphertools.calcpknoisehpz(nside,nzbin,zlims,signoise,True,signoisez,npix,cosmo)
  else:
    pknoise = 0.

########################################################################
# Add noise to the field.                                              #
########################################################################

  if (donoise):
    denshpz += sphertools.getnoisehpz(nzbin,ipixlst,signoise,True,signoisez)

########################################################################
# Convolve the field with the telescope beam.                          #
########################################################################

  if (dobeam):
    denshpz = sphertools.convhpzbeam(nside,ipixlst,nzbin,denshpz,sigdeg)

########################################################################
# Run foreground subtraction.                                          #
########################################################################

  if (dofg):
    denshpz = micetools.cleanFG(denshpz,N_IC=4)

########################################################################
# Convert the field from the (redshift,healpix) map to the FFT grid.   #
########################################################################

  densgrid,wingriddens = hppixtogrid.hppixtogrid(nzbin,nside,denshpz,winhpz,ipixlst,zlims,dobound,rmin,rmax,dmin,dmax,nx,ny,nz,lx,ly,lz,x0,y0,z0,cosmo)

########################################################################
# Read in and grid MICE galaxy data and window function.               #
########################################################################

  ras,dec,red,ngal = micetools.readmicegalcat(zmin,zmax)
  dxpos,dypos,dzpos = boxtools.getxyz(dobound,rmin,rmax,dmin,dmax,ras,dec,red,cosmo)
  galgrid = boxtools.discret(dxpos,dypos,dzpos,nx,ny,nz,lx,ly,lz,x0,y0,z0)
  wingridgal = boxtools.readwin(winfile,nx,ny,nz,lx,ly,lz)

########################################################################
# Measure the auto- and cross-power spectrum multipoles.               #
########################################################################

  pk0gal,pk2gal,pk4gal,pk0dens,pk2dens,pk4dens,pk0cross,pk2cross,pk4cross,pk0errgal,pk2errgal,pk4errgal,pk0errdens,pk2errdens,pk4errdens,pk0errcross,pk2errcross,pk4errcross,pk0congal,pk2congal,pk4congal,pk0condens,pk2condens,pk4condens,pk0concross,pk2concross,pk4concross = measpk.measpk(doconv,galgrid,wingridgal,densgrid,wingriddens,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,kmod,pkmod,betagal,betadens,sigvgal,sigvdens,bgal,bdens,pknoise,dobeam,sigdeg,lwin,pixwin,dzbin,cosmo)

########################################################################
# Read in power spectrum dataset (if it has been written out).         #
########################################################################

#  pkfile = 'pkpole.dat'
#  kmin,kmax,nkbin,kbin,pk0gal,pk2gal,pk4gal,pk0dens,pk2dens,pk4dens,pk0cross,pk2cross,pk4cross,pk0errgal,pk2errgal,pk4errgal,pk0errdens,pk2errdens,pk4errdens,pk0errcross,pk2errcross,pk4errcross,pk0modgal,pk2modgal,pk4modgal,pk0moddens,pk2moddens,pk4moddens,pk0modcross,pk2modcross,pk4modcross,pk0congal,pk2congal,pk4congal,pk0condens,pk2condens,pk4condens,pk0concross,pk2concross,pk4concross = pktools.readpolecross(pkfile)

########################################################################
# Convert from multipoles to P(k,mu) or P(kperp,kpar).                 #
########################################################################

  if ((binopt == 2) or (binopt == 3)):
    if (binopt == 2):
      nmu = nwedge
    else:
      nmu = 10
    pkmugal,pkmuerrgal,pkmucongal = pktools.pkpoletopkmu(nmu,pk0gal,pk2gal,pk4gal,pk0errgal,pk2errgal,pk4errgal,pk0congal,pk2congal,pk4congal)
    pkmudens,pkmuerrdens,pkmucondens = pktools.pkpoletopkmu(nmu,pk0dens,pk2dens,pk4dens,pk0errdens,pk2errdens,pk4errdens,pk0condens,pk2condens,pk4condens)
    pkmucross,pkmuerrcross,pkmuconcross = pktools.pkpoletopkmu(nmu,pk0cross,pk2cross,pk4cross,pk0errcross,pk2errcross,pk4errcross,pk0concross,pk2concross,pk4concross)
    if (binopt == 3):
      pk2dgal,pk2derrgal,pk2dcongal = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nmu,pkmugal,pkmuerrgal,pkmucongal)
      pk2ddens,pk2derrdens,pk2dcondens = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nmu,pkmudens,pkmuerrdens,pkmucondens)
      pk2dcross,pk2derrcross,pk2dconcross = pktools.pkmutopk2(kmin2,kmax2,nk2d,kmin,kmax,nkbin,nmu,pkmucross,pkmuerrcross,pkmuconcross)

########################################################################
# Plot power spectra.                                                  #
########################################################################

  ncase = 3
  labelcase = ['$P_{gg}$','$P_{TT}$','$P_{gT}$']
  if (binopt == 1):
    pk0case,pk0errcase,pk0concase,pk2case,pk2errcase,pk2concase,pk4case,pk4errcase,pk4concase = np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin)),np.empty((ncase,nkbin))
    pk0case[0,:],pk0errcase[0,:],pk0concase[0,:] = pk0gal,pk0errgal,pk0congal
    pk2case[0,:],pk2errcase[0,:],pk2concase[0,:] = pk2gal,pk2errgal,pk2congal
    pk4case[0,:],pk4errcase[0,:],pk4concase[0,:] = pk4gal,pk4errgal,pk4congal
    pk0case[1,:],pk0errcase[1,:],pk0concase[1,:] = pk0dens,pk0errdens,pk0condens
    pk2case[1,:],pk2errcase[1,:],pk2concase[1,:] = pk2dens,pk2errdens,pk2condens
    pk4case[1,:],pk4errcase[1,:],pk4concase[1,:] = pk4dens,pk4errdens,pk4condens
    pk0case[2,:],pk0errcase[2,:],pk0concase[2,:] = pk0cross,pk0errcross,pk0concross
    pk2case[2,:],pk2errcase[2,:],pk2concase[2,:] = pk2cross,pk2errcross,pk2concross
    pk4case[2,:],pk4errcase[2,:],pk4concase[2,:] = pk4cross,pk4errcross,pk4concross
    pktools.plotpkpole(kmin,kmax,nkbin,pk0case,pk0errcase,pk0concase,pk2case,pk2errcase,pk2concase,pk4case,pk4errcase,pk4concase,labelcase)
  elif (binopt == 2):
    pkmucase,pkmuerrcase,pkmuconcase = np.empty((ncase,nkbin,nwedge)),np.empty((ncase,nkbin,nwedge)),np.empty((ncase,nkbin,nwedge))
    pkmucase[0,:,:],pkmuerrcase[0,:,:],pkmuconcase[0,:,:] = pkmugal,pkmuerrgal,pkmucongal
    pkmucase[1,:,:],pkmuerrcase[1,:,:],pkmuconcase[1,:,:] = pkmudens,pkmuerrdens,pkmucondens
    pkmucase[2,:,:],pkmuerrcase[2,:,:],pkmuconcase[2,:,:] = pkmucross,pkmuerrcross,pkmuconcross
    pktools.plotpkwedge(nwedge,kmin,kmax,nkbin,pkmucase,pkmuerrcase,pkmuconcase,labelcase)
  elif (binopt == 3):
    pk2dcase,pk2derrcase,pk2dconcase = np.empty((ncase,nk2d,nk2d)),np.empty((ncase,nk2d,nk2d)),np.empty((ncase,nk2d,nk2d))
    pk2dcase[0,:,:],pk2derrcase[0,:,:],pk2dconcase[0,:,:] = pk2dgal,pk2derrgal,pk2dcongal
    pk2dcase[1,:,:],pk2derrcase[1,:,:],pk2dconcase[1,:,:] = pk2ddens,pk2derrdens,pk2dcondens
    pk2dcase[2,:,:],pk2derrcase[2,:,:],pk2dconcase[2,:,:] = pk2dcross,pk2derrcross,pk2dconcross
    pktools.plotpk2d(kmin2,kmax2,nk2d,pk2dcase,pk2derrcase,pk2dconcase,labelcase)
  return

if __name__ == '__main__':
  main()

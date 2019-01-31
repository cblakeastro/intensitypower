import numpy as np
import numpy_indexed as npi
import healpy as hp

########################################################################
# Bin (R.A., Dec., redshift) catalogue in (redshift,healpix) grid.     #
########################################################################

def gethpz(ras,dec,red,nside,ipixlst,nzbin,zlims):
  npix = len(ipixlst)
  print '\nMaking healpix grid of nside =',nside,'with',npix,'pixels in',nzbin,'slices...'
  phi,theta = np.radians(ras),np.radians(90.-dec)
  ipix = hp.ang2pix(nside,theta,phi)
  izbin = np.digitize(red,zlims) - 1
  cut = (np.isin(ipix,ipixlst)) & (izbin >= 0) & (izbin < nzbin)
  ipix,izbin = ipix[cut],izbin[cut]
  ipix = npi.indices(ipixlst,ipix)
  dathpz,edges = np.histogramdd(np.vstack([izbin,ipix]).transpose(),bins=(nzbin,npix))
  return dathpz

########################################################################
# Sample noise realization on (redshift,healpix) grid.                 #
########################################################################

def getnoisehpz(nzbin,ipixlst,sig,donoisez,signoisez):
  print '\nMaking Gaussian noise realization on healpix...'
  if (donoisez):
    print 'Applying z-dependent noise =',signoisez
  else:
    print 'Applying constant noise =',sig
  npix = len(ipixlst)
  noisehpz = np.empty((nzbin,npix))
  for iz in range(nzbin):
    if (donoisez):
      sig1 = signoisez[iz]
    else:
      sig1 = sig
    noisehpz[iz,:] = sig1*np.random.normal(size=npix)
  return noisehpz

########################################################################
# Get list of healpix pixels within given (R.A.,Dec.) boundaries.      #
########################################################################

def getipixlst(nside,dobound,rmin,rmax,dmin,dmax):
  npix = hp.nside2npix(nside)
  ipix = np.arange(npix)
  theta,phi = hp.pix2ang(nside,ipix)
  if (dobound):
    print '\nCutting to subset of pixels...'
    print '{:5.1f}'.format(rmin) + ' < R.A. < ' + '{:5.1f}'.format(rmax)
    print '{:5.1f}'.format(dmin) + ' < Dec. < ' + '{:5.1f}'.format(dmax)
    angres = np.degrees(hp.nside2resol(nside))
    print 'angres = ' + '{:4.2f}'.format(angres)
    cosd = np.cos(np.radians(0.5*(dmin+dmax)))
    rmin1,rmax1 = rmin+angres/cosd,rmax-angres/cosd
    dmin1,dmax1 = dmin+angres,dmax-angres
    print 'Range for pixel centres restricted to:'
    print '{:5.1f}'.format(rmin1) + ' < R.A. < ' + '{:5.1f}'.format(rmax1)
    print '{:5.1f}'.format(dmin1) + ' < Dec. < ' + '{:5.1f}'.format(dmax1)
    ras,dec = np.degrees(phi),90.-np.degrees(theta)
    cut = (ras > rmin1) & (ras < rmax1) & (dec > dmin1) & (dec < dmax1)
    ipixlst = ipix[cut]
    print 'Cut to',len(ipixlst),'/',npix,'pixels'
    npix = len(ipixlst)
  else:
    ipixlst = np.arange(npix)
  return ipixlst,npix

########################################################################
# Convolve (redshift,healpix) map with a telescope beam.               #
########################################################################

def convhpzbeam(nside,ipixlst,nzbin,dathpz,sigdeg):
  print '\nConvolving map with a beam...'
  print 'sigdeg =',sigdeg
  npixall = hp.nside2npix(nside)
  npix = len(ipixlst)
  sigrad = np.radians(sigdeg)
  dathpzconv = np.empty((nzbin,npix))
  for iz in range(nzbin):
    datallsky = np.zeros(npixall)
    datallsky[ipixlst] = dathpz[iz,:]
    datallsky = hp.sphtfunc.smoothing(datallsky,sigma=sigrad,verbose=False)
    dathpzconv[iz,:] = datallsky[ipixlst]
  return dathpzconv

########################################################################
# Determine redshift-dependent noise associated with a constant noise  #
# per unit volume.                                                     #
########################################################################

def getsignoisez(nside,nzbin,zlims,signoise,vpix,cosmo):
  print '\nDetermining z-dependent noise...'
  print 'vpixfid =',vpix,'signoise =',signoise
  pixarea = hp.nside2pixarea(nside)
  vpixz,signoisez = np.empty(nzbin),np.empty(nzbin)
  for iz in range(nzbin):
    z1,z2 = zlims[iz],zlims[iz+1]
    vpixz[iz] = (cosmo.comoving_volume(z2).value-cosmo.comoving_volume(z1).value)*(pixarea/(4.*np.pi))
    signoisez[iz] = signoise*np.sqrt(vpix/vpixz[iz])
  return signoisez

########################################################################
# Determine noise power spectrum.                                      #
########################################################################

def calcpknoisehpz(nside,nzbin,zlims,signoise,donoisez,signoisez,npix,cosmo):
  print '\nComputing noise power over healpix...'
  print 'donoisez =','{:1d}'.format(donoisez)
  print 'nzbin =',nzbin
  print 'npix =',npix
  if (donoisez):
    print 'signoisez =',signoisez
  else:
    print 'signoise =',signoise
  pixarea = hp.nside2pixarea(nside)
  pknoise,vsurv = 0.,0.
  for iz in range(nzbin):
    if (donoisez):
      sig1 = signoisez[iz]
    else:
      sig1 = signoise
    z1,z2 = zlims[iz],zlims[iz+1]
    vpix = (cosmo.comoving_volume(z2).value-cosmo.comoving_volume(z1).value)*(pixarea/(4.*np.pi))
    pknoise1 = npix*(vpix**2)*(sig1**2)
    vsurv1 = npix*vpix
    pknoise += pknoise1
    vsurv += vsurv1
  pknoise /= vsurv
  print 'Noise power =',pknoise
  return pknoise

########################################################################
# Determine 3D selection function associated with a constant galaxy    #
# number density.                                                      #
########################################################################

def getmodwinhpz(nside,nzbin,zlims,npix,ndens,cosmo):
  pixarea = hp.nside2pixarea(nside)
  winhpz = np.empty((nzbin,npix))
  for iz in range(nzbin):
    z1,z2 = zlims[iz],zlims[iz+1]
    vpix = (cosmo.comoving_volume(z2).value-cosmo.comoving_volume(z1).value)*(pixarea/(4.*np.pi))
    winhpz[iz,:] = ndens*vpix
  return winhpz

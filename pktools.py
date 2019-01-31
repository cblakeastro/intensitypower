import sys
import numpy as np
from scipy.integrate import quad
from scipy.special import sph_harm
import matplotlib.pyplot as plt

########################################################################
# Obtain model RSD power spectrum.                                     #
########################################################################

def getpkmod(k,domu,mu,kmod,pkmod,beta,sigv,b):
  if (domu):
    rsdfact = pkmuboost(mu,k,beta,sigv)
  else:
    rsdfact = np.array([pkboost(k1,beta,sigv) for k1 in k])
  pk = np.interp(k,kmod,pkmod)*(b**2)*rsdfact
  return pk

########################################################################
# Obtain model RSD cross-power spectrum.                               #
########################################################################

def getpkcrossmod(k,domu,mu,kmod,pkmod,beta1,beta2,sigv,b1,b2):
  if (domu):
    rsdfact = pkcrossmuboost(mu,k,beta1,beta2,sigv)
  else:
    rsdfact = np.array([pkcrossboost(k1,beta1,beta2,sigv) for k1 in k])
  pk = np.interp(k,kmod,pkmod)*b1*b2*rsdfact
  return pk

########################################################################
# Obtain model RSD power spectrum multipoles.                          #
########################################################################

def getpolemod(k,kmod,pkmod,beta,sigv,b):
  rsd0fact = np.array([poleboost(0,k1,beta,sigv) for k1 in k])
  rsd2fact = np.array([poleboost(2,k1,beta,sigv) for k1 in k])
  rsd4fact = np.array([poleboost(4,k1,beta,sigv) for k1 in k])
  norm2 = np.full(len(k),5.)
  norm4 = np.full(len(k),9.)
  pk = np.interp(k,kmod,pkmod)*(b**2)
  return rsd0fact*pk,norm2*rsd2fact*pk,norm4*rsd4fact*pk

########################################################################
# Obtain model RSD cross-power spectrum multipoles.                    #
########################################################################

def getpolecrossmod(k,kmod,pkmod,beta1,beta2,sigv,b1,b2):
  nk = len(k)
  pk = np.interp(k,kmod,pkmod)*b1*b2
  rsd0fact = np.array([polecrossboost(0,k1,beta1,beta2,sigv) for k1 in k])
  rsd2fact = np.array([polecrossboost(2,k1,beta1,beta2,sigv) for k1 in k])
  rsd4fact = np.array([polecrossboost(4,k1,beta1,beta2,sigv) for k1 in k])
  norm2 = np.full(nk,5.)
  norm4 = np.full(nk,9.)
  return pk*rsd0fact,norm2*pk*rsd2fact,norm4*pk*rsd4fact

########################################################################
# Multiplicative RSD factor to the power spectrum amplitude.           #
########################################################################

def pkboost(k,beta,sigv):
  boost,err = quad(pkmuboost,0.,1.,args=(k,beta,sigv))
  return boost

def pkcrossboost(k,beta1,beta2,sigv):
  boost,err = quad(pkcrossmuboost,0.,1.,args=(k,beta1,beta2,sigv))
  return boost

def poleboost(l,k,beta,sigv):
  boost,err = quad(polemuboost,0.,1.,args=(l,k,beta,sigv))
  return boost

def polecrossboost(l,k,beta1,beta2,sigv):
  boost,err = quad(polecrossmuboost,0.,1.,args=(l,k,beta1,beta2,sigv))
  return boost

def pkmuboost(mu,k,beta,sigv):
  nom = (1.+(beta*(mu**2)))**2
  den = 1. + ((k*0.01*sigv*mu)**2)
  boost = nom/den
  return boost

def pkcrossmuboost(mu,k,beta1,beta2,sigv):
  nom = (1.+(beta1*(mu**2)))*(1.+(beta2*(mu**2)))
  den = 1. + ((k*0.01*sigv*mu)**2)
  boost = nom/den
  return boost

def polemuboost(mu,l,k,beta,sigv):
  if (l == 2):
    leg = (3.*(mu**2)-1.)/2.
  elif (l == 4):
    leg = (35.*(mu**4)-30.*(mu**2)+3.)/8.
  else:
    leg = 1.
  boost = leg*pkmuboost(mu,k,beta,sigv)
  return boost

def polecrossmuboost(mu,l,k,beta1,beta2,sigv):
  if (l == 2):
    leg = (3.*(mu**2)-1.)/2.
  elif (l == 4):
    leg = (35.*(mu**4)-30.*(mu**2)+3.)/8.
  else:
    leg = 1.
  boost = leg*pkcrossmuboost(mu,k,beta1,beta2,sigv)
  return boost

########################################################################
# Obtain model RSD power spectrum over a grid.                         #
########################################################################

def getpkgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,beta,sigv,b):
  doindep = False
  kgrid,mugrid,indep = getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf)
  pkgrid = getpkmod(kgrid,True,mugrid,kmod,pkmod,beta,sigv,b)
  return pkgrid

########################################################################
# Obtain model RSD cross-power spectrum over a grid.                   #
########################################################################

def getpkcrossgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,beta1,beta2,sigv,b1,b2):
  doindep = False
  kgrid,mugrid,indep = getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf)
  pkgrid = getpkcrossmod(kgrid,True,mugrid,kmod,pkmod,beta1,beta2,sigv,b1,b2)
  return pkgrid

########################################################################
# Bin 3D power spectrum in angle-averaged bins.                        #
########################################################################

def binpk(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf):
  print 'Binning in angle-averaged bins...'
  kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf)
  pkspec = pkspec[indep == True]
  kspec = kspec[indep == True]
  ikbin = np.digitize(kspec,np.linspace(kmin,kmax,nkbin+1))
  nmodes,pk = np.zeros(nkbin,dtype=int),np.full(nkbin,-1.)
  for ik in range(nkbin):
    nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
    if (nmodes[ik] > 0):
      pk[ik] = np.mean(pkspec[ikbin == ik+1])
  return pk,nmodes

########################################################################
# Bin 3D power spectrum in angle-averaged bins, weighting by Legendre  #
# polynomials.                                                         #
########################################################################

def binpole(pkspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf):
  print 'Binning weighting by Legendre polynomials...'
  kspec,muspec,indep = getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf)
  pkspec = pkspec[indep == True]
  kspec = kspec[indep == True]
  muspec = muspec[indep == True]
  leg2spec = ((3.*(muspec**2))-1.)/2.
  leg4spec = ((35.*(muspec**4))-(30.*(muspec**2))+3.)/8.
  ikbin = np.digitize(kspec,np.linspace(kmin,kmax,nkbin+1))
  nmodes,pk0,pk2,pk4 = np.zeros(nkbin,dtype=int),np.full(nkbin,-1.),np.full(nkbin,-1.),np.full(nkbin,-1.)
  for ik in range(nkbin):
    nmodes[ik] = int(np.sum(np.array([ikbin == ik+1])))
    if (nmodes[ik] > 0):
      pk0[ik] = np.mean(pkspec[ikbin == ik+1])
      pk2[ik] = 5.*np.mean((pkspec*leg2spec)[ikbin == ik+1])
      pk4[ik] = 9.*np.mean((pkspec*leg4spec)[ikbin == ik+1])
  return pk0,pk2,pk4,nmodes

########################################################################
# Obtain 3D grid of k-modes.                                           #
########################################################################

def getkspec(nx,ny,nz,lx,ly,lz,doindep,dohalf):
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
  if (dohalf):
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:nz/2+1]
    indep = np.full((nx,ny,nz/2+1),True,dtype=bool)
    if (doindep):
      indep = getindep(nx,ny,nz)
  else:
    kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)
    indep = np.full((nx,ny,nz),True,dtype=bool)
  indep[0,0,0] = False
  kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  kspec[0,0,0] = 1.
  muspec = np.absolute(kx[:,np.newaxis,np.newaxis])/kspec
  kspec[0,0,0] = 0.
  return kspec,muspec,indep

########################################################################
# Obtain array of independent 3D modes.                                #
########################################################################

def getindep(nx,ny,nz):
  indep = np.full((nx,ny,nz/2+1),False,dtype=bool)
  indep[:,:,1:nz/2] = True
  indep[1:nx/2,:,0] = True
  indep[1:nx/2,:,nz/2] = True
  indep[0,1:ny/2,0] = True
  indep[0,1:ny/2,nz/2] = True
  indep[nx/2,1:ny/2,0] = True
  indep[nx/2,1:ny/2,nz/2] = True
  indep[nx/2,0,0] = True
  indep[0,ny/2,0] = True
  indep[nx/2,ny/2,0] = True
  indep[0,0,nz/2] = True
  indep[nx/2,0,nz/2] = True
  indep[0,ny/2,nz/2] = True
  indep[nx/2,ny/2,nz/2] = True
  return indep

########################################################################
# Fill full transform given half transform.                            #
########################################################################

def fthalftofull(nx,ny,nz,tempspec):
  tempgrid = np.empty((nx,ny,nz))
  ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(nz/2+1,nz)
  ixneg[0],iyneg[0] = 0,0
  tempgrid[:,:,:nz/2+1] = tempspec
  tempgrid[:,:,nz/2+1:nz] = tempgrid[:,:,izneg][:,iyneg][ixneg]
  return tempgrid

def fthalftofull2(nx,ny,nz,tempspec1,tempspec2):
  tempgrid = np.empty((nx,ny,nz))
  ixneg,iyneg,izneg = nx-np.arange(nx),ny-np.arange(ny),nz-np.arange(nz/2+1,nz)
  ixneg[0],iyneg[0] = 0,0
  tempgrid[:,:,:nz/2+1] = np.real(tempspec1*np.conj(tempspec2))
  tempgrid[:,:,nz/2+1:nz] = tempgrid[:,:,izneg][:,iyneg][ixneg]
  return tempgrid

########################################################################
# Obtain Fourier transforms of window functions.                       #
########################################################################

def getftwin(nx,ny,nz,wingridgal,wingriddens):
  print '\nComputing weighting for galaxy and density fields...'
  nc = float(nx*ny*nz)
# Normalize W(x)
  wingridgal /= np.sum(wingridgal)
  wingriddens /= np.sum(wingriddens)
# Set weights [currently set to unity]
  weigridgal = np.ones((nx,ny,nz))
  weigriddens = np.ones((nx,ny,nz))
# Determine normalization Sum W(x)^2 w(x)^2
  sumwsqgal = nc*np.sum((wingridgal*weigridgal)**2)
  sumwsqdens = nc*np.sum((wingriddens*weigriddens)**2)
  sumwsqcross = nc*np.sum(wingridgal*weigridgal*wingriddens*weigriddens)
  print 'Sum |Wgal(x)|^2   = ' + '{:5.3f}'.format(sumwsqgal)
  print 'Sum |Wdens(x)|^2  = ' + '{:5.3f}'.format(sumwsqdens)
  print 'Sum |Wcross(x)|^2 = ' + '{:5.3f}'.format(sumwsqcross)
# Determine FFT[w(x)*W(x)]
  winspecgal = np.fft.rfftn(weigridgal*wingridgal)
  winspecdens = np.fft.rfftn(weigriddens*wingriddens)
  return sumwsqgal,sumwsqdens,sumwsqcross,winspecgal,weigridgal,winspecdens,weigriddens

########################################################################
# Convolve model power spectrum with survey window function.           #
########################################################################

def getpkconv(nx,ny,nz,lx,ly,lz,sumwsq,winspec,kmod,pkmod,beta,sigv,b,donoise,pknoise):
  print '\nConvolving with window function...'
# Initializations
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)
  kgrid = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
# Determine model P(k) on grid
  kgrid[0,0,0] = 1.
  domu = True
  mugrid = np.absolute(kx[:,np.newaxis,np.newaxis])/kgrid
  tempgrid = getpkmod(kgrid,domu,mugrid,kmod,pkmod,beta,sigv,b)
  if (donoise):
    tempgrid += pknoise*np.ones_like(kgrid)
  tempgrid[0,0,0] = 0.
# FFT model P(k)
  pkmodspec = np.fft.rfftn(tempgrid)
# FFT |W(k)|^2
  tempgrid = fthalftofull2(nx,ny,nz,winspec,winspec)
  tempspec = np.fft.rfftn(tempgrid)
# Inverse Fourier transform
  pkcongrid = np.fft.irfftn(pkmodspec*tempspec)
# Normalization
  return pkcongrid/sumwsq

########################################################################
# Convolve model cross-power spectrum with survey window functions.    #
########################################################################

def getpkcrossconv(nx,ny,nz,lx,ly,lz,sumw1w2,winspec1,winspec2,kmod,pkmod,beta1,beta2,sigv,b1,b2):
  print '\nConvolving with window function...'
# Initializations
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)
  kgrid = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
# Determine model P(k) on grid
  kgrid[0,0,0] = 1.
  domu = True
  mugrid = np.absolute(kx[:,np.newaxis,np.newaxis])/kgrid
  tempgrid = getpkcrossmod(kgrid,domu,mugrid,kmod,pkmod,beta1,beta2,sigv,b1,b2)
  tempgrid[0,0,0] = 0.
# FFT model P(k)
  pkmodspec = np.fft.rfftn(tempgrid)
# FFT Re{W1(k).W2(k)^*}
  tempgrid = fthalftofull2(nx,ny,nz,winspec1,winspec2)
  tempspec = np.fft.rfftn(tempgrid)
# Inverse Fourier transform
  pkcongrid = np.fft.irfftn(pkmodspec*tempspec)
# Normalization
  return pkcongrid/sumw1w2

########################################################################
# Convolve power spectrum multipoles using spherical harmonics.        #
########################################################################

def getpoleconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsq,weigrid,wingrid,kmod,pkmod,beta,sigv,b,donoise,pknoise,pixcorrgrid):
  print '\nConvolving multipole power spectra with spherical harmonics...'
  nl = 3  # Number of multipoles to compute
  nlp = 3
  print 'nl =',nl,'nlmod =',nlp
# Check normalization
  if (np.absolute(np.sum(wingrid)-1.) > 0.01):
    print 'Sum W(x) =',np.sum(wingrid)
    print 'Window function is unnormalized!!'
    sys.exit()
# Initializations
  dx,dy,dz,vol = lx/nx,ly/ny,lz/nz,lx*ly*lz
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  x += 0.5*dx
  y += 0.5*dy
  z += 0.5*dz
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  kgrid = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  doindep,dohalf = False,False
# Unconvolved power spectrum multipoles
  dk = (kmax-kmin)/nkbin
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  pk0mod,pk2mod,pk4mod = getpolemod(kbin,kmod,pkmod,beta,sigv,b)
  if (donoise):
    pk0mod += pknoise
# Obtain spherical polar angles over the grid
  xthetagrid = np.arctan2(z[np.newaxis,np.newaxis,:],y[np.newaxis,:,np.newaxis])
  xphigrid = np.where(rgrid>0.,np.arccos(x[:,np.newaxis,np.newaxis]/rgrid),0.)
  kthetagrid = np.arctan2(kz[np.newaxis,np.newaxis,:],ky[np.newaxis,:,np.newaxis])
  kphigrid = np.where(kgrid>0.,np.arccos(kx[:,np.newaxis,np.newaxis]/kgrid),0.)
  winspec = np.fft.fftn(weigrid*wingrid)
# Compute convolutions
  pk0con,pk2con,pk4con = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
  for il in range(nl):
    l = 2*il
    pkcongrid = np.zeros((nx,ny,nz))
    for m in range(-l,l+1):
      xylmgrid = sph_harm(m,l,xthetagrid,xphigrid)
      kylmgrid = sph_harm(m,l,kthetagrid,kphigrid)
      for ilp in range(nlp):
        lp = 2*ilp
        norm = ((4.*np.pi)**2)/(sumwsq*float(2*lp+1))
        print 'Computing convolution for l =',l,'m =',m,'lp =',lp,'...'
        if (ilp == 0):
          plpgrid = np.interp(kgrid,kbin,pk0mod)
        elif (ilp == 1):
          plpgrid = np.interp(kgrid,kbin,pk2mod)
        elif (ilp == 2):
          plpgrid = np.interp(kgrid,kbin,pk4mod)
        for mp in range(-lp,lp+1):
          xylmpgrid = sph_harm(mp,lp,xthetagrid,xphigrid)
          kylmpgrid = sph_harm(mp,lp,kthetagrid,kphigrid)
          pkmodspec = np.fft.fftn(plpgrid*np.conj(kylmpgrid))
          slmlmpgrid = np.fft.fftn(weigrid*wingrid*xylmgrid*np.conj(xylmpgrid))
          tempspec = np.fft.fftn(winspec*np.conj(slmlmpgrid))
          pkcongrid += norm*np.real(kylmgrid*np.fft.ifftn(pkmodspec*tempspec))
# Include pixel correction
    pkcongrid *= pixcorrgrid
# Average over k modes
    pkcon,nmodes = binpk(pkcongrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
    if (il == 0):
      pk0con = pkcon
    elif (il == 1):
      pk2con = pkcon
    elif (il == 2):
      pk4con = pkcon
  return pk0con,pk2con,pk4con

########################################################################
# Convolve cross-power spectrum multipoles using spherical harmonics.  #
########################################################################

def getpolecrossconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumw1w2,weigrid1,wingrid1,weigrid2,wingrid2,kmod,pkmod,beta1,beta2,sigv,b1,b2,pixcorrgrid):
  print '\nConvolving multipole density cross-power spectra with spherical harmonics...'
  nl = 3  # Number of multipoles to compute
  nlp = 3 # Number of multipoles in model
  print 'nl =',nl,'nlmod =',nlp
# Initializations
  dx,dy,dz,vol = lx/nx,ly/ny,lz/nz,lx*ly*lz
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  kgrid = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  doindep,dohalf = False,False
# Unconvolved power spectrum multipoles
  dk = (kmax-kmin)/nkbin
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  pk0mod,pk2mod,pk4mod = getpolecrossmod(kbin,kmod,pkmod,beta1,beta2,sigv,b1,b2)
# Obtain spherical polar angles over the grid
  xthetagrid = np.arctan2(z[np.newaxis,np.newaxis,:],y[np.newaxis,:,np.newaxis])
  xphigrid = np.where(rgrid>0.,np.arccos(x[:,np.newaxis,np.newaxis]/rgrid),0.)
  kthetagrid = np.arctan2(kz[np.newaxis,np.newaxis,:],ky[np.newaxis,:,np.newaxis])
  kphigrid = np.where(kgrid>0.,np.arccos(kx[:,np.newaxis,np.newaxis]/kgrid),0.)
  pk0con,pk2con,pk4con = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
# Compute convolutions
  for il in range(nl):
    l = 2*il
    pkcongrid = np.zeros((nx,ny,nz))
    for m in range(-l,l+1):
      xylmgrid = sph_harm(m,l,xthetagrid,xphigrid)
      slmgrid = np.fft.fftn(weigrid1*wingrid1*xylmgrid)
      kylmgrid = sph_harm(m,l,kthetagrid,kphigrid)
      for ilp in range(nlp):
        lp = 2*ilp
        print 'Computing convolution for l =',l,'m =',m,'lp =',lp,'...'
        if (ilp == 0):
          plpgrid = np.interp(kgrid,kbin,pk0mod)
        elif (ilp == 1):
          plpgrid = np.interp(kgrid,kbin,pk2mod)
        elif (ilp == 2):
          plpgrid = np.interp(kgrid,kbin,pk4mod)
        for mp in range(-lp,lp+1):
          xylmpgrid = sph_harm(mp,lp,xthetagrid,xphigrid)
          slmpgrid = np.fft.fftn(weigrid2*wingrid2*xylmpgrid)
          kylmpgrid = sph_harm(mp,lp,kthetagrid,kphigrid)
          pkmodspec = np.fft.fftn(plpgrid*np.conj(kylmpgrid))
          tempspec = np.fft.fftn(np.conj(slmgrid)*slmpgrid)
          norm = ((4.*np.pi)**2)/(sumw1w2*float(2*lp+1))
          pkcongrid += norm*np.real(kylmgrid*np.fft.ifftn(pkmodspec*tempspec))
# Include pixel correction
    pkcongrid *= pixcorrgrid
# Average over k modes
    pkcon,nmodes = binpk(pkcongrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
    if (il == 0):
      pk0con = pkcon
    elif (il == 1):
      pk2con = pkcon
    elif (il == 2):
      pk4con = pkcon
  return pk0con,pk2con,pk4con

########################################################################
# Estimate the 3D power spectrum of a galaxy field.                    #
########################################################################

def getpkgalest(vol,ngal,datgrid,sumwsq,winspec,weigrid):
  print '\nEstimating FKP power spectrum...'
  sgal = np.sum((weigrid**2)*datgrid)
  datspec = np.fft.rfftn(weigrid*datgrid) - ngal*winspec
  pkspec = np.real(datspec)**2 + np.imag(datspec)**2
  pkspec = (pkspec-sgal)*vol/(sumwsq*(ngal**2))
  return pkspec

########################################################################
# Estimate the 3D power spectrum of a density field.                    #
########################################################################

def getpkdensest(vol,nc,vfrac,densgrid,weigrid,sumwsq):
  print '\nEstimating density power spectrum...'
  densspec = np.fft.rfftn(weigrid*densgrid)
  pkspec = np.real(densspec)**2 + np.imag(densspec)**2
  pkspec *= vol/(sumwsq*(vfrac**2)*(nc**2))
  return pkspec

########################################################################
# Estimate the 3D cross-power spectrum of galaxy and density fields.   #
########################################################################

def getpkcrossest(vol,nc,vfrac,ngal,datgrid,winspec,weigridgal,densgrid,weigriddens,sumwsqcross):
  print '\nEstimating density cross-power spectrum...'
  datspec = np.fft.rfftn(weigridgal*datgrid) - ngal*winspec
  densspec = np.fft.rfftn(weigriddens*densgrid)
  pkspec = np.real(datspec*np.conj(densspec))
  pkspec *= vol/(ngal*vfrac*nc*sumwsqcross)
  return pkspec

########################################################################
# Estimate power spectrum multipoles using Bianchi et al. (2015).      #
########################################################################

def getpoleest(opt,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,ngal,datgrid,sumwsq,weigrid,wingrid):
  print '\nEstimating multipole power spectra with Bianchi method...'
# Check normalization
  if (np.absolute(np.sum(wingrid)-1.) > 0.01):
    print 'Sum W(x) =',np.sum(wingrid)
    print 'Window function is unnormalized!!'
    sys.exit()
# Initializations
  dx,dy,dz,vol,nc = lx/nx,ly/ny,lz/nz,lx*ly*lz,float(nx*ny*nz)
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)[:nz/2+1]
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  rgrid[rgrid == 0.] = 1.
  kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  kspec[0,0,0] = 1.
# Determine F(x)
  if (opt == 1):
    fgrid = weigrid*(datgrid - ngal*wingrid)
# Determine shot noise factor Sum w(x)^2 N(x)
    sgal = np.sum((weigrid**2)*datgrid)
    den = sumwsq*(ngal**2)
  else:
    fgrid = datgrid
    sgal = 0.
    den = sumwsq*(nc**2)
# Determine A_0(k)
  print 'Determining A_0(k)...'
  a0spec = np.fft.rfftn(fgrid)
# Determine A_2(k)
  print 'Determining A_2(k)...'
  a2spec = np.zeros((nx,ny,nz/2+1),dtype=complex)
  for iterm in range(1,7):
    rfactgrid,kfactspec = geta2term(iterm,x,y,z,kx,ky,kz)
    tempgrid = (rfactgrid*fgrid)/(rgrid**2)
    tempspec = np.fft.rfftn(tempgrid)
    a2spec += (kfactspec*tempspec)/(kspec**2)
# Determine A_4(k)
  print 'Determining A_4(k)...'
  a4spec = np.zeros((nx,ny,nz/2+1),dtype=complex)
  for iterm in range(1,16):
    rfactgrid,kfactspec = geta4term(iterm,x,y,z,kx,ky,kz)
    tempgrid = (rfactgrid*fgrid)/(rgrid**4)
    tempspec = np.fft.rfftn(tempgrid)
    a4spec += (kfactspec*tempspec)/(kspec**4)
# Power spectrum estimators
  pk0spec = (np.real(a0spec*np.conj(a0spec))-sgal)*vol/den
  pk2spec = np.real(a0spec*np.conj(3.*a2spec-a0spec))*5.*vol/(2.*den)
  pk4spec = np.real(a0spec*np.conj(35.*a4spec-30.*a2spec+3.*a0spec))*9.*vol/(8.*den)
# Average power spectra in bins
  doindep,dohalf = True,True
  pk0,nmodes = binpk(pk0spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk2,nmodes = binpk(pk2spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk4,nmodes = binpk(pk4spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  return pk0,pk2,pk4,nmodes

def geta2term(iterm,x,y,z,kx,ky,kz):
  if (iterm == 1):
    rfact = x[:,np.newaxis,np.newaxis]**2
    kfact = kx[:,np.newaxis,np.newaxis]**2
  elif (iterm == 2):
    rfact = y[np.newaxis,:,np.newaxis]**2
    kfact = ky[np.newaxis,:,np.newaxis]**2
  elif (iterm == 3):
    rfact = z[np.newaxis,np.newaxis,:]**2
    kfact = kz[np.newaxis,np.newaxis,:]**2
  elif (iterm == 4):
    rfact = x[:,np.newaxis,np.newaxis]*y[np.newaxis,:,np.newaxis]
    kfact = 2.*kx[:,np.newaxis,np.newaxis]*ky[np.newaxis,:,np.newaxis]
  elif (iterm == 5):
    rfact = x[:,np.newaxis,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 2.*kx[:,np.newaxis,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 6):
    rfact = y[np.newaxis,:,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 2.*ky[np.newaxis,:,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  return rfact,kfact

def geta4term(iterm,x,y,z,kx,ky,kz):
  if (iterm == 1):
    rfact = x[:,np.newaxis,np.newaxis]**4
    kfact = kx[:,np.newaxis,np.newaxis]**4
  elif (iterm == 2):
    rfact = y[np.newaxis,:,np.newaxis]**4
    kfact = ky[np.newaxis,:,np.newaxis]**4
  elif (iterm == 3):
    rfact = z[np.newaxis,np.newaxis,:]**4
    kfact = kz[np.newaxis,np.newaxis,:]**4
  elif (iterm == 4):
    rfact = (x[:,np.newaxis,np.newaxis]**3)*y[np.newaxis,:,np.newaxis]
    kfact = 4.*(kx[:,np.newaxis,np.newaxis]**3)*ky[np.newaxis,:,np.newaxis]
  elif (iterm == 5):
    rfact = (x[:,np.newaxis,np.newaxis]**3)*z[np.newaxis,np.newaxis,:]
    kfact = 4.*(kx[:,np.newaxis,np.newaxis]**3)*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 6):
    rfact = (y[np.newaxis,:,np.newaxis]**3)*x[:,np.newaxis,np.newaxis]
    kfact = 4.*(ky[np.newaxis,:,np.newaxis]**3)*kx[:,np.newaxis,np.newaxis]
  elif (iterm == 7):
    rfact = (y[np.newaxis,:,np.newaxis]**3)*z[np.newaxis,np.newaxis,:]
    kfact = (ky[np.newaxis,:,np.newaxis]**3)*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 8):
    rfact = (z[np.newaxis,np.newaxis,:]**3)*x[:,np.newaxis,np.newaxis]
    kfact = (kz[np.newaxis,np.newaxis,:]**3)*kx[:,np.newaxis,np.newaxis]
  elif (iterm == 9):
    rfact = (z[np.newaxis,np.newaxis,:]**3)*y[np.newaxis,:,np.newaxis]
    kfact = (kz[np.newaxis,np.newaxis,:]**3)*ky[np.newaxis,:,np.newaxis]
  elif (iterm == 10):
    rfact = (x[:,np.newaxis,np.newaxis]**2)*(y[np.newaxis,:,np.newaxis]**2)
    kfact = 6.*(kx[:,np.newaxis,np.newaxis]**2)*(ky[np.newaxis,:,np.newaxis]**2)
  elif (iterm == 11):
    rfact = (x[:,np.newaxis,np.newaxis]**2)*(z[np.newaxis,np.newaxis,:]**2)
    kfact = 6.*(kx[:,np.newaxis,np.newaxis]**2)*(kz[np.newaxis,np.newaxis,:]**2)
  elif (iterm == 12):
    rfact = (y[np.newaxis,:,np.newaxis]**2)*(z[np.newaxis,np.newaxis,:]**2)
    kfact = 6.*(ky[np.newaxis,:,np.newaxis]**2)*(kz[np.newaxis,np.newaxis,:]**2)
  elif (iterm == 13):
    rfact = (x[:,np.newaxis,np.newaxis]**2)*y[np.newaxis,:,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 12.*(kx[:,np.newaxis,np.newaxis]**2)*ky[np.newaxis,:,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 14):
    rfact = (y[np.newaxis,:,np.newaxis]**2)*x[:,np.newaxis,np.newaxis]*z[np.newaxis,np.newaxis,:]
    kfact = 12.*(ky[np.newaxis,:,np.newaxis]**2)*kx[:,np.newaxis,np.newaxis]*kz[np.newaxis,np.newaxis,:]
  elif (iterm == 15):
    rfact = (z[np.newaxis,np.newaxis,:]**2)*x[:,np.newaxis,np.newaxis]*y[np.newaxis,:,np.newaxis]
    kfact = 12.*(kz[np.newaxis,np.newaxis,:]**2)*kx[:,np.newaxis,np.newaxis]*ky[np.newaxis,:,np.newaxis]
  return rfact,kfact

########################################################################
# Estimate cross-power spectrum multipoles.                            #
########################################################################

def getpolecrossest(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,ngal,datgrid,weigrid,wingrid,densgrid):
  print '\nEstimating galaxy-density cross-power spectrum multipoles with Bianchi method...'
# Initializations
  dx,dy,dz,vol,nc = lx/nx,ly/ny,lz/nz,lx*ly*lz,float(nx*ny*nz)
  x = dx*np.arange(nx) - x0
  y = dy*np.arange(ny) - y0
  z = dz*np.arange(nz) - z0
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=dx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=dy)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=dz)[:nz/2+1]
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  rgrid[rgrid == 0.] = 1.
  kspec = np.sqrt(kx[:,np.newaxis,np.newaxis]**2 + ky[np.newaxis,:,np.newaxis]**2 + kz[np.newaxis,np.newaxis,:]**2)
  kspec[0,0,0] = 1.
# Determine F(x) for galaxy and density data data
  fgrid1 = weigrid*(datgrid - ngal*wingrid)
  fgrid2 = densgrid
# Determine A_0(k)
  print 'Determining A_0(k)...'
  a0spec1 = np.fft.rfftn(fgrid1)
  a0spec2 = np.fft.rfftn(fgrid2)
# Determine A_2(k)
  print 'Determining A_2(k)...'
  a2spec1,a2spec2 = np.zeros((nx,ny,nz/2+1),dtype=complex),np.zeros((nx,ny,nz/2+1),dtype=complex)
  for iterm in range(1,7):
    rfactgrid,kfactspec = geta2term(iterm,x,y,z,kx,ky,kz)
    tempgrid = (rfactgrid*fgrid1)/(rgrid**2)
    tempspec = np.fft.rfftn(tempgrid)
    a2spec1 += (kfactspec*tempspec)/(kspec**2)
    tempgrid = (rfactgrid*fgrid2)/(rgrid**2)
    tempspec = np.fft.rfftn(tempgrid)
    a2spec2 += (kfactspec*tempspec)/(kspec**2)
# Determine A_4(k)
  print 'Determining A_4(k)...'
  a4spec1,a4spec2 = np.zeros((nx,ny,nz/2+1),dtype=complex),np.zeros((nx,ny,nz/2+1),dtype=complex)
  for iterm in range(1,16):
    rfactgrid,kfactspec = geta4term(iterm,x,y,z,kx,ky,kz)
    tempgrid = (rfactgrid*fgrid1)/(rgrid**4)
    tempspec = np.fft.rfftn(tempgrid)
    a4spec1 += (kfactspec*tempspec)/(kspec**4)
    tempgrid = (rfactgrid*fgrid2)/(rgrid**4)
    tempspec = np.fft.rfftn(tempgrid)
    a4spec2 += (kfactspec*tempspec)/(kspec**4)
# Power spectrum estimators
  tempspec = a0spec1*np.conj(a0spec2) + np.conj(a0spec1)*a0spec2
  pk0spec = (np.real(tempspec)*vol)/(2.*ngal*nc)
  tempspec = a0spec1*np.conj(3.*a2spec2-a0spec2) + np.conj(3.*a2spec1-a0spec1)*a0spec2
  pk2spec = (np.real(tempspec)*5.*vol)/(4.*ngal*nc)
  tempspec = a0spec1*np.conj(35.*a4spec2-30.*a2spec2+3.*a0spec2) + np.conj(35.*a4spec1-30.*a2spec1+3.*a0spec1)*a0spec2
  pk4spec = (np.real(tempspec)*9.*vol)/(16.*ngal*nc)
# Average power spectra in bins
  doindep,dohalf = True,True
  pk0,nmodes = binpk(pk0spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk2,nmodes = binpk(pk2spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk4,nmodes = binpk(pk4spec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  return pk0,pk2,pk4,nmodes

########################################################################
# Estimate error in power spectrum multipoles.                         #
########################################################################

def getpoleerr(vol,ngal,vfrac,nkbin,pk0gal,pk2gal,pk4gal,pk0dens,pk2dens,pk4dens,pk0cross,pk2cross,pk4cross,nmodes):
  ndens = ngal/vol
  pkdiagerr = np.zeros((9*nkbin,9*nkbin))
  for ik in range(nkbin):
    if (nmodes[ik] > 0):
      for istat in range(3):
        for il in range(3):
          ibin = (3*istat+il)*nkbin + ik
          for jstat in range(3):
            for jl in range(3):
              jbin = (3*jstat+jl)*nkbin + ik
              pkcrossvar,err = quad(pkcrossvarint,0.,1.,args=(istat,jstat,2*il,2*jl,pk0gal[ik],pk2gal[ik],pk4gal[ik],pk0dens[ik],pk2dens[ik],pk4dens[ik],pk0cross[ik],pk2cross[ik],pk4cross[ik],ndens))
              if (pkcrossvar > 0.):
                pkdiagerr[ibin,jbin] = np.sqrt(pkcrossvar/(vfrac*nmodes[ik]))
  return pkdiagerr

def pkcrossvarint(mu,stat1,stat2,l1,l2,pk0gal,pk2gal,pk4gal,pk0dens,pk2dens,pk4dens,pk0cross,pk2cross,pk4cross,ndens):
  leg2 = (3.*(mu**2)-1.)/2.
  leg4 = (35.*(mu**4)-30.*(mu**2)+3.)/8.
  pkgal = pk0gal + pk2gal*leg2 + pk4gal*leg4
  pkdens = pk0dens + pk2dens*leg2 + pk4dens*leg4
  pkcross = pk0cross + pk2cross*leg2 + pk4cross*leg4
  if (stat1 == 0):
    if (stat2 == 0):
      pkvar = (pkgal + (1./ndens))**2
    elif (stat2 == 1):
      pkvar = pkcross**2
    elif (stat2 == 2):
      pkvar = pkcross*(pkgal + (1./ndens))
  elif (stat1 == 1):
    if (stat2 == 0):
      pkvar = pkcross**2
    elif (stat2 == 1):
      pkvar = pkdens**2
    elif (stat2 == 2):
      pkvar = pkcross*pkdens
  elif (stat1 == 2):
    if (stat2 == 0):
      pkvar = pkcross*(pkgal + (1./ndens))
    elif (stat2 == 1):
      pkvar = pkcross*pkdens
    elif (stat2 == 2):
      pkvar = 0.5*((pkcross**2) + (pkgal + (1./ndens))*pkdens)
  if (l1 == 0):
    norm1 = 1.
  elif (l1 == 2):
    norm1 = 5.*leg2
  elif (l1 == 4):
    norm1 = 9.*leg4
  if (l2 == 0):
    norm2 = 1.
  elif (l2 == 2):
    norm2 = 5.*leg2
  elif (l2 == 4):
    norm2 = 9.*leg4
  return norm1*norm2*pkvar

########################################################################
# Plot multipole power spectra.                                        #
########################################################################

def plotpkpole(kmin,kmax,nkbin,pk0case,pk0errcase,pk0concase,pk2case,pk2errcase,pk2concase,pk4case,pk4errcase,pk4concase,labelcase):
  ncase = pk0case.shape[0]
  dk = (kmax-kmin)/nkbin
  ks = 0.01*(kmax-kmin)
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  norm = kbin
  fig = plt.figure()
  colorlst = ['black','red','green']
# P_0(k)
  plt.subplot(231)
  ymin,ymax = 0.,1000.
  plt.ylabel(r'$k \, P_0(k) \, [h^{-2} \, {\rm Mpc}^2]$')
  for icase in range(ncase):
    plt.plot(kbin,norm*pk0concase[icase,:],color=colorlst[icase])
    plt.errorbar(kbin+ks*icase,norm*pk0case[icase,:],yerr=norm*pk0errcase[icase,:],linestyle='None',color=colorlst[icase],label=labelcase[icase])
  plt.xticks([0.,0.1,0.2,0.3])
  plt.xlabel(r'$k \, [h \, {\rm Mpc}^{-1}]$')
  plt.xlim(kmin,kmax)
  plt.ylim(ymin,ymax)
  plt.legend(prop={'size':10})
# P_2(k)
  plt.subplot(232)
  ymin,ymax = 0.,700.
  plt.ylabel(r'$k \, P_2(k) \, [h^{-2} \, {\rm Mpc}^2]$')
  for icase in range(ncase):
    plt.plot(kbin,norm*pk2concase[icase,:],color=colorlst[icase])
    plt.errorbar(kbin+ks*icase,norm*pk2case[icase,:],yerr=norm*pk2errcase[icase,:],linestyle='None',color=colorlst[icase])
  plt.xticks([0.,0.1,0.2,0.3])
  plt.xlabel(r'$k \, [h \, {\rm Mpc}^{-1}]$')
  plt.xlim(kmin,kmax)
  plt.ylim(ymin,ymax)
# P_4(k)
  plt.subplot(233)
  ymin,ymax = -300.,300.
  plt.ylabel(r'$k \, P_4(k) \, [h^{-2} \, {\rm Mpc}^2]$')
  for icase in range(ncase):
    plt.plot(kbin,norm*pk4concase[icase,:],color=colorlst[icase])
    plt.errorbar(kbin+ks*icase,norm*pk4case[icase,:],yerr=norm*pk4errcase[icase,:],linestyle='None',color=colorlst[icase])
  plt.xticks([0.,0.1,0.2,0.3])
  plt.xlabel(r'$k \, [h \, {\rm Mpc}^{-1}]$')
  plt.xlim(kmin,kmax)
  plt.ylim(ymin,ymax)
  fig.tight_layout()
  fig.savefig('pkpole_runspherpk.png',bbox_inches='tight')
  return

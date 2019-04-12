import numpy as np
from scipy.interpolate import interpn
import pktools

########################################################################
# Obtain correction to model power spectrum from smoothing using a     #
# healpix grid, redshift bins and telescope beam.                      #
#                                                                      #
# Arguments:                                                           #
# docross -- True/False: calculate for cross-power spectrum?           #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# wingrid -- [0,1] array of window function (for volume-averaging)     #
# lwin,pixwin -- pixel window function array                           #
# dzbin -- width of redshift bins                                      #
# dobeam -- include telescope beam in model                            #
# sigdeg -- standard deviation of Gaussian beam [degrees]              #
# kmax -- maximum wavenumber required                                  #
# kmod,pkmod -- power spectrum array [k in h/Mpc, P in (Mpc/h)^3]      #
# beta -- value of RSD distortion parameter                            #
# sigv -- value of pairwise velocity dispersion                        #
# b -- galaxy bias                                                     #
# pknoise -- constant value of noise power spectrum in (Mpc/h)^3       #
# cosmo -- astropy fiducial cosmology                                  #
#                                                                      #
# Returns:                                                             #
# dampcorrspec -- power spectrum correction on FFT grid                #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

def getdampcorr(docross,nx,ny,nz,lx,ly,lz,x0,y0,z0,wingrid,lwin,pixwin,dzbin,dobeam,sigdeg,kmax,kmod,pkmod,beta,sigv,b,pknoise,cosmo):
  print '\nCalculating (healpix,redshift) gridding correction...'
  print 'docross =','{:1d}'.format(docross)
  print 'dzbin =',dzbin
  print 'dobeam =','{:1d}'.format(dobeam)
  if (dobeam):
    print 'sigdeg =',sigdeg
# If a sum over aliased k-modes is used, this sum is over k + n*k_Nyq where
# n is in the range [-nmax,+nmax].  If nmax=0, the correction is evaluated
# at the wavenumber k, without summing over additional modes
  nmax = 0
# Determine the co-moving distance rgrid at each point of the grid.  This is
# related to the multipole in the pixel window function as l = kperp*rgrid
  dx,dy,dz = lx/nx,ly/ny,lz/nz
  x = dx*np.arange(nx) - x0 + 0.5*dx
  y = dy*np.arange(ny) - y0 + 0.5*dy
  z = dz*np.arange(nz) - z0 + 0.5*dz
  rgrid = np.sqrt(x[:,np.newaxis,np.newaxis]**2 + y[np.newaxis,:,np.newaxis]**2 + z[np.newaxis,np.newaxis,:]**2)
  rgrid = rgrid[wingrid != 0.]
# Determine the co-moving separation corresponding to the redshift bin width
# at each point of the grid.  This is related to the Fourier transform of the
# radial binning function by u = kpar*drgrid
  zarr = np.linspace(0.,1.,1000)
  rarr = cosmo.comoving_distance(zarr).value
  drarr = 2997.9*cosmo.inv_efunc(zarr)*dzbin
  drgrid = np.interp(rgrid,rarr,drarr)
# Pre-compute the radial window function, it's a 1D Fourier transform of
# a top-hat, i.e. a sinc function
  umax,nuarr = 20.,2000
  uarr = np.linspace(0.,umax,nuarr)
  pixpararr = np.ones(nuarr)
  pixpararr[1:] = np.sin(0.5*uarr[1:])/(0.5*uarr[1:])
# Determine the co-moving smoothing length corresponding to the telescope
# beam.  This is related to the Fourier transform of the angular smoothing
# function by v = kperp*sigperpgrid
  sigperpgrid = rgrid*np.radians(sigdeg)
# Pre-compute the telescope beam window function, it's a 1D Fourier transform
# of a Gaussian, i.e. another Gaussian
  vmax,nvarr = 20.,2000
  varr = np.linspace(0.,vmax,nvarr)
  if (dobeam):
    beamperparr = np.exp(-0.5*(varr**2))
  else:
    beamperparr = np.ones(nvarr)
# Check that we are generating these arrays across a sufficient range to
# compute the smoothing correction
  nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
  kmaxgrid = np.sqrt(nyqx**2 + nyqy**2 + nyqz**2)
  print 'kmaxmeas =',kmax
  print 'kmaxgrid =',kmaxgrid
  print 'kmaxpixperp =',lwin[-1]/np.amax(rgrid)
  print 'kmaxpixpar  =',uarr[-1]/np.amax(drgrid)
  print 'kmaxbeamperp =',varr[-1]/np.amax(sigperpgrid)
  if ((kmax > lwin[-1]/np.amax(rgrid)) or (kmax > lwin[-1]/np.amax(rgrid)) or (kmax > varr[-1]/np.amax(sigperpgrid))):
    print 'Unable to perform correction, lower kmax'
    sys.exit()
# Pre-compute the power spectrum correction on a (k,mu) grid
  if (nmax == 0):
    kmaxfull = kmaxgrid
    kmaxgen = kmax
  else:
    kmaxfull = 3.*kmaxgrid
    kmaxgen = kmaxfull
  dk = 0.01
  nkfull,nkgen,nmugen = int(kmaxfull/dk)+2,int(kmaxgen/dk)+2,11
  kmaxfull,dmu = dk*(nkfull-1),1./float(nmugen-1)
  karr,muarr = np.linspace(0.,kmaxfull,nkfull),np.linspace(0.,1.,nmugen)
  pkdamparr = np.ones((nkfull,nmugen))
  print 'Pre-computing (k,mu) correction...'
  print 'nk  =',nkgen,'dk  =',dk
  print 'nmu =',nmugen,'dmu =',dmu
  for ik in range(nkgen):
    for imu in range(nmugen):
# Determine the correction factor for each (k,mu) value
      k,mu = karr[ik],muarr[imu]
      kperp,kpar = k*np.sqrt(1.-(mu**2)),k*mu
      if (docross):
        pkdamp = pkdampcross(kperp,kpar,rgrid,lwin,pixwin,drgrid,uarr,pixpararr,sigperpgrid,varr,beamperparr)
      else:
        pkdamp = pkdampauto(kperp,kpar,rgrid,lwin,pixwin,drgrid,uarr,pixpararr,sigperpgrid,varr,beamperparr)
      pkdamparr[ik,imu] = pkdamp
# Apply the correction to the full FFT grid
  print 'Applying (k,mu) correction...'
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:nz/2+1]
# Determine using sum over modes
  pkdampspec = dodampcorrsum(nmax,nx,ny,nz,nyqx,nyqy,nyqz,kx[:,np.newaxis,np.newaxis],ky[np.newaxis,:,np.newaxis],kz[np.newaxis,np.newaxis,:],docross,kmod,pkmod,beta,sigv,b,pknoise,karr,muarr,pkdamparr)
  pkdampspec = np.reshape(pkdampspec,(nx,ny,nz/2+1))
  return pkdampspec

########################################################################
# Volume-averaged correction for auto-power spectrum.                  #
########################################################################

def pkdampauto(kperp,kpar,rgrid,lwin,pixwin,drgrid,uarr,pixpararr,sigperpgrid,varr,beamperparr):
  pixperp = np.interp(kperp*rgrid,lwin,pixwin)
  pixpar = np.interp(kpar*drgrid,uarr,pixpararr)
  beamperp = np.interp(kperp*sigperpgrid,varr,beamperparr)
  return np.mean((pixperp*pixpar*beamperp)**2)

########################################################################
# Volume-averaged correction for cross-power spectrum.                 #
########################################################################

def pkdampcross(kperp,kpar,rgrid,lwin,pixwin,drgrid,uarr,pixpararr,sigperpgrid,varr,beamperparr):
  pixperp = np.interp(kperp*rgrid,lwin,pixwin)
  pixpar = np.interp(kpar*drgrid,uarr,pixpararr)
  beamperp = np.interp(kperp*sigperpgrid,varr,beamperparr)
  return np.mean(((pixperp*pixpar)**2)*beamperp)

########################################################################
# Sum correction for FFT grid aliasing.                                # 
########################################################################

def dodampcorrsum(nmax,nx,ny,nz,nyqx,nyqy,nyqz,kx,ky,kz,docross,kmod,pkmod,beta,sigv,b,pknoise,karr,muarr,pkdamparr):
  domu = True
  k = np.sqrt((kx**2)+(ky**2)+(kz**2))
  mu = np.divide(np.absolute(kx),k,out=np.zeros_like(k),where=k!=0.)
  if (docross):
    pk = pktools.getpkcrossmod(k,domu,mu,kmod,pkmod,beta,beta,sigv,b,b)
  else:
    pk = pktools.getpkmod(k,domu,mu,kmod,pkmod,beta,sigv,b,pknoise)
  sum1 = 0.
  for ix in range(-nmax,nmax+1):
    for iy in range(-nmax,nmax+1):
      for iz in range(-nmax,nmax+1):
        kx1 = kx + 2.*nyqx*ix
        ky1 = ky + 2.*nyqy*iy
        kz1 = kz + 2.*nyqz*iz
        k1 = np.sqrt((kx1**2)+(ky1**2)+(kz1**2))
        mu1 = np.divide(np.absolute(kx1),k1,out=np.zeros_like(k1),where=k1!=0.)
        if (docross):
          pk1 = pktools.getpkcrossmod(k1,domu,mu1,kmod,pkmod,beta,beta,sigv,b,b)
        else:
          pk1 = pktools.getpkmod(k1,domu,mu1,kmod,pkmod,beta,sigv,b,pknoise)
        qx1,qy1,qz1 = (np.pi*kx1)/(2.*nyqx),(np.pi*ky1)/(2.*nyqy),(np.pi*kz1)/(2.*nyqz)
        wx = np.divide(np.sin(qx1),qx1,out=np.ones_like(qx1),where=qx1!=0.)
        wy = np.divide(np.sin(qy1),qy1,out=np.ones_like(qy1),where=qy1!=0.)
        wz = np.divide(np.sin(qz1),qz1,out=np.ones_like(qz1),where=qz1!=0.)
        ngpwin = (wx*wy*wz)**2
        points = np.vstack([k1.flatten(),mu1.flatten()]).transpose()
        pkdamp = interpn((karr,muarr),pkdamparr,points)
        pkdamp = np.reshape(pkdamp,(nx,ny,nz/2+1))
        sum1 += ngpwin*pkdamp*pk1
  return sum1/pk

########################################################################
# Obtain correction to model power spectrum from NGP assignment.       #
########################################################################

def getngpcorr(nx,ny,nz,lx,ly,lz,kmod,pkmod,beta,sigv,b,pknoise):
  print '\nCalculating NGP gridding correction...'
  nmax = 1
  nyqx,nyqy,nyqz = nx*np.pi/lx,ny*np.pi/ly,nz*np.pi/lz
  kx = 2.*np.pi*np.fft.fftfreq(nx,d=lx/nx)
  ky = 2.*np.pi*np.fft.fftfreq(ny,d=ly/ny)
  kz = 2.*np.pi*np.fft.fftfreq(nz,d=lz/nz)[:nz/2+1]
  ngpcorrspec = dongpcorrsum(nmax,nyqx,nyqy,nyqz,kx[:,np.newaxis,np.newaxis],ky[np.newaxis,:,np.newaxis],kz[np.newaxis,np.newaxis,:],kmod,pkmod,beta,sigv,b,pknoise)
  return ngpcorrspec

########################################################################
# Sum correction for FFT grid aliasing.                                # 
########################################################################

def dongpcorrsum(nmax,nyqx,nyqy,nyqz,kx,ky,kz,kmod,pkmod,beta,sigv,b,pknoise):
  domu = True
  k = np.sqrt((kx**2)+(ky**2)+(kz**2))
  mu = np.divide(kx,k,out=np.zeros_like(k),where=k!=0.)
  pk = pktools.getpkmod(k,domu,mu,kmod,pkmod,beta,sigv,b,pknoise)
  sum1 = 0.
  for ix in range(-nmax,nmax+1):
    for iy in range(-nmax,nmax+1):
      for iz in range(-nmax,nmax+1):
        kx1 = kx + 2.*nyqx*ix
        ky1 = ky + 2.*nyqy*iy
        kz1 = kz + 2.*nyqz*iz
        k1 = np.sqrt((kx1**2)+(ky1**2)+(kz1**2))
        mu1 = np.divide(kx1,k1,out=np.zeros_like(k1),where=k1!=0.)
        pk1 = pktools.getpkmod(k1,domu,mu1,kmod,pkmod,beta,sigv,b,pknoise)
        qx1,qy1,qz1 = (np.pi*kx1)/(2.*nyqx),(np.pi*ky1)/(2.*nyqy),(np.pi*kz1)/(2.*nyqz)
        wx = np.divide(np.sin(qx1),qx1,out=np.ones_like(qx1),where=qx1!=0.)
        wy = np.divide(np.sin(qy1),qy1,out=np.ones_like(qy1),where=qy1!=0.)
        wz = np.divide(np.sin(qz1),qz1,out=np.ones_like(qz1),where=qz1!=0.)
        ww = wx*wy*wz
        sum1 += (ww**2)*pk1
  return sum1/pk

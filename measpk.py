import numpy as np
import pktools
import getcorr

########################################################################
# Measure auto- and cross-power spectra of a galaxy distribution       #
# and another density field such as an intensity-mapping distribution. #
#                                                                      #
# Arguments:                                                           #
#                                                                      #
# galgrid -- gridded galaxy number distribution                        #
# wingridgal -- gridded window function of galaxy data                 #
# densgrid -- gridded density field                                    #
# wingriddens -- gridded window function of density field              #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# kmin,kmax -- minimum,maximum wavenumber for binning [h/Mpc]          #
# nkbin -- number of k bins                                            #
# kmod,pkmod -- power spectrum array [k in h/Mpc, P in (Mpc/h)^3]      #
# betagal -- value of RSD distortion parameter for galaxies            #
# betadens -- value of RSD distortion parameter for density field      #
# sigv -- value of pairwise velocity dispersion                        #
# bgal -- bias parameter for galaxies                                  #
# bdens -- bias parameter for density field                            #
# pknoise -- constant value of noise power spectrum in (Mpc/h)^3       #
# sigdeg -- standard deviation of Gaussian beam [degrees]              #
# lwin,pixwin -- pixel window function array                           #
# dzbin -- width of redshift bins                                      #
# cosmo -- astropy fiducial cosmology                                  #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

def measpk(galgrid,wingridgal,densgrid,wingriddens,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,kmod,pkmod,betagal,betadens,sigv,bgal,bdens,pknoise,sigdeg,lwin,pixwin,dzbin,cosmo):

########################################################################
# Initializations.                                                     #
########################################################################

  vol,nc = lx*ly*lz,nx*ny*nz
  vfrac = np.sum(wingriddens)/nc
  dk = (kmax-kmin)/float(nkbin)
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  ngal = np.sum(galgrid)

########################################################################
# Obtain Fourier transforms of window functions, and normalizations    #
# used in power spectrum estimation.                                   #
########################################################################

  sumwsqgal,sumwsqdens,sumwsqcross,winspecgal,weigridgal,winspecdens,weigriddens = pktools.getftwin(nx,ny,nz,wingridgal,wingriddens)

########################################################################
# Determine corrections to the power spectrum over an FFT grid.        #
# The galaxy power spectrum just considers the NGP assignment.         #
# The density and cross-power spectra consider the binning in the      #
# spherical healpix/redshift grid, and telescope beam.                 #
########################################################################

  pixcorrgalspec = getcorr.getngpcorr(nx,ny,nz,lx,ly,lz,kmod,pkmod,betagal,sigv,bgal)
  pixcorrdensspec = getcorr.gethpcorr(False,nx,ny,nz,lx,ly,lz,x0,y0,z0,wingriddens,lwin,pixwin,dzbin,sigdeg,kmax,kmod,pkmod,betadens,sigv,bdens,pknoise,cosmo)
  pixcorrcrossspec = getcorr.gethpcorr(True,nx,ny,nz,lx,ly,lz,x0,y0,z0,wingriddens,lwin,pixwin,dzbin,sigdeg,kmax,kmod,pkmod,betagal,sigv,bgal,pknoise,cosmo)

########################################################################
# Obtain model auto- and cross-power spectra over the FFT grid,        #
# multiply them by the corrections computed above, and bin them        #
# in power spectrum multipoles.                                        #
########################################################################

  print '\nComputing model power spectrum...'
  doindep,dohalf = True,True
  pkmodgalspec = pktools.getpkgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,betagal,sigv,bgal)
  pkmoddensspec = pktools.getpkgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,betadens,sigv,bdens)
  pkmoddensspec += pknoise*np.ones_like(pkmoddensspec)
  pkmodcrossspec = pktools.getpkcrossgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,betagal,betadens,sigv,bgal,bdens)
  pkmodgalspec *= pixcorrgalspec
  pkmoddensspec *= pixcorrdensspec
  pkmodcrossspec *= pixcorrcrossspec
  pk0modgal,pk2modgal,pk4modgal,nmodes = pktools.binpole(pkmodgalspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0moddens,pk2moddens,pk4moddens,nmodes = pktools.binpole(pkmoddensspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0modcross,pk2modcross,pk4modcross,nmodes = pktools.binpole(pkmodcrossspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)

########################################################################
# Obtain convolved auto- and cross-power spectra over the FFT grid,    #
# multiply them by the corrections computed above, and bin them        #
# in power spectrum multipoles, in the flat-sky case.                  #
########################################################################

  pkcongalgrid = pktools.getpkconv(nx,ny,nz,lx,ly,lz,sumwsqgal,winspecgal,kmod,pkmod,betagal,sigv,bgal,False,pknoise)
  pkcondensgrid = pktools.getpkconv(nx,ny,nz,lx,ly,lz,sumwsqdens,winspecdens,kmod,pkmod,betadens,sigv,bdens,True,pknoise)
  pkconcrossgrid = pktools.getpkcrossconv(nx,ny,nz,lx,ly,lz,sumwsqcross,winspecgal,winspecdens,kmod,pkmod,betagal,betadens,sigv,bgal,bdens)
  pixcorrgalgrid = pktools.fthalftofull(nx,ny,ny,pixcorrgalspec)
  pixcorrdensgrid = pktools.fthalftofull(nx,ny,ny,pixcorrdensspec)
  pixcorrcrossgrid = pktools.fthalftofull(nx,ny,ny,pixcorrcrossspec)
  pkcongalgrid *= pixcorrgalgrid
  pkcondensgrid *= pixcorrdensgrid
  pkconcrossgrid *= pixcorrcrossgrid
  doindep,dohalf = False,False
  pk0flatcongal,pk2flatcongal,pk4flatcongal,nmodes = pktools.binpole(pkcongalgrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0flatcondens,pk2flatcondens,pk4flatcondens,nmodes = pktools.binpole(pkcondensgrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0flatconcross,pk2flatconcross,pk4flatconcross,nmodes = pktools.binpole(pkconcrossgrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)

########################################################################
# Obtain convolved auto- and cross-power spectrum multipoles in the    #
# curved-sky case.                                                     #
########################################################################

  pk0congal,pk2congal,pk4congal = pktools.getpoleconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsqgal,weigridgal,wingridgal,kmod,pkmod,betagal,sigv,bgal,False,pknoise,pixcorrgalgrid)
  pk0condens,pk2condens,pk4condens = pktools.getpoleconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsqdens,weigriddens,wingriddens,kmod,pkmod,betadens,sigv,bdens,True,pknoise,pixcorrdensgrid)
  pk0concross,pk2concross,pk4concross = pktools.getpolecrossconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsqcross,weigridgal,wingridgal,weigriddens,wingriddens,kmod,pkmod,betagal,betadens,sigv,bgal,bdens,pixcorrcrossgrid)

########################################################################
# Obtain auto- and cross-power spectrum estimates over the FFT grid,   #
# and bin them in power spectrum multipoles, in the flat-sky case.     #
########################################################################

  doindep,dohalf = True,True
  pkspecgal = pktools.getpkgalest(vol,ngal,galgrid,sumwsqgal,winspecgal,weigridgal)
  pkspecdens = pktools.getpkdensest(vol,nc,vfrac,densgrid,weigriddens,sumwsqdens)
  pkspeccross = pktools.getpkcrossest(vol,nc,vfrac,ngal,galgrid,winspecgal,weigridgal,densgrid,weigriddens,sumwsqcross)
  pk0flatgal,pk2flatgal,pk4flatgal,nmodes = pktools.binpole(pkspecgal,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0flatdens,pk2flatdens,pk4flatdens,nmodes = pktools.binpole(pkspecdens,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0flatcross,pk2flatcross,pk4flatcross,nmodes = pktools.binpole(pkspeccross,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)

########################################################################
# Obtain auto- and cross-power spectrum multipole estimates in the     #
# curved-sky case.                                                     #
########################################################################

  pk0gal,pk2gal,pk4gal,nmodes = pktools.getpoleest(1,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,ngal,galgrid,sumwsqgal,weigridgal,wingridgal)
  pk0dens,pk2dens,pk4dens,nmodes = pktools.getpoleest(2,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,0,densgrid,vfrac,weigriddens,wingriddens)
  pk0cross,pk2cross,pk4cross,nmodes = pktools.getpolecrossest(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,ngal,galgrid,weigridgal,wingridgal,densgrid)

########################################################################
# Determine errors in the power spectrum multipoles.                   #
########################################################################

  pkdiagerr = pktools.getpoleerr(vol,ngal,vfrac,nkbin,pk0congal,pk2congal,pk4congal,pk0flatcondens,pk2flatcondens,pk4flatcondens,pk0concross,pk2concross,pk4concross,nmodes)
  pk0errgal,pk2errgal,pk4errgal,pk0errdens,pk2errdens,pk4errdens,pk0errcross,pk2errcross,pk4errcross = np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin),np.zeros(nkbin)
  for ik in range(nkbin):
    pk0errgal[ik] = pkdiagerr[ik,ik]
    pk2errgal[ik] = pkdiagerr[nkbin+ik,nkbin+ik]
    pk4errgal[ik] = pkdiagerr[2*nkbin+ik,2*nkbin+ik]
    pk0errdens[ik] = pkdiagerr[3*nkbin+ik,3*nkbin+ik]
    pk2errdens[ik] = pkdiagerr[4*nkbin+ik,4*nkbin+ik]
    pk4errdens[ik] = pkdiagerr[5*nkbin+ik,5*nkbin+ik]
    pk0errcross[ik] = pkdiagerr[6*nkbin+ik,6*nkbin+ik]
    pk2errcross[ik] = pkdiagerr[7*nkbin+ik,7*nkbin+ik]
    pk4errcross[ik] = pkdiagerr[8*nkbin+ik,8*nkbin+ik]

########################################################################
# Display power spectrum results to screen.                            #
########################################################################

  print '\nAuto-power spectrum measurements for galaxies:'
  print '    k      est    flat      err       cov      mod      con     flat     pix'
  print 'l=0:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk0gal[ik]),'{:8.1f}'.format(pk0flatgal[ik]),'{:8.1f}'.format(pk0errgal[ik]),'{:8.1f}'.format(pk0errgal[ik]),'{:8.1f}'.format(pk0modgal[ik]),'{:8.1f}'.format(pk0congal[ik]),'{:8.1f}'.format(pk0flatcongal[ik])
  print 'l=2:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk2gal[ik]),'{:8.1f}'.format(pk2flatgal[ik]),'{:8.1f}'.format(pk2errgal[ik]),'{:8.1f}'.format(pk2errgal[ik]),'{:8.1f}'.format(pk2modgal[ik]),'{:8.1f}'.format(pk2congal[ik]),'{:8.1f}'.format(pk2flatcongal[ik])
  print 'l=4:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk4gal[ik]),'{:8.1f}'.format(pk4flatgal[ik]),'{:8.1f}'.format(pk4errgal[ik]),'{:8.1f}'.format(pk4errgal[ik]),'{:8.1f}'.format(pk4modgal[ik]),'{:8.1f}'.format(pk4congal[ik]),'{:8.1f}'.format(pk4flatcongal[ik])

  print '\nAuto-power spectrum measurements for density:'
  print '    k      est    flat      err       cov      mod      con     flat     pix'
  print 'l=0:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk0dens[ik]),'{:8.1f}'.format(pk0flatdens[ik]),'{:8.1f}'.format(pk0errdens[ik]),'{:8.1f}'.format(pk0errdens[ik]),'{:8.1f}'.format(pk0moddens[ik]),'{:8.1f}'.format(pk0condens[ik]),'{:8.1f}'.format(pk0flatcondens[ik])
  print 'l=2:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk2dens[ik]),'{:8.1f}'.format(pk2flatdens[ik]),'{:8.1f}'.format(pk2errdens[ik]),'{:8.1f}'.format(pk2errdens[ik]),'{:8.1f}'.format(pk2moddens[ik]),'{:8.1f}'.format(pk2condens[ik]),'{:8.1f}'.format(pk2flatcondens[ik])
  print 'l=4:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk4dens[ik]),'{:8.1f}'.format(pk4flatdens[ik]),'{:8.1f}'.format(pk4errdens[ik]),'{:8.1f}'.format(pk4errdens[ik]),'{:8.1f}'.format(pk4moddens[ik]),'{:8.1f}'.format(pk4condens[ik]),'{:8.1f}'.format(pk4flatcondens[ik])

  print '\nCross-power spectrum measurements:'
  print '    k      est    flat      err       cov      mod      con     flat     pix'
  print 'l=0:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk0cross[ik]),'{:8.1f}'.format(pk0flatcross[ik]),'{:8.1f}'.format(pk0errcross[ik]),'{:8.1f}'.format(pk0errcross[ik]),'{:8.1f}'.format(pk0modcross[ik]),'{:8.1f}'.format(pk0concross[ik]),'{:8.1f}'.format(pk0flatconcross[ik])
  print 'l=2:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk2cross[ik]),'{:8.1f}'.format(pk2flatcross[ik]),'{:8.1f}'.format(pk2errcross[ik]),'{:8.1f}'.format(pk2errcross[ik]),'{:8.1f}'.format(pk2modcross[ik]),'{:8.1f}'.format(pk2concross[ik]),'{:8.1f}'.format(pk2flatconcross[ik])
  print 'l=4:'
  for ik in range(nkbin):
    print '{:5.3f}'.format(kbin[ik]),'{:8.1f}'.format(pk4cross[ik]),'{:8.1f}'.format(pk4flatcross[ik]),'{:8.1f}'.format(pk4errcross[ik]),'{:8.1f}'.format(pk4errcross[ik]),'{:8.1f}'.format(pk4modcross[ik]),'{:8.1f}'.format(pk4concross[ik]),'{:8.1f}'.format(pk4flatconcross[ik])

########################################################################
# Plot power spectra.                                                  #
########################################################################

  ncase = 3
  labelcase = ['$P_{gg}$','$P_{TT}$','$P_{gT}$']
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

  return

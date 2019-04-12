import numpy as np
import pktools
import getcorr

########################################################################
# Measure auto- and cross-power spectra of a galaxy distribution       #
# and another density field such as an intensity-mapping distribution. #
#                                                                      #
# Arguments:                                                           #
#                                                                      #
# doconv -- generate convolution                                       #
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
# sigvgal -- value of pairwise velocity dispersion for galaxies        #
# sigvdens -- value of pairwise velocity dispersion for density field  #
# bgal -- bias parameter for galaxies                                  #
# bdens -- bias parameter for density field                            #
# pknoise -- constant value of noise power spectrum in (Mpc/h)^3       #
# dobeam -- include telescope beam in model                            #
# sigdeg -- standard deviation of Gaussian beam [degrees]              #
# lwin,pixwin -- pixel window function array                           #
# dzbin -- width of redshift bins                                      #
# cosmo -- astropy fiducial cosmology                                  #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

def measpk(doconv,galgrid,wingridgal,densgrid,wingriddens,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,kmod,pkmod,betagal,betadens,sigvgal,sigvdens,bgal,bdens,pknoise,dobeam,sigdeg,lwin,pixwin,dzbin,cosmo):

########################################################################
# Initializations.                                                     #
########################################################################

  vol,nc = lx*ly*lz,nx*ny*nz
  wmean = np.sum(wingriddens)/nc
  vfrac = float(len(wingriddens[wingriddens != 0.]))/nc
  dk = (kmax-kmin)/float(nkbin)
  kbin = np.linspace(kmin+0.5*dk,kmax-0.5*dk,nkbin)
  ngal = np.sum(galgrid)
  sigvcross = 0.5*(sigvgal+sigvdens)

########################################################################
# Obtain Fourier transforms of window functions, and normalizations    #
# used in power spectrum estimation.                                   #
########################################################################

  sumwsqgal,sumwsqdens,sumwsqcross,winspecgal,weigridgal,winspecdens,weigriddens = pktools.getftwin(nx,ny,nz,wingridgal,wingriddens)

########################################################################
# Compute noise power spectrum.                                        #
########################################################################

  pknoise1 = (pknoise*vfrac)/(sumwsqdens*(wmean**2))
  print 'pknoise =','{:7.1f}'.format(pknoise1)
  
########################################################################
# Determine corrections to the power spectrum over an FFT grid.        #
# The galaxy power spectrum just considers the NGP assignment.         #
# The density and cross-power spectra consider the binning in the      #
# spherical healpix/redshift grid, and telescope beam.                 #
########################################################################

  pkgaldampspec = getcorr.getngpcorr(nx,ny,nz,lx,ly,lz,kmod,pkmod,betagal,sigvgal,bgal,0.)
  pkdensdampspec = getcorr.getdampcorr(False,nx,ny,nz,lx,ly,lz,x0,y0,z0,wingriddens,lwin,pixwin,dzbin,dobeam,sigdeg,kmax,kmod,pkmod,betadens,sigvdens,bdens,pknoise1,cosmo)
  pkcrossdampspec = getcorr.getdampcorr(True,nx,ny,nz,lx,ly,lz,x0,y0,z0,wingriddens,lwin,pixwin,dzbin,dobeam,sigdeg,kmax,kmod,pkmod,betagal,sigvgal,bgal,pknoise1,cosmo)

########################################################################
# Obtain model auto- and cross-power spectra over the FFT grid,        #
# multiply them by the corrections computed above, and bin them        #
# in power spectrum multipoles.                                        #
########################################################################

  print '\nComputing model power spectrum...'
  doindep,dohalf = True,True
  pkmodgalspec = pktools.getpkgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,betagal,sigvgal,bgal,0.)
  pkmoddensspec = pktools.getpkgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,betadens,sigvdens,bdens,pknoise1)
  pkmodcrossspec = pktools.getpkcrossgrid(nx,ny,nz,lx,ly,lz,dohalf,kmod,pkmod,betagal,betadens,sigvcross,bgal,bdens)
  pkmodgalspec *= pkgaldampspec
  pkmoddensspec *= pkdensdampspec
  pkmodcrossspec *= pkcrossdampspec
  pk0modgal,pk2modgal,pk4modgal,nmodes = pktools.binpole(pkmodgalspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0moddens,pk2moddens,pk4moddens,nmodes = pktools.binpole(pkmoddensspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0modcross,pk2modcross,pk4modcross,nmodes = pktools.binpole(pkmodcrossspec,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)

########################################################################
# Obtain convolved auto- and cross-power spectra over the FFT grid,    #
# multiply them by the corrections computed above, and bin them        #
# in power spectrum multipoles, in the flat-sky case.                  #
########################################################################

  if (doconv):
    doindep,dohalf = False,False
    pkcongalgrid = pktools.getpkconv(nx,ny,nz,lx,ly,lz,sumwsqgal,winspecgal,kmod,pkmod,betagal,sigvgal,bgal,0.)
    pkcondensgrid = pktools.getpkconv(nx,ny,nz,lx,ly,lz,sumwsqdens,winspecdens,kmod,pkmod,betadens,sigvdens,bdens,pknoise1)
    pkconcrossgrid = pktools.getpkcrossconv(nx,ny,nz,lx,ly,lz,sumwsqcross,winspecgal,winspecdens,kmod,pkmod,betagal,betadens,sigvcross,bgal,bdens)
    pkgaldampgrid = pktools.fthalftofull(nx,ny,ny,pkgaldampspec)
    pkdensdampgrid = pktools.fthalftofull(nx,ny,ny,pkdensdampspec)
    pkcrossdampgrid = pktools.fthalftofull(nx,ny,ny,pkcrossdampspec)
    pkcongalgrid *= pkgaldampgrid
    pkcondensgrid *= pkdensdampgrid
    pkconcrossgrid *= pkcrossdampgrid
    pk0flatcongal,pk2flatcongal,pk4flatcongal,nmodes = pktools.binpole(pkcongalgrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
    pk0flatcondens,pk2flatcondens,pk4flatcondens,nmodes = pktools.binpole(pkcondensgrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
    pk0flatconcross,pk2flatconcross,pk4flatconcross,nmodes = pktools.binpole(pkconcrossgrid,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  else:
    pk0flatcongal,pk2flatcongal,pk4flatcongal = pk0modgal,pk2modgal,pk4modgal
    pk0flatcondens,pk2flatcondens,pk4flatcondens = pk0moddens,pk2moddens,pk4moddens
    pk0flatconcross,pk2flatconcross,pk4flatconcross = pk0modcross,pk2modcross,pk4modcross

########################################################################
# Obtain convolved auto- and cross-power spectrum multipoles in the    #
# curved-sky case.                                                     #
########################################################################

  if (doconv):
    pk0congal,pk2congal,pk4congal = pktools.getpoleconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsqgal,weigridgal,wingridgal,kmod,pkmod,betagal,sigvgal,bgal,0.,pkgaldampgrid)
    pk0condens,pk2condens,pk4condens = pktools.getpoleconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsqdens,weigriddens,wingriddens,kmod,pkmod,betadens,sigvdens,bdens,pknoise1,pkdensdampgrid)
    pk0concross,pk2concross,pk4concross = pktools.getpolecrossconvharm(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,sumwsqcross,weigridgal,wingridgal,weigriddens,wingriddens,kmod,pkmod,betagal,betadens,sigvcross,bgal,bdens,pkcrossdampgrid)
  else:
    pk0congal,pk2congal,pk4congal = pk0modgal,pk2modgal,pk4modgal
    pk0condens,pk2condens,pk4condens = pk0moddens,pk2moddens,pk4moddens
    pk0concross,pk2concross,pk4concross = pk0modcross,pk2modcross,pk4modcross

########################################################################
# Obtain auto- and cross-power spectrum estimates over the FFT grid,   #
# and bin them in power spectrum multipoles, in the flat-sky case.     #
########################################################################

  doindep,dohalf = True,True
  pkspecgal = pktools.getpkgalest(vol,ngal,galgrid,sumwsqgal,winspecgal,weigridgal)
  pkspecdens = pktools.getpkdensest(vol,nc,wmean,densgrid,weigriddens,sumwsqdens)
  pkspeccross = pktools.getpkcrossest(vol,nc,wmean,ngal,galgrid,winspecgal,weigridgal,densgrid,weigriddens,sumwsqcross)
  pk0flatgal,pk2flatgal,pk4flatgal,nmodes = pktools.binpole(pkspecgal,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0flatdens,pk2flatdens,pk4flatdens,nmodes = pktools.binpole(pkspecdens,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)
  pk0flatcross,pk2flatcross,pk4flatcross,nmodes = pktools.binpole(pkspeccross,nx,ny,nz,lx,ly,lz,kmin,kmax,nkbin,doindep,dohalf)

########################################################################
# Obtain auto- and cross-power spectrum multipole estimates in the     #
# curved-sky case.                                                     #
########################################################################

  pk0gal,pk2gal,pk4gal,nmodes = pktools.getpoleest(1,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,ngal,galgrid,wmean,sumwsqgal,weigridgal,wingridgal)
  pk0dens,pk2dens,pk4dens,nmodes = pktools.getpoleest(2,nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,0,densgrid,wmean,sumwsqdens,weigriddens,wingriddens)
  pk0cross,pk2cross,pk4cross,nmodes = pktools.getpolecrossest(nx,ny,nz,lx,ly,lz,x0,y0,z0,kmin,kmax,nkbin,ngal,galgrid,weigridgal,wingridgal,densgrid,wmean,sumwsqcross,weigriddens)

########################################################################
# Determine errors in the power spectrum multipoles.                   #
########################################################################

  pkdiagerr = pktools.getpoleerr(vol,ngal,wmean,nkbin,pk0congal,pk2congal,pk4congal,pk0flatcondens,pk2flatcondens,pk4flatcondens,pk0concross,pk2concross,pk4concross,nmodes)
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
  print '    k      est    flat      err       cov      mod      con     flat'
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
  print '    k      est    flat      err       cov      mod      con     flat'
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
  print '    k      est    flat      err       cov      mod      con     flat'
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
# Write out data.                                                      #
########################################################################

#  pkfile = 'pkpole.dat'
#  pktools.writepolecross(pkfile,kmin,kmax,nkbin,ngal,0,nx,ny,nz,pk0gal,pk2gal,pk4gal,pk0dens,pk2dens,pk4dens,pk0cross,pk2cross,pk4cross,pk0errgal,pk2errgal,pk4errgal,pk0errdens,pk2errdens,pk4errdens,pk0errcross,pk2errcross,pk4errcross,pk0modgal,pk2modgal,pk4modgal,pk0moddens,pk2moddens,pk4moddens,pk0modcross,pk2modcross,pk4modcross,pk0congal,pk2congal,pk4congal,pk0condens,pk2condens,pk4condens,pk0concross,pk2concross,pk4concross,np.ones(nkbin),nmodes)

########################################################################
# Return results.                                                      #
########################################################################

  return pk0gal,pk2gal,pk4gal,pk0dens,pk2dens,pk4dens,pk0cross,pk2cross,pk4cross,pk0errgal,pk2errgal,pk4errgal,pk0errdens,pk2errdens,pk4errdens,pk0errcross,pk2errcross,pk4errcross,pk0congal,pk2congal,pk4congal,pk0condens,pk2condens,pk4condens,pk0concross,pk2concross,pk4concross

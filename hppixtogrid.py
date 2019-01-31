########################################################################
# Code to transfer (redshift, healpix) binning to (x,y,z) binning      #
# using Monte Carlo random catalogues.                                 #
#                                                                      #
# Arguments:                                                           #
# nzbin -- number of redshift bins of input map                        #
# nside -- healpix resolution                                          #
# denshpz[nzbin,npix] -- (redshift, healpix) data map                  #
# ipixlst[npix] -- list of healpix pixel IDs in the map                #
# zlims[nzbin+1] -- redshift values of bin divisions                   #
# dobound -- take centre of field as (rmin+rmax)/2,(dmin+dmax)/2?      #
# rmin,rmax -- minimum, maximum R.A. of the field                      #
# dmin,dmax -- minimum, maximum Dec. of the field                      #
# nx,ny,nz -- size of gridded cuboid                                   #
# lx,ly,lz -- dimensions of gridded cuboid [Mpc/h]                     #
# x0,y0,z0 -- co-ordinate origin in cuboid co-ordinates [Mpc/h]        #
# cosmo -- astropy fiducial cosmology                                  #
#                                                                      #
# Returns:                                                             #
# densgrid -- density map in (x,y,z) binning                           #
# wingrid -- (0,1) coverage map in (x,y,z) binning                     #
#                                                                      #
# Original code by Chris Blake (cblake@swin.edu.au).                   #
########################################################################

import numpy as np
import numpy_indexed as npi
import healpy as hp
import boxtools

def hppixtogrid(nzbin,nside,denshpz,ipixlst,zlims,dobound,rmin,rmax,dmin,dmax,nx,ny,nz,lx,ly,lz,x0,y0,z0,cosmo):
  rsets = 10       # Number of random sets to average
  nran = 5000000   # Number of points in each random set
  npix = len(ipixlst)
  print '\nMapping (healpix,redshift) binning to (x,y,z) binning...'
  print 'Healpix grid with nzbin =',nzbin,'npix =',npix,'ntot =',nzbin*npix
  print 'Cuboid grid with nx =',nx,'ny =',ny,'nz =',nz,'ntot =',nx*ny*nz
  print 'Sampling with random points...'
  print 'rsets =',rsets
  print 'nran =',nran
  countgrid,densgrid,wingrid = np.zeros((nzbin,npix)),np.zeros((nx,ny,nz)),np.zeros((nx,ny,nz))
  for iset in range(rsets):
    print 'Generating random set',iset+1,'...'
# Generate random points in cuboid
    rxpos,rypos,rzpos = boxtools.genransim(nran,lx,ly,lz)
# Convert random points to spherical co-ordinates
    rras,rdec,rred = boxtools.getradecred(dobound,rmin,rmax,dmin,dmax,rxpos,rypos,rzpos,x0,y0,z0,cosmo)
# Cut points within redshift bins
    izbin = np.digitize(rred,zlims) - 1
    cut = (izbin >= 0) & (izbin < nzbin)
    rxpos,rypos,rzpos,rras,rdec,izbin = rxpos[cut],rypos[cut],rzpos[cut],rras[cut],rdec[cut],izbin[cut]
# Cut points within healpix pixels
    rphi,rtheta = np.radians(rras),np.radians(90.-rdec)
    ipix = hp.ang2pix(nside,rtheta,rphi)
    cut = np.isin(ipix,ipixlst)
    rxpos,rypos,rzpos,izbin,ipix = rxpos[cut],rypos[cut],rzpos[cut],izbin[cut],ipix[cut]
# Re-index pixels to run from 1 to len(ipixlst)
    ipix = npi.indices(ipixlst,ipix)
# Count numbers in each (healpix,redshift) cell
    tempgrid,edges = np.histogramdd(np.vstack([izbin,ipix]).transpose(),bins=(nzbin,npix))
    countgrid += tempgrid
# Bin densities in each (x,y,z) cell
    rdens = denshpz[izbin,ipix]
    tempgrid,edges = np.histogramdd(np.vstack([rxpos,rypos,rzpos]).transpose(),bins=(nx,ny,nz),range=((0.,lx),(0.,ly),(0.,lz)),normed=False,weights=rdens)
    densgrid += tempgrid
# Count numbers in each (x,y,z) cell
    tempgrid,edges = np.histogramdd(np.vstack([rxpos,rypos,rzpos]).transpose(),bins=(nx,ny,nz),range=((0.,lx),(0.,ly),(0.,lz)))
    wingrid += tempgrid
  print 'Number of randoms in healpix grid mean =',np.mean(countgrid),'std =',np.std(countgrid),'nullfrac =',float(len(countgrid[countgrid == 0]))/float(nzbin*len(ipixlst))
  print 'Number of randoms in cuboid grid mean =',np.mean(wingrid[wingrid>0.]),'std =',np.std(wingrid[wingrid>0.])
# Average densities on (x,y,z) grid
  densgrid = np.where(wingrid>0.,densgrid/wingrid,0.)
# Determine (0,1) window function on (x,y,z) grid
  wingrid = np.where(wingrid>0.,1.,0.)
  return densgrid,wingrid

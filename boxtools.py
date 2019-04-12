import numpy as np

########################################################################
# Get dimensions of close-fitting cuboid for given range of (R.A.,     #
# Dec., redshift).                                                     #
########################################################################

def boxsize(redmin,redmax,dobound,rmin,rmax,dmin,dmax,cosmo):
  print '\nSurvey box:'
  rminr,rmaxr = np.radians(rmin),np.radians(rmax)
  dminr,dmaxr = np.radians(dmin),np.radians(dmax)
  distmin,distmax = cosmo.comoving_distance(redmin).value,cosmo.comoving_distance(redmax).value
  if (dobound):
    rcenr,dcenr = 0.5*(rminr+rmaxr),0.5*(dminr+dmaxr)
    crcen,srcen = np.cos(rcenr),np.sin(rcenr)
    cdcen,sdcen = np.cos(dcenr),np.sin(dcenr)
    rasrlst = np.array([rminr,rcenr,rmaxr])
    decrlst = np.array([dminr,dcenr,dmaxr])
    distlst = np.array([distmin,distmax])
  else:
    if ((rmax-rmin) > 180.):
      rasrlst = np.radians([0.,90.,180.,270])
    else:
      rasrlst = np.radians([rmin,rmax])
    decrlst = np.array([dminr,0.,dmaxr])
    distlst = np.array([distmax])
  nras = len(rasrlst)
  ndec = len(decrlst)
  ndist = len(distlst)
  xlst,ylst,zlst = np.empty(shape=(ndist,nras,ndec)),np.empty(shape=(ndist,nras,ndec)),np.empty(shape=(ndist,nras,ndec))
  for i in range(nras):
    for j in range(ndec):
      if (dobound):
        rasr,decr = dorot(np.array([rasrlst[i]]),np.array([decrlst[j]]),crcen,srcen,cdcen,sdcen)
      else:
        rasr,decr = rasrlst[i],decrlst[j]
      for k in range(ndist):
        xlst[k,i,j],ylst[k,i,j],zlst[k,i,j] = xyzconv(distlst[k],rasr,decr)
  xmin,xmax = np.amin(xlst),np.amax(xlst)
  ymin,ymax = np.amin(ylst),np.amax(ylst)
  zmin,zmax = np.amin(zlst),np.amax(zlst)
  lx,ly,lz = xmax-xmin,ymax-ymin,zmax-zmin
  x0,y0,z0 = -xmin,-ymin,-zmin
  print 'dobound =',dobound
  print ' ' + '{:5.1f}'.format(rmin) + ' < RA   < ' + '{:4.1f}'.format(rmax)
  print ' ' + '{:5.1f}'.format(dmin) + ' < Dec  < ' + '{:4.1f}'.format(dmax)
  print ' ' + '{:5.2f}'.format(redmin) + ' < red  < ' + '{:4.2f}'.format(redmax)
  print '{:6.1f}'.format(distmin) + ' < dist < ' + '{:6.1f}'.format(distmax)
  print '{:7.1f}'.format(xmin) + ' < x < ' + '{:6.1f}'.format(xmax) + ' L_x = ' + '{:6.1f}'.format(lx) + ' x_0 = ' + '{:6.1f}'.format(x0)
  print '{:7.1f}'.format(ymin) + ' < y < ' + '{:6.1f}'.format(ymax) + ' L_y = ' + '{:6.1f}'.format(ly) + ' y_0 = ' + '{:6.1f}'.format(y0)
  print '{:7.1f}'.format(zmin) + ' < z < ' + '{:6.1f}'.format(zmax) + ' L_z = ' + '{:6.1f}'.format(lz) + ' z_0 = ' + '{:6.1f}'.format(z0)
  return lx,ly,lz,x0,y0,z0

########################################################################
# Read in GiggleZ simulation data.                                     #
########################################################################

def readgiggquick():
  print '\nReading in GiggleZ simulation...'
  infile = 'GiggleZ_z0pt000_dark_subsample.ascii'
  print infile
  f = open(infile,'r')
  lines = f.readlines()[8:]
  xpos,ypos,zpos,xvel,yvel,zvel = [],[],[],[],[],[]
  for line in lines:
    fields = line.split()
    xpos.append(float(fields[1]))
    ypos.append(float(fields[2]))
    zpos.append(float(fields[3]))
    xvel.append(float(fields[4]))
    yvel.append(float(fields[5]))
    zvel.append(float(fields[6]))
  f.close()
  xpos,ypos,zpos = np.array(xpos),np.array(ypos),np.array(zpos)
  xvel,yvel,zvel = np.array(xvel),np.array(yvel),np.array(zvel)
  ndat = len(xpos)
  print ndat,'dark matter particles read in'
  return xpos,ypos,zpos,xvel,yvel,zvel,ndat

########################################################################
# Read in selection function.                                          #
########################################################################

def readwin(winfile,nx,ny,nz,lx,ly,lz):
  print '\nReading in window function...'
  print winfile
  f = open(winfile,'r')
  f.readline()
  fields = f.readline().split()
  nxwin,nywin,nzwin = int(fields[0]),int(fields[1]),int(fields[2])
  lxwin,lywin,lzwin = float(fields[3]),float(fields[4]),float(fields[5])
  if ((nxwin != nx) | (nywin != ny) | (nzwin != nz) | (np.abs(lxwin-lx) > 0.1) | (np.abs(lywin-ly) > 0.1) | (np.abs(lzwin-lz) > 0.1)):
    print '** Window function file wrong size!!'
    print nx,ny,nz,nxwin,nywin,nzwin
    print lx,ly,lz,lxwin,lywin,lzwin
    sys.exit()
  if (len(fields) < 9):
    f.readline()
  wingrid = np.zeros(shape=(nx,ny,nz))
  for iz in range(nz):
    for iy in range(ny):
      for ix in range(nx):
        wingrid[ix,iy,iz] = float(f.readline())
  f.close()
  print 'Number of randoms =','{:.2e}'.format(np.sum(wingrid))
  return wingrid

########################################################################
# Convert redshift / angular catalogue to (x,y,z) co-ordinates.        #
########################################################################

def getxyz(dobound,rmin,rmax,dmin,dmax,ras,dec,red,cosmo):
  rasr,decr = np.radians(ras),np.radians(dec)
  zarr = np.linspace(0.,2.,1000)
  rarr = cosmo.comoving_distance(zarr).value
  dist = np.interp(red,zarr,rarr)
  if (dobound):
    rcenr,dcenr = np.radians(0.5*(rmin+rmax)),np.radians(0.5*(dmin+dmax))
    crcen,srcen = np.cos(rcenr),np.sin(rcenr)
    cdcen,sdcen = np.cos(dcenr),np.sin(dcenr)
    rasr,decr = dorot(rasr,decr,crcen,srcen,cdcen,sdcen)
  xpos,ypos,zpos = xyzconv(dist,rasr,decr)
  return xpos,ypos,zpos

########################################################################
# Convert angles to new (ra,dec) centre.                               #
########################################################################

def dorot(r1r,d1r,crcen,srcen,cdcen,sdcen):
  cd1r = np.cos(d1r)
  x = cd1r*np.cos(r1r)
  y = cd1r*np.sin(r1r)
  z = np.sin(d1r)
  x1 = x*crcen + y*srcen
  y1 = -x*srcen + y*crcen
  z1 = z
  x = x1*cdcen + z1*sdcen
  y = y1
  z = -x1*sdcen + z1*cdcen
  r2r,d2r = radecconv(x,y,z)
  return r2r,d2r

########################################################################
# Convert (x,y,z) co-ordinates to distance and angles.                 #
########################################################################

def radecconv(x,y,z):
  rasr = np.arctan(abs(y/x))
  rasr[x < 0] = np.pi - rasr[x < 0]
  rasr[y < 0] = -rasr[y < 0]
  decr = np.arctan(z/np.sqrt(x**2 + y**2))
  return rasr,decr

########################################################################
# Convert distance and angles to (x,y,z) co-ordinates.                 #
########################################################################

def xyzconv(r,rasr,decr):
  cdecr = np.cos(decr)
  x = r*cdecr*np.cos(rasr)
  y = r*cdecr*np.sin(rasr)
  z = r*np.sin(decr)
  return x,y,z

########################################################################
# Apply RSD along the radial direction to (x,y,z) positions.           #
########################################################################

def applyrsd(dxpos,dypos,dzpos,dxvel,dyvel,dzvel,lx,ly,lz,x0,y0,z0):
  print '\nApplying RSD along radial direction...'
  dxpos -= x0
  dypos -= y0
  dzpos -= z0
  drvel = (dxvel*dxpos + dyvel*dypos + dzvel*dzpos)/((dxpos**2) + (dypos**2) + (dzpos**2))
  dxpos *= (1. + drvel/100.)
  dypos *= (1. + drvel/100.)
  dzpos *= (1. + drvel/100.)
  dxpos += x0
  dypos += y0
  dzpos += z0
  dxpos[dxpos < 0.] += lx
  dxpos[dxpos > lx] -= lx
  dypos[dypos < 0.] += ly
  dypos[dypos > ly] -= ly
  dzpos[dzpos < 0.] += lz
  dzpos[dzpos > lz] -= lz
  return dxpos,dypos,dzpos

########################################################################
# Randomly select half the particles for a second tracer.              #
########################################################################

def makecrosssample(dxpos,dypos,dzpos,ndat):
  print '\nSplitting data into two random sub-samples...'
  insamp1 = np.random.choice([True,False],size=ndat)
  insamp2 = np.invert(insamp1)
  dxpos1,dypos1,dzpos1 = dxpos[insamp1],dypos[insamp1],dzpos[insamp1]
  dxpos2,dypos2,dzpos2 = dxpos[insamp2],dypos[insamp2],dzpos[insamp2]
  ndat1,ndat2 = len(dxpos1),len(dxpos2)
  print 'ndat1 =',ndat1,'ndat2 =',ndat2
  return dxpos1,dypos1,dzpos1,ndat1,dxpos2,dypos2,dzpos2,ndat2

########################################################################
# Determine (0,1) window function for a survey cone embedded in a      #
# cuboid.                                                              #
########################################################################

def getmodwingrid(nx,ny,nz,lx,ly,lz,x0,y0,z0,dobound,rmin,rmax,dmin,dmax,zmin,zmax,cosmo):
  print '\nGenerating model window function...'
  print '{:5.1f}'.format(rmin) + ' < R.A. < ' + '{:5.1f}'.format(rmax)
  print '{:5.1f}'.format(dmin) + ' < Dec. < ' + '{:5.1f}'.format(dmax)
  print '{:5.2f}'.format(zmin) + ' < z    < ' + '{:5.2f}'.format(zmax)
  dx,dy,dz = lx/nx,ly/ny,lz/nz
  x = dx*np.arange(nx) - x0 + 0.5*dx
  y = dy*np.arange(ny) - y0 + 0.5*dy
  z = dz*np.arange(nz) - z0 + 0.5*dz
  rasgrid,decgrid,redgrid = getradecred(dobound,rmin,rmax,dmin,dmax,x[:,np.newaxis,np.newaxis],y[np.newaxis,:,np.newaxis],z[np.newaxis,np.newaxis,:],0.,0.,0.,cosmo)
  cutgrid = (rasgrid > rmin) & (rasgrid < rmax) & (decgrid > dmin) & (decgrid < dmax) & (redgrid > zmin) & (redgrid < zmax)
  wingrid = np.where(cutgrid,1.,0.)
  print len(wingrid[wingrid == 1.]),'cells with W=1'
  print len(wingrid[wingrid == 0.]),'cells with W=0'
  return wingrid

########################################################################
# Generate uniform random distribution in box.                         #
########################################################################

def genransim(nran,lx,ly,lz):
  rxpos = lx*np.random.rand(nran)
  rypos = ly*np.random.rand(nran)
  rzpos = lz*np.random.rand(nran)
  return rxpos,rypos,rzpos

########################################################################
# Convert (x,y,z) co-ordinates to (R.A.,Dec.,redshift) in curved sky.  #
########################################################################

def getradecred(dobound,rmin,rmax,dmin,dmax,xpos,ypos,zpos,x0,y0,z0,cosmo):
  ras,dec = getradec(dobound,rmin,rmax,dmin,dmax,xpos,ypos,zpos,x0,y0,z0)
  dist = np.sqrt(((xpos-x0)**2)+((ypos-y0)**2)+((zpos-z0)**2))
  redarr = np.linspace(0.,2.,1000)
  distarr = cosmo.comoving_distance(redarr).value
  red = np.interp(dist,distarr,redarr)
  return ras,dec,red

########################################################################
# Convert (x,y,z) co-ordinates to (R.A.,Dec.) in curved sky.           #
########################################################################

def getradec(dobound,rmin,rmax,dmin,dmax,xpos,ypos,zpos,x0,y0,z0):
  if (dobound):
    rcenr,dcenr = np.radians(0.5*(rmin+rmax)),np.radians(0.5*(dmin+dmax))
    crcen,srcen = np.cos(rcenr),np.sin(rcenr)
    cdcen,sdcen = np.cos(dcenr),np.sin(dcenr)
    xpos1 = (xpos-x0)*cdcen - (zpos-z0)*sdcen
    ypos1 = ypos-y0
    zpos1 = (xpos-x0)*sdcen + (zpos-z0)*cdcen
    xpos2 = xpos1*crcen - ypos1*srcen
    ypos2 = xpos1*srcen + ypos1*crcen
    zpos2 = zpos1
  else:
    xpos2 = xpos-x0
    ypos2 = ypos-y0
    zpos2 = zpos-z0
  ras,dec = radecconv(xpos2,ypos2,zpos2)
  ras,dec = np.degrees(ras),np.degrees(dec)
  ras[ras < 0] += 360.
  return ras,dec

########################################################################
# Grid galaxy distribution.                                            #
########################################################################

def discret(xpos,ypos,zpos,nx,ny,nz,lx,ly,lz,x0,y0,z0):
  print '\nGridding',len(xpos),'objects...'
  datgrid,edges = np.histogramdd(np.vstack([xpos+x0,ypos+y0,zpos+z0]).transpose(),bins=(nx,ny,nz),range=((0.,lx),(0.,ly),(0.,lz)))
  return datgrid

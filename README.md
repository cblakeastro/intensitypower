# intensitypower
This package is a set of python functions to measure and model the auto- 
and cross-power spectrum multipoles of galaxy catalogues and radio 
intensity maps presented in spherical co-ordinates.

This package accompanies the paper "Power spectrum modelling of galaxy 
and radio intensity maps including observational effects".

https://arxiv.org/abs/1902.07439

We assume that the galaxy catalogue is a set of discrete points, and the 
radio intensity map is a pixelized continuous field which includes:

* angular pixelization using healpix
* binning in redshift channels
* smoothing by a Gaussian telescope beam
* addition of a Gaussian noise in each cell

The galaxy catalogue and radio intensity map are transferred onto an FFT 
grid, and power spectrum multipoles are measured using curved-sky 
effects.  Both maps include redshift-space distortions.

The python codes presented are:

* runspherpk.py -- build mock galaxy and intensity mapping datasets from 
a dark matter simulation, and measure and model their auto- and 
cross-power spectrum multipoles.

* hppixtogrid.py -- transfer (redshift, healpix) binning to (x,y,z) 
binning using Monte Carlo random catalogues.

* getcorr.py -- generate corrections to the power spectra for the 
various observational effects.

* measpk.py -- model and measure the auto- and cross-power spectrum 
multipoles of the galaxy catalogue and density field.

* pktools.py -- set of functions for modelling and measuring the power 
spectra.

* boxtools.py -- set of functions for manipulating the survey cone 
within the Fourier cuboid.

* sphertools.py -- set of functions for manipulating the (redshift, 
healpix) density field.

The python libraries needed to run the functions are:

* numpy

* scipy

* healpy

* astropy

* matplotlib

* numpy_indexed

The other accompanying files are:

* GiggleZ_z0pt000_dark_subsample.ascii.gz -- dark matter subsample of the 
GiggleZ N-body simulation, used to produce the example mock datasets.

* pkcambhalofit_zeq0_gigglez.dat -- CAMB halofit model power spectrum 
with the same fiducial cosmology as GiggleZ.

* pixwin_nside128.dat -- healpix window function for nside=128 (extended 
to higher multipoles than provided with healpix).

* pkpole_runspherpk.png -- plot of power spectrum model and measurements 
for the default run, produced by the code.

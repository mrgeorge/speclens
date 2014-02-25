speclens
========

speclens is a package for simulating galaxy images and velocity maps
for spectroscopic weak lensing. The goal is to determine optimal 
approaches to measuring galaxy kinematics for the suppression of shape noise, including the
dependence on spatial and spectral resolution, and uncertainties in
seeing and pointing.


External Dependencies
---------------------

speclens currently uses the following packages:

* python, and common libaries like numpy, matplotlib, scipy

* [emcee](http://dan.iel.fm/emcee/) Dan Foreman-Mackey's parallel
  sampler for fast MCMC

Optional:

* [acorr](https://github.com/dfm/acor) For MCMC convergence testing

* [GalSim](https://github.com/GalSim-developers/GalSim) The GalSim
  package for galaxy images, PSFs, shear. Currently GalSim is used as
  an optional path to generating, convolving, and shearing galaxy
  images. It will be used for testing against the mainline approach
  of pixel arrays.

The following packages are imported by some scripts in scratch/ but
aren't essential for the main functions:

* [fitsio](https://github.com/esheldon/fitsio) Erin Sheldon's fitsio
  for python

* [esutil](http://code.google.com/p/esutil/) Erin Sheldon's python
  utilities for some cosmological distance calculations

* [astropy](http://www.astropy.org/) For ascii table io

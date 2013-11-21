speclens
========

speclens is a package for simulating galaxy images and velocity maps
for spectroscopic weak lensing. The code is in an early development
stage. The goal is to determine optimal approaches to measuring galaxy
kinematics for the suppression of shape noise, including the
dependence on spatial and spectral resolution, and uncertainties in
seeing and pointing.

speclens is modeled on the
[GalSim](https://github.com/GalSim-developers/GalSim) package.


External Dependencies
---------------------

speclens currently uses the following packages. Some of these can probably be removed.

* python, and common libaries like numpy, matplotlib, scipy

* [GalSim](https://github.com/GalSim-developers/GalSim) The GalSim package for galaxy images, PSFs, shear. There have been some issues attempting to integrate with this code due to real-space convolution bugs with velocity maps that have total flux=0. Current trend has been to reduce/eliminate dependency on GalSim, but re-integration may eventually be desired.

* [emcee](http://dan.iel.fm/emcee/) Dan Foreman-Mackey's parallel sampler for fast MCMC

* [fitsio](https://github.com/esheldon/fitsio) Erin Sheldon's fitsio for python

* [esutil](http://code.google.com/p/esutil/) Erin Sheldon's python utilities for some cosmological distance calculations

* [astropy](http://www.astropy.org/) For ascii table io

#! env python

import numpy as np
import os
import sys
import copy
import scipy.stats
from scipy.signal import fftconvolve

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens
import speclens.sim as sim


class Aperture(object):
    """
    Labels the positions, orientations, and dimensions of the slits,
    IFU pixels, or fibers used to observe the object.  Positions,
    dimensions are meant to be given in pixel coordinates, cented on
    the image center.

    Each aperture consists of a list of the coordinates of the
    pixels included in the observation.

    The "size" parameter here is the fiber diameter, the slit width, or the side length of an ifu pixel.
    
    TODO: Add keyword to allow positions and dimensions to be given in
    sky coordinates as well.
    """
    def __init__(self, model = None, xcenter = 0., ycenter= 0., size = 2., obsType="fiber", position_angle = 0.):
        availableObsTypes = ["fiber","slit","pixel"]
        if model == None:  # model should be provided. If not, make a default model.
            model = speclens.Model("B")
            xx, yy = np.meshgrid(np.arange(model.nPix) - float(model.nPix)/2.,np.arange(model.nPix) - float(self.nPix)/2.)

        if type(obsType) is str:
            assert obsType in availableObsTypes, "Problem: obsType argument to Aperture must be one of: "+", ".join(availableObsTypes)+"."
            if obsType == "slit":
                #xnew =  xx * np.cos(x) + yy * np.sin(x)
                ynew = -(xx - xcenter) * np.sin(position_angle) + (yy - ycenter) * np.cos(position_angle)
                aperture_pixels = np.where(ynew <= size/2.)
            if obsType == "fiber":
                dist = np.sqrt((xx - xcenter) * (xx - xcenter)  +  ( yy - ycenter ) * ( yy - ycenter))
                aperture_pixels = np.where(dist <= size/2.)
            if obsType == "pixel":
                xnew =  (xx - xcenter) * np.cos(position_angle) + (yy - ycenter) * np.sin(position_angle)
                ynew = -(xx - xcenter) * np.sin(position_angle) + (yy - ycenter) * np.cos(position_angle)
                aperture_pixels = np.where( (np.abs(xnew) <= size/2.) and (np.abs(ynew) <= size/2.))
                
        if type(obsType) is list:
            assert (len(center) == len(size) == len(obsType)), "Problem: Lengths of center, size, and obsType lists should be the same."
            for i in np.arange(len(obsType)): 
                assert obsType[i] in availableObsTypes, "Problem: All obsType arguments to Aperture must be one of: "+", ".join(availableObsTypes)+"."
                aperture_pixels = []
                if obsType[i] == "slit":
                    #xnew =  xx * np.cos(position_angle) + yy * np.sin(position_angle)
                    ynew = -(xx - xcenter[i]) * np.sin(position_angle[i]) + (yy - ycenter[i]) * np.cos(position_angle[i])
                    if i == 0: 
                        aperture_pixels = [np.where(ynew <= size/2.)]
                    else:


                        aperture_pixels.append(np.where(ynew <= size/2.))
                if obsType[i] == "fiber":
                    dist = np.sqrt((xx - xcenter) * (xx - xcenter)  +  ( yy - ycenter ) * ( yy - ycenter))
                    if i == 0:
                        aperture_pixels = np.where(dist <= size/2.)
                    else:
                        aperture_pixels.append(np.where(dist <= size/2.))
                if obsType == "pixel":
                    xnew =  (xx - xcenter) * np.cos(position_angle) + (yy - ycenter) * np.sin(position_angle)
                    ynew = -(xx - xcenter) * np.sin(position_angle) + (yy - ycenter) * np.cos(position_angle)
                    if i == 0:
                        aperture_pixels = np.where( (np.abs(xnew) <= size/2.) and (np.abs(ynew) <= size/2.))
                    else:
                        aperture_pixels.append((np.abs(xnew) <= size/2.) and (np.abs(ynew) <= size/2.))
        self.aperture_pixels = aperture_pixels



            
class EmissionLine(object):
    """
    EmissionLine holds the basic information about a single emission
    line. These objects are meant to be backed into arrays, which are
    iterated over when constructing galaxy spectra.  

    TODO: Add function which uses provided parameters to return the
    line profile evaluated at provided wavelengths. This will allow us
    to use arbitrary line profiles.
    """

    def __init__(self,center = 3727.30, width = 1., peak = 1e-17, name="OII"):
        self.Center = center
        self.Width = width # Linewidth (angstroms)
        self.PeakFlux = peak
        self.Name = name

        
class Datacube(object):
    """
    DataCube class storing parameters for, and instantiating, a
    spectral data cube.  Use a Model class to specify the
    surface-brightness profile; if none provided, then instantiate a
    default Model.
    
    Set additional parameters necessary for constructing the spectrum,
    but don't actually build until needed.
    """

    def __init__(self, model = None, linelist = None):
        if model == None:
            model = speclens.Model("B")
        assert isinstance(model,speclens.Model), "Argument to Datacube, if provided, must be a Model class instance"
        self.galaxyModel = model

        # Parameters describing the observation.
        self.expTime = 1. # Exposure time (s)
        self.aperture = 10. # Collecting area diameter (m)
        self.spectralResolution = 10000. # Spectral resolution, Delta lambda / lambda
        self.Npix = model.nPix # Number of spatial pixels on image side
        self.seeingScale = 40. # characteristic wavenumber of seeing disk, in arcsec^-1. 40 gets us a ~1'' psf
        self.pixelScale = 0.2  # Image Pixel scale, in arcsec per pixel
        self.deltaLambdaObs = 1. # wavelength scale, in Angstroms per pixel
        self.LambdaObsMin = 3000. # Starting wavelength, in Angstroms
        self.nSpectralPixels = 100.
        self.lambdaObs = self.deltaLambdaObs * np.arange(self.nSpectralPixels)+ self.LambdaObsMin
        
        #Parameters describing the galaxy spectrum.
        self.baseModel = model
        self.z = model.redshift # galaxy redshift
        self.ContinuumReferenceWavelength = 5000. # Wavelength at which to normalize the continuum, in Angstroms
        self.ContinuumReferenceNorm = 1e-18 # Value of continuum at reference wavelength, in erg/s/cm^2
        
        #Set up the default linelist. The user can change this.
        if linelist == None:
            self._defaultLinelist = ["OII","Hbeta","OIIIa","OIIIb","Halpha"] # Name of emission line.
            self._defaultLineCenters = [3727.30, 4861.33, 4958.91, 5006.84, 6562.82] # Location of line center, in Angstroms
            self._defaultLinewidth =  [1.,1.,1.,1.,1.] # 1-sigma width of emission line, in Angstroms
            self._defaultLinePeakFlux = [1e-17, 1e-17, 1e-17, 1e-17, 1e-17] # peak line flux, in erg/s/cm^2
            self.LineList = []
            for i in range(len(self._defaultLinePeakFlux)): 
                self.LineList.append(EmissionLine(center = self._defaultLineCenters[i],\
                                                  width  = self._defaultLinewidth[i],\
                                                  peak = self._defaultLinePeakFlux[i],\
                                                  name = self._defaultLinelist[i]))
        else:
            self.LineList = linelist
        
 
 
    def _makePsfConvKernel(self):
        # Draw the galaxy surface-brightness profile.
        image = sim.makeImageBessel(self.galaxyModel)
        #Make a seeing disk.
        kx, ky = np.meshgrid(np.arange(self.Npix) - float(self.Npix)/2.,np.arange(self.Npix) - float(self.Npix)/2.)
        kk2 = (kx*kx + ky*ky)/(self.seeingScale)**2
        Pk = np.exp(-(kk2)**(5./6.)) #Kolmogorov power spectrum.
        Pr = np.real(np.fft.fft2(Pk))
        psf = np.fft.fftshift(Pr)
        # Convolve this model profile with a sensible seeing disk.
        return psf

    
    def makeDataCube(self):
        # draw the psf-convolved galaxy surface-brightness profile.
        image = sim.makeImageBessel(self.galaxyModel)
        xx, yy = np.meshgrid(np.arange(self.Npix) - float(self.Npix)/2.,np.arange(self.Npix) - float(self.Npix)/2.)
        vmap = sim.vmapModel(self.baseModel,xx,yy)
        self._loadTransparency()
        self._loadThroughput()
        self._loadSky()
        self._loadGalaxySpectrum()
        # Loop  over the  spatial  grid;  in each  pixel,  look up  the
        # velocity,  and  interpolate  the  galaxy  spectrum  onto  the
        # redshift wavelength scale.
        c = 300000. # Speed of light, km/s
        dataCube = np.zeros([self.Npix,self.Npix,self.nSpectralPixels])
        thisSky = np.interp(self.lambdaObs,self.lambdaSky,self.skyFluxTemplate)
        thisTransparency = np.interp(self.lambdaObs,self.lambdaTransparency,self.transparencyTemplate)
        for i in np.arange(self.Npix):
            for j in np.arange(self.Npix):
                z_velocity = vmap[i,j] / c 
                lambdaRest = self.lambdaObs * (1 + self.z) * (1 + z_velocity)
                dataCube[i,j,:] = image[i,j] * (np.interp(lambdaRest,self.lambdaGalaxy,self.GalaxyFlux) * \
                  thisTransparency + thisSky) / np.max(image)
        self.dataCube = dataCube

        # Finally, step through the datacube and convolve each spectral slice with the psf.
        psf = self._makePsfConvKernel()
        for i in np.arange(self.nSpectralPixels):
            dataCube[:,:,i] = fftconvolve(dataCube[:,:,i],psf,mode="same")

        return dataCube
        

        
    def _loadSky(self):
        '''
        Read the sky spectral template, convert to erg/s/cm^2, and return
        the spectrum interpolated to observer wavelength grid.
        '''
        path,filename = os.path.split(sim.__file__)
        datadir = os.path.join(os.path.abspath(path),"../data/")
        skyFile = datadir+"kpno_sky.txt"
        skyTemplate = np.loadtxt(skyFile)
        self.lambdaSky = skyTemplate[:,0]
        self.skyFluxTemplate = 10.**((21.572-skyTemplate[:,1])/2.5)*1e-17

    def _loadTransparency(self):
        '''
        Read the sky transparency template, return the transparency
        interpolated to observer wavelength grid.
        '''
        path,filename = os.path.split(sim.__file__)
        datadir = os.path.join(os.path.abspath(path),"../data/")
        transparencyFile = datadir+"atmtrans_default.dat"
        transparencyData = np.loadtxt(transparencyFile)
        self.lambdaTransparency = transparencyData[:,0]
        self.transparencyTemplate = transparencyData[:,1]

    def _loadThroughput(self):
        '''
        Read the instrumental throughput model, return the fractional
        throughput interpolated to observer wavelength grid.
        '''
        throughput = np.zeros(self.nSpectralPixels)+1.
        return throughput
        

    def _loadGalaxySpectrum(self, templateFile="ssp_100Myr_z008.spec"):
        '''
        Read the galaxy spectrum, convert to flux (erg/s/cm^2), return
        galaxy spectrum normalized to some fiducial distance.
        '''
        # First, go and read in the continuum.
        path,filename = os.path.split(sim.__file__)
        datadir = os.path.join(os.path.abspath(path),"../data/bc03_templates/")
        galaxyFile = datadir+templateFile
        galaxyData = np.loadtxt(galaxyFile)
        self.lambdaGalaxy = galaxyData[:,0]
        self.GalaxyFlux = galaxyData[:,1]
        fluxNorm = self.ContinuumReferenceNorm  / np.interp(self.ContinuumReferenceWavelength ,self.lambdaGalaxy, self.GalaxyFlux)
        self.GalaxyFlux *= fluxNorm

        # Then, add lines from our internal linelist at the appropriate wavelengths.
        for line in self.LineList:
            self.GalaxyFlux += line.PeakFlux /(np.sqrt(2* np.pi) * line.Width) * \
              np.exp(-( (self.lambdaGalaxy - line.Center)**2 )/(2 * line.Width))


import matplotlib.pyplot as plt

data = Datacube()
cube = data.makeDataCube()
plot = plt.plot(cube[50,50,:])
plt.savefig("cube_peak_spectrum")

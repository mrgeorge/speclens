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


class EmissionLine(object,center = 3727.30, width = 1., peak = 1e-17, name="OII"):
    def __init__(self):
        self.Center = center
        self.Width = width #Linewidth (angstroms)
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

    def __init__(self, model, linelist):
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
        self.LambaObsMin = 3000. # Starting wavelength, in Angstroms
        self.nSpectralPixels = 1000.
        self.lambdaObs = self.deltaLambdaObs * np.arange(self.nSpectralPixels)+ self.LambdaObsMin
        
        #Parameters describing the galaxy spectrum.
        self.baseModel = Model
        self.z = z #galaxy Redshift
        self.ContinuumReferenceWavelength = 5000. # Wavelength at which to normalize the continuum, in Angstroms
        self.ContinuumReferenceNorm = 1e-18 # Value of continuum at reference wavelength, in erg/s/cm^2
        
        #Set up the default linelist. The user can change this.
        if linelist == None:
            self._defaultLinelist = ["OII","Hbeta","OIIIa","OIIIb","Halpha"] # Name of emission line.
            self._detaultLineCenters = [3727.30, 4861.33, 4958.91, 5006.84, 6562.82] # Location of line center, in Angstroms
            self._defaultLinewidth =  [1.,1.,1.,1.,1.] # 1-sigma width of emission line, in Angstroms
            self._defaultLinePeakFlux = [1e-17, 1e-17, 1e-17, 1e-17, 1e-17] # peak line flux, in erg/s/cm^2
            self.LineList = []
            for i in range(len(self._defaultLinePeakFux)): 
                self.LineList.append(EmissionLine(center = self._defaultLineCenters[i],\
                                                  width  = self._defaultLinewidth[i],\
                                                  peak = self._defaultLinePeakFlux[i],\
                                                  name = self._defaultLinelist[i]))
        else:
            self.LineList = linelist
        
 
 
    def makePsfConvImage(self):
        # Draw the galaxy surface-brightness profile.
        image = sim.makeImageBessel(self.galaxyModel)
        #Make a seeing disk.
        kx, ky = np.meshgrid(np.arange(self.Npix) - float(self.Npix)/2.,np.arange(self.Npix) - float(self.Npix)/2.)
        kk2 = (kx*kx + ky*ky)/(self.SeeingScale)**2
        Pk = np.exp(-(kk2)**(5./6.)) #Kolmogorov power spectrum.
        Pr = np.real(np.fft.fft2(Pk))
        Pr = np.fft.fftshift(Pr)
        # Convolve this model profile with a sensible seeing disk.
        image_smeared = fftconvolve(image,Pr,mode='same')
        return image_smeared

    
    def makeDataCube(self):
        # draw the psf-convolved galaxy surface-brightness profile.
        image_smeared = self.makePsfConvImage()
        xx, yy = np.meshgrid(np.arange(self.Npix) - float(self.Npix)/2.,np.arange(self.Npix) - float(self.Npix)/2.)
        vmap = sim.vmapModel(model,xx,yy)
        self._loadTransparency()
        self._loadThroughput()
        self._loadSky()
        self._loadGalaxySpectrum()
        


        galaxySpec = (self._loadGalaxySpectrum() * transparency + skyspec) * throughput
        dataCubeSky = np.einsum('...ij','...k',image_smeared,skyspec)
        dataCube = np.einsum('..ij','...k',image_smeared,galaxySpec)
        return dataCube, dataCubeSky
    

        
    def _loadSky(self):
        '''
        Read the sky spectral template, convert to erg/s/cm^2, and return
        the spectrum interpolated to observer wavelength grid.
        '''
        path,filename = os.path.split(sim.__file__)
        datadir = os.path.join(os.path.abspath(path),"../data/")
        skyFile = datadir+"kpno_sky.txt"
        skyTemplate = np.loadtxt(skyFile)
        self.lambdaSky = sky[:,0]
        self.skyFluxTemplate = 10.^((21.572-skyTemplate[:,1])/2.5)*1e-17

    def _loadTransparency(self):
        '''
        Read the sky transparency template, return the transparency
        interpolated to observer wavelength grid.
        '''
        path,filename = os.path.split(sim.__file__)
        datadir = os.path.join(os.path.abspath(path),"../data/")
        transparencyFile = datadir+"atmtrans_default.dat"
        transparencyData = np.loadtxt(skyFile)
        self.lambdaTransparency = transparency[:,0]
        self.transparencyTemplate = transparency[:,1]

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
        datadir = os.path.join(os.path.abspath(path),"../data/")
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



        



            
model = speclens.Model("B")
#Draw an image.



# Make a spectrum.

# Interpolate the sky spectrum onto the wavelength range under consideration.

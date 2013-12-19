import numpy as np
import sim

class Galaxy(object):
    """Galaxy class defining intrinsic parameters of a galaxy object"""

    def __init__(self, diskRadius=1., diskVRadius=2.2, diskCA=0.2,
                 diskSersic=1., diskVCirc=200., diskFlux=1.,
                 bulgeFraction=0., bulgeRadius=1., bulgeSersic=4.):
        self.diskRadius=diskRadius
        self.diskVRadius=diskVRadius
        self.diskCA=diskCA
        self.diskSersic=diskSersic
        self.diskVCirc=diskVCirc
        self.diskFlux=diskFlux
        self.bulgeFraction=bulgeFraction
        self.bulgeRadius=bulgeRadius
        self.bulgeSersic=bulgeSersic

class Observation(object):
    """Observation class takes a galaxy and produces observables"""

    def __init__(self, galaxy, redshift=0.5, diskPA=0., diskBA=0.7,
                 g1=0., g2=0., atmosFWHM=None, pixScale=0.1, nPix=100,
                 obsConfig="crossSlit", obsSize=1., nObs=10):
        self.galaxy=galaxy
        self.redshift=redshift
        self.diskPA=diskPA
        self.diskBA=diskBA
        self.g1=g1
        self.g2=g2
        self.atmosFWHM=atmosFWHM
        self.pixScale=pixScale
        self.nPix=nPix
        self.obsConfig=obsConfig
        self.obsSize=obsSize
        self.nObs=nObs

    def _drawMaps(self):
        vmapArr,fluxVMapArr,imgArr=sim.makeGalVMap2(self.galaxy.bulgeSersic,
            self.galaxy.bulgeRadius, self.galaxy.diskSersic,
            self.galaxy.diskRadius, self.galaxy.bulgeFraction,
            np.cos(np.deg2rad(self.diskInclination)), self.diskPA,
            self.galaxy.diskFlux, "flat", [self.galaxy.diskVCirc],
            self.g1, self.g2)
        self.vmap=vmapArr
        self.fvmap=fluxVMapArr
        self.image=imgArr
    
    def getImage(self):
        try:
            return self.image
        except AttributeError:
            self._drawMaps()
            return self.image

    def getVMap(self):
        try:
            return self.vmap
        except AttributeError:
            self._drawMaps()
            return self.vmap

    def getFVMap(self):
        try:
            return self.fvmap
        except AttributeError:
            self._drawMaps()
            return self.fvmap

    def getVelocities(self):
        pass

    def addNoise(self, sigma):
        pass

class Model(object):
    """Model class defining fit parameters

    Define all galaxy and observational parameters and describe
    which ones will be fit or fixed.
    """

    def __init__(self, name):
        self.name=name
        self.setDefaultVals()

        if(name=="A"):
            self.description="""Thin disk, flat rotation curve

                gal_beta - disk position angle in degrees [0,360)
                gal_q - projected image axis ratio (0,1)
                vmax - circular velocity (>0)
                g1 - shear 1 (abs<0.5)
                g2 - shear 2 (abs<0.5)
            """
            self.pars=[self.diskPA, self.diskBA, self.vCirc, self.g1, self.g2]
            self.labels=np.array(["PA","b/a","vmax","g1","g2"])
            self.guess=np.array([10.,0.1,100.,0.,0.])
            self.guessScale=np.array([10.,0.3,50.,0.02,0.02])

        else:
            raise ValueError(name)

    def setDefaultVals(self):
        self.diskRadius=1.
        self.diskVRadius=2.2
        self.diskCA=0.2
        self.diskSersic=1.
        self.diskFlux=1.
        self.bulgeFraction=0.
        self.bulgeRadius=1.
        self.bulgeSersic=4.
        self.rotCurveOpt="flat"
        self.rotCurvePars=[200.]
        self.vCirc=self.rotCurvePars[0]
        self.redshift=0.5
        self.diskPA=0.
        self.diskBA=0.7
        self.g1=0.
        self.g2=0.
        self.atmosFWHM=None
        self.pixScale=0.1  # arcseconds per pixel
        self.nPix=100
        self.vSampConfig="crossSlit"
        self.vSampSize=0.5  # arcseconds
        self.nVSamp=20
        self.vSampPA=self.diskPA
        self.vSampConvolve=True
        self.convOpt="pixel"
            
    def fixPars(self, fixed):
        self.fixed=fixed

    def setPriors(self, priors):
        self.priors=priors

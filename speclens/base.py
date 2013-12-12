import numpy as np
import sim

class Galaxy(object):
    """Galaxy class defining intrinsic parameters of a galaxy object"""

    def __init__(self, diskRadius=1., diskBA=1., diskCA=0.2,
                 diskSersic=1., diskVCirc=100., diskVRadius=1.,
                 diskFlux=1., bulgeFraction=0., bulgeRadius=1.,
                 bulgeSersic=4., bulgeVDisp=100.):
        self.diskRadius=diskRadius
        self.diskBA=diskBA
        self.diskCA=diskCA
        self.diskSersic=diskSersic
        self.diskVCirc=diskVCirc
        self.diskVRadius=diskVRadius
        self.diskFlux=diskFlux
        self.bulgeFraction=bulgeFraction
        self.bulgeRadius=bulgeRadius
        self.bulgeSersic=bulgeSersic
        self.bulgeVDisp=bulgeVDisp

class Observation(object):
    """Observation class takes a galaxy and produces observables"""

    def __init__(self, galaxy, redshift=0.5, diskPA=0., diskInclination=0.,
                 g1=0., g2=0., atmosFWHM=None, pixScale=0.1, nPix=100,
                 obsConfig="crossSlit", obsSize=1., nObs=10):
        self.galaxy=galaxy
        self.redshift=redshift
        self.diskPA=diskPA
        self.diskInclination=diskInclination
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


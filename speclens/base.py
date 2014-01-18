import numpy as np
import sim

class Galaxy(object):
    """Galaxy class defining intrinsic parameters of a galaxy object"""

    def __init__(self, diskRadius=1., diskVRadius=2.2, diskCA=0.2,
                 diskSersic=1., diskVCirc=200., bulgeFraction=0.,
                 bulgeRadius=1., bulgeSersic=4., galFlux=1.):
        self.diskRadius=diskRadius
        self.diskVRadius=diskVRadius
        self.diskCA=diskCA
        self.diskSersic=diskSersic
        self.diskVCirc=diskVCirc
        self.bulgeFraction=bulgeFraction
        self.bulgeRadius=bulgeRadius
        self.bulgeSersic=bulgeSersic
        self.galFlux=galFlux

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
            self.galaxy.galFlux, "flat", [self.galaxy.diskVCirc],
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

    def __init__(self, modelName, galName="default"):
        self.modelName=modelName
        self.setupAttr(galName)

        # Define priors and guess for each type of model
        # Note: priors are those used for fitting, 
        #       inputPriors are the distributions for generating ensembles.
        if(modelName=="A"):
            self.description="""Fixed disk thickness, flat rotation curve

                diskPA - disk position angle in degrees [0,360)
                cosi - cosine of the disk inclination (0=edge on, 1=face on)
                vmax - circular velocity (>0)
                g1 - shear 1 (abs<0.5)
                g2 - shear 2 (abs<0.5)
            """
            self.origPars=[self.diskPA, self.cosi, self.vCirc, self.g1, self.g2]
            self.labels=np.array(["PA","cos(i)","log10(vmax)","g1","g2"])
            self.origGuess=np.array([10.,0.5,np.log10(200.),0.,0.])
            self.origGuessScale=np.array([30.,0.2,0.06,0.02,0.02])
            self.origPriors=[("wrap",0.,360.), ("uniform",0.01,0.99),
                             ("norm",np.log10(200.),0.06),
                             ("uniform",-0.5,0.5),
                             ("uniform",-0.5,0.5)]
            self.inputPriors=[("uniform",0.0,360.),
                             ("uniform",0.01,0.99), ("fixed",np.log10(200.)),
                             ("norm",0.,0.05), ("norm",0.,0.05)]

        elif(modelName=="B"):
            self.description="""Free disk thickness, flat rotation curve

                diskPA - disk position angle in degrees [0,360)
                cosi - cosine of the disk inclination (0=edge on, 1=face on)
                diskCA - edge on disk thickness ratio (0=thin,1=sphere)
                log10(vmax) - circular velocity
                g1 - shear 1 (abs<0.5)
                g2 - shear 2 (abs<0.5)
            """
            self.origPars=[self.diskPA, self.cosi, self.diskCA,
                           np.log10(self.vCirc), self.g1, self.g2]
            self.labels=np.array(["PA","cos(i)","c/a","log10(vmax)","g1","g2"])
            self.origGuess=np.array([10.,0.5,0.2,np.log10(200.),0.,0.])
            self.origGuessScale=np.array([30.,0.2,0.1,0.06,0.02,0.02])
            self.origPriors=[("wrap",0.,360.), ("uniform",0.01,0.99),
                             ("truncnorm",0.2,0.05,0.,1.),
                             ("norm",np.log10(200.),0.06),
                             ("uniform",-0.5,0.5),
                             ("uniform",-0.5,0.5)]
            self.inputPriors=[("uniform",0.0,360.),
                             ("uniform",0.01,0.99), ("fixed",0.2),
                             ("fixed",np.log10(200.)),
                             ("norm",0.,0.05), ("norm",0.,0.05)]

        else:
            raise ValueError(modelName)

    def updatePars(self, pars):
        """Take a set of model pars and update stored values

        Since fit.lnProbVMapModel requires a pars array and other
        functions require a model object, this function takes a given pars
        array and reassigns the stored values in the model object.
        """
        if(self.modelName=="A"):
            self.diskPA, self.cosi, log10vCirc, self.g1, self.g2 = pars
            self.diskBA = sim.convertInclination(diskCA=self.diskCA, inc=np.arccos(self.cosi))
            self.vCirc = 10.**log10vCirc
        elif(self.modelName=="B"):
            self.diskPA, self.cosi, self.diskCA, log10vCirc, self.g1, self.g2 = pars
            self.diskBA = sim.convertInclination(diskCA=self.diskCA, inc=np.arccos(self.cosi))
            self.vCirc = 10.**log10vCirc
        else:
            raise ValueError(self.modelName)

    def setupAttr(self, galName="default"):
        """Define Model attributes"""

        if(galName == "default"):
            self.diskRadius=1.
            self.diskVRadius=2.2
            self.diskCA=0.2
            self.diskNu=0.5
            self.bulgeFraction=0.
            self.bulgeRadius=1.
            self.bulgeNu=-0.6
            self.galFlux=1.
            self.rotCurveOpt="arctan"
            self.vCirc=200.
            self.rotCurvePars=[self.vCirc,self.diskVRadius]
            self.redshift=0.5
            self.diskPA=0.
            self.cosi=np.cos(np.deg2rad(30.))
            self.diskBA=sim.convertInclination(diskCA=self.diskCA,inc=np.arccos(self.cosi))
            self.g1=0.
            self.g2=0.
            self.atmosFWHM=1.
            self.pixScale=0.1  # arcseconds per pixel
            self.nPix=100
            self.vSampConfig="hexNoCen"
            self.vSampSize=1.  # arcseconds (radius for fibers, side length for pixels)
            self.nVSamp=6
            self.vSampPA=self.diskPA
            self.vSampConvolve=True
            self.convOpt="pixel"
        else:
            raise ValueError(galName)

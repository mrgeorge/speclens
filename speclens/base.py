import numpy as np
import sim

class Detector(object):
    """Properties describing imaging and spectroscopic detectors

    (typically fixed, no errors)
    """
    def __init__(self):
        self.pixScale=0.1  # arcseconds per pixel
        self.nPix=100
        self.vSampConfig="crossslit"
        self.vSampSize=0.5  # arcseconds (radius for fibers, side length for pixels)
        self.nVSamp=20
        self.vSampConvolve=True

    def __repr__(self):
        return "%s.%s(%r)" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__)
    def __str__(self):
        return "%s.%s\n%s" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__.__str__().replace(',','\n'))

    @property
    def vSampShape(self):
        # see sim.getSamplePos for acceptable vSampConfig values
        shapes={"hex":"circle",
                "hexNoCen":"circle",
                "triNoCen":"circle",
                "slit":"square",
                "crossslit":"square",
                "ifu":"square"
                }
        return shapes[self.vSampConfig]


class PSF(object):
    def __init__(self):
        self.atmosFWHM=1.
        self.atmosFWHMErr=0.
        
    def __repr__(self):
        return "%s.%s(%r)" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__)

    def __str__(self):
        return "%s.%s\n%s" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__.__str__().replace(',','\n'))

class Observable(object):
    """Collection of data and self-description for lnP calculation

    Though a Model object may produce simulated Observable data, this
    object contains only the information that is in principle
    available from a telescope, e.g. datacube (flux vs x,y,lambda),
    derived velocity samples, image properties like measured PA, noise
    properties, and detector characteristics like pixel scale, etc. It
    does not contain model parameters such as intrinsic PA or shear.

    The likelihood calculation for model fitting is done in Observable
    space. This object also contains a description of what type of
    observable is used in the fit (e.g. imaging only, spectroscopy
    only, datacube vs derived velocities) and any special treatment
    needed (e.g. phase wrapping for measured PA).
    """

    def __init__(self):
        # parameters derived from raw data, with errors
        #   (typically fixed in the model)
        self.redshift=0.5
        self.diskRadius=1.
        self.diskRadiusErr=0.
        self.diskNu=0.5
        self.diskNuErr=0.
        self.bulgeFraction=0.
        self.bulgeFractionErr=0.
        self.bulgeRadius=1.
        self.bulgeRadiusErr=0.
        self.bulgeNu=-0.6
        self.bulgeNuErr=0
        self.galFlux=1.
        self.galFluxErr=0.

        # parameters derived from raw data, with errors
        #   (typically free in the model)
        self.diskPA=0.
        self.diskPAErr=10.
        self.diskBA=0.5
        self.diskBAErr=0.1
        self.vObs=None
        self.vObsErr=None

        # raw data, with errors
        self.image=None
        self.imageErr=None
        self.datacubeErr=None

        # parameters describing detector and seeing
        self.detector=Detector()
        self.psf=PSF()

    def __repr__(self):
        return "%s.%s(%r)" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__)

    def __str__(self):
        return "%s.%s\n%s" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__.__str__().replace(',','\n'))

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(**kwargs)

    def readData(self, inputFile, dataType):
        """Fill object attributes from a data file"""
        pass

    def setPointing(self, xObs=None, yObs=None, xObsErr=0., yObsErr=0., vSampPA=None):
        """Assign sampling positions for velocity measurements

        Inputs:
            Set xObs and yObs to ndarrays to assign positions directly.
            Defaults use sim.getSamplePos to determine xObs and yObs
                from vSampConfig and other attributes.
        """

        if((xObs is not None) & (yObs is not None)):
            self.xObs=xObs
            self.yObs=yObs
        else:
            pos = sim.getSamplePos(self.detector.nVSamp,
                self.detector.vSampSize, self.detector.vSampConfig, sampPA=vSampPA)
            self.xObs, self.yObs = pos

        self.detector.vSampPA=vSampPA
        self.xObsErr=xObsErr
        self.yObsErr=yObsErr

    def makeConvolutionKernel(self, convOpt):
        if(convOpt == "pixel"):
            self.kernel = sim.makeConvolutionKernel(self.xObs,
                self.yObs, self.detector, self.psf)
        else:  #convOpt is "galsim" or None
            self.kernel=None

    def defineDataVector(self, dataType):
        """Define data and error vectors for likelihood calculation

        Inputs:
            dataType - string describing data format
                       "imgPar" - derived PA and axis ratio
                       "velocities" - derived velocities
                       "imgPar+velocities" - combined
                       "datacube" - flux(x, y, lambda)
                       None - use model priors only
        Returns:
            Nothing, self is updated with dataVector, errVector, and
            wrapVector
        """

        if(dataType is None):
            self.dataVector=None
            self.errVector=None
            self.wrapVector=None
        elif(dataType == "imgPar"):
            self.dataVector=np.array([self.diskPA, self.diskBA])
            self.errVector=np.array([self.diskPAErr, self.diskBAErr])
            self.wrapVector=[(0.,180.), None]
        elif(dataType == "velocities"):
            self.dataVector=self.vObs            
            self.errVector=self.vObsErr
            self.wrapVector=None
        elif(dataType == "imgPar+velocities"):
            self.dataVector=np.concatenate([np.array([self.diskPA,
                self.diskBA]), self.vObs])
            self.errVector=np.concatenate([np.array([self.diskPAErr,
                self.diskBAErr]), self.vObsErr])
            self.wrapVector=[(0.,180.), None]
            [self.wrapVector.append(None) for xx in range(self.vObs.size)]
        elif(dataType == "datacube"):
            self.dataVector=self.datacube
            self.errVector=self.datacubeErr
            self.wrapVector=None
        else:
            raise ValueError(dataType)


class Galaxy(object):
    """Intrinsic properties for a model galaxy

    Note: some parameters overlap with Observable object,
    e.g. diskRadius. These are both intrinsic and observable
    properties. For instance, we assume size, flux, and surface
    brightness profile shape are both necessary to generate a model
    galaxy and also observationally accessible. On the other hand,
    cosi and diskCA generally are not observable. Another case is
    diskPA, which is stored here as the intrinsic source value,
    whereas in the corresponding Observable object it may differ due
    to shear.
    """
    def __init__(self):
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
        self.redshift=0.5
        self.diskPA=0.
        self.cosi=np.cos(np.deg2rad(30.))
        self.g1=0.
        self.g2=0.

    def __repr__(self):
        return "%s.%s(%r)" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__)

    def __str__(self):
        return "%s.%s\n%s" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__.__str__().replace(',','\n'))

    def setAttr(self, **kwargs):
        """Set/update arbitrary attribute list with **kwargs"""
        self.__dict__.update(**kwargs)

    # Use property decorators to generate attributes on the fly that
    # depend on other attributes. These are accessed with the normal
    # syntax (e.g. model.diskBA) but the setter is not implemented
    # (e.g. model.diskBA=x fails)
    @property
    def diskBA(self):
        return sim.convertInclination(diskCA=self.diskCA,
            inc=np.arccos(self.cosi))

    @property
    def rotCurvePars(self):
        # see sim.getOmega for name conventions
        if(self.rotCurveOpt == "flat"):
            return [self.vCirc]
        elif(self.rotCurveOpt == "solid"):
            return [self.vCirc, self.diskVRadius]
        elif(self.rotCurveOpt == "arctan"):
            return [self.vCirc, self.diskVRadius]
        elif(self.rotCurveOpt == "arctan"):
            return [self.vCirc, self.diskVRadius]
        else:
            raise ValueError(self.rotCurveOpt)

            
class Model(object):
    """Container for simulated observable, generator source galaxy, and model pars"""
    def __init__(self):
        self.obs=Observable()
        self.source=Galaxy()
        self.convOpt="pixel"
        
    def __repr__(self):
        return "%s.%s(%r)" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__)

    def __str__(self):
        return "%s.%s\n%s" % (self.__class__.__module__,
                              self.__class__.__name__,
                              self.__dict__.__str__().replace(',','\n'))

    def defineModelPars(self, modelName):
        self.modelName=modelName
        if(modelName=="A"):
            self.description="""Fixed disk thickness, fixed rotation scale

                diskPA - disk position angle in degrees [0,180)
                cosi - cosine of the disk inclination (0=edge on, 1=face on)
                log10(vmax) - circular velocity
                g1 - shear 1 (abs<0.5)
                g2 - shear 2 (abs<0.5)
            """
            self.origPars=[self.source.diskPA, self.source.cosi,
                np.log10(self.source.vCirc), self.source.g1,
                self.source.g2]
            self.labels=np.array(["PA","cos(i)","lg10(vc)","g1","g2"])
            self.origGuess=np.array([10.,0.5,np.log10(200.),0.,0.])
            self.origGuessScale=np.array([30.,0.2,0.06,0.02,0.02])
            self.origPriors=[("wrap",0.,180.), ("uniform",0.01,0.99),
                             ("norm",np.log10(200.),0.06),
                             ("uniform",-0.5,0.5),
                             ("uniform",-0.5,0.5)]
            self.inputPriors=[("uniform",0.0,180.),
                              ("uniform",0.01,0.99),
                              ("fixed",np.log10(200.)),
                              ("norm",0.,0.05), ("norm",0.,0.05)]

        elif(modelName=="B"):
            self.description="""Free disk thickness, fixed rotation scale

                diskPA - disk position angle in degrees [0,180)
                cosi - cosine of the disk inclination (0=edge on, 1=face on)
                diskCA - edge on disk thickness ratio (0=thin,1=sphere)
                log10(vmax) - circular velocity
                g1 - shear 1 (abs<0.5)
                g2 - shear 2 (abs<0.5)
            """
            self.origPars=[self.source.diskPA, self.source.cosi,
                self.source.diskCA, np.log10(self.source.vCirc),
                self.source.g1, self.source.g2]
            self.labels=np.array(["PA","cos(i)","c/a","lg10(vc)","g1","g2"])
            self.origGuess=np.array([10.,0.5,0.2,np.log10(200.),0.,0.])
            self.origGuessScale=np.array([30.,0.2,0.1,0.06,0.02,0.02])
            self.origPriors=[("wrap",0.,180.), ("uniform",0.01,0.99),
                             ("truncnorm",0.2,0.05,0.,1.),
                             ("norm",np.log10(200.),0.06),
                             ("uniform",-0.5,0.5),
                             ("uniform",-0.5,0.5)]
            self.inputPriors=[("uniform",0.0,180.),
                              ("uniform",0.01,0.99), ("fixed",0.2),
                              ("fixed",np.log10(200.)),
                              ("norm",0.,0.05), ("norm",0.,0.05)]

        elif(modelName=="C"):
            self.description="""Free disk thickness, free rotation scale

                diskPA - disk position angle in degrees [0,180)
                cosi - cosine of the disk inclination (0=edge on, 1=face on)
                diskCA - edge on disk thickness ratio (0=thin,1=sphere)
                vRadRatio - ratio of diskVRadius / diskRadius
                log10(vmax) - circular velocity
                g1 - shear 1 (abs<0.5)
                g2 - shear 2 (abs<0.5)
            """
            self.origPars=[self.source.diskPA, self.source.cosi,
                self.source.diskCA,
                self.source.diskVRadius/self.source.diskRadius,
                np.log10(self.source.vCirc), self.source.g1,
                self.source.g2]
            self.labels=np.array(["PA","cos(i)","c/a","Rv/R","lg10(vc)","g1","g2"])
            self.origGuess=np.array([10.,0.5,0.2,2.2,np.log10(200.),0.,0.])
            self.origGuessScale=np.array([30.,0.2,0.1,0.2,0.06,0.02,0.02])
            self.origPriors=[("wrap",0.,180.), ("uniform",0.01,0.99),
                             ("truncnorm",0.2,0.05,0.,1.),
                             ("uniform",1.,4.),
                             ("norm",np.log10(200.),0.06),
                             ("uniform",-0.5,0.5),
                             ("uniform",-0.5,0.5)]
            self.inputPriors=[("uniform",0.0,180.),
                              ("uniform",0.01,0.99), ("fixed",0.2),
                              ("fixed",2.2), ("fixed",np.log10(200.)),
                              ("norm",0.,0.05), ("norm",0.,0.05)]
        elif(modelName=="D"):
            self.description="""Free disk thickness, free rotation scale

                phig - 0.5 * arctan(g2/g1) (0-pi)
                cosi - cosine of the disk inclination (0=edge on, 1=face on)
                diskCA - edge on disk thickness ratio (0=thin,1=sphere)
                vRadRatio - ratio of diskVRadius / diskRadius
                log10(vmax) - circular velocity
                gcosPA - shear (abs < 1)
                gsinPA - shear (abs < 1)
            """
            self.origPars=[np.sqrt(self.source.g1**2 + self.source.g2**2),
                self.source.cosi,
                self.source.diskCA,
                self.source.diskVRadius/self.source.diskRadius,
                np.log10(self.source.vCirc),
                self.source.g1 * np.cos(np.deg2rad(self.source.diskPA)),
                self.source.g2 * np.cos(np.deg2rad(self.source.diskPA))]
            self.labels=np.array(["phig","cos(i)","c/a","Rv/R","lg10(vc)","gcosPA","gsinPA"])
            self.origGuess=np.array([0.,0.5,0.2,2.2,np.log10(200.),0.,0.])
            self.origGuessScale=np.array([0.03,0.2,0.1,0.2,0.06,0.02,0.02])
            self.origPriors=[("wrap",0.,np.pi), ("uniform",0.01,0.99),
                             ("truncnorm",0.2,0.05,0.,1.),
                             ("uniform",1.,4.),
                             ("norm",np.log10(200.),0.06),
                             ("uniform",-0.5,0.5),
                             ("uniform",-0.5,0.5)]
            self.inputPriors=[("uniform",0.0,np.pi),
                              ("uniform",0.01,0.99), ("fixed",0.2),
                              ("fixed",2.2), ("fixed",np.log10(200.)),
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
            diskPA, cosi, log10vCirc, g1, g2 = pars
            self.source.diskPA = diskPA
            self.source.cosi = cosi
            self.source.vCirc = 10.**log10vCirc
            self.source.g1 = g1
            self.source.g2 = g2
        elif(self.modelName=="B"):
            diskPA, cosi, diskCA, log10vCirc, g1, g2 = pars
            self.source.diskPA = diskPA
            self.source.cosi = cosi
            self.source.diskCA = diskCA
            self.source.vCirc = 10.**log10vCirc
            self.source.g1 = g1
            self.source.g2 = g2
        elif(self.modelName=="C"):
            diskPA, cosi, diskCA, vRadRatio, log10vCirc, g1, g2 = pars
            self.source.diskPA = diskPA
            self.source.cosi = cosi
            self.source.diskCA = diskCA
            self.source.diskVRadius = self.source.diskRadius * vRadRatio
            self.source.vCirc = 10.**log10vCirc
            self.source.g1 = g1
            self.source.g2 = g2
        elif(self.modelName=="D"):
            phig, cosi, diskCA, vRadRatio, log10vCirc, gcosPA, gsinPA = pars
            g = np.sqrt(gcosPA**2 + gsinPA**2)
            tan2phig = np.tan(2*phig)
            if g==0:
                self.source.diskPA = 0.
            else:
                self.source.diskPA = np.arccos(gcosPA/g)
            self.source.cosi = cosi
            self.source.diskCA = diskCA
            self.source.diskVRadius = self.source.diskRadius * vRadRatio
            self.source.vCirc = 10.**log10vCirc
            self.source.g1 = g / np.sqrt(1. + tan2phig**2)
            self.source.g2 = self.source.g1 * tan2phig
        else:
            raise ValueError(self.modelName)

    def updateObservable(self, dataType):
        """Generate new observable based on self.source

        Inputs:
            dataType - string describing data format
                       "imgPar" - derived PA and axis ratio
                       "velocities" - derived velocities
                       "imgPar+velocities" - combined
                       "datacube" - flux(x, y, lambda)
                       None - use model priors only
        """
        self.dataType = dataType

        if(dataType in ("imgPar", "imgPar+velocities")):
            diskPASheared, diskBASheared = sim.ellModel(self.source)
            self.obs.diskPA = diskPASheared
            self.obs.diskBA = diskBASheared

        if(dataType in ("velocities", "imgPar+velocities")):
            if(self.convOpt is not None):
                galFibFlux, vmapFibFlux = sim.vmapObs(self, self.obs.xObs,
                    self.obs.yObs)
                vObs = vmapFibFlux / galFibFlux
            else:  # faster, don't need to convolve with psf or fiber
                vObs = sim.vmapModel(self.source, self.obs.xObs,
                    self.obs.yObs)
            self.obs.vObs = vObs

        if(dataType == "datacube"):
            pass # TO DO - generate simulated datacube here

        # update dataVector, errVector, wrapVector attributes
        self.obs.defineDataVector(dataType)

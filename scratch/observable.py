#! env python


# test out the Observable object in base.py
# instantiate an object
# assign its parameters
# generate observed velocities with sim or makeObs and assign
# generate a similar Model object and compare values for likelihood

import numpy as np
import os
import sys

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens


class TestObservable(object):
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
        self.atmosFWHM=1.
        self.atmosFWHMErr=0.

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

        # parameters describing detector
        self.detector=TestDetector()


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
            self.vSampPA=vSampPA
            pos,self.vSampShape = speclens.sim.getSamplePos(self.detector.nVSamp,
                self.detector.vSampSize, self.detector.vSampConfig, sampPA=self.vSampPA)
            self.xObs, self.yObs = pos

        self.xObsErr=xObsErr
        self.yObsErr=yObsErr

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
            self.wrapVector=[(0.,180.), None, np.repeat(None, len(self.vObs))]
        elif(dataType == "datacube"):
            self.dataVector=self.datacube
            self.errVector=self.datacubeErr
            self.wrapVector=None
        else:
            raise ValueError(dataType)

class TestDetector(object):
    """Properties describing imaging and spectroscopic detectors

    (typically fixed, no errors)
    """
    def __init__(self):
        self.pixScale=0.1  # arcseconds per pixel
        self.nPix=100
        self.vSampConfig="crossslit"
        self.vSampSize=0.5  # arcseconds (radius for fibers, side length for pixels)
        self.nVSamp=20


class TestGalaxy(object):
    """Properties needed to generate a model galaxy, including unobservables

    e.g. include cosi and diskCA, which aren't generally known
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
        self.rotCurvePars=[self.vCirc,self.diskVRadius]
        self.redshift=0.5
        self.diskPA=0.
        self.cosi=np.cos(np.deg2rad(30.))
        self.g1=0.
        self.g2=0.


class TestModel(object):
    """Container for simulated observable, generator source galaxy, and model pars"""
    def __init__(self):
        self.obs=TestObservable()
        self.source=TestGalaxy()



def computeLikelihood(model, observable):
    pass

vObsErr=np.repeat(10.,20)
diskPAErr=10.
diskBAErr=0.1
to=TestObservable()
to.setPointing(vSampPA=to.diskPA)
to.setAttr(vObsErr=vObsErr, diskPAErr=diskPAErr, diskBAErr=diskBAErr)

model=speclens.Model("B")

model.redshift=to.redshift
model.diskRadius=to.diskRadius
model.diskNu=to.diskNu
model.bulgeFraction=to.bulgeFraction
model.bulgeRadius=to.bulgeRadius
model.bulgeNu=to.bulgeNu
model.galFlux=to.galFlux
model.atmosFWHM=to.atmosFWHM

model.pixScale=to.detector.pixScale
model.nPix=to.detector.nPix
model.vSampConfig=to.detector.vSampConfig
model.vSampSize=to.detector.vSampSize
model.nVSamp=to.detector.nVSamp

xvals,yvals,vvals,ellObs,inputPars=speclens.ensemble.makeObs(model,
    sigma=to.vObsErr, ellErr=np.array([to.diskPAErr, to.diskBAErr]),
    randomPars=False)
np.testing.assert_allclose(to.xObs, xvals)
np.testing.assert_allclose(to.yObs, yvals)

to.vObs=vvals

dataType="imgPar+velocities"
to.defineDataVector(dataType)

computeLikelihood(model, to)

print "done"

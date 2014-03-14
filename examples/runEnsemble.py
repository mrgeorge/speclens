#! env python

import numpy as np
import os
import sys
import copy

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens

def exampleGalaxies(modelName, exampleName):
    model = speclens.Model()

    # Current examples just set cosi and diskPA
    # all other params are left to defaults
    examples = {"edge":(0.02, 0.),
                "face":(0.98, 0.),
                "horizontal":(0.5, 0.),
                "diagonal":(0.5, 45.),
                "vertical":(0.5, 90.)
                }
    cosi, diskPA = examples[exampleName]
    model.source.setAttr(cosi=cosi, diskPA=diskPA)
    model.defineModelPars(modelName)

    return model.origPars

def ensemblePlots(modelName, dataDir, plotDir, figExt="pdf", showPlot=False,
                  randomPars=True):
    """Run fits for an ensemble, store chains and make plots for each

    Generate a large sample of galaxy orientations and shears, fit
    their observables with parametric model, and compute offsets
    in the recovered values from the input values. This provides an
    estimate of the precision with which shear and other variables
    can be estimated from the data.
    """

    # Measurement errors
    vObsErr = 10.
    diskPAErr = 10.
    diskBAErr = 0.1

    # model to use for generating observables
    inputModel = speclens.Model()
    if randomPars:
        nGal=10
    else:
        exampleNames = ("edge", "face", "horizontal", "diagonal", "vertical")
        nGal = len(exampleNames)

    inputModel.defineModelPars(modelName)
    inputModel.obs.vObsErr = np.repeat(vObsErr,
        inputModel.obs.detector.nVSamp)
    inputModel.obs.diskPAErr = diskPAErr
    inputModel.obs.diskBAErr = diskBAErr

    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)

        thisInputModel = copy.deepcopy(inputModel)
        if not randomPars: # set pars from list of examples
            inputPars = exampleGalaxies(modelName, exampleNames[ii])
            thisInputModel.origPars = inputPars
            thisInputModel.updatePars(inputPars)

        speclens.ensemble.makeObs(thisInputModel,
            "imgPar+velocities", randomPars=randomPars, seed=ii)

        # model to use for fitting
        fitModel=speclens.Model()
        fitModel.defineModelPars(modelName)
        # set up fitModel matching observation parameters like psf
        fitModel.obs = copy.deepcopy(thisInputModel.obs)

        # Fit these data with a model
        speclens.ensemble.runGal(dataDir, plotDir, ii,
            thisInputModel.origPars, fitModel, thisInputModel.obs,
            figExt=figExt, addNoise=False, nWalkers=2000, nBurn=50,
            nSteps=1000, nThreads=8, seed=ii, minAF=None, maxAF=None,
            nEff=None, walkerOpt="prior")


if __name__ == "__main__":

    modelName="C"

    # set up paths for output dirs
    speclensDir="../"
    if not os.path.isdir(speclensDir):
        raise NameError(speclensDir)
    plotDir=speclensDir+"plots/"+modelName
    dataDir=speclensDir+"chains/"+modelName
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

    figExt="pdf" # pdf or png
    showPlot=False
    randomPars=False

    ensemblePlots(modelName, dataDir, plotDir, figExt=figExt,
                  showPlot=showPlot, randomPars=randomPars)

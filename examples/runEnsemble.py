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

def ensemblePlots(modelName, dataDir, plotDir, figExt="pdf", showPlot=False):
    """Run fits for an ensemble, store chains and make plots for each

    Generate a large sample of galaxy orientations and shears, fit
    their observables with parametric model, and compute offsets
    in the recovered values from the input values. This provides an
    estimate of the precision with which shear and other variables
    can be estimated from the data.
    """

    nGal=10

    # model to use for generating fake data
    inputModel=speclens.Model()
    inputModel.defineModelPars(modelName)

    # model to use for fitting
    fitModel=speclens.Model()
    fitModel.defineModelPars(modelName)

    vObsErr = 10.
    diskPAErr = 10.
    diskBAErr = 0.1

    inputModel.obs.vObsErr = np.repeat(vObsErr,
        inputModel.obs.detector.nVSamp)
    inputModel.obs.diskPAErr = diskPAErr
    inputModel.obs.diskBAErr = diskBAErr

    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)
        thisInputModel = copy.deepcopy(inputModel)
        thisFitModel = copy.deepcopy(fitModel)

        speclens.ensemble.makeObs(thisInputModel,
            "imgPar+velocities", randomPars=True, seed=ii)

        # set up fitModel matching observation parameters like psf
        thisFitModel.obs = copy.deepcopy(thisInputModel.obs)

        # Fit these data with a model
        speclens.ensemble.runGal(dataDir, plotDir, ii,
            thisInputModel.origPars, thisFitModel, thisInputModel.obs,
            figExt=figExt, addNoise=True, nWalkers=20, nBurn=5,
            nSteps=25, nThreads=1, seed=ii)


if __name__ == "__main__":

    modelName="B"

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

    ensemblePlots(modelName,dataDir,plotDir,figExt=figExt,showPlot=showPlot)

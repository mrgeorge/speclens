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
    their observables with a 5-parameter model, and compute offsets
    in the recovered values from the input values. This provides an
    estimate of the precision with which shear and other variables
    can be estimated from the data.
    """

    nGal=10
    model=speclens.Model(modelName)

    sigma=30.
    ellErr=np.array([10.,0.1])

    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)
        thisModel=copy.deepcopy(model)

        # Get model galaxy and observables
        xvals,yvals,vvals,ellObs,inputPars=speclens.ensemble.makeObs(thisModel,sigma=sigma,ellErr=ellErr,seed=ii,randomPars=True)

        # Fit these data with a model
        speclens.ensemble.runGal(dataDir,plotDir,ii,inputPars,vvals,sigma,ellObs,ellErr,thisModel,figExt=figExt,addNoise=True,nWalkers=20,nBurn=5,nSteps=25,seed=ii)


if __name__ == "__main__":

    modelName="B"

    # set up paths for output dirs
    speclensDir="../"
    if not os.path.isdir(speclensDir):
        raise NameError(speclensDir)
    plotDir=speclensDir+"plots/"+modelName
    dataDir=speclensDir+"data/"+modelName
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

    figExt="pdf" # pdf or png
    showPlot=False

    ensemblePlots(modelName,dataDir,plotDir,figExt=figExt,showPlot=showPlot)

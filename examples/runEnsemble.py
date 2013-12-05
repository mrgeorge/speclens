#! env python

import numpy as np
import os
import sys

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens

def ensemblePlots(dataDir, plotDir, figExt="pdf", showPlot=False):
    """Run fits for an ensemble, store chains and make plots for each

    Generate a large sample of galaxy orientations and shears, fit
    their observables with a 5-parameter model, and compute offsets
    in the recovered values from the input values. This provides an
    estimate of the precision with which shear and other variables
    can be estimated from the data.
    """

    nGal=100
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]
    obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]

    disk_r=1.
    convOpt="pixel"
    atmos_fwhm=1.
    numFib=6
    fibRad=1
    fibConvolve=True
    fibConfig="hexNoCen"
    sigma=30.
    ellErr=np.array([10.,0.1])

    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)
        # Get model galaxy and observables
        xvals,yvals,vvals,ellObs,inputPars=speclens.ensemble.makeObs(inputPriors=inputPriors,disk_r=disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr,seed=ii)

        # Fit these data with a model
        speclens.ensemble.runGal(dataDir,plotDir,ii,inputPars,labels,vvals,sigma,ellObs,ellErr,obsPriors,figExt=figExt,disk_r=disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,fibPA=ellObs[0],addNoise=True,seed=ii)


if __name__ == "__main__":

    # set up paths for output dirs
    speclensDir="../"
    if not os.path.isdir(speclensDir):
        raise NameError(speclensDir)
    plotDir=speclensDir+"/plots"
    dataDir=speclensDir+"/data"
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

    figExt="pdf" # pdf or png
    showPlot=False

    ensemblePlots(dataDir,plotDir,figExt=figExt,showPlot=showPlot)

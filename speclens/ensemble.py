#! env python

import io
import plot
import numpy as np
import os
import glob

import sim
import fit

def generatePars(nGal,priors,seed=None):
    """Generate set of galaxies with shapes following a set of priors

    Inputs:
        nGal - number of galaxies to generate
        priors - see fit.getPriorFuncs for conventions, must not be None here
        seed - used for repeatable random number generation (default None)
    Returns:
        pars - ndarray (e.g. nGal x [diskPA, diskBA, vmax, g1, g2])
    """

    nPars=len(priors)
    pars=np.zeros((nGal,nPars))
    np.random.seed(seed)
    
    # getPriorFuncs can't handle fixed values
    # so they need to be treated separately
    # otherwise use the function's rvs method to generate random deviates
    for ii,prior in enumerate(priors):
        if(prior[0] == "fixed"):
            pars[:,ii]=np.copy(prior[1])
        else:
            priorFunc=fit.getPriorFuncs([prior])[0]
            pars[:,ii]=priorFunc.rvs(nGal)

    return pars
    
def makeObs(model, dataType, randomPars=False, seed=None):
    """Generate input model parameters and observables for one galaxy"""

    # Setup galaxy properties
    if(randomPars):
        inputPars=generatePars(1,model.inputPriors,seed=seed).squeeze()
        model.origPars=inputPars # overwrite the default pars array used
                                 # to initialize model
        model.updatePars(inputPars) # overwrite individual attributes
    else:
        inputPars=model.origPars

    # get observed PA for image ellipse and use it for spec PA
    model.updateObservable("imgPar")
    if(dataType != "imgPar"):
        model.obs.setPointing(vSampPA=model.obs.diskPA)
        model.obs.makeConvolutionKernel(model.convOpt)
        model.updateObservable(dataType)

    model.obs.defineDataVector(dataType)

def runGal(chainDir, plotDir, galID, inputPars, model, observation,
           figExt="pdf", **kwargs):
    """Call fit.fitObs to run MCMC for a galaxy and save the resulting chains

    This is what create_qsub_galArr calls to run each galaxy

    Inputs:
        chainDir - directory to write output files
        plotDir - directory to write output plots
        galID - label to name each galaxy file separately
        inputPars - ndarray of nGal sets of model parameters
                    from makeObs or generatePars
        vvals - observed velocity values
        sigma - error on vvals
        ellObs - observed image values
        ellErr - error on ellObs
        model object used to describe fit priors and pars

        figExt - plot file format (default "pdf", or "png")
        **kwargs - args passed on to fit.fitObs
    Returns:
        nothing, chains and plots written to chainDir, plotDir
    """

    io.writeObj(model, chainDir+"/model_{:03d}.dat".format(galID))
    io.writeObj(observation, chainDir+"/obs_{:03d}.dat".format(galID))

    chains,lnprobs,iterations,accfracs,nWalkers=fit.fitObs(model, observation,
        **kwargs)
    headers = [io.makeHeader(iterations[ii], accfracs[ii],
                             nWalkers[ii]) for ii in range(3)]

    for ii,opt in enumerate(['I','S','IS']):
        io.writeRec(io.chainToRec(chains[ii], lnprobs[ii], labels=model.labels),
            chainDir+"/chain{}_{:03d}.fits.gz".format(opt, galID),
            header=headers[ii], compress="GZIP")

    plot.contourPlotAll(chains, lnprobs=lnprobs, inputPars=inputPars,
        showMax=True, showPeakKDE=True, show68=True, smooth=3,
        percentiles=[0.68,0.95], labels=model.labels, showPlot=False,
        filename=plotDir+"/gal_{:03d}.{}".format(galID, figExt))


def getScatter(dir,nGal,inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],labels=np.array(["PA","b/a","vmax","g1","g2"]),free=np.array([0,1,2,3,4]),fileType="chain"):
    """Read output chains or chain summaries and compute scatter in fits

    This should work while an ensemble is still running, as it tries to
    read all the output files but ignores ones that are missing.

    Inputs:
        dir - directory where chains are saved
        nGal - number of galaxies in ensemble
        inputPriors - priors used when generating galaxies
                      (note, these may differ from obsPriors used in fitting)
                      default [[0,360],[0,1],150,(0,0.05),(0,0.05)]
        labels - string parameter names for plot axes
                 default ndarray["PA","b/a","vmax","g1","g2"]
                 Note - labels includes *all* pars, not just free ones
        free - ndarray listing indices of fit parameters
                 default ndarray[0,1,2,3,4]
        fileType - "chain" or "stats" 
    Returns:
        tuple of ndarrays summarizing offsets and scatter of 
            parameters recovered from fits to noisy data
    """
    
    chainIFiles=glob.glob(dir+"/chainI_*.fits.gz")
    chainSFiles=glob.glob(dir+"/chainS_*.fits.gz")
    chainISFiles=glob.glob(dir+"/chainIS_*.fits.gz")
    statsIFiles=glob.glob(dir+"/statsI_*.fits.gz")
    statsSFiles=glob.glob(dir+"/statsS_*.fits.gz")
    statsISFiles=glob.glob(dir+"/statsIS_*.fits.gz")

    dI=np.zeros((nGal,len(free)))
    dS=np.zeros_like(dI)
    dIS=np.zeros_like(dI)
    dIkde=np.zeros_like(dI)
    dSkde=np.zeros_like(dI)
    dISkde=np.zeros_like(dI)
    hwI=np.zeros_like(dI)
    hwS=np.zeros_like(dI)
    hwIS=np.zeros_like(dI)
    inputPars=np.zeros_like(dI)

    # check if inputPriors is a list of lists (i.e. different inputs for each galaxy)
    if(len(inputPriors) != len(labels)):
        if(len(inputPriors[0]) == len(labels)):
            listInput=True
        else:
            print "Error in getScatter: inputPriors should be a list of len={} or a list of such lists".format(len(labels))
    else:
        listInput=False
    
    for ii in range(nGal):
        print ii
        if(fileType=="chain"):
            filesExist=((dir+"chainI_{:03d}.fits.gz".format(ii) in chainIFiles) &
                        (dir+"chainS_{:03d}.fits.gz".format(ii) in chainSFiles) &
                        (dir+"chainIS_{:03d}.fits.gz".format(ii) in chainISFiles))
        elif(fileType=="stats"):
            filesExist=((dir+"statsI_{:03d}.fits.gz".format(ii) in statsIFiles) &
                        (dir+"statsS_{:03d}.fits.gz".format(ii) in statsSFiles) &
                        (dir+"statsIS_{:03d}.fits.gz".format(ii) in statsISFiles))
        if(filesExist):
            if(listInput):
                inputPars[ii,:]=generatePars(1,inputPriors[ii],seed=ii).squeeze()[free]
            else:
                inputPars[ii,:]=generatePars(1,inputPriors,seed=ii).squeeze()[free]

            if(fileType=="chain"):
                recI=io.readRec(dir+"chainI_{:03d}.fits.gz".format(ii))
                recS=io.readRec(dir+"chainS_{:03d}.fits.gz".format(ii))
                recIS=io.readRec(dir+"chainIS_{:03d}.fits.gz".format(ii))

                chainI=io.recToPars(recI,labels=labels[free])
                chainS=io.recToPars(recS,labels=labels[free])
                chainIS=io.recToPars(recIS,labels=labels[free])

                obsI=fit.getMaxProb(chainI,recI['lnprob'])
                obsS=fit.getMaxProb(chainS,recS['lnprob'])
                obsIS=fit.getMaxProb(chainIS,recIS['lnprob'])
            
                obsIkde=fit.getPeakKDE(chainI,obsI)
                obsSkde=fit.getPeakKDE(chainS,obsS)
                obsISkde=fit.getPeakKDE(chainIS,obsIS)

                hwI[ii,:]=fit.get68(chainI,opt="hw")
                hwS[ii,:]=fit.get68(chainS,opt="hw")
                hwIS[ii,:]=fit.get68(chainIS,opt="hw")
            elif(fileType=="stats"):
                statsI=io.readRec(dir+"statsI_{:03d}.fits.gz".format(ii))
                statsS=io.readRec(dir+"statsS_{:03d}.fits.gz".format(ii))
                statsIS=io.readRec(dir+"statsIS_{:03d}.fits.gz".format(ii))

                obsI=statsI['mp']
                obsS=statsS['mp']
                obsIS=statsIS['mp']

                obsIkde=statsI['kde']
                obsSkde=statsS['kde']
                obsISkde=statsIS['kde']

                hwI[ii,:]=statsI['hw']
                hwS[ii,:]=statsS['hw']
                hwIS[ii,:]=statsIS['hw']

            dI[ii,:]=obsI-inputPars[ii,:]
            dS[ii,:]=obsS-inputPars[ii,:]
            dIS[ii,:]=obsIS-inputPars[ii,:]
            dIkde[ii,:]=obsIkde-inputPars[ii,:]
            dSkde[ii,:]=obsSkde-inputPars[ii,:]
            dISkde[ii,:]=obsISkde-inputPars[ii,:]
        else:
            dI[ii,:]=np.repeat(np.nan,len(free))
            dS[ii,:]=np.repeat(np.nan,len(free))
            dIS[ii,:]=np.repeat(np.nan,len(free))
            hwI[ii,:]=np.repeat(np.nan,len(free))
            hwS[ii,:]=np.repeat(np.nan,len(free))
            hwIS[ii,:]=np.repeat(np.nan,len(free))

    good=~np.isnan(dI[:,0])

    print "STD Max"
    print np.std(dI[good,:],axis=0)
    print np.std(dS[good,:],axis=0)
    print np.std(dIS[good,:],axis=0)

    return (dI[good],dS[good],dIS[good],dIkde[good],dSkde[good],dISkde[good],hwI[good],hwS[good],hwIS[good],inputPars[good])

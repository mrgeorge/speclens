#! env python

import io
import plot
import numpy as np
import os
import glob

import galsim # for PS or NFW shear prior

import sim
import fit

def generatePars(nGal,priors,shearOpt=None,seed=None):
    """Generate set of galaxies with shapes following a set of priors

    Inputs:
        nGal - number of galaxies to generate
        priors - see fit.interpretPriors for conventions, must not be None here
        shearOpt - distribution of shears, if not set by priors ("PS", "NFW", None)
        seed - used for repeatable random number generation (default None)
    Returns:
        pars - ndarray (nGal x [gal_beta, gal_q, vmax, g1, g2])
    """

    nPars=len(priors)
    pars=np.zeros((nGal,nPars))
    np.random.seed(seed)
    for ii in xrange(nPars):
        prior=priors[ii]
        # note: each of the assignments below needs to *copy* aspects of prior to avoid pointer overwriting
        if((isinstance(prior, int)) | (isinstance(prior, float))): # fixed
            fixVal=np.copy(prior)
            pars[:,ii]=fixVal
        elif(isinstance(prior, list)): # flat prior
            priorRange=np.copy(prior)
            pars[:,ii]=np.random.rand(nGal)*(priorRange[1]-priorRange[0]) + priorRange[0]
        elif(isinstance(prior, tuple)): # gaussian
            priorMean=np.copy(prior[0])
            priorSigma=np.copy(prior[1])
            pars[:,ii]=np.random.randn(nGal)*priorSigma + priorMean

    if(shearOpt is not None):
        # define area
        density=150./3600 # 150/sq deg in /sq arcsec (~BOSS target density)
        area=nGal/density # sq arcsec
        gridLength=np.ceil(np.sqrt(area)) # arcsec
        gridSpacing=1. # arcsec

        # assign random uniform positions with origin at center
        xpos=np.random.rand(nGal)*gridLength - 0.5*gridLength
        ypos=np.random.rand(nGal)*gridLength - 0.5*gridLength
        
        if(shearOpt == "PS"):
            ps=galsim.PowerSpectrum(lambda k: k**2)
            ps.buildGrid(grid_spacing=gridSpacing, ngrid=gridLength)
            g1, g2 = ps.getShear((xpos,ypos))
            pars[:,-2]=g1
            pars[:,-1]=g2

        elif(shearOpt == "NFW"):
            pass

    return pars
    
def makeObs(model,sigma=30.,ellErr=np.array([10.,0.1]),seed=None,randomPars=True):
    """Generate input model parameters and observables for one galaxy

    Inputs:
        inputPriors - list of priors for input pars, 
                      see fit.interpretPriors for format
                      default [[0,360],[0,1],150,(0,0.05),(0,0.05)]
        disk_r - galaxy size, float or ndarray (default None)
        convOpt - None (default), "galsim", or "pixel"
        atmos_fwhm - gaussian PSF FWHM (default None)
        numFib - number of fibers (default 6)
        fibRad - fiber radius in arcsecs (default 1)
        fibConvolve - bool for whether to convolved with fiber (default False)
                      (note PSF and fiber convolution controlled separately)
        fibConfig - string used by sim.getFiberPos to describe 
                    slit/ifu/fiber config (default "hexNoCen")
        fibPA - fiber position angle if shape is square (default None)
        sigma - rms velocity error in km/s (default 30.)
        ellErr - ndarray (gal_beta_err degrees, gal_q_err), default [10,0.1]
        seed - random number generator repeatability (default None)
        randomPars - if True (default), make a random instance of inputPars
                       given inputPriors, else use model.origPars
    Returns:
        (xvals,yvals,vvals,ellObs,inputPars) tuple    
    """

    # Setup galaxy properties
    if(randomPars):
        inputPars=generatePars(1,model.inputPriors,shearOpt=None,seed=seed).squeeze()
        model.origPars=inputPars # overwrite the default pars array used
                                 # to initialize model
        model.updatePars(inputPars) # overwrite individual attributes
    else:
        inputPars=model.origPars
    
    # get imaging and spectroscopic observables
    # note, no noise added here 
    # if PA offsets are desired, set model.vSampPA first
    ellObs=sim.ellModel(model)

    pos,fibShape=sim.getFiberPos(model.nVSamp,model.vSampSize,
                                 model.vSampConfig,model.vSampPA)
    xvals,yvals=pos
    model.vSampShape=fibShape
    if(model.convOpt is not None):
        model.kernel=sim.makeConvolutionKernel(xvals,yvals,model)
        vvals=sim.vmapObs(model,xvals,yvals)
    else: # this is faster if we don't need to convolve with psf or fiber
        vvals=sim.vmapModel(model,xvals,yvals)

    return (xvals,yvals,vvals,ellObs,inputPars)

def runGal(dataDir,plotDir,galID,inputPars,vvals,sigma,ellObs,ellErr,model,figExt="pdf",**kwargs):
    """Call fit.fitObs to run MCMC for a galaxy and save the resulting chains

    This is what create_qsub_galArr calls to run each galaxy

    Inputs:
        dataDir - directory to write output files
        plotDir - directory to write output plots
        galID - label to name each galaxy file separately
        inputPars - ndarray of nGal sets of model parameters
                    from makeObs or generatePars
        labels - string parameter names for plot axes
        vvals - observed velocity values
        sigma - error on vvals
        ellObs - observed image values
        ellErr - error on ellObs
        obsPriors - priors used when fitting data
                    (note: these may be different than inputPriors)
        figExt - plot file format (default "pdf", or "png")
        **kwargs - args passed on to fit.fitObs
    Returns:
        nothing, chains and plots written to dataDir, plotDir
    """

    chains,lnprobs=fit.fitObs(vvals,sigma,ellObs,ellErr,model,**kwargs)
    io.writeRec(io.chainToRec(chains[0],lnprobs[0],labels=model.labels),dataDir+"/chainI_{:03d}.fits.gz".format(galID),compress="GZIP")
    io.writeRec(io.chainToRec(chains[1],lnprobs[1],labels=model.labels),dataDir+"/chainS_{:03d}.fits.gz".format(galID),compress="GZIP")
    io.writeRec(io.chainToRec(chains[2],lnprobs[2],labels=model.labels),dataDir+"/chainIS_{:03d}.fits.gz".format(galID),compress="GZIP")
    plot.contourPlotAll(chains,lnprobs=lnprobs,inputPars=inputPars,showMax=True,showPeakKDE=True,show68=True,smooth=3,percentiles=[0.68,0.95],labels=model.labels,showPlot=False,filename=plotDir+"/gal_{:03d}.{}".format(galID,figExt))


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
                inputPars[ii,:]=generatePars(1,inputPriors[ii],shearOpt=None,seed=ii).squeeze()[free]
            else:
                inputPars[ii,:]=generatePars(1,inputPriors,shearOpt=None,seed=ii).squeeze()[free]

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

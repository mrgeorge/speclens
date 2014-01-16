#! env python

import numpy as np
import emcee
import scipy.stats
import sim

####
# Priors
####
# These generate functions used in lnProbVMapModel to compute chisq_prior

def makeFlatPrior(range):
    return lambda x: priorFlat(x, range)

def priorFlat(arg, range):
    if((arg >= range[0]) & (arg < range[1])):
        return 0
    else:
        return np.Inf

def makeGaussPrior(mean, sigma):
    return lambda x: ((x-mean)/sigma)**2

def makeGaussTruncPrior(mean, sigma, range):
    return lambda x: ((x-mean)/sigma)**2 + priorFlat(x, range)
    
def interpretPriors(model):
    """Generate functions to evaluate priors and fix variables.

    Inputs:
        priors - a list or tuple with an entry for each of M parameters 
          in the full model. Each parameter can have a prior set by one of
          the following formats:
            None - leave this variable completely free
            float - fix the variable to this value
            list[a,b] - flat prior between a and b
            tuple(a,b) - gaussian prior with mean a and stddev b
            tuple(a,b,c,d) - gaussian prior with mean a and stddev b, truncated at c and d

    Returns:
        model object is updated with the following arrays:
            priorFuncs - ndarray of length N (# of free parameters to fit)
            fixed - ndarray of length M; None for free pars, float for fixed pars
            guess - ndarray of length N with initial guesses
            guessScale - ndarray of length N with scale for range of initial guesses
    """

    # Define initial guess and range for emcee
    # these arrays will be shortened if there are fixed parameters
    guess=np.copy(model.origGuess)
    guessScale=np.copy(model.origGuessScale)
    nPars=len(model.priors) # Number of pars in FULL MODEL (some may get fixed and not be sent to emcee)

    fixed=np.repeat(None, nPars)
    priorFuncs=np.repeat(None, nPars)
    if(model.priors is not None):
        for ii in xrange(nPars):
            prior=model.priors[ii]
            # note: each of the assignments below needs to *copy* aspects of prior to avoid pointer overwriting
            if(prior is not None):
                if((isinstance(prior, int)) | (isinstance(prior, float))):
                # entry will be removed from list of pars and guess but value is still sent to evauluate function
                    fixVal=np.copy(prior)
                    fixed[ii]=fixVal
                elif(isinstance(prior, list)):
                    priorRange=np.copy(prior)
                    priorFuncs[ii]=makeFlatPrior(priorRange)
                elif(isinstance(prior, tuple)):
                    if(len(prior)==2):
                        priorMean=np.copy(prior[0])
                        priorSigma=np.copy(prior[1])
                        priorFuncs[ii]=makeGaussPrior(priorMean,priorSigma)
                    elif(len(prior)==4):
                        priorMean=np.copy(prior[0])
                        priorSigma=np.copy(prior[1])
                        priorRange=np.copy(prior[2:])
                        priorFuncs[ii]=makeGaussTruncPrior(priorMean,priorSigma,priorRange)
                    else:
                        raise ValueError(prior)
                else:
                    raise ValueError(ii,prior,type(prior))

    # remove fixed entries from list of pars to fit
    delarr=np.array([])
    for ii in xrange(nPars):
        if(fixed[ii] is not None):
            delarr=np.append(delarr,ii)
    if(len(delarr) > 0):
        guess=np.delete(guess,delarr)
        guessScale=np.delete(guessScale,delarr)
        priorFuncs=np.delete(priorFuncs,delarr)

    model.priorFuncs=priorFuncs
    model.guess=guess
    model.guessScale=guessScale
    model.fixed=fixed

    return


####
# Evaluate Likelihood
####

def lnProbVMapModel(pars, model, xobs, yobs, vobs, verr, ellobs, ellerr):
    """Return ln(P(model|data)) = -0.5*chisq to evaluate likelihood surface.

    Take model parameters, priors, and data and compute chisq=sum[((model-data)/error)**2].

    Note: if you only want to consider spectroscopic or imaging observables,
          set xobs or ellobs to None

    Inputs:
        pars - ndarray of N model parameters to be fit (N<=M)
        priors - see interpretPriors for format
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        vobs - float or ndarray of fiber velocities
        verr - errors on vobs
        ellobs - imaging observables
        ellerr - errors on ellobs
        
    Returns:
        lnP - a float (this is what emcee needs)
    """

    # wrap PA to fall between 0 and 360
    if(model.fixed[0] is None):
        pars[0]=pars[0] % 360.
    else:
        model.fixed[0]=model.fixed[0] % 360.

    # First evaluate the prior to see if this set of pars should be ignored
    chisq_prior=0.
    if(model.priorFuncs is not None):
        for ii in range(len(model.priorFuncs)):
            func=model.priorFuncs[ii]
            if(func is not None):
                chisq_prior+=func(pars[ii])
        if(chisq_prior == np.Inf):
            return -np.Inf

    # re-insert any fixed parameters into pars array
    nPars=len(model.fixed)
    fullPars=np.zeros(nPars)
    parsInd=0
    for ii in xrange(nPars):
        if(model.fixed[ii] is None):
            fullPars[ii]=pars[parsInd]
            parsInd+=1
        else:
            fullPars[ii]=model.fixed[ii]

    # Reassign model object attributes with pars array
    model.updatePars(fullPars)

    if((xobs is None) & (ellobs is None)): # no data, only priors
        chisq_like=0.
    else:
        if((xobs is None) & (ellobs is not None)): # use only imaging data
            modelVals=sim.ellModel(model)
            dataVals=ellobs
            errorVals=ellerr
        elif((xobs is not None) & (ellobs is None)): # use only velocity data
            if(model.convOpt is not None):
                modelVals=sim.vmapObs(model,xobs,yobs)
            else: # this is faster if we don't need to convolve with psf or fiber
                modelVals=sim.vmapModel(model,xobs,yobs)
            dataVals=vobs
            errorVals=verr
        elif((xobs is not None) & (ellobs is not None)): # use both imaging and velocity data
            if(model.convOpt is not None):
                vmodel=sim.vmapObs(model,xobs,yobs)
            else: # this is faster if we don't need to convolve with psf or fiber
                vmodel=sim.vmapModel(model,xobs,yobs)
            modelVals=np.concatenate([vmodel,sim.ellModel(model)])
            dataVals=np.concatenate([vobs,ellobs])
            errorVals=np.concatenate([verr,ellerr])

        chisq_like=np.sum(((modelVals-dataVals)/errorVals)**2)

    return -0.5*(chisq_like+chisq_prior)


def vmapFit(vobs,sigma,imObs,imErr,model,addNoise=True,nWalkers=2000,nBurn=50,nSteps=250,seed=None):
    """Call emcee and return sampler to fit model to velocity and/or imaging data

    Inputs:
        vobs - velocity data array to be fit
        sigma - errorbars on vobs (e.g. 30 km/s)
        imObs - imaging data array to be fit
        imErr - errorbars on imObs
        priors - see interpretPriors for format
        disk_r, convOpt, atmos_fwhm, fibRad, fibConvolve, fibConfig, fibPA 
            - see sim.vmapObs
        addNoise - bool for whether to fit noisy or noise-free observations
        nWalkers, nBurn, nSteps - see emcee documentation

    Returns:
        sampler - emcee object with posterior chains
    """

    # SETUP DATA
    if(vobs is not None):
        numFib=vobs.size
        pos,fibShape=sim.getSamplePos(model.nVSamp,model.vSampSize,model.vSampConfig,sampPA=model.vSampPA)
        xobs,yobs=pos
        vel=np.array(vobs).copy()
        velErr=np.repeat(sigma,numFib)

        # SETUP CONVOLUTION KERNEL
        if(model.convOpt=="pixel"):
            model.kernel=sim.makeConvolutionKernel(xobs,yobs,model)
        else: #convOpt is "galsim" or None
            model.kernel=None
    else:
        xobs=None
        yobs=None
        vel=None
        velErr=None
        model.kernel=None

    if(imObs is not None):
        ellObs=np.array(imObs).copy()
        ellErr=np.array(imErr).copy()
    else:
        ellObs=None
        ellErr=None

    if(addNoise): # useful when simulating many realizations to project parameter constraints
        np.random.seed(seed)
        # NOTE: imObs will always be 2 number, but len(vobs) may vary with fiber configuration
        #       To preserve random seed, generate imObs noise first
    if(imObs is not None):
        imNoise=np.random.randn(ellObs.size)*ellErr
        ellObs+=imNoise
    if(vobs is not None):
        specNoise=np.random.randn(numFib)*sigma
        vel+=specNoise


    # SETUP PARS and PRIORS
    interpretPriors(model)
    nPars=len(model.guess)

    # RUN MCMC
    walkerStart=np.array([np.random.randn(nWalkers)*model.guessScale[ii]+model.guess[ii] for ii in xrange(nPars)]).T
    sampler=emcee.EnsembleSampler(nWalkers,nPars,lnProbVMapModel,args=[model, xobs, yobs, vel, velErr, ellObs, ellErr])
    print "emcee burnin"
    pos, prob, state = sampler.run_mcmc(walkerStart,nBurn)
    sampler.reset()

    print "emcee running"
    sampler.run_mcmc(pos, nSteps)

    return sampler

def fitObs(specObs,specErr,imObs,imErr,model,**kwargs):
    """Wrapper to vmapFit to compute chains for each set of observables

    Passes kwargs to vmapFit
    """

    print "Imaging"
    samplerI=vmapFit(None,specErr,imObs,imErr,model,**kwargs)
    print "Spectroscopy"
    samplerS=vmapFit(specObs,specErr,None,imErr,model,**kwargs)
    print "Combined"
    samplerIS=vmapFit(specObs,specErr,imObs,imErr,model,**kwargs)
    
    flatchainI=samplerI.flatchain
    flatlnprobI=samplerI.flatlnprobability
    flatchainS=samplerS.flatchain
    flatlnprobS=samplerS.flatlnprobability
    flatchainIS=samplerIS.flatchain
    flatlnprobIS=samplerIS.flatlnprobability
    
    goodI=(flatlnprobI > -np.Inf)
    goodS=(flatlnprobS > -np.Inf)
    goodIS=(flatlnprobIS > -np.Inf)

    chains=[flatchainI[goodI], flatchainS[goodS], flatchainIS[goodIS]]
    lnprobs=[flatlnprobI[goodI],flatlnprobS[goodS],flatlnprobIS[goodIS]]
    return (chains,lnprobs)

####
# Chain statistics
####

def getMaxProb(chain,lnprob):
    """Return chain pars that give max lnP"""
    maxP=(lnprob == np.max(lnprob)).nonzero()[0][0]
    return chain[maxP]

def getPeakKDE(chain,guess):
    """Return chain pars that give peak of posterior PDF, using KDE"""
    if(len(chain.shape)==1):
        nPars=1
        kern=scipy.stats.gaussian_kde(chain)
        peakKDE=scipy.optimize.fmin(lambda x: -kern(x), guess,disp=False)
        return peakKDE
    else:
        nPars=chain.shape[1]
        peakKDE=np.zeros(nPars)
        for ii in range(nPars):
            kern=scipy.stats.gaussian_kde(chain[:,ii])
            peakKDE[ii]=scipy.optimize.fmin(lambda x: -kern(x), guess[ii],disp=False)
        return peakKDE

def getMedPost(chain):
    """Return chain pars that give median of posterior PDF

    Warning! this is not a good estimator when posteriors are flat
    """
    return np.median(chain,axis=0)

def get68(chain,opt="hw"):
    """Return description of 16-84 percentile range in posterior PDF

    Inputs: opt - "hw" returns half-width. For a gaussian
                       distribution, this is 1-sigma
                  "low" returns the 16th percentile
                  "high" returns the 84th percentile
                  "lowhigh" returns a tuple (16th, 84th) percentile range
    """
    nSteps=len(chain)
    chainSort=np.sort(chain,axis=0)
    low68=chainSort[0.16*nSteps]
    high68=chainSort[0.84*nSteps]
    hw68=0.5*(high68-low68)
    if(opt=="hw"):
        return hw68
    elif(opt=="low"):
        return low68
    elif(opt=="high"):
        return high68
    elif(opt=="lowhigh"):
        return (low68,high68)

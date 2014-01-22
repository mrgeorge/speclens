#! env python

import numpy as np
import emcee
import scipy.stats
import sim

####
# Priors
####

def getPriorFuncs(priors):
    """Translate human-readable description of priors into functions

    Inputs:
        priors - a list of tuples [(string, arg1, arg2...), ...]
                each tuple is for one parameter
                string options:
                    "norm" - gaussian, arg1=mean, arg2=sigma
                    "truncnorm" - truncated gaussian, arg1=mean,
                                      arg2=sigma, arg3=min,
                                      arg4=max
                    "uniform" - arg1=min, arg2=max
                    "wrap" similar to uniform, but lnProb will also 
                           wrap values within bounds. 
                           arg1=min, arg2=max
            Note: "fixed" values should be removed beforehand
    Returns:
        priorFuncs - array of functions from scipy.stats
                     useful sub-methods include pdf(x) and logpdf(x)
                     which evaluate the distribution at x and rvs(N)
                     which produces N random deviates.
    """

    nPars=len(priors)
    priorFuncs=np.repeat(None, nPars)
    for ii,prior in enumerate(priors):
        # note: each of the assignments below needs to *copy*
        # aspects of prior to avoid pointer overwriting
        if(prior is not None):
            if(prior[0]=="norm"):
                priorFuncs[ii]=scipy.stats.norm(
                        loc=np.float64(np.copy(prior[1])),
                        scale=np.float64(np.copy(prior[2])))
            elif(prior[0]=="truncnorm"):
                pmean=np.float64(np.copy(prior[1]))
                psigma=np.float64(np.copy(prior[2]))
                pmin=np.float64(np.copy(prior[3]))
                pmax=np.float64(np.copy(prior[4]))
                priorFuncs[ii]=scipy.stats.truncnorm((pmin-pmean)/psigma,
                        (pmax-pmean)/psigma, loc=pmean, scale=psigma)
            elif(prior[0] in ("uniform", "wrap")):
                pmin=np.float64(np.copy(prior[1]))
                pmax=np.float64(np.copy(prior[2]))
                priorFuncs[ii]=scipy.stats.uniform(loc=pmin,
                        scale=pmax-pmin)
            else:
                raise ValueError(prior[0])
        else:
            priorFuncs[ii]=None

    return priorFuncs

def wrapPars(priors, pars):
    """Handle pars that should be wrapped around a given range

    Parameters can be wrapped to allow a finite range without
    messy sampling effects near the boundaries.

    To wrap a parameter, specify its prior as ("wrap",min,max)
    The distribution is assumed uniform over the range [min,max)

    Inputs:
        priors - see getPriorFuncs for description
        pars - ndarray of parameters

    Returns:
        pars is updated with any wraps
    """
    for ii,prior in enumerate(priors):
        if(prior is not None):
            if(prior[0]=="wrap"):
                pmin=np.float64(np.copy(prior[1]))
                pmax=np.float64(np.copy(prior[2]))
                pars[ii]=(pars[ii]-pmin) % (pmax-pmin) + pmin
    return

def removeFixedPars(model):
    """Find pars to be fixed and removed from MCMC

    Values can be fixed by setting their prior as ("fixed", value).
    As a result, they are removed from the chain to speed up
    evaluation.

    Model object is inspected for origPriors and updated with the addition
    of the following arrays:
        fixed - same length as origGuess, entries are None for free
                pars and the fixed value for fixed pars. This array can be 
                used to to reconstruct the full parameter array when coupled 
                with an entry from the Markov chain.

        priors, guess, guessScale - copies of origPriors, origGuess,
                origGuessScale with any fixed pars removed
    """
    
    # store initial guess and range for emcee
    # these arrays will be shortened if there are fixed parameters
    priors=np.copy(model.origPriors)
    guess=np.copy(model.origGuess)
    guessScale=np.copy(model.origGuessScale)
    nPars=len(priors) # Number of pars in FULL MODEL (some may get fixed and not be sent to emcee)

    # find fixed entries and record values
    fixed=np.repeat(None, nPars)
    delarr=np.array([])
    for ii,prior in enumerate(priors):
        if(prior is not None):
            if(prior[0]=="fixed"):
                fixed[ii]==np.copy(prior[1])
                delarr=np.append(delarr,ii)

    # remove fixed entries from list of pars to fit
    if(len(delarr) > 0):
        priors=np.delete(priors,delarr)
        guess=np.delete(guess,delarr)
        guessScale=np.delete(guessScale,delarr)

    # update model object
    model.fixed=fixed
    model.priors=priors
    model.guess=guess
    model.guessScale=guessScale

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
        model object with fixed array defined
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        vobs - float or ndarray of fiber velocities
        verr - errors on vobs
        ellobs - imaging observables
        ellerr - errors on ellobs
        
    Returns:
        lnP - a float (this is what emcee needs)
    """

    # wrap any pars that need wrapping, e.g. PA
    wrapPars(model.priors, pars)

    # First evaluate the prior to see if this set of pars should be ignored
    lnp_prior=0.
    priorFuncs=getPriorFuncs(model.priors)
    if(priorFuncs is not None):
        for ii in range(len(priorFuncs)):
            func=priorFuncs[ii]
            if(func is not None):
                lnp_prior+=func.logpdf(pars[ii])  # logpdf requires a modern version of scipy
        if(lnp_prior == -np.Inf):  # we can skip likelihood evaluation
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

    return -0.5*chisq_like + lnp_prior


def vmapFit(vobs,sigma,imObs,imErr,model,addNoise=True,nWalkers=2000,nBurn=50,nSteps=250,nThreads=1,seed=None):
    """Call emcee and return sampler to fit model to velocity and/or imaging data

    Inputs:
        vobs - velocity data array to be fit
        sigma - errorbars on vobs (e.g. 30 km/s)
        imObs - imaging data array to be fit
        imErr - errorbars on imObs
        model object with priors
        addNoise - bool for whether to fit noisy or noise-free observations
        nWalkers, nBurn, nSteps, nThreads - see emcee documentation
        seed - optional for random number repeatability
    
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

    if(addNoise): # useful when simulating many realizations to
                  # project parameter constraints
        np.random.seed(seed)
        # NOTE: imObs will always be 2 number, but len(vobs) may vary
        # with fiber configuration. To preserve random seed, generate
        # imObs noise first
        if(imObs is not None):
            imNoise=np.random.randn(ellObs.size)*ellErr
            ellObs+=imNoise
        if(vobs is not None):
            specNoise=np.random.randn(numFib)*sigma
            vel+=specNoise

    # SETUP CHAIN SHAPE
    removeFixedPars(model)
    nPars=len(model.guess) # number of FREE pars to fit

    # RUN MCMC
    walkerStart=np.array([np.random.randn(nWalkers)*model.guessScale[ii]+model.guess[ii] for ii in xrange(nPars)]).T
    sampler=emcee.EnsembleSampler(nWalkers,nPars,lnProbVMapModel,args=[model, xobs, yobs, vel, velErr, ellObs, ellErr],threads=nThreads)
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

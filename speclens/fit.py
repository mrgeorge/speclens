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
                try:  # assume pars is a chain (nsteps x npars)
                    pars[:,ii]=(pars[:,ii]-pmin) % (pmax-pmin) + pmin
                except IndexError:  # allow if pars is a single entry
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

def chisq(modelVector, dataVector, errVector, wrapVector):
    if(wrapVector is None):
        return np.sum(((modelVector - dataVector) / errVector)**2)

    # handle wrapped pars
    chisq = 0.
    for model, data, err, wrap in zip(modelVector, dataVector,
                                        errVector, wrapVector):
        if(wrap is None):
            chisq += ((model - data) / err)**2
        else:
            width = wrap[1] - wrap[0]
            assert(err < width)
            model = (model-wrap[0]) % width + wrap[0]
            data = (data-wrap[0]) % width + wrap[0]
            delta = np.min([np.abs(model - data),
                            np.abs(width - (model-data))])
            chisq += (delta / err)**2
    return chisq

def lnProbVMapModel(pars, model, observation):
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

    # First evaluate prior
    # If out of range, ignore (return -np.Inf)
    lnPPrior=0.
    priorFuncs=getPriorFuncs(model.priors)
    if(priorFuncs is not None):
        for ii in range(len(priorFuncs)):
            func=priorFuncs[ii]
            if(func is not None):
                lnPPrior+=func.logpdf(pars[ii])  # logpdf requires a
                                                  # modern version of
                                                  # scipy
        if(lnPPrior == -np.Inf):  # we can skip likelihood evaluation
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

    # Project new model into observable space (time-consuming step)
    model.updateObservable(observation.dataType)

    # Compute likelihood
    if(observation.dataVector is None):  # no data, only priors
        chisqLike = 0.
    else:
        chisqLike = chisq(model.obs.dataVector,
            observation.dataVector, observation.errVector,
            observation.wrapVector)

    # Compute lnP
    lnP = -0.5 * chisqLike + lnPPrior

    return lnP


def vmapFit(model, observation, addNoise=True, nWalkers=2000,
            nBurn=50, nSteps=250, nThreads=1, seed=None):
    """Run MCMC to fit model to observation

    Inputs:
        (observation object contains)
            dataVector
            errVector
            wrapVector
        model object with priors
        addNoise - bool for noisy or noise-free observations
        nWalkers, nBurn, nSteps, nThreads - see emcee documentation
        seed - optional int for random number repeatability
    
    Returns:
        sampler - emcee object with posterior chains
    """

    # ADD OPTIONAL NOISE
    if(addNoise): # useful when simulating many realizations to
                  # project parameter constraints
        np.random.seed(seed)
        noise = np.random.randn(observation.dataVector.size)
        observation.dataVector += (noise.reshape(observation.dataVector.shape) *
                observation.errVector)

    # SETUP CHAIN SHAPE
    removeFixedPars(model)
    nPars=len(model.guess) # number of FREE pars to fit

    # RUN MCMC

    walkerStart = np.array([np.random.randn(nWalkers) *
        model.guessScale[ii] + model.guess[ii] for ii in
        xrange(nPars)]).T
    sampler = emcee.EnsembleSampler(nWalkers, nPars, lnProbVMapModel,
        args=[model, observation], threads=nThreads)
    print "emcee burnin"
    pos, prob, state = sampler.run_mcmc(walkerStart,nBurn)
    sampler.reset()

    print "emcee running"
    sampler.run_mcmc(pos, nSteps)

    # wrap chain entries this handles cases where emcee guesses a
    # value outside of the wrap range. The lnP returned may be good,
    # but we want to store the wrapped values, e.g. for plotting
    wrapPars(model.priors, sampler.flatchain)

    # we probably only care about flatchain, but let's reset sampler's
    # chain attr too. To do this, need to set _chain since chain can't
    # be set directly
    sampler._chain=sampler.flatchain.reshape((nWalkers,nSteps,nPars))
    
    return sampler

def fitObs(model, observation, **kwargs):
    """Wrapper to vmapFit to compute chains for each set of observables

    Passes kwargs to vmapFit
    """

    print "Imaging"
    observation.dataType = "imgPar"
    observation.defineDataVector(observation.dataType)
    samplerI=vmapFit(model, observation, **kwargs)
    print "Spectroscopy"
    observation.dataType = "velocities"
    observation.defineDataVector(observation.dataType)
    samplerS=vmapFit(model, observation, **kwargs)
    print "Combined"
    observation.dataType = "imgPar+velocities"
    observation.defineDataVector(observation.dataType)
    samplerIS=vmapFit(model, observation, **kwargs)
    
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

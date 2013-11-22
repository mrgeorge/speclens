#! env python

import numpy as np
import emcee
import scipy.stats
import sim

####
# Priors
####

def makeFlatPrior(range):
    return lambda x: priorFlat(x, range)

def priorFlat(arg, range):
    if((arg >= range[0]) & (arg < range[1])):
	return 0
    else:
	return np.Inf

def makeGaussPrior(mean, sigma):
    return lambda x: ((x-mean)/sigma)**2

def interpretPriors(priors):
# priors is a list or tuple with an entry for each fit parameter: gal_beta, gal_q, vmax, g1,g2
#    None: leave this variable completely free
#    float: fix the variable to this value
#    list[a,b]: flat prior between a and b
#    tuple(a,b): gaussian prior with mean a and stddev b

    guess=np.array([10.,0.1,100.,0.,0.])
    guessScale=np.array([10.,0.3,50.,0.02,0.02])
    nPars=len(guess)
    fixed=np.repeat(None, nPars)
    priorFuncs=np.repeat(None, nPars)
    if(priors is not None):
	for ii in xrange(nPars):
	    prior=priors[ii]
	    # note: each of the assignments below needs to *copy* aspects of prior to avoid pointer overwriting
	    if(prior is not None):
		if((type(prior) is int) | (type(prior) is float)):
		# entry will be removed from list of pars and guess but value is still sent to evauluate function
		    fixVal=np.copy(prior)
		    fixed[ii]=fixVal
		elif(type(prior) is list):
		    priorRange=np.copy(prior)
		    priorFuncs[ii]=makeFlatPrior(priorRange)
		elif(type(prior) is tuple):
		    priorMean=np.copy(prior[0])
		    priorSigma=np.copy(prior[1])
		    priorFuncs[ii]=makeGaussPrior(priorMean,priorSigma)

    # remove fixed entries from list of pars to fit
    delarr=np.array([])
    for ii in xrange(nPars):
	if(fixed[ii] is not None):
            delarr=np.append(delarr,ii)
    if(len(delarr) > 0):
        guess=np.delete(guess,delarr)
        guessScale=np.delete(guessScale,delarr)
        priorFuncs=np.delete(priorFuncs,delarr)

    return (priorFuncs,fixed,guess,guessScale)


####
# Evaluate Likelihood
####

def lnProbVMapModel(pars, xobs, yobs, vobs, verr, ellobs, ellerr, priorFuncs, fixed, disk_r, convOpt, atmos_fwhm, fibRad, fibConvolve, kernel):
# pars are the free parameters to be fit (some or all of [PA, b/a, vmax, g1, g2])
# xobs, yobs are the fiber positions for velocity measurements
# vobs are the measured velocities
# verr are the uncertainties in measured velocities
# ellobs are the imaging observables sheared (PA, b/a)
# ellerr are the uncertainties on the imaging observables
# priorFuncs are the functions that define the priors (from interpretPriors)
# fixed is an array with an entry for each of [PA, b/a, vmax, g1, g2]. float means fix, None means free
# returns ln(P(model|data))

# Note: if you only want to consider spectroscopic or imaging observables, set xobs or ellobs to None
	
    # wrap PA to fall between 0 and 360
    if(fixed[0] is None):
	pars[0]=pars[0] % 360.
    else:
	fixed[0]=fixed[0] % 360.

    # First evaluate the prior to see if this set of pars should be ignored
    chisq_prior=0.
    if(priorFuncs is not None):
	for ii in range(len(priorFuncs)):
	    func=priorFuncs[ii]
	    if(func is not None):
		chisq_prior+=func(pars[ii])
    if(chisq_prior == np.Inf):
	return -np.Inf

    # re-insert any fixed parameters into pars array
    nPars=len(fixed)
    fullPars=np.zeros(nPars)
    parsInd=0
    for ii in xrange(nPars):
	if(fixed[ii] is None):
	    fullPars[ii]=pars[parsInd]
	    parsInd+=1
        else:
            fullPars[ii]=fixed[ii]

    if((xobs is None) & (ellobs is None)): # no data, only priors
	chisq_like=0.
    else:
        if((xobs is None) & (ellobs is not None)): # use only imaging data
	    model=sim.ellModel(fullPars)
	    data=ellobs
	    error=ellerr
	elif((xobs is not None) & (ellobs is None)): # use only velocity data
            if(convOpt is not None):
                model=sim.vmapObs(fullPars,xobs,yobs,disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
            else: # this is faster if we don't need to convolve with psf or fiber
                model=sim.vmapModel(fullPars,xobs,yobs)
	    data=vobs
	    error=verr
        elif((xobs is not None) & (ellobs is not None)): # use both imaging and velocity data
            if(convOpt is not None):
                vmodel=sim.vmapObs(fullPars,xobs,yobs,disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
            else: # this is faster if we don't need to convolve with psf or fiber
                vmodel=sim.vmapModel(fullPars,xobs,yobs)
	    model=np.concatenate([vmodel,ellModel(fullPars)])
	    data=np.concatenate([vobs,ellobs])
	    error=np.concatenate([verr,ellerr])

	chisq_like=np.sum(((model-data)/error)**2)

    return -0.5*(chisq_like+chisq_prior)



def vmapFit(vobs,sigma,imObs,imErr,priors,disk_r=None,convOpt=None,atmos_fwhm=None,fibRad=1.,fibConvolve=False,fibConfig="hexNoCen",fibPA=None,addNoise=True,nWalkers=2000,nBurn=50,nSteps=250,seed=None):
# fit model to fiber velocities
# vobs is the data to be fit
# sigma is the errorbar on that value (e.g. 30 km/s)

# priors is a list or tuple with an entry for each fit parameter: gal_beta, gal_q, vmax
#    None: leave this variable completely free
#    float: fix the variable to this value
#    list[a,b]: flat prior between a and b
#    tuple(a,b): gaussian prior with mean a and stddev b

    # SETUP DATA
    if(vobs is not None):
	numFib=vobs.size
        pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig,fibPA=fibPA)
        xobs,yobs=pos
	vel=np.array(vobs).copy()
	velErr=np.repeat(sigma,numFib)

        # SETUP CONVOLUTION KERNEL
        if(convOpt=="pixel"):
            kernel=sim.makeConvolutionKernel(xobs,yobs,atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA)
        else: #convOpt is "galsim" or None
            kernel=None
    else:
	xobs=None
        yobs=None
	vel=None
	velErr=None
        kernel=None

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
    priorFuncs,fixed,guess,guessScale = interpretPriors(priors)
    nPars=len(guess)

    # RUN MCMC
    walkerStart=np.array([np.random.randn(nWalkers)*guessScale[ii]+guess[ii] for ii in xrange(nPars)]).T
    sampler=emcee.EnsembleSampler(nWalkers,nPars,lnProbVMapModel,args=[xobs, yobs, vel, velErr, ellObs, ellErr, priorFuncs, fixed, disk_r, convOpt, atmos_fwhm, fibRad, fibConvolve, kernel])
    print "emcee burnin"
    pos, prob, state = sampler.run_mcmc(walkerStart,nBurn)
    sampler.reset()

    print "emcee running"
    sampler.run_mcmc(pos, nSteps)

    return sampler

def fitObs(specObs,specErr,imObs,imErr,priors,**kwargs):
# wrapper to vmapFit to compare chains with imaging, spectroscopy, and combined observables
# passes kwargs to vmapFit

    print "Imaging"
    samplerI=vmapFit(None,specErr,imObs,imErr,priors,**kwargs)
    print "Spectroscopy"
    samplerS=vmapFit(specObs,specErr,None,imErr,priors,**kwargs)
    print "Combined"
    samplerIS=vmapFit(specObs,specErr,imObs,imErr,priors,**kwargs)
    
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

def getMaxProb(chain,lnprob):
    maxP=(lnprob == np.max(lnprob)).nonzero()[0][0]
    return chain[maxP]

def getPeakKDE(chain,guess):
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
# this is not a good estimator when posteriors are flat
    return np.median(chain,axis=0)

def get68(chain,opt="hw"):
# get half-width of 68% confidence range
# for a gaussian distribution, this is 1-sigma
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
    

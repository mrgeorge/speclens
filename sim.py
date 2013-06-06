#! env python

import galsim
import scipy.integrate
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches
import emcee

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
plt.rc('text', usetex=True)
plt.rc('axes',linewidth=1.5)

def getFiberPos(fibID,numFib,fibRad):
    # returns fiber position relative to center in same units as fibRad (arcsec)
    if(fibID == 0):
        pos=galsim.PositionD(x=0,y=0)
    else:
        theta=(fibID-1.)/(numFib-1.) * 2.*np.pi
        rad=2.*fibRad
        pos=galsim.PositionD(x=rad*np.cos(theta),y=rad*np.sin(theta))
    return pos

# These functions take a galsim object <image> and integrate over the area of a fiber
def radIntegrand(rad,theta,fiberPos,image):
    pos=galsim.PositionD(x=fiberPos.x+rad*np.cos(theta),y=fiberPos.y+rad*np.sin(theta))
    return rad*image.xValue(pos)
def thetaIntegrand(theta,fiberPos,image,fibRad,tol):
    return scipy.integrate.quad(radIntegrand,0,fibRad,args=(theta,fiberPos,image),epsabs=tol,epsrel=tol)[0]
def getFiberFlux(fibID,numFib,fibRad,image,tol=1.e-4):
    fiberPos=getFiberPos(fibID,numFib,fibRad)
    return scipy.integrate.quad(thetaIntegrand,0,2.*np.pi, args=(fiberPos,image,fibRad,tol), epsabs=tol, epsrel=tol)

def showImage(profile,numFib,fibRad,filename=None,colorbar=True,cmap=matplotlib.cm.jet,plotScale="linear",trim=0,xlabel="x (arcsec)",ylabel="y (arcsec)",ellipse=None):
# Plot image given by galsim object <profile> with fiber pattern overlaid

    pixScale=0.1
    imgSizePix=int(10.*fibRad/pixScale)
    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)
    img=profile.draw(image=imgFrame,dx=pixScale)
    halfWidth=0.5*imgSizePix*pixScale # arcsec
    #    img.setCenter(0,0)

    if(plotScale=="linear"):
	plotArr=img.array
    elif(plotScale=="log"):
	plotArr=np.log(img.array)

    plt.imshow(plotArr,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest',cmap=cmap)
    for ii in range(numFib):
        pos=getFiberPos(ii,numFib,fibRad)
        circ=plt.Circle((pos.x,pos.y),radius=fibRad,fill=False,color='white',lw=2)
        ax=plt.gca()
        ax.add_patch(circ)
    if(colorbar):
	plt.colorbar()

    if(ellipse is not None): # ellipse is either None or np.array([disk_r,gal_q,gal_beta])
	ax=plt.gca()
	rscale=2
	ell=matplotlib.patches.Ellipse(xy=(0,0),width=rscale*ellipse[0]*ellipse[1],height=rscale*ellipse[0],angle=ellipse[2]-90,fill=False,color="white",lw=2)
	ax.add_artist(ell)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(trim>0): # trim all edges by this amount in arcsec
	plt.xlim((-halfWidth+trim,halfWidth-trim))
	plt.ylim((-halfWidth+trim,halfWidth-trim))

    if(filename):
	plt.savefig(filename)
    plt.show()

def contourPlot(xvals,yvals,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],xlabel="X",ylabel="Y",filename=None):
# make a 2d contour plot of parameter posteriors

    n2dbins=300
    zz,xx,yy=np.histogram2d(xvals,yvals,bins=n2dbins)
    xxbin=xx[1]-xx[0]
    yybin=yy[1]-yy[0]
    xx=xx[1:]+0.5*xxbin
    yy=yy[1:]+0.5*yybin

    if(smooth > 0):
	kernSize=int(10*smooth)
	sx,sy=scipy.mgrid[-kernSize:kernSize+1, -kernSize:kernSize+1]
	kern=np.exp(-(sx**2 + sy**2)/(2.*smooth**2))
	zz=scipy.signal.convolve2d(zz,kern/np.sum(kern),mode='same')
	
    hist,bins=np.histogram(zz.flatten(),bins=1000)
    sortzz=np.sort(zz.flatten())
    cumhist=np.cumsum(sortzz)*1./np.sum(zz)
    levels=np.array([sortzz[(cumhist>(1-pct)).nonzero()[0][0]] for pct in percentiles])

    plt.contour(xx,yy,zz.T,levels=levels,colors=["red","green","blue"])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(filename):
	plt.savefig(filename)
    plt.show()
    

def makeGalImage(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm):
    # Define the galaxy profile
    bulge=galsim.Sersic(bulge_n, half_light_radius=bulge_r)
    disk=galsim.Sersic(disk_n, half_light_radius=disk_r)

    gal=bulge_frac * bulge + (1.-bulge_frac) * disk
    gal.setFlux(gal_flux)
    
    # Set shape of galaxy from axis ratio and position angle
    gal_shape=galsim.Shear(q=gal_q, beta=gal_beta*galsim.degrees)
    gal.applyShear(gal_shape)

    # Define atmospheric PSF
    atmos=galsim.Kolmogorov(fwhm=atmos_fwhm)

    # Convolve galaxy with PSF
    nopixX=galsim.Convolve([atmos, gal],real_space=True) # used real-space convolution for easier real-space integration
    nopixK=galsim.Convolve([atmos, gal],real_space=False) # used fourier-space convolution for faster drawing

    return (nopixX, nopixK)

def getInclination(gal_q):
# see http://eo.ucar.edu/staff/dward/sao/spirals/methods.htm
    # just give simple flat thin disk case for now
    return np.arccos(gal_q) # radians

def getOmega(rad,pars,option='flat'):
# return angular rotation rate, i.e. v(r)/r

    if(option=='flat'): # v(r)=pars[0]
        return pars[0]/rad
    elif(option=='solid'): # v(r)=pars[0]*rad/pars[1]
        return pars[0]/pars[1]
    elif(option=='nfw'):
        # v ~ sqrt(M(<r)/r)
        # M(<r) ~ [log(1 + r/rs) - r/(r+rs)]
        mass=np.log(1.+rad/pars[1]) - rad/(rad+pars[1])
        vel=np.sqrt(mass/rad)
        return pars[0]*vel/rad

def makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,e1,e2):
# Construct galsim objects for galaxy (image), velocity map, and flux-weighted velocity map
# with optional lensing shear and PSF convolution
	
    # Define the galaxy velocity map
    bulge=galsim.Sersic(bulge_n, half_light_radius=bulge_r)
    disk=galsim.Sersic(disk_n, half_light_radius=disk_r)

    gal=bulge_frac * bulge + (1.-bulge_frac) * disk
    gal.setFlux(gal_flux)
    
    # Set shape of galaxy from axis ratio and position angle
    gal_shape=galsim.Shear(q=gal_q, beta=gal_beta*galsim.degrees)
    gal.applyShear(gal_shape)

    # Generate galaxy image and empty velocity map array
    halfWidth=0.5*imgSizePix*pixScale
    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)
    galImg=gal.draw(image=imgFrame,dx=pixScale)
    imgArr=galImg.array.copy()	# must store these arrays as copies to avoid overwriting with shared imgFrame

    vmapArr=np.zeros_like(imgArr)
    fluxVMapArr=np.zeros_like(imgArr)

    # Set up velocity map parameters
    xCen=0.5*(galImg.xmax-galImg.xmin)
    yCen=0.5*(galImg.ymax-galImg.ymin)

    inc=getInclination(gal_q)
    sini=np.sin(inc)
    tani=np.tan(inc)
    gal_beta_rad=np.deg2rad(gal_beta)

    if(rotCurveOpt=='flat'):
        rotCurvePars=np.array([100])
    elif(rotCurveOpt=='solid'):
        rotCurvePars=np.array([100,5])
    elif(rotCurveOpt=='nfw'):
        rotCurvePars=np.array([100,5])

    # Fill velocity map array pixel by pixel
    for xx in range(galImg.xmin,galImg.xmax):
        for yy in range(galImg.ymin,galImg.ymax):
            # primed image coordinates centered on galaxy, rotated so xp is major axis
            xp=(xx-xCen)*np.cos(gal_beta_rad)+(yy-yCen)*np.sin(gal_beta_rad)
            yp=-(xx-xCen)*np.sin(gal_beta_rad)+(yy-yCen)*np.cos(gal_beta_rad)

            # coordinates in the plane of the galaxy
            radvec=np.array([xp,yp,yp*tani])
            kvec=np.array([1,0,0])
            vmapArr[yy,xx]=getOmega(np.linalg.norm(radvec),rotCurvePars,option=rotCurveOpt) * sini * np.dot(radvec,kvec)

    # Weight velocity map by galaxy flux and make galsim object
    fluxVMapArr=vmapArr*imgArr
    fluxVMapImg=galsim.ImageViewF(fluxVMapArr,scale=pixScale)
    fluxVMap=galsim.InterpolatedImage(fluxVMapImg)
    vmap=galsim.InterpolatedImage(galsim.ImageViewF(vmapArr,scale=pixScale)) # not flux-weighted

    # Apply lensing shear to galaxy and velocity maps
    gal.applyShear(e1=e1,e2=e2)
    galX=gal
    galK=gal
    fluxVMap.applyShear(e1=e1,e2=e2)
    vmap.applyShear(e1=e1,e2=e2)

    # Convolve velocity map and galaxy with PSF
    if(atmos_fwhm > 0):

        # Define atmospheric PSF
        #    atmos=galsim.Kolmogorov(fwhm=atmos_fwhm)
	atmos=galsim.Gaussian(fwhm=atmos_fwhm)

        # note: real-space convolution with InterpolatedImage doesn't seem to work,
        #       so just use scipy's convolve2d to convolve the arrays
	# must store these arrays as copies to avoid overwriting with shared imgFrame
	fluxVMapArr=fluxVMap.draw(image=imgFrame,dx=pixScale).array.copy()
	atmosArr=atmos.draw(image=imgFrame,dx=pixScale).array.copy()
	fluxVMapArrPSF=scipy.signal.convolve2d(fluxVMapArr,atmosArr,mode='same')
	fluxVMap=galsim.InterpolatedImage(galsim.ImageViewF(fluxVMapArrPSF,scale=pixScale))
        #    fluxVMapPSFK=galsim.Convolve([atmos, fluxVMap],real_space=False) # used fourier-space convolution for faster drawing
    
        galX=galsim.Convolve([atmos, gal],real_space=True) # used real-space convolution for easier real-space integration
        galX=galsim.Convolve([atmos, gal],real_space=False) # used real-space convolution for easier real-space integration

    return (vmap,fluxVMap,galX,galK)

def vmapObs(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,e1,e2,pixScale,fibRad,numFib,showPlot=False):
# get flux-weighted fiber-averaged velocities
	
    imgSizePix=int(10.*fibRad/pixScale)
    vmap,fluxVMap,gal,galK=makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,e1,e2)

    if(showPlot):
	showImage(galK,numFib,fibRad)
	showImage(vmap,numFib,fibRad)
	showImage(fluxVMap,numFib,fibRad)

    # Get the flux in each fiber
    galFibFlux=np.zeros(numFib)
    if((numFib-1) % 2 != 0):
        for ii in range(numFib):
            print "{}/{}".format(ii,numFib)
            galFibFlux[ii], error=getFiberFlux(ii,numFib,fibRad,gal)
            print galFibFlux[ii],error
    else: # take advantage of symmetry of outer fibers
        for ii in range(1+(numFib-1)/2):
            print "{}/{}".format(ii,numFib)
            galFibFlux[ii], error=getFiberFlux(ii,numFib,fibRad,gal)
            if(ii > 0):
                print "{}/{}".format(ii+(numFib-1)/2,numFib)
                galFibFlux[ii+(numFib-1)/2]=galFibFlux[ii]
            print galFibFlux[ii],error

    vmapFibFlux=np.zeros(numFib)
    for ii in range(numFib):
        print "{}/{}".format(ii,numFib)
        vmapFibFlux[ii], error=getFiberFlux(ii,numFib,fibRad,fluxVMap)
        print vmapFibFlux[ii],error

    print vmapFibFlux/galFibFlux

    return vmapFibFlux/galFibFlux

def lnProbVMapModel(pars, xobs, yobs, yerr, priorFuncs, fixed):
#
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

    chisq_like=np.sum(((vmapModel(fullPars,xobs)-yobs)/yerr)**2)

    return -0.5*(chisq_like+chisq_prior)

#def vmapModel(xvals, gal_beta, gal_q, vmax):
def vmapModel(pars, xvals):
# evaluate velocity field at azimuthal angles around center, called by curve_fit
# pars: PA in deg., gal_q, vmax
# xvals are the azimuthal angles (in radians) at which the field is sampled

    gal_beta,gal_q,vmax=pars

    fibRad=1.
    rad=2.*fibRad # assume the fibers sample the v field at their center

    # Cartesian coords
    xx=rad*np.cos(xvals)
    yy=rad*np.sin(xvals)    

    # rotated coords aligned with PA guess of major axis
    xCen,yCen=0,0 # assume centroid is well-measured
    PArad=np.deg2rad(gal_beta)
    xp=(xx-xCen)*np.cos(PArad)+(yy-yCen)*np.sin(PArad)
    yp=-(xx-xCen)*np.sin(PArad)+(yy-yCen)*np.cos(PArad)
    # projection along apparent major axis in rotated coords
    kvec=np.array([1,0,0])
    
    inc=getInclination(gal_q)
    sini=np.sin(inc)
    tani=np.tan(inc)

    rotCurvePars=np.array([vmax])
    nSamp=xvals.size
    vmodel=np.zeros(nSamp)
    for ii in range(xvals.size):
        # coordinates in the plane of the galaxy
	radvec=np.array([xp[ii],yp[ii],yp[ii]*tani])
	vmodel[ii]=getOmega(np.linalg.norm(radvec),rotCurvePars,option='flat') * sini * np.dot(radvec,kvec)

    return vmodel

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
# priors is a list or tuple with an entry for each fit parameter: gal_beta, gal_q, vmax
#    None: leave this variable completely free
#    float: fix the variable to this value
#    list[a,b]: flat prior between a and b
#    tuple(a,b): gaussian prior with mean a and stddev b

    guess=np.array([10.,0.1,100.])
    guessScale=np.array([10.,0.3,50.])
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
		    fixVal=np.copy(prior)[0]
		    fixed[ii]=fixVal
		elif(type(prior) is list):
		    priorRange=np.copy(prior)
		    priorFuncs[ii]=makeFlatPrior(priorRange)
		elif(type(prior) is tuple):
		    priorMean=np.copy(prior[0])
		    priorSigma=np.copy(prior[1])
		    priorFuncs[ii]=makeGaussPrior(priorMean,priorSigma)

    # remove fixed entries from list of pars to fit
    for ii in xrange(nPars):
	if(fixed[ii] is not None):
	    guess=np.delete(guess,ii)
	    guessScale=np.delete(guess,ii)
	    priorFuncs=np.delete(priorFuncs,ii)
	    nPars-=1

    return (priorFuncs,fixed,guess,guessScale)

def vmapFit(vfibFlux,sigma,priors,addNoise=True,showPlot=False):
# fit model to fiber velocities
# vfibFlux is the data to be fit
# sigma is the errorbar on that value (e.g. 30 km/s)

# priors is a list or tuple with an entry for each fit parameter: gal_beta, gal_q, vmax
#    None: leave this variable completely free
#    float: fix the variable to this value
#    list[a,b]: flat prior between a and b
#    tuple(a,b): gaussian prior with mean a and stddev b

    # SETUP DATA
    numFib=vfibFlux.size
    ang=np.linspace(0,2.*np.pi,num=numFib,endpoint=False)
    vel=vfibFlux.copy()
    velErr=np.repeat(sigma,numFib)

    if(addNoise): # useful when simulating many realizations to project parameter constraints
	noise=np.random.randn(numFib)*sigma
	vel+=noise


    # SETUP PARS and PRIORS
    priorFuncs,fixed,guess,guessScale = interpretPriors(priors)
    nPars=len(guess)


    # RUN MCMC
    nWalkers=500
    walkerStart=np.array([np.random.randn(nWalkers)*guessScale[ii]+guess[ii] for ii in xrange(nPars)]).T
    sampler=emcee.EnsembleSampler(nWalkers,nPars,lnProbVMapModel,args=[ang, vel, velErr, priorFuncs, fixed])

    walkerVals=np.array([lnProbVMapModel(walkerStart[ii,:],ang,vel,velErr,priorFuncs,fixed) for ii in xrange(nWalkers)])

    print "emcee burnin"
    nBurn=10
    pos, prob, state = sampler.run_mcmc(walkerStart,nBurn)
    sampler.reset()

    print "emcee running"
    nSteps=100
    sampler.run_mcmc(pos, nSteps)

    #    err= lambda pars, xvals, yvals: vmapModel(pars, xvals) - yvals
    #    pars, success = scipy.optimize.leastsq(err, guess[:], args=(ang,vel))
    #    pars, pcov=scipy.optimize.curve_fit(vmapModel, ang, vel, guess, sigma=weight, maxfev=100000)
    #    print "gal_beta={}, gal_q={}, vmax={}. success={}".format(pars[0],pars[1],pars[2],success)

    #    if(showPlot):
    #	fitX=np.linspace(0,2.*np.pi)
    #	fitY=vmapModel(fitX,pars[0],pars[1])
    #	plt.plot(ang,vel,'bo',fitX,fitY,'r-')
    #	plt.show()

    return sampler

def main(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,numFib,showPlot=False):

    #    bulge_n=4.
    #    bulge_r=1. # arcsec
    #    disk_n=1
    #    disk_r=2. # arcsec
    #    bulge_frac=0.3
    #    gal_q=0.1 # axis ratio 0<q<1 (1 for circular)
    #    gal_beta=30.*np.pi/180 # radians (position angle on the sky)
    atmos_fwhm=1.5 # arcsec
    fibRad=1. # fiber radius in arcsec
    #    numFib=7 # number of fibers, symmetric with one in center

    nopixX, nopixK = makeGalImage(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm)

    # Plot the image with fibers overlaid
    if(showPlot):
        showImage(nopixK,numFib,fibRad)

    # Get the flux in each fiber
    fibFlux=np.zeros(numFib)
    if((numFib-1) % 2 != 0):
        for ii in range(numFib):
            print "{}/{}".format(ii,numFib)
            fibFlux[ii], error=getFiberFlux(ii,numFib,fibRad,nopixX)
            print fibFlux[ii],error
    else: # take advantage of symmetry of outer fibers
        for ii in range(1+(numFib-1)/2):
            print "{}/{}".format(ii,numFib)
            fibFlux[ii], error=getFiberFlux(ii,numFib,fibRad,nopixX)
            if(ii > 0):
                print "{}/{}".format(ii+(numFib-1)/2,numFib)
                fibFlux[ii+(numFib-1)/2]=fibFlux[ii]
            print fibFlux[ii],error
        
    return fibFlux

if __name__ == "__main__":
    main(4,1,1,1,0,1,0,1,7)

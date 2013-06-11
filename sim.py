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

def shearEllipse(ellipse,g1,g2):
# Following Supri & Harari 1999
    disk_r,gal_q,gal_beta=ellipse
    gamma=np.sqrt(g1**2 + g2**2)
    epsilon=(1-gal_q**2)/(1+gal_q**2)
    psi=np.deg2rad(gal_beta)
    phi=0.5*np.arctan2(g2,g1)

    dpsi=0.5*np.arctan2(2*(gamma/epsilon)*np.sin(2.*(phi-psi)), 1.+2*(gamma/epsilon)*np.cos(2*(phi-psi)))
    epsilon_prime=np.sqrt((epsilon + 2.*gamma*np.cos(2.*(phi-psi)))**2 + 4*(gamma*np.sin(2*(phi-psi)))**2) / (1.+2.*epsilon*gamma*np.cos(2.*(phi-psi)))

    disk_r_prime=disk_r*(1+gamma)
    gal_q_prime=np.sqrt((1-epsilon_prime**2)/(1+epsilon_prime)**2)
    gal_beta_prime=np.rad2deg(psi+dpsi)

    return (disk_r_prime,gal_q_prime,gal_beta_prime)

def shearPairs(pairs,g1,g2):
    pairs_prime=pairs.copy()
    gSq=g1**2+g2**2
    if(pairs.shape == (2,)): # only one pair
	x1,y1=pairs
	#	x1p=((1+g1)*x1 -     g2*y1)
	#	y1p=(   -g2*x1 + (1-g1)*y1)
	#	x1p=((1-kappa-gamma1)*x1 -           gamma2*y1)
	#	y1p=(         -gamma2*x1 + (1-kappa+gamma1)*y1)
	x1p=1./np.sqrt(1.-gSq)*((1+g1)*x1 -     g2*y1)
	y1p=1./np.sqrt(1.-gSq)*(   -g2*x1 + (1-g1)*y1)
	pairs_prime=np.array([x1p,y1p])
    else:
	for ii in range(len(pairs)):
	    x1,y1=pairs[ii]
	    #	    x1p=((1+g1)*x1 -     g2*y1)
	    #	    y1p=(   -g2*x1 + (1-g1)*y1)
	    #	x1p=((1-kappa-gamma1)*x1 -           gamma2*y1)
	    #	y1p=(         -gamma2*x1 + (1-kappa+gamma1)*y1)
	    x1p=1./np.sqrt(1.-gSq)*((1+g1)*x1 -     g2*y1)
	    y1p=1./np.sqrt(1.-gSq)*(   -g2*x1 + (1-g1)*y1)
	    pairs_prime[ii]=np.array([x1p,y1p])

    return pairs_prime
 
def shearLines(lines,g1,g2):
# lines is either None or np.array([[x1,x2,y1,y2],...]) or np.array([x1,x2,y1,y2])
    lines_prime=lines.copy()
    if(lines.shape == (4,)): # only one line		
	x1,x2,y1,y2=lines
	x1p,y1p=shearPairs(np.array([x1,y1]),g1,g2)
	x2p,y2p=shearPairs(np.array([x2,y2]),g1,g2)
	lines_prime=np.array([x1p,x2p,y1p,y2p])
    else:
	for ii in range(len(lines)):
	    x1,x2,y1,y2=lines[ii]
	    x1p,y1p=shearPairs(np.array([x1,y1]),g1,g2)
	    x2p,y2p=shearPairs(np.array([x2,y2]),g1,g2)
	    lines_prime[ii]=np.array([x1p,x2p,y1p,y2p])
	
    return lines_prime

def getEllipseAxes(ellipse):
# returns endpoints of major and minor axis of an ellipse
    disk_r,gal_q,gal_beta=ellipse
    gal_beta_rad=np.deg2rad(gal_beta)
    xmaj=disk_r*np.cos(gal_beta_rad)
    ymaj=disk_r*np.sin(gal_beta_rad)
    xmin=disk_r*gal_q*np.cos(gal_beta_rad+np.pi/2)
    ymin=disk_r*gal_q*np.sin(gal_beta_rad+np.pi/2.)
    lines=np.array([[-xmaj,xmaj,-ymaj,ymaj],[-xmin,xmin,-ymin,ymin]])

    return lines

def showImage(profile,numFib,fibRad,filename=None,colorbar=True,cmap=matplotlib.cm.jet,plotScale="linear",trim=0,xlabel="x (arcsec)",ylabel="y (arcsec)",ellipse=None,lines=None,lcolors="white",lstyles="--"):
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
    
    if(lines is not None): # lines is either None or np.array([[x1,x2,y1,y2],...]) or np.array([x1,x2,y1,y2])
	    if(lines.shape == (4,)): # only one line		
		plt.plot(lines[0:2],lines[2:4],color=lcolors,lw=2,ls=lstyles)
            else:
		if(type(lcolors) is str):
		    lcolors=np.repeat(lcolors,len(lines))
		if(type(lstyles) is str):
		    lstyles=np.repeat(lstyles,len(lines))
		for line,color,style in zip(lines,lcolors,lstyles):
		    plt.plot(line[0:2],line[2:4],color=color,lw=2,ls=style)
		    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(trim>0): # trim all edges by this amount in arcsec
	plt.xlim((-halfWidth+trim,halfWidth-trim))
	plt.ylim((-halfWidth+trim,halfWidth-trim))

    if(filename):
	plt.savefig(filename)
    plt.show()

def contourPlot(xvals,yvals,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],xlabel=None,ylabel=None,xlim=None,ylim=None,filename=None,show=False):
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

    plt.contour(xx,yy,zz.T,levels=levels,colors=colors)

    if(xlabel is not None):
	plt.xlabel(xlabel)
    if(ylabel is not None):
	plt.ylabel(ylabel)
    if(xlim is not None):
	plt.xlim(xlim)
    if(ylim is not None):
	plt.ylim(ylim)

    if(filename):
	plt.savefig(filename)
    if(show):
	plt.show()
    
def contourPlotAll(chain,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],labels=None,figsize=(8,6),filename=None):
# make a grid of contour plots for each pair of parameters

    nPars=chain.shape[1]
    fig,axarr=plt.subplots(nPars,nPars,figsize=figsize)
    fig.subplots_adjust(hspace=0,wspace=0)

    if(labels is None):
	labels=np.repeat(None,nPars)

    for row in range(nPars):
	for col in range(nPars):
	    fig.sca(axarr[row,col])

	    if(row == nPars-1):
		xlabel=labels[col]
		plt.setp(axarr[row,col].get_xticklabels(), rotation="vertical", fontsize="xx-small")
            else:
		xlabel=None
		plt.setp(axarr[row,col].get_xticklabels(),visible=False)
	    if(col == 0):
		ylabel=labels[row]
		plt.setp(axarr[row,col].get_yticklabels(), fontsize="xx-small")
            else:
		ylabel=None
		plt.setp(axarr[row,col].get_yticklabels(),visible=False)

	    xarr=chain[:,col]
	    yarr=chain[:,row]
	    xlim=(np.min(xarr),np.max(xarr))
	    ylim=(np.min(yarr),np.max(yarr))
	    if(row == col):
		axarr[row,col].hist(xarr,bins=50,histtype="step")
		if(xlabel is not None):
		    axarr[row,col].set_xlabel(xlabel)
		if(ylabel is not None):
		    axarr[row,col].set_ylabel(ylabel)
		axarr[row,col].set_xlim(xlim)
		plt.setp(axarr[row,col].get_yticklabels(),visible=False)
            elif(col < row):
		contourPlot(xarr,yarr,smooth=smooth,percentiles=percentiles,colors=colors,xlabel=xlabel,ylabel=ylabel)
		xlim=(np.min(xarr),np.max(xarr))
		ylim=(np.min(yarr),np.max(yarr))
		axarr[row,col].set_xlim(xlim)
		axarr[row,col].set_ylim(ylim)
            else:
		axarr[row,col].axis("off")

    fig.subplots_adjust(bottom=0.15)
    if(filename):
	fig.savefig(filename)
    fig.show()


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

def makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,g1,g2):
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
    gal.applyShear(g1=g1,g2=g2)
    galX=gal
    galK=gal
    fluxVMap.applyShear(g1=g1,g2=g2)
    vmap.applyShear(g1=g1,g2=g2)

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

def vmapObs(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,g1,g2,pixScale,fibRad,numFib,showPlot=False):
# get flux-weighted fiber-averaged velocities
	
    imgSizePix=int(10.*fibRad/pixScale)
    vmap,fluxVMap,gal,galK=makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,g1,g2)

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

def lnProbVMapModel(pars, xobs, yobs, yerr, ellobs, ellerr, priorFuncs, fixed):
# pars are the free parameters to be fit (some or all of [PA, b/a, vmax, g1, g2])
# xobs are the fiber position angles for velocity measurements
# yobs are the measured velocities
# yerr are the uncertainties in measured velocities
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

    vmodel,ellmodel=vmapModel(fullPars,xobs)

    if(xobs is None): # use only imaging data
	model=ellmodel
	data=ellobs
	error=ellerr
    elif(ellobs is None): # use only velocity data
	model=vmodel
	data=yobs
	error=yerr
    else: # use both imaging and velocity data
	model=np.concatenate(vmodel,ellmodel)
	data=np.concatenate(yobs,ellobs)
	error=np.concatenate(yerr,ellerr)
	

    chisq_like=np.sum(((model-data)/error)**2)

    return -0.5*(chisq_like+chisq_prior)

def vmapModel(pars, fiberAngles):
# evaluate velocity field at azimuthal angles around center, called by curve_fit
# pars: PA in deg., gal_q, vmax, g1, g2 [PA, gal_q, and vmax are the *unsheared* parameters]
# fiberAngles are the N azimuthal angles (in radians) at which the *sheared* (observed) field is sampled
# returns (vmodel, ellmodel), where vmodel is an N array of fiber velocities, ellmodel is the sheared (gal_beta,gal_q)

    gal_beta,gal_q,vmax,g1,g2=pars

    # compute spectroscopic observable
    if(fiberAngles is not None):
	fibRad=1.
	rad=2.*fibRad # assume the fibers sample the v field at their center

	# Cartesian coords
	xobs=rad*np.cos(fiberAngles)
	yobs=rad*np.sin(fiberAngles)    

	# convert coords to source plane
	pairs=shearPairs(np.array(zip(xobs,yobs)),-g1,-g2)
	xx=pairs[:,0]
	yy=pairs[:,1]

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
	nSamp=fiberAngles.size
	vmodel=np.zeros(nSamp)
	for ii in range(nSamp):
	    # coordinates in the plane of the galaxy
	    radvec=np.array([xp[ii],yp[ii],yp[ii]*tani])
	    vmodel[ii]=getOmega(np.linalg.norm(radvec),rotCurvePars,option='flat') * sini * np.dot(radvec,kvec)
    else:
	vmodel=None

    # compute imaging observable
    disk_r=1. # we're not modeling sizes now
    ellipse=(disk_r,gal_beta,gal_q) # unsheared ellipse
    disk_r_prime,gal_beta_prime,gal_q_prime=shearEllipse(ellipse,g1,g2)
    ellmodel=np.array([gal_beta_prime,gal_q_prime]) # model sheared ellipse observables

    return (vmodel,ellmodel)

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
	    guessScale=np.delete(guessScale,ii)
	    priorFuncs=np.delete(priorFuncs,ii)

    return (priorFuncs,fixed,guess,guessScale)

def vmapFit(vfibFlux,sigma,imObs,imErr,priors,addNoise=True,showPlot=False):
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
    ellObs=imObs.copy()
    ellErr=imErr.copy()

    if(addNoise): # useful when simulating many realizations to project parameter constraints
	specNoise=np.random.randn(numFib)*sigma
	vel+=specNoise
	imNoise=np.random.randn(ellObs.size)*ellErr
	ellObs+=imNoise

    # SETUP PARS and PRIORS
    priorFuncs,fixed,guess,guessScale = interpretPriors(priors)
    nPars=len(guess)


    # RUN MCMC
    nWalkers=500
    walkerStart=np.array([np.random.randn(nWalkers)*guessScale[ii]+guess[ii] for ii in xrange(nPars)]).T
    sampler=emcee.EnsembleSampler(nWalkers,nPars,lnProbVMapModel,args=[ang, vel, velErr, ellObs, ellErr, priorFuncs, fixed])

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

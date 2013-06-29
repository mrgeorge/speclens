#! env python

import galsim
import scipy.integrate
import scipy.signal
import scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') # must appear before importing pyplot to get plots w/o GUI
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches
import emcee
import fitsio

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
plt.rc('text', usetex=True)
plt.rc('axes',linewidth=1.5)

pixScale=0.1
imgSizePix=100

def getFiberPos(numFib,fibRad,fibConfig,fibPA=None):
# returns fiber center positions relative to origin in same units as fibRad (arcsec)
# also returns fibShape (circle or square)
# for configurations with square fibers, PA sets position angle (e.g. slit, IFU grid)
# for circular fibers, fibRad=fiber radius. for square fibers, fibRad=edge length

    pos=np.zeros((2,numFib))

    if(fibConfig=="hex"):
        fibShape="circle"
        pos[:,0]=np.array([0.,0.])
        theta=np.linspace(0,2*np.pi,num=numFib-1,endpoint=False)
        rad=2.*fibRad
        pos[0,1:]=rad*np.cos(theta)
        pos[1,1:]=rad*np.sin(theta)
    elif(fibConfig=="hexNoCen"):
        fibShape="circle"
        theta=np.linspace(0,2*np.pi,num=numFib,endpoint=False)
        rad=2.*fibRad
        pos[0,:]=rad*np.cos(theta)
        pos[1,:]=rad*np.sin(theta)
    elif(fibConfig=="slit"):
        fibShape="square"
        slitX=np.linspace(-1,1,num=numFib)*0.5*fibRad*(numFib-1)
        pos[0,:]=slitX*np.cos(np.deg2rad(fibPA))
        pos[1,:]=slitX*np.sin(np.deg2rad(fibPA))
    elif(fibConfig=="ifu"):
        fibShape="square"
        numSide=np.sqrt(numFib)
        if(np.int(numSide) != numSide):
            print "Error: ifu config needs a square number of fibers"
        else:
            ifuX=np.linspace(-1,1,num=numSide)*0.5*fibRad*(numSide-1)
            xx,yy=np.meshgrid(ifuX,ifuX)
            xx=xx.flatten()
            yy=yy.flatten()
            PArad=np.deg2rad(fibPA)
            pos[0,:]=xx*np.cos(PArad)-yy*np.sin(PArad)
            pos[1,:]=xx*np.sin(PArad)+yy*np.cos(PArad)
    else:
        # TO DO - add other configs - e.g. circle, extend hex for MaNGA style
        pass 

    return (pos,fibShape)


# These functions take a galsim object <image> and integrate over the area of a fiber
def radIntegrand(rad,theta,fiberPos,image):
    pos=galsim.PositionD(x=fiberPos[0]+rad*np.cos(theta),y=fiberPos[1]+rad*np.sin(theta))
    return rad*image.xValue(pos)
def thetaIntegrand(theta,fiberPos,image,fibRad,tol):
    return scipy.integrate.quad(radIntegrand,0,fibRad,args=(theta,fiberPos,image),epsabs=tol,epsrel=tol)[0]
def getFiberFlux(fibID,numFib,fibRad,fibConfig,image,tol=1.e-4):
    fiberPos=getFiberPos(numFib,fibRad,fibConfig)[:,fibID]
    return scipy.integrate.quad(thetaIntegrand,0,2.*np.pi, args=(fiberPos,image,fibRad,tol), epsabs=tol, epsrel=tol)

def getFiberFluxes(xobs,yobs,fibRad,fibConvolve,image):
# convolve image with fiber area and return fiber flux
# if fibConvolve is False, just sample the image at central position without convolving
 
    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)

    coordsPix=np.array([xobs,yobs])/pixScale + 0.5*imgSizePix # converted to pixels

    if(fibConvolve):
        scale_radius=10.*fibRad
        beta=0.
        fiber=galsim.Moffat(beta=beta,scale_radius=scale_radius,trunc=fibRad) # a kludgy way to get a circular tophat

        fibImage=galsim.Convolve([fiber,image])
        fibImageArr=fibImage.draw(image=imgFrame,dx=pixScale).array.copy()
    else:
        fibImageArr=image.draw(image=imgFrame,dx=pixScale).array.copy()

    return scipy.ndimage.map_coordinates(fibImageArr.T,coordsPix)
    
def shearEllipse(ellipse,g1,g2):
# Following Supri & Harari 1999
    disk_r,gal_q,gal_beta=ellipse
    gamma=np.sqrt(g1**2 + g2**2)
    epsilon=(1-gal_q**2)/(1+gal_q**2)
    psi=np.deg2rad(gal_beta)
    phi=0.5*np.arctan2(g2,g1)

    dpsi=0.5*np.arctan2(2*(gamma/epsilon)*np.sin(2.*(phi-psi)), 1.+2*(gamma/epsilon)*np.cos(2*(phi-psi)))
    assert((epsilon + 2.*gamma*np.cos(2.*(phi-psi)))**2 + 4*(gamma*np.sin(2*(phi-psi)))**2 >= 0)
    epsilon_prime=np.sqrt((epsilon + 2.*gamma*np.cos(2.*(phi-psi)))**2 + 4*(gamma*np.sin(2*(phi-psi)))**2) / (1.+2.*epsilon*gamma*np.cos(2.*(phi-psi)))

    disk_r_prime=disk_r*(1+gamma)
    #    assert(epsilon_prime < 1.1)
    if(epsilon_prime>1):
	epsilon_prime=1.
    assert(epsilon_prime <= 1.)
    gal_q_prime=np.sqrt((1.-epsilon_prime)/(1.+epsilon_prime))
    gal_beta_prime=np.rad2deg(psi+dpsi)

    return (disk_r_prime,gal_q_prime,gal_beta_prime)

def shearPairs(pairs,g1,g2):
    pairs_prime=pairs.copy()
    gSq=g1**2+g2**2
    assert(gSq < 1)
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

def showImage(profile,xfib,yfib,fibRad,fibShape="circle",fibPA=None,filename=None,colorbar=True,colorbarLabel=r"v$_{LOS}$ (km/s)",cmap=matplotlib.cm.jet,plotScale="linear",trim=0,xlabel="x (arcsec)",ylabel="y (arcsec)",ellipse=None,lines=None,lcolors="white",lstyles="--",showPlot=False):
# Plot image given by galsim object <profile> with fiber pattern overlaid

    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)
    img=profile.draw(image=imgFrame,dx=pixScale)
    halfWidth=0.5*imgSizePix*pixScale # arcsec
    #    img.setCenter(0,0)

    if(plotScale=="linear"):
	plotArr=img.array
    elif(plotScale=="log"):
	plotArr=np.log(img.array)

    plt.imshow(plotArr,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest',cmap=cmap)

    if(xfib is not None):
        numFib=xfib.size
        for pos in zip(xfib,yfib):
            if(fibShape=="circle"):
                circ=plt.Circle((pos[0],pos[1]),radius=fibRad,fill=False,color='white',lw=2)
                ax=plt.gca()
                ax.add_patch(circ)
            elif(fibShape=="square"):
                PArad=np.deg2rad(fibPA)
                corners=np.zeros((4,2))
                xx=np.array([-1,1,1,-1])
                yy=np.array([-1,-1,1,1])
                corners[:,0]=(xx*np.cos(PArad)-yy*np.sin(PArad))*0.5*fibRad+pos[0]
                corners[:,1]=(xx*np.sin(PArad)+yy*np.cos(PArad))*0.5*fibRad+pos[1]
                sq=plt.Polygon(corners,fill=False,color='white',lw=2)
                ax=plt.gca()
                ax.add_patch(sq)

    if(colorbar):
	cbar=plt.colorbar()
	if(colorbarLabel is not None):
	    cbar.set_label(colorbarLabel)

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
    if(showPlot):
	plt.show()

def contourPlot(xvals,yvals,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],xlabel=None,ylabel=None,xlim=None,ylim=None,filename=None,showPlot=False):
# make a 2d contour plot of parameter posteriors

    n2dbins=300

    # if it's a single ndarray wrapped in a list, convert to ndarray to use full color list
    if((type(xvals) is list) & (len(xvals) ==1)):
	xvals=xvals[0]
	yvals=yvals[0]
	
    if(type(xvals) is list):
	for ii in range(len(xvals)):
	    zz,xx,yy=np.histogram2d(xvals[ii],yvals[ii],bins=n2dbins)
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

	    plt.contour(xx,yy,zz.T,levels=levels,colors=colors[ii])
    else: #we just have single ndarrays for xvals and yvals
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
    if(showPlot):
	plt.show()
    
def contourPlotAll(chains,inputPars=None,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],labels=None,figsize=(8,6),filename=None,showPlot=False):
# make a grid of contour plots for each pair of parameters
# chain is actually a list of 1 or more chains from emcee sampler

    nChains=len(chains)
    nPars=chains[0].shape[1]

    fig,axarr=plt.subplots(nPars,nPars,figsize=figsize)
    fig.subplots_adjust(hspace=0,wspace=0)

    if(labels is None):
	labels=np.repeat(None,nPars)

    # find max and min for all pars across chains
    limArr=np.tile((np.Inf,-np.Inf),nPars).reshape(nPars,2)
    for ch in chains:
	for par in range(nPars):
	    lo,hi=np.min(ch[:,par]), np.max(ch[:,par])
	    if(lo < limArr[par,0]):
		limArr[par,0]=lo.copy()
	    if(hi > limArr[par,1]):
		limArr[par,1]=hi.copy()

    # handle colors
    if(len(colors) == len(chains)):
	histColors=colors
	contourColors=colors
    if((nChains == 1) & (len(colors) == len(percentiles))):
	histColors=colors[0]
	contourColors=colors
	    
    # fill plot panels
    for row in range(nPars):
	for col in range(nPars):
	    fig.sca(axarr[row,col])

	    # setup axis labels
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
		    
	    xarrs=[chain[:,col] for chain in chains]
	    yarrs=[chain[:,row] for chain in chains]
	    xlim=limArr[col]
	    ylim=limArr[row]
	    if(row == col):
		axarr[row,col].hist(xarrs,bins=50,range=xlim,histtype="step",color=histColors)
		if(xlabel is not None):
		    axarr[row,col].set_xlabel(xlabel)
		if(ylabel is not None):
		    axarr[row,col].set_ylabel(ylabel)
		axarr[row,col].set_xlim(xlim)
		plt.setp(axarr[row,col].get_yticklabels(),visible=False)
                if(inputPars is not None):
                    # add vertical lines marking the input value
                    plt.plot(np.repeat(inputPars[col],2),np.array(plt.gca().get_ylim()),color="yellow",ls="--")
	    elif(col < row):
		contourPlot(xarrs,yarrs,smooth=smooth,percentiles=percentiles,colors=contourColors,xlabel=xlabel,ylabel=ylabel)
		axarr[row,col].set_xlim(xlim)
		axarr[row,col].set_ylim(ylim)
                if(inputPars is not None):
                    # add lines marking the input values
                    plt.plot(np.repeat(inputPars[col],2),ylim,color="yellow",ls="--")
                    plt.plot(xlim,np.repeat(inputPars[row],2),color="yellow",ls="--")
	    else:
		axarr[row,col].axis("off")

    fig.subplots_adjust(bottom=0.15)
    if(filename):
	fig.savefig(filename)
    if(showPlot):
	fig.show()

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

def makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,g1,g2):
# Construct galsim objects for galaxy (image), velocity map, and flux-weighted velocity map
# with optional lensing shear and PSF convolution
	
    # Define the galaxy velocity map
    if(0 < bulge_frac < 1):
        bulge=galsim.Sersic(bulge_n, half_light_radius=bulge_r)
        disk=galsim.Sersic(disk_n, half_light_radius=disk_r)

        gal=bulge_frac * bulge + (1.-bulge_frac) * disk
    elif(bulge_frac == 0):
        gal=galsim.Sersic(disk_n, half_light_radius=disk_r)
    elif(bulge_frac == 1):
        gal=galsim.Sersic(bulge_n, half_light_radius=bulge_r)

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

    # Fill velocity map array
    xx, yy=np.meshgrid(range(galImg.xmin-1,galImg.xmax),range(galImg.ymin-1,galImg.ymax))
    xp=(xx-xCen)*np.cos(gal_beta_rad)+(yy-yCen)*np.sin(gal_beta_rad)
    yp=-(xx-xCen)*np.sin(gal_beta_rad)+(yy-yCen)*np.cos(gal_beta_rad)
    radNorm=np.sqrt(xp**2 + yp**2 * (1.+tani**2))
    vmapArr=getOmega(radNorm,rotCurvePars,option=rotCurveOpt) * sini * xp
    vmapArr[0,:]=0 # galsim.InterpolatedImage has a problem with this array if I don't do something weird at the edge like this

    # Weight velocity map by galaxy flux and make galsim object
    fluxVMapArr=vmapArr*imgArr

    sumFVM=np.sum(fluxVMapArr)
    if(np.abs(sumFVM) < 0.01):
        print "experimental renorm"
        fluxVMapArr/=sumFVM
        gal.scaleFlux(1./sumFVM)

    fluxVMapImg=galsim.ImageViewD(fluxVMapArr,scale=pixScale)
    fluxVMap=galsim.InterpolatedImage(fluxVMapImg,pad_factor=6.)
    vmap=galsim.InterpolatedImage(galsim.ImageViewD(vmapArr,scale=pixScale)) # not flux-weighted

    # Apply lensing shear to galaxy and velocity maps
    if((g1 != 0.) | (g2 != 0.)):
        gal.applyShear(g1=g1,g2=g2)
        fluxVMap.applyShear(g1=g1,g2=g2)
        vmap.applyShear(g1=g1,g2=g2)

    # Convolve velocity map and galaxy with PSF
    if(atmos_fwhm > 0):
        # Define atmospheric PSF
        #    atmos=galsim.Kolmogorov(fwhm=atmos_fwhm)
	atmos=galsim.Gaussian(fwhm=atmos_fwhm)
        fluxVMap=galsim.Convolve([atmos, fluxVMap])
        gal=galsim.Convolve([atmos, gal])

    return (vmap,fluxVMap,gal)

def makeGalVMap2(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,rotCurveOpt,rotCurvePars,g1,g2):
    # Define the galaxy velocity map
    if(0 < bulge_frac < 1):
        bulge=galsim.Sersic(bulge_n, half_light_radius=bulge_r)
        disk=galsim.Sersic(disk_n, half_light_radius=disk_r)

        gal=bulge_frac * bulge + (1.-bulge_frac) * disk
    elif(bulge_frac == 0):
        gal=galsim.Sersic(disk_n, half_light_radius=disk_r)
    elif(bulge_frac == 1):
        gal=galsim.Sersic(bulge_n, half_light_radius=bulge_r)

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

    # Fill velocity map array
    xx, yy=np.meshgrid(range(galImg.xmin-1,galImg.xmax),range(galImg.ymin-1,galImg.ymax))
    xp=(xx-xCen)*np.cos(gal_beta_rad)+(yy-yCen)*np.sin(gal_beta_rad)
    yp=-(xx-xCen)*np.sin(gal_beta_rad)+(yy-yCen)*np.cos(gal_beta_rad)
    radNorm=np.sqrt(xp**2 + yp**2 * (1.+tani**2))
    vmapArr=getOmega(radNorm,rotCurvePars,option=rotCurveOpt) * sini * xp

    # Weight velocity map by galaxy flux
    fluxVMapArr=vmapArr*imgArr

    # Apply lensing shear to galaxy and velocity maps
    if((g1 != 0.) | (g2 != 0.)):
        shear=np.array([[1-g1,-g2],[-g2,1+g1]])/np.sqrt(1.-g1**2-g2**2)
        xs=shear[0,0]*(xx-xCen) + shear[0,1]*(yy-yCen) + xCen
        ys=shear[1,0]*(xx-xCen) + shear[1,1]*(yy-yCen) + yCen
        fluxVMapArr=scipy.ndimage.map_coordinates(fluxVMapArr.T,(xs,ys))
        imgArr=scipy.ndimage.map_coordinates(imgArr.T,(xs,ys))

    return (fluxVMapArr,imgArr)

def makeConvolutionKernel(xobs,yobs,atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA):
# construct the fiber x PSF convolution kernel
# this saves time since you only need to calculate it once and can multiply it by the flux map,
#    rather than convolving each model galaxy and sampling at a position
# returns kernel, an ndarray of size [numFib, imgSizePix, imgSizePix]

    numFib=xobs.size
    half=imgSizePix/2
    xx,yy=np.meshgrid((np.arange(imgSizePix)-half)*pixScale,(np.arange(imgSizePix)-half)*pixScale)
    if(atmos_fwhm > 0):
        atmos_sigma=atmos_fwhm/(2.*np.sqrt(2.*np.log(2.)))
        if(fibConvolve): # PSF and Fiber convolution
            psfArr=np.exp(-(xx**2 + yy**2)/(2.*atmos_sigma**2))
            fibArrs=np.zeros((numFib,imgSizePix,imgSizePix))
            if(fibShape=="circle"):
                sel=np.array([((xx-pos[0])**2 + (yy-pos[1])**2 < fibRad**2) for pos in zip(xobs,yobs)])
            elif(fibShape=="square"):
                PArad=np.deg2rad(fibPA)
                sel=np.array([((np.abs((xx-pos[0])*np.cos(PArad) - (yy-pos[1])*np.sin(PArad)) < 0.5*fibRad) & (np.abs((xx-pos[0])*np.sin(PArad) + (yy-pos[1])*np.cos(PArad)) < 0.5*fibRad)) for pos in zip(xobs,yobs)])
            fibArrs[sel]=1.
            kernel=np.array([scipy.signal.fftconvolve(psfArr,fibArrs[ii],mode="same") for ii in range(numFib)])
        else:
            # this is basically the psf convolved with a delta function at the center of each fiber
            kernel=np.array([np.exp(-((xx-pos[0])**2 + (yy-pos[1])**2)/(2.*atmos_sigma**2)) for pos in zip(xobs,yobs)])
    else:
        # Fiber only
        kernel=np.zeros((numFib,imgSizePix,imgSizePix))
        if(fibShape=="circle"):
            sel=np.array([((xx-pos[0])**2 + (yy-pos[1])**2 < fibRad**2) for pos in zip(xobs,yobs)])
        elif(fibShape=="square"):
            PArad=np.deg2rad(fibPA)
            sel=np.array([((np.abs((xx-pos[0])*np.cos(PArad) - (yy-pos[1])*np.sin(PArad)) < 0.5*fibRad) & (np.abs((xx-pos[0])*np.sin(PArad) + (yy-pos[1])*np.cos(PArad)) < 0.5*fibRad)) for pos in zip(xobs,yobs)])
        kernel[sel]=1.
        
    return kernel

def vmapObs(pars,xobs,yobs,disk_r,showPlot=False,convOpt="galsim",atmos_fwhm=None,fibRad=None,fibConvolve=False,kernel=None):
# get flux-weighted fiber-averaged velocities
#for convOpt=galsim, need to specify atmos_fwhm,fibRad,fibConvolve
#for convOpt=pixel, need to specify kernel
	
    gal_beta,gal_q,vmax,g1,g2=pars
    
    numFib=xobs.size
    bulge_n=4.
    bulge_r=1.
    disk_n=1.
    bulge_frac=0.
    gal_flux=1.e6
    rotCurveOpt="flat"
    rotCurvePars=np.array([vmax])

    if(convOpt=="galsim"):
        vmap,fluxVMap,gal=makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,g1,g2)

        if(showPlot):
            showImage(gal,xobs,yobs,fibRad,showPlot=True)
            showImage(vmap,xobs,yobs,fibRad,showPlot=True)
            showImage(fluxVMap,xobs,yobs,fibRad,showPlot=True)

        # Get the flux in each fiber
        galFibFlux=getFiberFluxes(xobs,yobs,fibRad,fibConvolve,gal)
        vmapFibFlux=getFiberFluxes(xobs,yobs,fibRad,fibConvolve,fluxVMap)

    elif(convOpt=="pixel"):
        fluxVMapArr,imgArr=makeGalVMap2(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,rotCurveOpt,rotCurvePars,g1,g2)
        if(showPlot):
            halfWidth=0.5*imgSizePix*pixScale # arcsec
            plt.clf()
            plt.imshow(imgArr,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest',cmap=matplotlib.cm.jet)
            plt.show()
            plt.imshow(fluxVMapArr,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest',cmap=matplotlib.cm.jet)
            plt.show()

        vmapFibFlux=np.array([np.sum(kernel[ii]*fluxVMapArr) for ii in range(numFib)])
        galFibFlux=np.array([np.sum(kernel[ii]*imgArr) for ii in range(numFib)])

    return vmapFibFlux/galFibFlux

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
	    model=ellModel(fullPars)
	    data=ellobs
	    error=ellerr
	elif((xobs is not None) & (ellobs is None)): # use only velocity data
            if(convOpt is not None):
                model=vmapObs(fullPars,xobs,yobs,disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
            else: # this is faster if we don't need to convolve with psf or fiber
                model=vmapModel(fullPars,xobs,yobs)
	    data=vobs
	    error=verr
        elif((xobs is not None) & (ellobs is not None)): # use both imaging and velocity data
            if(convOpt is not None):
                vmodel=vmapObs(fullPars,xobs,yobs,disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
            else: # this is faster if we don't need to convolve with psf or fiber
                vmodel=vmapModel(fullPars,xobs,yobs)
	    model=np.concatenate([vmodel,ellModel(fullPars)])
	    data=np.concatenate([vobs,ellobs])
	    error=np.concatenate([verr,ellerr])

	chisq_like=np.sum(((model-data)/error)**2)

    return -0.5*(chisq_like+chisq_prior)

def vmapModel(pars, xobs, yobs):
# evaluate velocity field at azimuthal angles around center, called by curve_fit
# pars: PA in deg., gal_q, vmax, g1, g2 [PA, gal_q, and vmax are the *unsheared* parameters]
# xobs, yobs are the N positions (in arcsec) relative to the center at which
#    the *sheared* (observed) field is sampled
# returns (vmodel, ellmodel), where vmodel is an N array of fiber velocities,
#    ellmodel is the array of sheared imaging observables (gal_beta,gal_q)

    gal_beta,gal_q,vmax,g1,g2=pars

    # compute spectroscopic observable
    if(xobs is not None):
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
	nSamp=xobs.size
	vmodel=np.zeros(nSamp)
        radNorm=np.sqrt(xp**2 + yp**2 * (1.+tani**2))
        vmodel=getOmega(radNorm,rotCurvePars,option="flat") * sini * xp
    else:
	vmodel=None

    return vmodel

def ellModel(pars):
    # compute imaging observable
    gal_beta,gal_q,vmax,g1,g2=pars

    disk_r=1. # we're not modeling sizes now
    ellipse=(disk_r,gal_q,gal_beta) # unsheared ellipse
    disk_r_prime,gal_q_prime,gal_beta_prime=shearEllipse(ellipse,g1,g2)
    ellmodel=np.array([gal_beta_prime,gal_q_prime]) # model sheared ellipse observables

    return ellmodel

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
    for ii in xrange(nPars):
	if(fixed[ii] is not None):
	    guess=np.delete(guess,ii)
	    guessScale=np.delete(guessScale,ii)
	    priorFuncs=np.delete(priorFuncs,ii)

    return (priorFuncs,fixed,guess,guessScale)

def generateEnsemble(nGal,priors,shearOpt="PS",seed=None):
# generate a set of galaxies with intrinsic shapes following the prior distribution
# (priors follow same convention as used by interpretPriors but must not be None)
# and generate shear parameters following an approach set by shearOpt
#     shearOpt="PS", "NFW", None - shears can also be defined using uniform or gaussian priors

    nPars=len(priors)
    pars=np.zeros((nGal,nPars))
    np.random.seed(seed)
    for ii in xrange(nPars):
        prior=priors[ii]
        # note: each of the assignments below needs to *copy* aspects of prior to avoid pointer overwriting
        if((type(prior) is int) | (type(prior) is float)): # fixed
            fixVal=np.copy(prior)
            pars[:,ii]=fixVal
        elif(type(prior) is list): # flat prior
            priorRange=np.copy(prior)
            pars[:,ii]=np.random.rand(nGal)*(priorRange[1]-priorRange[0]) + priorRange[0]
        elif(type(prior) is tuple): # gaussian
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
        pos,fibShape=getFiberPos(numFib,fibRad,fibConfig,fibPA=fibPA)
        xobs,yobs=pos
	vel=np.array(vobs).copy()
	velErr=np.repeat(sigma,numFib)

        # SETUP CONVOLUTION KERNEL
        if(convOpt=="pixel"):
            kernel=makeConvolutionKernel(xobs,yobs,atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA)
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

def parsToRec(pars,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    dtype=[(label,float) for label in labels]
    rec=np.recarray(len(pars),dtype=dtype)
    for ii in range(len(labels)):
        rec[labels[ii]]=pars[:,ii]
    return rec

def chainToRec(chain,lnprob,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    nGal=chain.shape[0]
    nPars=chain.shape[1]
    arr=np.zeros((nGal,nPars+1))
    arr[:,:-1]=chain
    arr[:,-1]=lnprob
    labels=np.append(labels,"lnprob")
    rec=parsToRec(arr,labels=labels)
    return rec

def recToPars(rec,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    recLabels=rec.dtype.fields.keys() # note, this list is unordered since rec is a dict, so we need to use parsLabels (which should be sorted to match the order of columns in pars array)
    pars=np.zeros((len(rec),len(labels)))
    for ii in range(len(labels)):
        pars[:,ii]=rec[labels[ii]]
    return pars

def obsToRec(xvals,yvals,vvals,ellObs):
    dtype=[("xvals",(xvals.dtype.type,xvals.shape)),("yvals",(yvals.dtype.type,yvals.shape)),("vvals",(vvals.dtype.type,vvals.shape)),("ellObs",(ellObs.dtype.type,ellObs.shape))]
    rec=np.recarray(1,dtype=dtype)
    rec["xvals"]=xvals
    rec["yvals"]=yvals
    rec["vvals"]=vvals
    rec["ellObs"]=ellObs
    return rec

def recToObs(rec):
    xvals=rec["xvals"].squeeze()
    yvals=rec["yvals"].squeeze()
    vvals=rec["vvals"].squeeze()
    ellObs=rec["ellObs"].squeeze()
    return (xvals,yvals,vvals,ellObs)

def writeRec(rec,filename,clobber=True):
    fitsio.write(filename,rec,clobber=clobber)

def readRec(filename):
    rec=fitsio.read(filename)
    return rec

if __name__ == "__main__":
    print "use one of the functions - no main written"
    

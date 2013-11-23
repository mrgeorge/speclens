#! env python

import galsim
import scipy.integrate
import scipy.signal
import scipy.ndimage
import numpy as np
import plot

pixScale=0.1
imgSizePix=100

def getFiberPos(numFib,fibRad,fibConfig,fibPA=None):
    """Return fiber center positions and fiber shape given config string.

    Inputs:
        numFib - number of fibers
        fibRad - fiber radius in arcsec for circular fibers
                 or edge length for square fibers
        fibConfig - configuration string (hex|hexNoCen|slit|ifu|triNoCen)
        fibPA - position angle for configs with square fibers (default None)
    Returns:
        (pos, fibShape)
            pos - 2 x numFib ndarray with fiber centers in arcsecs from origin
            fibShape - string "circle" or "square"
    """        

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
    elif(fibConfig=="triNoCen"):
        fibShape="circle"
        theta=np.linspace(0,2*np.pi,num=numFib,endpoint=False)
        rad=fibRad
        pos[0,:]=rad*np.cos(theta)
        pos[1,:]=rad*np.sin(theta)
    else:
        # TO DO - add other configs - e.g. circle, extend hex for MaNGA style
        pass 

    return (pos,fibShape)


# These functions take a galsim object <image> and integrate over the area of a fiber (currently obsolete)
def radIntegrand(rad,theta,fiberPos,image):
    """Evaluate radial integrand used by galsim version of getFiberFlux"""
    pos=galsim.PositionD(x=fiberPos[0]+rad*np.cos(theta),y=fiberPos[1]+rad*np.sin(theta))
    return rad*image.xValue(pos)
def thetaIntegrand(theta,fiberPos,image,fibRad,tol):
    """Evaluate theta integrand used by galsim version of getFiberFlux"""
    return scipy.integrate.quad(radIntegrand,0,fibRad,args=(theta,fiberPos,image),epsabs=tol,epsrel=tol)[0]
def getFiberFlux(fibID,numFib,fibRad,fibConfig,image,tol=1.e-4):
    """Integrate image flux over fiber area, used by galsim version of getFiberFlux"""
    fiberPos=getFiberPos(numFib,fibRad,fibConfig)[:,fibID]
    return scipy.integrate.quad(thetaIntegrand,0,2.*np.pi, args=(fiberPos,image,fibRad,tol), epsabs=tol, epsrel=tol)


def getFiberFluxes(xobs,yobs,fibRad,fibConvolve,image):
    """Convolve image with fiber area and return fiber flux.

    Inputs:
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        fibRad - fiber radius in arcsecs
        fibConvolve - if False, just sample the image at central position 
                      without convolving, else convolve
        image - galsim object
    Returns:
        fiberFluxes - ndarray of values sampled at fiber centers in convolved image.

    Note: getFiberFluxes is meant as an update to getFiberFlux since 
          sampling the image after galsim's Convolve with scipy's map_coordinates
          should be faster than the real-space integral over each fiber.
          But, getFiberFluxes currently assumes circular fibers only, and
          is being superceded by makeConvolutionKernel with convOpt=pixel.
    """

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
    """Shear ellipse parameters following Supri & Harari 1999.

    Inputs:
        ellipse - (disk_r, gal_q, gal_beta) unsheared ellipse
        g1 - shear 1
        g2 - shear 2
    Returns:
        (disk_r_prime, gal_q_prime, gal_beta_prime) - sheared ellipse parameters

    Note: This is only used for visualization
    """
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
    """Shear coordinate pairs

    Inputs:
        pairs - list or ndarray of [(x1,y1), ...] coordinates
        g1 - shear 1
        g2 - shear 2
    Returns:
        pairs_prime - shear coordinates in same format as pairs
    Note: This shear matrix is different from the one in makeGalVMap2
          because images get flipped around when plotted
          This convention is meant for visualization
    """

    pairs_prime=pairs.copy()
    gSq=g1**2+g2**2
    assert(gSq < 1)
    if(pairs.shape == (2,)): # only one pair
        x1,y1=pairs
        x1p=1./np.sqrt(1.-gSq)*((1+g1)*x1 +     g2*y1)
        y1p=1./np.sqrt(1.-gSq)*(    g2*x1 + (1-g1)*y1)
        pairs_prime=np.array([x1p,y1p])
    else:
        for ii in range(len(pairs)):
            x1,y1=pairs[ii]
            x1p=1./np.sqrt(1.-gSq)*((1+g1)*x1 +     g2*y1)
            y1p=1./np.sqrt(1.-gSq)*(    g2*x1 + (1-g1)*y1)
            pairs_prime[ii]=np.array([x1p,y1p])

    return pairs_prime
 
def shearLines(lines,g1,g2):
    """Shear lines by calling shearPairs on each pair of coords

    Inputs:
        lines - ndarray([[x1,x2,y1,y2],...]) or ndarray([x1,x2,y1,y2])
        g1 - shear 1
        g2 - shear 2
    Returns:
        lines_prime - shear coordinates in same format as lines
    """
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
    """Return endpoints of major and minor axis of an ellipse

    Input:
        ellipse - (disk_r, gal_q, gal_beta)
    Returns:
        lines - ndarray([[xa1,xa2,ya1,ya2],[xb1,xb2,yb1,yb2]])
    """
    disk_r,gal_q,gal_beta=ellipse
    gal_beta_rad=np.deg2rad(gal_beta)
    xmaj=disk_r*np.cos(gal_beta_rad)
    ymaj=disk_r*np.sin(gal_beta_rad)
    xmin=disk_r*gal_q*np.cos(gal_beta_rad+np.pi/2)
    ymin=disk_r*gal_q*np.sin(gal_beta_rad+np.pi/2.)
    lines=np.array([[-xmaj,xmaj,-ymaj,ymaj],[-xmin,xmin,-ymin,ymin]])

    return lines


def getInclination(gal_q):
    """Return inclination in radians for a flat think disk

    See http://eo.ucar.edu/staff/dward/sao/spirals/methods.htm
    """
    return np.arccos(gal_q) # radians

def getOmega(rad,pars,option='flat'):
    """Return angular rotation rate, i.e. v(r)/r

    Inputs:
        rad - ndarray of radii at which to sample
        pars - float or array of option-dependent rotation curve parameters
        option - "flat", "solid", or "nfw" (default "flat")
    Returns:
        omega - ndarray same length as rad with rotation rates
    """

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
    """Construct galsim objects for image, velocity map, and flux-weighted velocity map.

    Inputs:
        bulge_n - bulge Sersic index
        bulge_r - bulge half-light radius
        disk_n - disk Sersic index
        disk_r - disk half-light radius
        bulge_frac - bulge fraction (0=pure disk, 1=pure bulge)
        gal_q - axis ratio
        gal_beta - position angle in degrees
        gal_flux - normalization of image flux
        atmos_fwhm - FWHM of gaussian PSF
        rotCurveOpt - option for getOmega ("flat", "solid", or "nfw")
        rotCurvePars - parameters for getOmega (depends on rotCurveOpt)
        g1 - shear 1
        g2 - shear 2
    Returns:
        (vmap,fluxVMap,gal) - tuple of galsim objects

    Note: See also makeGalVMap2, which uses pixel arrays instead of galsim objects
    """
    
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
    """Construct pixel arrays for image, velocity map, and flux-weighted velocity map.

    Inputs:
        bulge_n - bulge Sersic index
        bulge_r - bulge half-light radius
        disk_n - disk Sersic index
        disk_r - disk half-light radius
        bulge_frac - bulge fraction (0=pure disk, 1=pure bulge)
        gal_q - axis ratio
        gal_beta - position angle in degrees
        gal_flux - normalization of image flux
        rotCurveOpt - option for getOmega ("flat", "solid", or "nfw")
        rotCurvePars - parameters for getOmega (depends on rotCurveOpt)
        g1 - shear 1
        g2 - shear 2
    Returns:
        (vmapArr,fluxVMapArr,imgArr) - tuple of ndarray images

    Note: See also makeGalVMap, which uses galsim objects instead of pixel arrays
    """

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
        vmapArr=scipy.ndimage.map_coordinates(vmapArr.T,(xs,ys))
        fluxVMapArr=scipy.ndimage.map_coordinates(fluxVMapArr.T,(xs,ys))
        imgArr=scipy.ndimage.map_coordinates(imgArr.T,(xs,ys))

    return (vmapArr,fluxVMapArr,imgArr)

def makeConvolutionKernel(xobs,yobs,atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA):
    """Construct the fiber x PSF convolution kernel

    When using pixel arrays (instead of galsim objects),
    makeConvolutionKernel saves time since you only need to 
    calculate the kernel once and can multiply it by the flux map,
    rather than convolving each model galaxy and sampling at a position

    Inputs:
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        atmos_fwhm - FWHM of gaussian PSF
        fibRad - fiber radius in arcsecs
        fibConvolve - if False, just sample the image at central position 
                      without convolving, else convolve
        fibShape - string "circle" or "square"
        fibPA - position angle for configs with square fibers 
                (can be None for fibShape="circle")
    Returns:
        kernel - an ndarray of size [xobs.size, imgSizePix, imgSizePix]
    """
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
    """Get flux-weighted fiber-averaged velocities

    vmapObs computes fiber sampling in two ways, depending on convOpt
        for convOpt=galsim, need to specify atmos_fwhm,fibRad,fibConvolve
        for convOpt=pixel, need to specify kernel

    Inputs:
        pars - [gal_beta, gal_q, vmax, g1, g2] *unsheared* values
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        disk_r - disk half-light radius
        showPlot - bool for presenting plots (default False)
        convOpt - how to compute images and convolutions ("galsim" or "pixel")
        atmos_fwhm - FWHM of gaussian PSF (default None)
        fibRad - fiber radius in arcsecs (default None)
        fibConvolve - if False (default), just sample the image at central position 
                      without convolving, else convolve
        kernel - an ndarray of size [xobs.size, imgSizePix, imgSizePix]
                 from makeConvolutionKernel
    Returns:
        ndarray of flux-weighted fiber-averaged velocities

    Note: see vmapModel for faster vmap evaluation without PSF and fiber convolution
    """
    	
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
            plot.showImage(gal,xobs,yobs,fibRad,showPlot=True)
            plot.showImage(vmap,xobs,yobs,fibRad,showPlot=True)
            plot.showImage(fluxVMap,xobs,yobs,fibRad,showPlot=True)

        # Get the flux in each fiber
        galFibFlux=getFiberFluxes(xobs,yobs,fibRad,fibConvolve,gal)
        vmapFibFlux=getFiberFluxes(xobs,yobs,fibRad,fibConvolve,fluxVMap)

    elif(convOpt=="pixel"):
        fluxVMapArr,imgArr=makeGalVMap2(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,rotCurveOpt,rotCurvePars,g1,g2)
        if(showPlot):
            plot.showArr(imgArr)
            plot.showArr(fluxVMapArr)
        vmapFibFlux=np.array([np.sum(kernel[ii]*fluxVMapArr) for ii in range(numFib)])
        galFibFlux=np.array([np.sum(kernel[ii]*imgArr) for ii in range(numFib)])

    return vmapFibFlux/galFibFlux

def vmapModel(pars, xobs, yobs):
    """Evaluate model velocity field at given coordinates

    Inputs:
        pars - [gal_beta, gal_q, vmax, g1, g2] *unsheared* values
        xobs, yobs - the N positions (in arcsec) relative to the center at which
                     the *sheared* (observed) field is sampled
    Returns:
        (vmodel, ellmodel)
            vmodel is an N array of fiber velocities
            ellmodel is the array of sheared imaging observables (gal_beta,gal_q)

    Note: vmapModel is like vmapObs without PSF and fiber convolution
    """
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
    """Compute sheared ellipse pars for a model galaxy

    Inputs:
        pars - [gal_beta, gal_q, vmax, g1, g2] *unsheared* values
    Returns:
        ndarray([gal_beta, gal_q]) *sheared* values
    """

    gal_beta,gal_q,vmax,g1,g2=pars

    disk_r=1. # we're not modeling sizes now
    ellipse=(disk_r,gal_q,gal_beta) # unsheared ellipse
    disk_r_prime,gal_q_prime,gal_beta_prime=shearEllipse(ellipse,g1,g2)
    ellmodel=np.array([gal_beta_prime,gal_q_prime]) # model sheared ellipse observables

    return ellmodel

def generateEnsemble(nGal,priors,shearOpt=None,seed=None):
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
    
if __name__ == "__main__":
    print "use one of the functions - no main written"
    

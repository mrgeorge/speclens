#! env python

import galsim
import scipy.integrate
import scipy.signal
import scipy.ndimage
import numpy as np
import plot

pixScale=0.1
imgSizePix=100

def getSamplePos(nSamp,sampSize,sampConfig,sampPA=None):
    """Return fiber center positions and fiber shape given config string.

    Inputs:
        nSamp - number of fibers
        sampSize - fiber radius in arcsec for circular fibers
                 or edge length for square fibers
        sampConfig - configuration string (hex|hexNoCen|slit|ifu|triNoCen)
        sampPA - position angle for configs with square fibers (default None)
    Returns:
        (pos, fibShape)
            pos - 2 x nSamp ndarray with fiber centers in arcsecs from origin
            fibShape - string "circle" or "square"
    """        

    pos=np.zeros((2,nSamp))

    if(sampConfig=="hex"):
        fibShape="circle"
        pos[:,0]=np.array([0.,0.])
        theta=np.linspace(0,2*np.pi,num=nSamp-1,endpoint=False)
        rad=2.*sampSize
        pos[0,1:]=rad*np.cos(theta)
        pos[1,1:]=rad*np.sin(theta)
    elif(sampConfig=="hexNoCen"):
        fibShape="circle"
        theta=np.linspace(0,2*np.pi,num=nSamp,endpoint=False)
        rad=2.*sampSize
        pos[0,:]=rad*np.cos(theta)
        pos[1,:]=rad*np.sin(theta)
    elif(sampConfig=="slit"):
        fibShape="square"
        slitX=np.linspace(-1,1,num=nSamp)*0.5*sampSize*(nSamp-1)
        pos[0,:]=slitX*np.cos(np.deg2rad(sampPA))
        pos[1,:]=slitX*np.sin(np.deg2rad(sampPA))
    elif(sampConfig=="crossslit"):
        fibShape="square"
        slit1X=np.linspace(-1,1,num=0.5*nSamp)*0.5*sampSize*(0.5*nSamp-1)
        slit2X=np.linspace(-1,1,num=0.5*nSamp)*0.5*sampSize*(0.5*nSamp-1)
        pos[0,:]=np.append(slit1X*np.cos(np.deg2rad(sampPA)), slit2X*np.cos(np.deg2rad(sampPA+90.)))
        pos[1,:]=np.append(slit1X*np.sin(np.deg2rad(sampPA)), slit2X*np.sin(np.deg2rad(sampPA+90.)))
    elif(sampConfig=="ifu"):
        fibShape="square"
        numSide=np.sqrt(nSamp)
        if(np.int(numSide) != numSide):
            print "Error: ifu config needs a square number of fibers"
        else:
            ifuX=np.linspace(-1,1,num=numSide)*0.5*sampSize*(numSide-1)
            xx,yy=np.meshgrid(ifuX,ifuX)
            xx=xx.flatten()
            yy=yy.flatten()
            PArad=np.deg2rad(sampPA)
            pos[0,:]=xx*np.cos(PArad)-yy*np.sin(PArad)
            pos[1,:]=xx*np.sin(PArad)+yy*np.cos(PArad)
    elif(sampConfig=="triNoCen"):
        fibShape="circle"
        theta=np.linspace(0,2*np.pi,num=nSamp,endpoint=False)
        rad=sampSize
        pos[0,:]=rad*np.cos(theta)
        pos[1,:]=rad*np.sin(theta)
    else:
        # TO DO - add other configs - e.g. circle, extend hex for MaNGA style
        pass 

    return (pos,fibShape)


def getFiberFluxes(xobs,yobs,sampSize,fibConvolve,image):
    """Convolve image with fiber area and return fiber flux.

    Inputs:
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        sampSize - fiber radius in arcsecs
        fibConvolve - if False, just sample the image at central position 
                      without convolving, else convolve
        image - galsim object
    Returns:
        fiberFluxes - ndarray of values sampled at fiber centers in convolved image.

    Note: getFiberFluxes currently assumes circular fibers only, and
          is being superceded by makeConvolutionKernel with convOpt=pixel.
    """

    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)

    coordsPix=np.array([xobs,yobs])/pixScale + 0.5*imgSizePix # converted to pixels

    if(fibConvolve):
        scale_radius=10.*sampSize
        beta=0.
        fiber=galsim.Moffat(beta=beta,scale_radius=scale_radius,trunc=sampSize) # a kludgy way to get a circular tophat

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

def convertInclination(diskBA=None, diskCA=None, inc=None):
    """Convert between inclination and 3d axis ratios

    Given 2 of 3 inputs, return the 3rd
    
    Input: 
        diskBA - projected minor/major axis ratio (i.e. the imaging
                observable)
        diskCA - edge-on disk thickness/major axis ratio
        inc - disk inclination in radians (0 = face on)
    """
    assert((diskBA is None) + (diskCA is None) + (inc is None) == 1)

    if(diskBA is None):
        diskBA = np.sqrt(1. - np.sin(inc)**2 * (1. - diskCA**2))
        return diskBA
    elif(diskCA is None):
        diskCA = np.sqrt(1. - (1. - diskBA**2)/np.sin(inc)**2)
        return diskCA
    else:
        inc = np.arcsin(np.sqrt((1. - diskBA**2)/(1. - diskCA**2)))
        return inc

def getOmega(rad,rotCurvePars,rotCurveOpt='flat'):
    """Return angular rotation rate, i.e. v(r)/r

    Inputs:
        rad - ndarray of radii at which to sample
        rotCurvePars - float or array of option-dependent rotation curve parameters
        rotCurveOpt - "flat", "solid", or "nfw" (default "flat")
    Returns:
        omega - ndarray same length as rad with rotation rates
    """

    if(rotCurveOpt=='flat'): # v(r)=rotCurvePars[0]
        return rotCurvePars[0]/rad
    elif(rotCurveOpt=='solid'): # v(r)=rotCurvePars[0]*rad/rotCurvePars[1]
        return rotCurvePars[0]/rotCurvePars[1]
    elif(rotCurveOpt=='nfw'):
        # v ~ sqrt(M(<r)/r)
        # M(<r) ~ [log(1 + r/rs) - r/(r+rs)]
        mass=np.log(1.+rad/rotCurvePars[1]) - rad/(rad+rotCurvePars[1])
        vel=np.sqrt(mass/rad)
        return rotCurvePars[0]*vel/rad

def makeGalVMap(model):
    """Construct galsim objects for image, velocity map, and flux-weighted velocity map.

    Inputs (model object must contain the following):
        bulgeSersic - bulge Sersic index
        bulgeRadius - bulge half-light radius
        diskSersic - disk Sersic index
        diskRadius - disk half-light radius
        bulgeFraction - bulge fraction (0=pure disk, 1=pure bulge)
        diskBA - projected image axis ratio
        diskCA - edge-on axis ratio
        diskPA - position angle in degrees
        galFlux - normalization of image flux
        atmosFWHM - FWHM of gaussian PSF
        rotCurveOpt - option for getOmega ("flat", "solid", or "nfw")
        rotCurvePars - parameters for getOmega (depends on rotCurveOpt)
        g1 - shear 1
        g2 - shear 2
        pixScale - arcseconds per pixel
        nPix - number of pixels per side of each image
    Returns:
        (vmap,fluxVMap,gal) - tuple of galsim objects

    Note: See also makeGalVMap2, which uses pixel arrays instead of galsim objects
    """
    
    # Define the galaxy velocity map
    if(0 < model.bulgeFraction < 1):
        bulge=galsim.Sersic(model.bulgeSersic, half_light_radius=model.bulgeRadius)
        disk=galsim.Sersic(model.diskSersic, half_light_radius=model.diskRadius)

        gal=model.bulgeFraction * bulge + (1.-model.bulgeFraction) * disk
    elif(model.bulgeFraction == 0):
        gal=galsim.Sersic(model.diskSersic, half_light_radius=model.diskRadius)
    elif(model.bulgeFraction == 1):
        gal=galsim.Sersic(model.bulgeSersic, half_light_radius=model.bulgeRadius)

    gal.setFlux(model.galFlux)
    
    # Set shape of galaxy from axis ratio and position angle
    gal_shape=galsim.Shear(q=model.diskBA, beta=model.diskPA*galsim.degrees)
    gal.applyShear(gal_shape)

    # Generate galaxy image and empty velocity map array
    halfWidth=0.5*model.nPix*model.pixScale
    imgFrame=galsim.ImageF(model.nPix,model.nPix)
    galImg=gal.draw(image=imgFrame,dx=model.pixScale)
    imgArr=galImg.array.copy()   # must store these arrays as copies to avoid overwriting with shared imgFrame

    vmapArr=np.zeros_like(imgArr)
    fluxVMapArr=np.zeros_like(imgArr)

    # Set up velocity map parameters
    xCen=0.5*(galImg.xmax-galImg.xmin)
    yCen=0.5*(galImg.ymax-galImg.ymin)

    inc=convertInclination(diskBA=model.diskBA, diskCA=model.diskCA)
    sini=np.sin(inc)
    tani=np.tan(inc)
    gal_beta_rad=np.deg2rad(model.diskPA)

    # Fill velocity map array
    xx, yy=np.meshgrid(range(galImg.xmin-1,galImg.xmax),range(galImg.ymin-1,galImg.ymax))
    xp=(xx-xCen)*np.cos(gal_beta_rad)+(yy-yCen)*np.sin(gal_beta_rad)
    yp=-(xx-xCen)*np.sin(gal_beta_rad)+(yy-yCen)*np.cos(gal_beta_rad)
    radNorm=np.sqrt(xp**2 + yp**2 * (1.+tani**2))
    vmapArr=getOmega(radNorm,model.rotCurvePars,option=model.rotCurveOpt) * sini * xp
    vmapArr[0,:]=0 # galsim.InterpolatedImage has a problem with this array if I don't do something weird at the edge like this

    # Weight velocity map by galaxy flux and make galsim object
    fluxVMapArr=vmapArr*imgArr

    sumFVM=np.sum(fluxVMapArr)
    if(np.abs(sumFVM) < 0.01):
        print "experimental renorm"
        fluxVMapArr/=sumFVM
        gal.scaleFlux(1./sumFVM)

    fluxVMapImg=galsim.ImageViewD(fluxVMapArr,scale=model.pixScale)
    fluxVMap=galsim.InterpolatedImage(fluxVMapImg,pad_factor=6.)
    vmap=galsim.InterpolatedImage(galsim.ImageViewD(vmapArr,scale=model.pixScale)) # not flux-weighted

    # Apply lensing shear to galaxy and velocity maps
    if((model.g1 != 0.) | (model.g2 != 0.)):
        gal.applyShear(g1=model.g1,g2=model.g2)
        fluxVMap.applyShear(g1=model.g1,g2=model.g2)
        vmap.applyShear(g1=model.g1,g2=model.g2)

    # Convolve velocity map and galaxy with PSF
    if(model.atmosFWHM > 0):
        # Define atmospheric PSF
        #    atmos=galsim.Kolmogorov(fwhm=atmos_fwhm)
        atmos=galsim.Gaussian(fwhm=model.atmosFWHM)
        fluxVMap=galsim.Convolve([atmos, fluxVMap])
        gal=galsim.Convolve([atmos, gal])

    return (vmap,fluxVMap,gal)

def makeGalVMap2(model):
    """Construct pixel arrays for image, velocity map, and flux-weighted velocity map.

    Inputs (model object must contain the following):
        bulgeSersic - bulge Sersic index
        bulgeRadius - bulge half-light radius
        diskSersic - disk Sersic index
        diskRadius - disk half-light radius
        bulgeFraction - bulge fraction (0=pure disk, 1=pure bulge)
        diskBA - projected image axis ratio
        diskCA - edge-on axis ratio
        diskPA - position angle in degrees
        galFlux - normalization of image flux
        rotCurveOpt - option for getOmega ("flat", "solid", or "nfw")
        rotCurvePars - parameters for getOmega (depends on rotCurveOpt)
        g1 - shear 1
        g2 - shear 2
        pixScale - arcseconds per pixel
        nPix - number of pixels per side of each image
    Returns:
        (vmapArr,fluxVMapArr,thinImgArr,imgArr) - tuple of ndarray images

    Note: See also makeGalVMap, which uses galsim objects instead of pixel arrays
    """

    # Define the galaxy
#    if(0 < model.bulgeFraction < 1):
#        bulge=galsim.Sersic(model.bulgeSersic, half_light_radius=model.bulgeRadius)
#        disk=galsim.Sersic(model.diskSersic, half_light_radius=model.diskRadius)
#
#        gal=model.bulgeFraction * bulge + (1.-model.bulgeFraction) * disk
#    elif(model.bulgeFraction == 0):
#        gal=galsim.Sersic(model.diskSersic, half_light_radius=model.diskRadius)
#    elif(model.bulgeFraction == 1):
#        gal=galsim.Sersic(model.bulgeSersic, half_light_radius=model.bulgeRadius)
#
#    gal.setFlux(model.galFlux)
#    
#    # Set shape of galaxy from axis ratio and position angle
#    gal_shape=galsim.Shear(q=model.diskBA, beta=model.diskPA*galsim.degrees)
#    gal.applyShear(gal_shape)
#
#    # Generate galaxy image and empty velocity map array
#    halfWidth=0.5*model.nPix*model.pixScale
#    imgFrame=galsim.ImageF(model.nPix,model.nPix)
#    galImg=gal.draw(image=imgFrame,dx=model.pixScale)
#    imgArr=galImg.array.copy()    # must store these arrays as copies to avoid overwriting with shared imgFrame

    imgArr=makeImageBessel(model)
    
    vmapArr=np.zeros_like(imgArr)
    fluxVMapArr=np.zeros_like(imgArr)

    # Set up velocity map parameters
    xCen=0.5*model.nPix-0.5 # half-pixel offsets help avoid nans at r=0
    yCen=0.5*model.nPix-0.5

    inc=np.arccos(model.cosi)
    sini=np.sin(inc)
    tani=np.tan(inc)
    gal_beta_rad=np.deg2rad(model.diskPA)

    # Fill velocity map array
    xx, yy=np.meshgrid(range(model.nPix),range(model.nPix))
    xp=(xx-xCen)*np.cos(gal_beta_rad)+(yy-yCen)*np.sin(gal_beta_rad)
    yp=-(xx-xCen)*np.sin(gal_beta_rad)+(yy-yCen)*np.cos(gal_beta_rad)
    radNorm=np.sqrt(xp**2 + yp**2 * (1.+tani**2))
    vmapArr=getOmega(radNorm,model.rotCurvePars,rotCurveOpt=model.rotCurveOpt) * sini * xp

    # Weight velocity map by galaxy flux
    # Note we assume the emission line flux is from a thin disk
    # rather than weighting by the flux from the stellar image
    thinImgArr=makeImageBessel(model,diskCA=0.,bulgeFraction=0.)
    fluxVMapArr=vmapArr*thinImgArr

    # Apply lensing shear to galaxy and velocity maps
    if((model.g1 != 0.) | (model.g2 != 0.)):
        shear=np.array([[1-model.g1,-model.g2],[-model.g2,1+model.g1]])/np.sqrt(1.-model.g1**2-model.g2**2)
        xs=shear[0,0]*(xx-xCen) + shear[0,1]*(yy-yCen) + xCen
        ys=shear[1,0]*(xx-xCen) + shear[1,1]*(yy-yCen) + yCen
        vmapArr=scipy.ndimage.map_coordinates(vmapArr.T,(xs,ys))
        fluxVMapArr=scipy.ndimage.map_coordinates(fluxVMapArr.T,(xs,ys))
        thinImgArr=scipy.ndimage.map_coordinates(thinImgArr.T,(xs,ys))
        imgArr=scipy.ndimage.map_coordinates(imgArr.T,(xs,ys))

    return (vmapArr,fluxVMapArr,thinImgArr,imgArr)

def makeImageBessel(model,diskCA=None,bulgeFraction=None):
    """Draw a galaxy image using Bessel functions

    Follows Spergel 2010 to generate analytic surface brightness
    profiles. To be tested against galsim approach.

    Inputs (model object must contain the following):
        bulgeNu - bulge slope (Sersic index 4 -> nu~-0.6)
        bulgeRadius - bulge half-light radius
        diskNu - disk slope (Sersic index 1 -> nu~0.5)
        diskRadius - disk half-light radius
        bulgeFraction - bulge fraction (0=pure disk, 1=pure bulge)
        cosi - cosine of disk inclination (cosi=1 == i=0 == face on)
        diskCA - edge-on axis ratio
        diskPA - position angle in degrees
        galFlux - normalization of image flux
        pixScale - arcseconds per pixel
        nPix - number of pixels per side of each image

    Optional inputs (to use different thickness or bulge fraction than
    model value (model is unchanged)). Allows easy creation of thick
    and thin disk images without altering model object (e.g. for
    flux-weighting the emission line velocity map using a thin disk)

        diskCA - default None
        bulgeFraction - default None
                 
    Returns:
        image - 2d array of observable image
    """

    if(diskCA is None):
        diskCA=model.diskCA
    
    xx, yy = np.meshgrid(np.arange(model.nPix)-0.5*model.nPix,
                         np.arange(model.nPix)-0.5*model.nPix)

    if(model.bulgeFraction < 1.):
        paRad=np.deg2rad(model.diskPA)
        xp_disk = xx * np.cos(paRad) + yy * np.sin(paRad)
        yp_disk = -xx * np.sin(paRad) + yy * np.cos(paRad)
        phi_r = np.arctan2(yp_disk,xp_disk)-np.pi/2. # rotated by pi/2 so PA=0 is at x,y=1,0
        rr_disk = np.sqrt(xp_disk**2 + yp_disk**2)
        eps_disk = np.sqrt(1.-(1.-diskCA**2)*model.cosi**2)
        uu_disk = (rr_disk*model.pixScale)/model.diskRadius*np.sqrt((1.+eps_disk*np.cos(2.*phi_r))/(1.-eps_disk**2))
        f_disk = (uu_disk/2.)**model.diskNu*scipy.special.kv(model.diskNu,uu_disk)/scipy.special.gamma(model.diskNu+1.)

        # handle the r=0 case (at least for nu>0)
        # Note error in Spergel 2010 following Eq. 7,
        #   small u behavior is f->1/(2nu), not 1/(2(nu+1))
        #   (check examples in Eqs 6 & 7 to verify)
        if(model.diskNu > 0):
            f_disk[uu_disk==0] = 1./(2.*model.diskNu)

    else:
        f_disk=0.
        
    if(model.bulgeFraction > 0.):
        rr_bulge = np.sqrt(xx**2 + yy**2)
        uu_bulge = (rr_bulge*model.pixScale)/model.bulgeRadius
        f_bulge = (uu_bulge/2.)**model.bulgeNu*scipy.special.kv(model.bulgeNu,uu_bulge)/scipy.special.gamma(model.bulgeNu+1.)
    else:
        f_bulge=0.

    image = (1.-model.bulgeFraction)*f_disk + model.bulgeFraction*f_bulge
    return image

    
def makeConvolutionKernel(xobs,yobs,model):
    """Construct the fiber x PSF convolution kernel

    When using pixel arrays (instead of galsim objects),
    makeConvolutionKernel saves time since you only need to 
    calculate the kernel once and can multiply it by the flux map,
    rather than convolving each model galaxy and sampling at a position

    Inputs:
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        model - object containing the following attributes
            atmosFWHM - FWHM of gaussian PSF
            vSampSize - fiber radius in arcsecs or IFU/slit pixel size
            vSampConvolve - if False, just sample the image at central position 
                          without convolving, else convolve
            vSampShape - string "circle" or "square"
            vSampPA - position angle for configs with square fibers 
                    (can be None for fibShape="circle")
            nPix - number of pixels on image side
            pixScale - arcseconds per pixel
    Returns:
        kernel - an ndarray of size [xobs.size, imgSizePix, imgSizePix]
    """

    half=model.nPix/2
    xx,yy=np.meshgrid((np.arange(model.nPix)-half)*model.pixScale,(np.arange(model.nPix)-half)*model.pixScale)
    if(model.atmosFWHM > 0):
        atmos_sigma=model.atmosFWHM/(2.*np.sqrt(2.*np.log(2.)))
        if(model.vSampConvolve): # PSF and Fiber convolution
            psfArr=np.exp(-(xx**2 + yy**2)/(2.*atmos_sigma**2))
            fibArrs=np.zeros((model.nVSamp,model.nPix,model.nPix))
            if(model.vSampShape=="circle"):
                sel=np.array([((xx-pos[0])**2 + (yy-pos[1])**2 < model.vSampSize**2) for pos in zip(xobs,yobs)])
            elif(model.vSampShape=="square"):
                PArad=np.deg2rad(model.vSampPA)
                sel=np.array([((np.abs((xx-pos[0])*np.cos(PArad) - (yy-pos[1])*np.sin(PArad)) < 0.5*model.vSampSize) & (np.abs((xx-pos[0])*np.sin(PArad) + (yy-pos[1])*np.cos(PArad)) < 0.5*model.vSampSize)) for pos in zip(xobs,yobs)])
            fibArrs[sel]=1.
            kernel=np.array([scipy.signal.fftconvolve(psfArr,fibArrs[ii],mode="same") for ii in range(model.nVSamp)])
        else:
            # this is basically the psf convolved with a delta function at the center of each fiber
            kernel=np.array([np.exp(-((xx-pos[0])**2 + (yy-pos[1])**2)/(2.*atmos_sigma**2)) for pos in zip(xobs,yobs)])
    else:
        # Fiber only
        kernel=np.zeros((model.nVSamp,model.nPix,model.nPix))
        if(model.vSampShape=="circle"):
            sel=np.array([((xx-pos[0])**2 + (yy-pos[1])**2 < model.vSampSize**2) for pos in zip(xobs,yobs)])
        elif(model.vSampShape=="square"):
            PArad=np.deg2rad(model.vSampPA)
            sel=np.array([((np.abs((xx-pos[0])*np.cos(PArad) - (yy-pos[1])*np.sin(PArad)) < 0.5*model.vSampSize) & (np.abs((xx-pos[0])*np.sin(PArad) + (yy-pos[1])*np.cos(PArad)) < 0.5*model.vSampSize)) for pos in zip(xobs,yobs)])
        kernel[sel]=1.
        
    return kernel

def vmapObs(model,xobs,yobs,showPlot=False):
    """Get flux-weighted fiber-averaged velocities

    vmapObs computes fiber sampling in two ways, depending on convOpt
        for convOpt=galsim, need to specify atmos_fwhm,vSampSize,fibConvolve
        for convOpt=pixel, need to specify kernel

    Inputs:
        model - object with galaxy and observable parameters
        xobs - float or ndarray of fiber x-centers
        yobs - float or ndarray of fiber y-centers
        showPlot - bool for presenting plots (default False)
    Returns:
        ndarray of flux-weighted fiber-averaged velocities

    Note: see vmapModel for faster vmap evaluation without PSF and fiber convolution
    """

    if(model.convOpt=="galsim"):
        vmap,fluxVMap,gal=makeGalVMap(model)

        if(showPlot):
            plot.showImage(gal,xobs,yobs,model.vSampSize,showPlot=True)
            plot.showImage(vmap,xobs,yobs,model.vSampSize,showPlot=True)
            plot.showImage(fluxVMap,xobs,yobs,model.vSampSize,showPlot=True)

        # Get the flux in each fiber
        galFibFlux=getFiberFluxes(xobs,yobs,model.vSampSize,model.vSampConvolve,gal)
        vmapFibFlux=getFiberFluxes(xobs,yobs,model.vSampSize,model.vSampConvolve,fluxVMap)

    elif(model.convOpt=="pixel"):
        vmapArr,fluxVMapArr,thinImgArr,imgArr=makeGalVMap2(model)
        if(showPlot):
            plot.showArr(imgArr)
            plot.showArr(fluxVMapArr)
        vmapFibFlux=np.array([np.sum(model.kernel[ii]*fluxVMapArr) for ii in range(model.nVSamp)])
        galFibFlux=np.array([np.sum(model.kernel[ii]*thinImgArr) for ii in range(model.nVSamp)])

    return vmapFibFlux/galFibFlux

def vmapModel(model, xobs, yobs):
    """Evaluate model velocity field at given coordinates

    Inputs:
        model - object with these values 
          [diskPA, diskBA, diskCA, rotCurvePars rotCurveOpt, g1, g2] 
          *unsheared* values
        xobs, yobs - the N positions (in arcsec) relative to the center at which
                     the *sheared* (observed) field is sampled
    Returns:
        vmodel is an N array of fiber velocities

    Note: vmapModel is like vmapObs without PSF and fiber convolution
    """

    # compute spectroscopic observable
    if(xobs is not None):
        # convert coords to source plane
        pairs=shearPairs(np.array(zip(xobs,yobs)),-model.g1,-model.g2)
        xx=pairs[:,0]
        yy=pairs[:,1]

        # rotated coords aligned with PA guess of major axis
        xCen,yCen=0,0 # assume centroid is well-measured
        PArad=np.deg2rad(model.diskPA)
        xp=(xx-xCen)*np.cos(PArad)+(yy-yCen)*np.sin(PArad)
        yp=-(xx-xCen)*np.sin(PArad)+(yy-yCen)*np.cos(PArad)
        # projection along apparent major axis in rotated coords
        kvec=np.array([1,0,0])
    
        inc=convertInclination(diskBA=model.diskBA, diskCA=model.diskCA)
        sini=np.sin(inc)
        tani=np.tan(inc)

        rotCurvePars=np.array([vmax])
        nSamp=xobs.size
        vmodel=np.zeros(nSamp)
        radNorm=np.sqrt(xp**2 + yp**2 * (1.+tani**2))
        vmodel=getOmega(radNorm,model.rotCurvePars,rotCurveOpt=model.rotCurveOpt) * sini * xp
    else:
        vmodel=None

    return vmodel

def ellModel(model):
    """Compute sheared ellipse pars for a model galaxy

    Inputs:
        model - object with these values 
          [diskPA, diskBA, diskCA, rotCurvePars rotCurveOpt, g1, g2] 
          *unsheared* values
    Returns:
        ndarray([gal_beta, gal_q]) *sheared* values
    """

    ellipse=(model.diskRadius,model.diskBA,model.diskPA) # unsheared ellipse
    disk_r_prime,gal_q_prime,gal_beta_prime=shearEllipse(ellipse,model.g1,model.g2)
    ellmodel=np.array([gal_beta_prime,gal_q_prime]) # model sheared ellipse observables

    return ellmodel

if __name__ == "__main__":
    print "use one of the functions - no main written"
    

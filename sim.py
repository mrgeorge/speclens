#! env python

import galsim
import scipy.integrate
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def getFiberPos(fibID,numFib,fibRad):
    # returns fiber position relative to center in same units as fibRad (arcsec)
    if(fibID == 0):
        pos=galsim.PositionD(x=0,y=0)
    else:
        theta=(fibID-1.)/(numFib-1.) * 2.*np.pi
        rad=2.*fibRad
        pos=galsim.PositionD(x=rad*np.cos(theta),y=rad*np.sin(theta))
    return pos

def radIntegrand(rad,theta,fiberPos,image):
    pos=galsim.PositionD(x=fiberPos.x+rad*np.cos(theta),y=fiberPos.y+rad*np.sin(theta))
    return rad*image.xValue(pos)
def thetaIntegrand(theta,fiberPos,image,fibRad,tol):
    return scipy.integrate.quad(radIntegrand,0,fibRad,args=(theta,fiberPos,image),epsabs=tol,epsrel=tol)[0]
def getFiberFlux(fibID,numFib,fibRad,image,tol=1.e-4):
    fiberPos=getFiberPos(fibID,numFib,fibRad)
    return scipy.integrate.quad(thetaIntegrand,0,2.*np.pi, args=(fiberPos,image,fibRad,tol), epsabs=tol, epsrel=tol)

def showImage(profile,numFib,fibRad):
    pixScale=0.1
    imgSizePix=int(10.*fibRad/pixScale)
    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)
    img=profile.draw(image=imgFrame,dx=pixScale)
    halfWidth=0.5*imgSizePix*pixScale
    #    img.setCenter(0,0)
    plt.imshow(img.array,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest')
    for ii in range(numFib):
        pos=getFiberPos(ii,numFib,fibRad)
        circ=plt.Circle((pos.x,pos.y),radius=fibRad,fill=False)
        ax=plt.gca()
        ax.add_patch(circ)
    plt.colorbar()
    plt.show()

def makeGalImage(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm):
    # Define the galaxy profile
    bulge=galsim.Sersic(bulge_n, half_light_radius=bulge_r)
    disk=galsim.Sersic(disk_n, half_light_radius=disk_r)

    gal=bulge_frac * bulge + (1.-bulge_frac) * disk
    gal.setFlux(gal_flux)
    
    # Set shape of galaxy from axis ratio and position angle
    gal_shape=galsim.Shear(q=gal_q, beta=gal_beta*galsim.radians)
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
    # Define the galaxy velocity map
    bulge=galsim.Sersic(bulge_n, half_light_radius=bulge_r)
    disk=galsim.Sersic(disk_n, half_light_radius=disk_r)

    gal=bulge_frac * bulge + (1.-bulge_frac) * disk
    gal.setFlux(gal_flux)
    
    # Set shape of galaxy from axis ratio and position angle
    gal_shape=galsim.Shear(q=gal_q, beta=gal_beta*galsim.radians)
    gal.applyShear(gal_shape)

    # Generate galaxy image and empty velocity map array
    halfWidth=0.5*imgSizePix*pixScale
    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)
    galImg=gal.draw(image=imgFrame,dx=pixScale)
    imgArr=galImg.array
    vmapArr=np.zeros_like(imgArr)
    fluxVMapArr=np.zeros_like(imgArr)

    # Set up velocity map parameters
    xCen=0.5*(galImg.xmax-galImg.xmin)
    yCen=0.5*(galImg.ymax-galImg.ymin)

    inc=getInclination(gal_q)
    sini=np.sin(inc)
    tani=np.tan(inc)

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
            xp=(xx-xCen)*np.cos(gal_beta)+(yy-yCen)*np.sin(gal_beta)
            yp=-(xx-xCen)*np.sin(gal_beta)+(yy-yCen)*np.cos(gal_beta)

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
    galX.applyShear(e1=e1,e2=e2)
    galK=galX
    fluxVMap.applyShear(e1=e1,e2=e2)
    vmap.applyShear(e1=e1,e2=e2)

    # Convolve velocity map and galaxy with PSF
    if(atmos_fwhm > 0):

        # Define atmospheric PSF
        #    atmos=galsim.Kolmogorov(fwhm=atmos_fwhm)
	atmos=galsim.Gaussian(fwhm=atmos_fwhm)

        # note: real-space convolution with InterpolatedImage doesn't seem to work,
        #       so just use scipy's convolve2d to convolve the arrays
	fluxVMapArr=scipy.signal.convolve2d(fluxVMap.draw(dx=pixScale).array,atmos.draw(dx=pixScale).array,mode='same')
	fluxVMap=galsim.InterpolatedImage(galsim.ImageViewF(fluxVMapArr,scale=pixScale))
        #    fluxVMapPSFK=galsim.Convolve([atmos, fluxVMap],real_space=False) # used fourier-space convolution for faster drawing
    
        galX=galsim.Convolve([atmos, gal],real_space=True) # used real-space convolution for easier real-space integration
        galX=galsim.Convolve([atmos, gal],real_space=False) # used real-space convolution for easier real-space integration

    return (vmap,fluxVMap,galX,galK)

def vmapObs(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,e1,e2,pixScale,fibRad,numFib,plotIm=False):
# get flux-weighted fiber-averaged velocities
	
    imgSizePix=int(10.*fibRad/pixScale)
    vmap,fluxVMap,gal,galK=makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,e1,e2)

    if(plotIm):
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

def vmapModel(pars, xvals):
# function to fit velocity field
# pars: 0=gal_beta (PA), 1=gal_q, vmax
# xvals are the azimuthal angles at which the field is sampled
	
    fibRad=1.
    rad=2.*fibRad # assume the fibers sample the v field at their center

    # Cartesian coords
    xx=rad*np.cos(xvals)
    yy=rad*np.sin(xvals)    

    # rotated coords aligned with pars[0] guess of major axis
    xCen,yCen=0,0 # assume centroid is well-measured
    xp=(xx-xCen)*np.cos(pars[0])+(yy-yCen)*np.sin(pars[0])
    yp=-(xx-xCen)*np.sin(pars[0])+(yy-yCen)*np.cos(pars[0])
    # projection along apparent major axis in rotated coords
    kvec=np.array([1,0,0])
    
    inc=getInclination(pars[1])
    sini=np.sin(inc)
    tani=np.tan(inc)

    rotCurvePars=np.array([pars[2]])
    nSamp=xvals.size
    vmodel=np.zeros(nSamp)
    for ii in range(xvals.size):
        # coordinates in the plane of the galaxy
	radvec=np.array([xp[ii],yp[ii],yp[ii]*tani])
	vmodel[ii]=getOmega(np.linalg.norm(radvec),rotCurvePars,option='flat') * sini * np.dot(radvec,kvec)

    return vmodel

def vmapFit(vfibFlux):
# fit model to fiber velocities

    numFib=vfibFlux.size - 1
    ang=np.linspace(0,2.*np.pi,num=numFib,endpoint=False)
    vel=vfibFlux[1:] # ignore the central pointing

    guess=np.array([np.pi/6,0.1,100.])

    err= lambda pars, xvals, yvals: vmapModel(pars, xvals) - yvals
    pars, success = scipy.optimize.leastsq(err, guess[:], args=(ang,vel))
    print "gal_beta={}, gal_q={}, vmax={}. success={}".format(pars[0],pars[1],pars[2],success)
    fitX=np.linspace(0,2.*np.pi)
    fitY=vmapModel(pars,fitX)
    plt.plot(ang,vel,'bo',fitX,fitY,'r-')
    plt.show()

def main(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,numFib,plotIm=False):

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
    if(plotIm):
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

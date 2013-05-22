#! env python

import galsim
import scipy.integrate
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

def integrand(theta,rad,fiberPos,image):
    pos=galsim.PositionD(x=fiberPos.x+rad*np.cos(theta),y=fiberPos.y+rad*np.sin(theta))
    return rad*image.xValue(pos)

def getFiberFlux(fibID,numFib,fibRad,image):
    fiberPos=getFiberPos(fibID,numFib,fibRad)
    tol=1.e-2
    return scipy.integrate.dblquad(integrand,0,fibRad,lambda x: 0,lambda x: 2.*np.pi, args=(fiberPos,image), epsabs=tol, epsrel=tol)

def showImage(profile,numFib,fibRad):
    pixScale=0.1
    imgSizePix=int(10.*fibRad/pixScale)
    imgFrame=galsim.ImageF(imgSizePix,imgSizePix)
    img=profile.draw(image=imgFrame,dx=pixScale)
    halfWidth=0.5*imgSizePix*pixScale
    #    img.setCenter(0,0)
    plt.imshow(img.array,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth))
    for ii in range(numFib):
        pos=getFiberPos(ii,numFib,fibRad)
        circ=plt.Circle((pos.x,pos.y),radius=fibRad,fill=False)
        ax=plt.gca()
        ax.add_patch(circ)
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

    # Plot the image with fibers overlaid
    if(plotIm):
        showImage(nopixK,numFib,fibRad)

    # Get the flux in each fiber
    fibFlux=np.zeros(numFib)
    for ii in range(numFib):
        fibFlux[ii]=getFiberFlux(ii,numFib,fibRad,nopixX)[0]

    return fibFlux

if __name__ == "__main__":
    main()

#! env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.ndimage

# stackoverflow's nd meshgrid
# http://stackoverflow.com/questions/1827489/numpy-meshgrid-in-3d
def ndmesh(*args):
   args = map(np.asarray,args)
   return np.broadcast_arrays(*[x[(slice(None),)+(None,)*i] 
                                for i, x in enumerate(args)])

def fnu(nu,u):
    """Spergel 2010 Eq 4"""
    ff = (u/2.)**nu * scipy.special.kv(nu,u) / scipy.special.gamma(nu+1.)

    # handle the u=0 case (at least for nu>0)
    # Note error in Spergel 2010 following Eq. 7,
    #   small u behavior is f->1/(2nu), not 1/(2(nu+1))
    #   (check examples in Eqs 6 & 7 to verify)
    # First try to catch u==0 assuming u is an array
    #   then handle the scalar case if exception is thrown
    try:
        if(nu > 0):
            ff[u==0] = 1./(2.*nu)
    except IndexError:
        if(nu > 0):
            if(u==0):
                ff = 1./(2.*nu)
    return ff
            
def bessel3d(nu, arg):
    const=1.
    return const*fnu(nu-0.5,arg)
    
def r3d(xx,yy,zz,xcen,ycen,zcen,rx,ry,rz):
    return np.sqrt(((xx-xcen)/rx)**2 + ((yy-ycen)/ry)**2 +
                    ((zz-zcen)/rz)**2)

def rexp2rhalf(rexp):
    return 1.67835 * rexp

def rhalf2rexp(rhalf):
    return rhalf/1.67835

def exp3d(arg): 
    return np.exp(-arg)
    
if __name__ == "__main__":

    # define x,y,z grid
    xx,yy,zz=ndmesh(range(100),range(100),range(100))
    # centroid positions aren't integer 50 because
    #   bessel3d diverges at r=0 for nu<=0.5
    xcen,ycen,zcen=49.9,49.9,49.9

    # define scale lengths (in pixels)
    # play with rz to change thickness
    rx,ry,rz=10.,10.,0.1

    # play with inclination for tilt (rotation about x-axis)
    inc=np.deg2rad(0.)

    # define inclined coordinates
    xp=xx
    yp=(yy-ycen)*np.cos(inc) - (zz-zcen)*np.sin(inc) + ycen
    zp=(yy-ycen)*np.sin(inc) + (zz-zcen)*np.cos(inc) + zcen

    # 3d radius scaled by each dimension's scale length
    arg=r3d(xx,yy,zz,xcen,ycen,zcen,rx,ry,rz)
    
    # evaluate 3d luminosity density with bessel functions
    nu=0.5 # 0.5 for exponential, -0.6 for ~deVaucouleur's
    cnu=1.67835 # convert half-light to exponential radius
    bb=bessel3d(nu,arg)

    # evaluate luminosity density at inclined coordinates
    bp=scipy.ndimage.map_coordinates(bb.T,[xp,yp,zp])

    # compare original and inclined images in each projected dimension
    plt.imshow(bb.sum(axis=0),interpolation="nearest") # xy
    plt.show()
    plt.imshow(bp.sum(axis=0),interpolation="nearest")
    plt.show()
    plt.imshow(bb.sum(axis=1),interpolation="nearest") # xz
    plt.show()
    plt.imshow(bp.sum(axis=1),interpolation="nearest")
    plt.show()
    plt.imshow(bb.sum(axis=2),interpolation="nearest") # yz
    plt.show()
    plt.imshow(bp.sum(axis=2),interpolation="nearest")
    plt.show()

    # vertical slice in xy plane
    # this tests what the effective scale radius is in the 
    #    inclined (y) dimension. To a good approximation,
    #    ry' = ry * sqrt[1 - sin(inc)**2 * (1-(rz/ry)**2)]
    #    which gives ry * cos(inc) as rz->0 (thin disk)
    #    and ry or rz as rz->ry (spherical case)
    # But face on bessel disks appear less cuspy then exp
    yarr=np.arange(100)
    plt.plot(bp.sum(axis=0)[:,50]/np.max(bp.sum(axis=0)[:,50]),'blue')
    plt.plot(bp.sum(axis=0)[:,50],'red')
    plt.plot(np.exp(-np.sqrt(((yarr-ycen)/(ry*np.sqrt(1.-np.sin(inc)**2*(1.-(rz/ry)**2))))**2)),'green')
    plt.show()

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
    return (u/2.)**nu * scipy.special.kv(nu,u) / scipy.special.gamma(nu+1.)

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

    xx,yy,zz=ndmesh(range(100),range(100),range(100))

    xcen,ycen,zcen=49.5,49.5,49.5
    rx,ry,rz=10.,10.,10.


    inc=np.deg2rad(0.)
    xp=xx
    yp=(yy-ycen)*np.cos(inc) - (zz-zcen)*np.sin(inc) + ycen
    zp=(yy-ycen)*np.sin(inc) + (zz-zcen)*np.cos(inc) + zcen

    arg=r3d(xx,yy,zz,xcen,ycen,zcen,rx,ry,rz)
    
    ff=exp3d(arg)
    fp=scipy.ndimage.map_coordinates(ff.T,[xp,yp,zp])

    nu=0.5
    cnu=1.67835
    bb=bessel3d(nu,arg)
    bp=scipy.ndimage.map_coordinates(bb.T,[xp,yp,zp])

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
    yarr=np.arange(100)
    plt.plot(bp.sum(axis=0)[:,50]/np.max(bp.sum(axis=0)[:,50]))
    plt.plot(np.exp(-np.sqrt(((yarr-ycen)/(ry*np.cos(inc)))**2)))
    plt.show()
    

#! env python

import numpy as np
import os
import sys

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens
from speclens import sim

rtol=1.e-2
atol=0.
rtolDeg=0.
atolDeg=1.

def test_ellModel0():
    model=speclens.Model("B")
    model.g1=0.
    model.g2=0.
    expected=np.array([model.diskPA,model.diskBA])

    ell=sim.ellModel(model)

    np.testing.assert_allclose(expected, ell, rtol=rtol, atol=atol,
        err_msg="ellModel with no shear should return original values")

def test_shearEllipse0():
    g1=0.
    g2=0.

    diskRadius=1.
    diskBA=0.5
    diskPA=0.

    expected=[diskRadius, diskBA, diskPA]
    obs=sim.shearEllipse(expected, g1, g2)

    np.testing.assert_allclose(expected, obs, rtol=rtol, atol=atol,
        err_msg="shearEllipse with no shear should return original values")

def test_shearEllipse1():
    g1=0.01
    g2=0.

    diskRadius=1.
    diskBA=0.5
    diskPA=0.
    
    expectedRad=diskRadius
    # note sign difference with Eq 6 from 1311.1489v1
    expectedBA = diskBA * (1. - 2.*g1)
    # Eq 7 from 1311.1489v1
    expectedPA = np.rad2deg(np.deg2rad(diskPA) + (1. + diskBA**2)/(1. - diskBA**2) * g2) 

    obsRad,obsBA,obsPA=sim.shearEllipse([diskRadius,diskBA,diskPA],g1,g2)
    
    np.testing.assert_allclose(expectedRad, obsRad, rtol=rtol,
        atol=atol, err_msg="shearEllips does not return expected radius in weak limit on gplus")
    np.testing.assert_allclose(expectedBA, obsBA, rtol=rtol,
        atol=atol, err_msg="ellModel does not return expected axis ratio in weak limit on gplus")
    np.testing.assert_allclose(expectedPA, obsPA, rtol=rtolDeg,
        atol=atolDeg, err_msg="ellModel does not return expected PA in weak limit on gplus")

def test_shearEllipse2():
    g1=0.01
    g2=0.

    diskRadius=1.
    diskBA=0.5
    diskPA=0.
    
    expectedRad=diskRadius
    # note sign difference with Eq 6 from 1311.1489v1
    expectedBA = diskBA * (1. - 2.*g1)
    # Eq 7 from 1311.1489v1
    expectedPA= np.rad2deg(np.deg2rad(diskPA) + (1. + diskBA**2)/(1. - diskBA**2) * g2) 

    obsRad,obsBA,obsPA=sim.shearEllipse([diskRadius,diskBA,diskPA],g1,g2)
    
    np.testing.assert_allclose(expectedRad, obsRad, rtol=rtol,
        atol=atol, err_msg="shearEllips does not return expected radius in weak limit on gcross")
    np.testing.assert_allclose(expectedBA, obsBA, rtol=rtol,
        atol=atol, err_msg="ellModel does not return expected axis ratio in weak limit on gcross")
    np.testing.assert_allclose(expectedPA, obsPA, rtol=rtolDeg,
        atol=atolDeg, err_msg="ellModel does not return expected PA in weak limit on gcross")


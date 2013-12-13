#! env python

import numpy as np
import os
import sys
import galsim

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens


def test_image(): 

    diskRadius=1.
    diskSersic=1.
    diskFlux=1.
    bulgeFraction=0.
    bulgeRadius=1.
    bulgeSersic=4.

    redshift=0.5
    diskPA=0.
    g1=0.
    g2=0.
    atmosFWHM=None

    galBA=1.
    galCA=0.2
    diskInclination=speclens.sim.convertInclination(galBA=galBA, 
                                                    galCA=galCA)
    
    gal=speclens.Galaxy(diskRadius=diskRadius, diskSersic=diskSersic,
                        diskFlux=diskFlux,
                        bulgeFraction=bulgeFraction,
                        bulgeRadius=bulgeRadius,
                        bulgeSersic=bulgeSersic)
    obs=speclens.Observation(gal, redshift=redshift, diskPA=diskPA.,
                             diskInclination=diskInclination, g1=0., 
                             g2=0., atmosFWHM=None) 
    im1=obs.getImage()

    im2=speclens.sim.makeImageBessel(bulgeSersic, bulgeRadius,
                                     diskSersic, diskRadius,
                                     bulgeFraction, galBA, diskPA,
                                     galCA)

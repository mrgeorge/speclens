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
    gal=speclens.Galaxy()
    obs=speclens.Observation(gal)
    im1=obs.getImage()

    im2=

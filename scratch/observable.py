#! env python


# test out the Observable object in base.py
# instantiate an object
# assign its parameters
# generate observed velocities with sim or makeObs and assign
# generate a similar Model object and compare values for likelihood

import numpy as np
import os
import sys

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens

def computeLikelihood(model, observable):
    pass

vObsErr=np.repeat(10.,20)
diskPAErr=10.
diskBAErr=0.1
to=speclens.TestObservable()
to.setPointing(vSampPA=to.diskPA)
to.setAttr(vObsErr=vObsErr, diskPAErr=diskPAErr, diskBAErr=diskBAErr)

model=speclens.Model("B")

model.redshift=to.redshift
model.diskRadius=to.diskRadius
model.diskNu=to.diskNu
model.bulgeFraction=to.bulgeFraction
model.bulgeRadius=to.bulgeRadius
model.bulgeNu=to.bulgeNu
model.galFlux=to.galFlux
model.atmosFWHM=to.atmosFWHM

model.pixScale=to.pixScale
model.nPix=to.nPix
model.vSampConfig=to.vSampConfig
model.vSampSize=to.vSampSize
model.nVSamp=to.nVSamp

xvals,yvals,vvals,ellObs,inputPars=speclens.ensemble.makeObs(model,
    sigma=to.vObsErr, ellErr=np.array([to.diskPAErr, to.diskBAErr]),
    randomPars=False)
np.testing.assert_allclose(to.xObs, xvals)
np.testing.assert_allclose(to.yObs, yvals)

to.vObs=vvals

dataType="imgPar+velocities"
to.defineDataVector(dataType)

computeLikelihood(model, to)

print "done"

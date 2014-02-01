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
from speclens import fit

rtol=1.e-2
atol=0.
rtolDeg=0.
atolDeg=1.


def test_chisq0():
    nVals = 10
    modelVector = np.zeros(nVals)
    dataVector = np.ones(nVals)
    errVector = np.ones(nVals)
    wrapVector = np.repeat(None, nVals)
    
    expected = np.sum(((modelVector - dataVector) / errVector)**2)
    result1 = fit.chisq(modelVector, dataVector, errVector, None)
    result2 = fit.chisq(modelVector, dataVector, errVector, wrapVector)

    np.testing.assert_allclose(expected, result1, rtol=rtol, atol=atol,
        err_msg = "chisq fails with wrapVector = None")
    np.testing.assert_allclose(expected, result2, rtol=rtol, atol=atol,
        err_msg = "chisq fails with wrapVector = None array")

def test_chisq1():
    modelVector = np.array([0.])
    dataVector = np.array([1.])
    errVector = np.array([0.1])
    wrapVector1 = [(0.,0.3)]
    wrapVector2 = [(0.,0.5)]
    wrapVector3 = [(0.,1.0)]
    wrapVector4 = [(0.,1.1)]
    wrapVector5 = [(0.,2.0)]
    
    expected1 = 1.**2
    expected2 = 0.
    expected3 = 0.
    expected4 = 1.**2
    expected5 = 10.**2
    result1 = fit.chisq(modelVector, dataVector, errVector, wrapVector1)
    result2 = fit.chisq(modelVector, dataVector, errVector, wrapVector2)
    result3 = fit.chisq(modelVector, dataVector, errVector, wrapVector3)
    result4 = fit.chisq(modelVector, dataVector, errVector, wrapVector4)
    result5 = fit.chisq(modelVector, dataVector, errVector, wrapVector5)

    np.testing.assert_allclose(expected1, result1, rtol=rtol, atol=atol)
    np.testing.assert_allclose(expected2, result2, rtol=rtol, atol=atol)
    np.testing.assert_allclose(expected3, result3, rtol=rtol, atol=atol)
    np.testing.assert_allclose(expected4, result4, rtol=rtol, atol=atol) # FAILS - because chisq uses mod operator with floats - check this
    np.testing.assert_allclose(expected5, result5, rtol=rtol, atol=atol)

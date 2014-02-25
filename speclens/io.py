#! env python

import fitsio
import numpy as np

def writeRec(rec, filename, header=None, clobber=True, compress="GZIP"):
    """Write recarray to fits file"""
    fitsio.write(filename, rec, header=header, clobber=clobber,
        compress=compress)

def readRec(filename, header=False):
    """Read fits file into recarray

    header - if True, return (rec, header) tuple, else return rec
    """
    return fitsio.read(filename, header=header)

def parsToRec(pars,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    """Convert pars ndarray to recarray"""
    dtype=[(label,float) for label in labels]
    rec=np.recarray(len(pars),dtype=dtype)
    for ii in range(len(labels)):
        rec[labels[ii]]=pars[:,ii]
    return rec

def chainToRec(chain,lnprob,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    """Convert chain and lnprob to recarray"""
    nGal=chain.shape[0]
    nPars=chain.shape[1]
    arr=np.zeros((nGal,nPars+1))
    arr[:,:-1]=chain
    arr[:,-1]=lnprob
    labels=np.append(labels,"lnprob")
    rec=parsToRec(arr,labels=labels)
    return rec

def recToPars(rec,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    """Convert recarray to pars array"""
    recLabels=rec.dtype.fields.keys() # note, this list is unordered since rec is a dict, so we need to use parsLabels (which should be sorted to match the order of columns in pars array)
    pars=np.zeros((len(rec),len(labels)))
    for ii in range(len(labels)):
        pars[:,ii]=rec[labels[ii]]
    return pars

def obsToRec(xvals,yvals,vvals,ellObs):
    """Convert array of observables to recarray"""
    dtype=[("xvals",(xvals.dtype.type,xvals.shape)),("yvals",(yvals.dtype.type,yvals.shape)),("vvals",(vvals.dtype.type,vvals.shape)),("ellObs",(ellObs.dtype.type,ellObs.shape))]
    rec=np.recarray(1,dtype=dtype)
    rec["xvals"]=xvals
    rec["yvals"]=yvals
    rec["vvals"]=vvals
    rec["ellObs"]=ellObs
    return rec

def recToObs(rec):
    """Convert recarray of ndarrays of observables"""
    xvals=rec["xvals"].squeeze()
    yvals=rec["yvals"].squeeze()
    vvals=rec["vvals"].squeeze()
    ellObs=rec["ellObs"].squeeze()
    return (xvals,yvals,vvals,ellObs)

def statsToRec(inputPars,mp,kde,hw):
    """Store input pars array and some chain stats as recarray"""
    dtype=[("inputPars",(inputPars.dtype.type,inputPars.shape)),("mp",(mp.dtype.type,mp.shape)),("kde",(kde.dtype.type,kde.shape)),("hw",(hw.dtype.type,hw.shape))]
    rec=np.recarray(1,dtype=dtype)
    rec["inputPars"]=inputPars
    rec["mp"]=mp
    rec["kde"]=kde
    rec["hw"]=hw
    return rec

def makeHeader(iterations, accfrac):
    """Generate dictionary to store in fits header"""
    return {"ITERATIONS":iterations, "ACCFRAC":accfrac}

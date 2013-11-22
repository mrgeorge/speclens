#! env python

import fitsio
import numpy as np

def writeRec(rec,filename,clobber=True,compress="GZIP"):
    fitsio.write(filename,rec,clobber=clobber,compress=compress)

def readRec(filename):
    rec=fitsio.read(filename)
    return rec

def parsToRec(pars,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    dtype=[(label,float) for label in labels]
    rec=np.recarray(len(pars),dtype=dtype)
    for ii in range(len(labels)):
        rec[labels[ii]]=pars[:,ii]
    return rec

def chainToRec(chain,lnprob,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    nGal=chain.shape[0]
    nPars=chain.shape[1]
    arr=np.zeros((nGal,nPars+1))
    arr[:,:-1]=chain
    arr[:,-1]=lnprob
    labels=np.append(labels,"lnprob")
    rec=parsToRec(arr,labels=labels)
    return rec

def recToPars(rec,labels=np.array(["PA","b/a","vmax","g1","g2"])):
    recLabels=rec.dtype.fields.keys() # note, this list is unordered since rec is a dict, so we need to use parsLabels (which should be sorted to match the order of columns in pars array)
    pars=np.zeros((len(rec),len(labels)))
    for ii in range(len(labels)):
        pars[:,ii]=rec[labels[ii]]
    return pars

def obsToRec(xvals,yvals,vvals,ellObs):
    dtype=[("xvals",(xvals.dtype.type,xvals.shape)),("yvals",(yvals.dtype.type,yvals.shape)),("vvals",(vvals.dtype.type,vvals.shape)),("ellObs",(ellObs.dtype.type,ellObs.shape))]
    rec=np.recarray(1,dtype=dtype)
    rec["xvals"]=xvals
    rec["yvals"]=yvals
    rec["vvals"]=vvals
    rec["ellObs"]=ellObs
    return rec

def recToObs(rec):
    xvals=rec["xvals"].squeeze()
    yvals=rec["yvals"].squeeze()
    vvals=rec["vvals"].squeeze()
    ellObs=rec["ellObs"].squeeze()
    return (xvals,yvals,vvals,ellObs)

def statsToRec(inputPars,mp,kde,hw):
    dtype=[("inputPars",(inputPars.dtype.type,inputPars.shape)),("mp",(mp.dtype.type,mp.shape)),("kde",(kde.dtype.type,kde.shape)),("hw",(hw.dtype.type,hw.shape))]
    rec=np.recarray(1,dtype=dtype)
    rec["inputPars"]=inputPars
    rec["mp"]=mp
    rec["kde"]=kde
    rec["hw"]=hw
    return rec


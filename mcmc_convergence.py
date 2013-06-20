#! env python
import sim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import sys

def makeObs(nGal,numFib=6,fibRad=1,fibConfig="hexNoCen",sigma=30.,ellErr=np.array([0.1,10])):

    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]
    inputPars=sim.generateEnsemble(nGal,inputPriors,shearOpt=None)

    xvals,yvals=sim.getFiberPos(numFib,fibRad,fibConfig)
    vvals=np.array([sim.vmapModel(inputPars[ii,:],xvals,yvals) for ii in range(nGal)])
    ellObs=np.array([sim.ellModel(inputPars[ii,:]) for ii in range(nGal)])

    specNoise=np.random.randn(numFib*nGal).reshape((nGal,numFib))*sigma
    imNoise=np.random.randn(2*nGal).reshape(nGal,2)*ellErr
    vvals+=specNoise
    ellObs+=imNoise

    return (xvals,yvals,vvals,ellObs,inputPars)

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


if __name__ == "__main__":

    if(sys.argv[1]=="first"):
        nGal=int(sys.argv[2])
        numFib=6.
        fibRad=1.
        sigma=30.
        ellErr=np.array([0.1,10])
        xvals,yvals,vvals,ellObs,inputPars=makeObs(nGal,numFib=numFib,fibRad=fibRad)
        recObs=obsToRec(xvals,yvals,vvals,ellObs)
        recPars=sim.parsToRec(inputPars)
        sim.writeRec(recObs,"mcmc_convergence_obs.fits")
        sim.writeRec(recPars,"mcmc_convergence_inputPars.fits")
    else:
        recObs=sim.readRec("mcmc_convergence_obs.fits")
        recPars=sim.readRec("mcmc_convergence_inputPars.fits")
        xvals,yvals,vvals,ellObs=recToObs(recObs)
        inputPars=sim.recToPars(recPars)

    obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]

    numFib=len(xvals)
    nGal=len(vvals)
    nPars=len(obsPriors)
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    figExt="pdf"

    nBurn=np.array([5,5,10,100,200])
    nSteps=np.array([5,10,50,500,1000])
    nMCMC=len(nBurn)

    obsParsI=np.zeros((nMCMC,nGal,nPars))
    obsParsS=np.zeros_like(obsParsI)
    obsParsIS=np.zeros_like(obsParsI)
    
    for mm in range(nMCMC):
        print "************MCMC {}".format(mm)
        for ii in range(nGal):
            print "************Running Galaxy {}".format(ii)
            chains,lnprobs=sim.fitObs(vvals[ii,:],sigma,ellObs[ii,:],ellErr,obsPriors,fibRad=fibRad,addNoise=False,nBurn=nBurn[mm],nSteps=nSteps[mm])
            obsParsI[mm,ii,:]=sim.getMaxProb(chains[0],lnprobs[0])
            obsParsS[mm,ii,:]=sim.getMaxProb(chains[1],lnprobs[1])
            obsParsIS[mm,ii,:]=sim.getMaxProb(chains[2],lnprobs[2])
            print inputPars[ii,:]
            print obsParsI[mm,ii,:]
            print obsParsS[mm,ii,:]
            print obsParsIS[mm,ii,:]

            sim.contourPlotAll(chains,inputPars=inputPars[ii],smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename="mcmc_{}_{}.{}".format(nSteps[mm],ii,figExt))


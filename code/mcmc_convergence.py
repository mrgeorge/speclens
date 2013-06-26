#! env python
import sim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import sys

def makeObs(nGal,inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],disk_r=None,convOpt=None,atmos_fwhm=None,numFib=6,fibRad=1,fibConvolve=False,fibConfig="hexNoCen",sigma=30.,ellErr=np.array([10.,0.1]),seed=None):

    inputPars=sim.generateEnsemble(nGal,inputPriors,shearOpt=None,seed=seed)

    xvals,yvals=sim.getFiberPos(numFib,fibRad,fibConfig)
    if(convOpt is not None):
        if((type(disk_r) is int) | (type(disk_r) is float)):
            disk_r=np.repeat(disk_r,nGal)
        kernel=sim.makeConvolutionKernel(xvals,yvals,atmos_fwhm,fibRad,fibConvolve)
        vvals=np.array([sim.vmapObs(inputPars[ii,:],xvals,yvals,disk_r[ii],convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel) for ii in range(nGal)])
    else: # this is faster if we don't need to convolve with psf or fiber
        vvals=np.array([sim.vmapModel(inputPars[ii,:],xvals,yvals) for ii in range(nGal)])
    ellObs=np.array([sim.ellModel(inputPars[ii,:]) for ii in range(nGal)])

    return (xvals,yvals,vvals,ellObs,inputPars)


if __name__ == "__main__":

    figExt="pdf"
    plotDir="/data/mgeorge/speclens/plots"
    dataDir="/data/mgeorge/speclens/data"

    if(sys.argv[1]=="first"):
        print "creating new input files"
        nGal=int(sys.argv[2])
        numFib=6.
        fibRad=1.
        sigma=30.
        ellErr=np.array([10.,0.1])
        xvals,yvals,vvals,ellObs,inputPars=makeObs(nGal,numFib=numFib,fibRad=fibRad)
        recObs=obsToRec(xvals,yvals,vvals,ellObs)
        labels=np.array(["PA","b/a","vmax","g1","g2"])
        recPars=sim.parsToRec(inputPars,labels=labels)
        sim.writeRec(recObs,"{}/mcmc_convergence_obs.fits".format(dataDir))
        sim.writeRec(recPars,"{}/mcmc_convergence_inputPars.fits".format(dataDir))
    else:
        print "opening old input files"
        fibRad=1.
        sigma=30.
        ellErr=np.array([10.,0.1])
        recObs=sim.readRec("{}/mcmc_convergence_obs.fits".format(dataDir))
        recPars=sim.readRec("{}/mcmc_convergence_inputPars.fits".format(dataDir))
        xvals,yvals,vvals,ellObs=recToObs(recObs)
        labels=np.array(["PA","b/a","vmax","g1","g2"])
        inputPars=sim.recToPars(recPars,labels=labels)

    obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]

    numFib=len(xvals)
    nGal=len(vvals)
    nPars=len(obsPriors)
    nBurn=np.array([20,50,50,100,100])
    nSteps=np.array([100,200,300,500,700])
    nMCMC=len(nBurn)

    obsParsI=np.zeros((nMCMC,nGal,nPars))
    obsParsS=np.zeros_like(obsParsI)
    obsParsIS=np.zeros_like(obsParsI)

    atmos_fwhm=1.5
    disk_r=1.
    fibConvolve=True
    convOpt="pixel"
    
    for mm in range(nMCMC):
        print "************MCMC {}".format(mm)
        for ii in range(nGal):
            print "************Running Galaxy {}".format(ii)
            chains,lnprobs=sim.fitObs(vvals[ii,:],sigma,ellObs[ii,:],ellErr,obsPriors,fibRad=fibRad,addNoise=False,nBurn=nBurn[mm],nSteps=nSteps[mm],atmos_fwhm=atmos_fwhm,disk_r=disk_r,fibConvolve=fibConvolve,convOpt=convOpt)
            obsParsI[mm,ii,:]=sim.getMaxProb(chains[0],lnprobs[0])
            obsParsS[mm,ii,:]=sim.getMaxProb(chains[1],lnprobs[1])
            obsParsIS[mm,ii,:]=sim.getMaxProb(chains[2],lnprobs[2])
            print inputPars[ii,:]
            print obsParsI[mm,ii,:]
            print obsParsS[mm,ii,:]
            print obsParsIS[mm,ii,:]

            sim.contourPlotAll(chains,inputPars=inputPars[ii],smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename="{}/mcmc_{}_{}_conv.{}".format(plotDir,ii,nSteps[mm],figExt))


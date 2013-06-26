#! env python
import sim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import os

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

def runEnsemble(nGal,**kwargs):
    figExt="pdf"
    plotDir="/data/mgeorge/speclens/plots/"
    dataDir="/data/mgeorge/speclens/data/"
    subDir="opt_{}_{}_{}_{}_{}_{}".format(convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig)
    if(not(os.path.exists(dataDir+subDir))):
        os.makedirs(dataDir+subDir)
    if(not(os.path.exists(plotDir+subDir))):
        os.makedirs(plotDir+subDir)

    obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    nPars=len(obsPriors)
    obsParsI=np.zeros((nGal,nPars))
    obsParsS=np.zeros_like(obsParsI)
    obsParsIS=np.zeros_like(obsParsI)

    xvals,yvals,vvals,ellObs,inputPars=makeObs(nGal,**kwargs)
    sim.writeRec(sim.parsToRec(inputPars,labels=labels),dataDir+subDir+"/inputPars.fits")

    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)

        chains,lnprobs=sim.fitObs(vvals[ii,:],sigma,ellObs[ii,:],ellErr,obsPriors,disk_r=disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,addNoise=True,seed=seed)
        obsParsI[ii,:]=sim.getMaxProb(chains[0],lnprobs[0])
        obsParsS[ii,:]=sim.getMaxProb(chains[1],lnprobs[1])
        obsParsIS[ii,:]=sim.getMaxProb(chains[2],lnprobs[2])

        # write (and re-write) fits file output as we go
        sim.writeRec(sim.parsToRec(obsParsI[0:ii+1],labels=labels),dataDir+subDir+"/obsParsI.fits")
        sim.writeRec(sim.parsToRec(obsParsS[0:ii+1],labels=labels),dataDir+subDir+"/obsParsS.fits")
        sim.writeRec(sim.parsToRec(obsParsIS[0:ii+1],labels=labels),dataDir+subDir+"/obsParsIS.fits")

        sim.contourPlotAll(chains,inputPars=inputPars[ii],smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=plotDir+subDir+"/gal_{}.{}".format(ii,figExt))



if __name__ == "__main__":


    nGal=100
    disk_r=1.
    convOpt=np.array([None,"pixel"])
    atmos_fwhm=np.array([None,1.5])
    numFib=np.array([6,6])
    fibRad=np.array([1.,1.])
    fibConvolve=np.array([False,True])
    fibConfig=np.array(["hexNoCen","hexNoCen"])
    seed=7
    
    runEnsemble(nGal,disk_r=disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,seed=seed)
    



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


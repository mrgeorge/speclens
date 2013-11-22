#! env python
import sim
import io
import plot
import numpy as np
import os
import glob

def makeObs(inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],disk_r=None,convOpt=None,atmos_fwhm=None,numFib=6,fibRad=1,fibConvolve=False,fibConfig="hexNoCen",fibPA=None,sigma=30.,ellErr=np.array([10.,0.1]),seed=None):
# Generate input pars and observed values (w/o noise added) for a galaxy

    inputPars=sim.generateEnsemble(1,inputPriors,shearOpt=None,seed=seed).squeeze()

    # first get imaging observables (with noise) to get PA for slit/ifu alignment
    ellObs=sim.ellModel(inputPars)
    np.random.seed(100*seed)
    imNoise=np.random.randn(ellObs.size)*ellErr
    ellObs+=imNoise

    fibPA=ellObs[0] # align fibers to observed PA (no effect for circular fibers)

    pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig,fibPA)
    xvals,yvals=pos
    if(convOpt is not None):
        kernel=sim.makeConvolutionKernel(xvals,yvals,atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA)
        vvals=sim.vmapObs(inputPars,xvals,yvals,disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
    else: # this is faster if we don't need to convolve with psf or fiber
        vvals=sim.vmapModel(inputPars,xvals,yvals)

    specNoise=np.random.randn(numFib)*sigma
    vvals+=specNoise

    return (xvals,yvals,vvals,ellObs,inputPars)

def runGal(outDir,galID,inputPars,labels,vvals,sigma,ellObs,ellErr,obsPriors,figExt="pdf",**kwargs):
# call fitObs to run MCMC for a galaxy and save the resulting chains
# this is what create_qsub_galArr calls to run each galaxy

    chains,lnprobs=fit.fitObs(vvals,sigma,ellObs,ellErr,obsPriors,**kwargs)
    io.writeRec(io.chainToRec(chains[0],lnprobs[0],labels=labels),outDir+"/chainI_{:03d}.fits.gz".format(galID),compress="GZIP")
    io.writeRec(io.chainToRec(chains[1],lnprobs[1],labels=labels),outDir+"/chainS_{:03d}.fits.gz".format(galID),compress="GZIP")
    io.writeRec(io.chainToRec(chains[2],lnprobs[2],labels=labels),outDir+"/chainIS_{:03d}.fits.gz".format(galID),compress="GZIP")
    plot.contourPlotAll(chains,lnprobs=lnprobs,inputPars=inputPars,showMax=True,showPeakKDE=True,show68=True,smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

def create_qsub_galArr(outDir,nGal,inputPriors,convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig,sigma,ellErr):
# make a job array that generates a list of galaxies and runs each one as a separate job
    
    # text for qsub file
    jobHeader=("#!/clusterfs/riemann/software/Python/2.7.1/bin/python\n"
               "#PBS -j oe\n"
               "#PBS -m bea\n"
               "#PBS -M mgeorge@astro.berkeley.edu\n"
               "#PBS -V\n"
               "\n"
               "import os\n"
               "import numpy as np\n"
               "import ensemble\n"
               "\n"
               "os.system('date')\n"
               "os.system('echo `hostname`')\n"
        )
    jobTail="os.system('date')\n"
    
    jobFile="{}/qsub".format(outDir)

    if(convOpt is not None):
        convOpt="\"{}\"".format(convOpt)

    commands=("\n"
              "thisGal=int(os.environ['PBS_ARRAYID'])\n"
              "disk_r=1.\n"
              "obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]\n"
              "labels=np.array(['PA','b/a','vmax','g1','g2'])\n"
              "xvals,yvals,vvals,ellObs,inputPars=ensemble.makeObs(inputPriors={inputPriors},disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},numFib={numFib},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",sigma={sigma},ellErr={ellErr},seed=thisGal)\n"
              "ensemble.runGal(\"{outDir}\",thisGal,inputPars,labels,vvals,{sigma},ellObs,{ellErr},obsPriors,figExt=\"{figExt}\",disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",fibPA=ellObs[0],addNoise=False,seed=thisGal)\n\n".format(nGal=nGal,inputPriors=inputPriors,convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr.tolist(),outDir=outDir,figExt=figExt)
        )

    # create qsub file
    jf=open(jobFile,'w')
    jf.write(jobHeader)
    jf.write(commands)
    jf.write(jobTail)
    jf.close()

    return jobFile

def getScatter(dir,nGal,inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],labels=np.array(["PA","b/a","vmax","g1","g2"]),free=np.array([0,1,2,3,4]),fileType="chain"):
    
    chainIFiles=glob.glob(dir+"/chainI_*.fits.gz")
    chainSFiles=glob.glob(dir+"/chainS_*.fits.gz")
    chainISFiles=glob.glob(dir+"/chainIS_*.fits.gz")
    statsIFiles=glob.glob(dir+"/statsI_*.fits.gz")
    statsSFiles=glob.glob(dir+"/statsS_*.fits.gz")
    statsISFiles=glob.glob(dir+"/statsIS_*.fits.gz")

    dI=np.zeros((nGal,len(free)))
    dS=np.zeros_like(dI)
    dIS=np.zeros_like(dI)
    dIkde=np.zeros_like(dI)
    dSkde=np.zeros_like(dI)
    dISkde=np.zeros_like(dI)
    hwI=np.zeros_like(dI)
    hwS=np.zeros_like(dI)
    hwIS=np.zeros_like(dI)
    inputPars=np.zeros_like(dI)

    # check if inputPriors is a list of lists (i.e. different inputs for each galaxy)
    if(len(inputPriors) != len(labels)):
        if(len(inputPriors[0]) == len(labels)):
            listInput=True
        else:
            print "Error in getScatter: inputPriors should be a list of len={} or a list of such lists".format(len(labels))
    else:
        listInput=False
    
    for ii in range(nGal):
        print ii
        if(fileType=="chain"):
            filesExist=((dir+"chainI_{:03d}.fits.gz".format(ii) in chainIFiles) &
                        (dir+"chainS_{:03d}.fits.gz".format(ii) in chainSFiles) &
                        (dir+"chainIS_{:03d}.fits.gz".format(ii) in chainISFiles))
        elif(fileType=="stats"):
            filesExist=((dir+"statsI_{:03d}.fits.gz".format(ii) in statsIFiles) &
                        (dir+"statsS_{:03d}.fits.gz".format(ii) in statsSFiles) &
                        (dir+"statsIS_{:03d}.fits.gz".format(ii) in statsISFiles))
        if(filesExist):
            if(listInput):
                inputPars[ii,:]=sim.generateEnsemble(1,inputPriors[ii],shearOpt=None,seed=ii).squeeze()[free]
            else:
                inputPars[ii,:]=sim.generateEnsemble(1,inputPriors,shearOpt=None,seed=ii).squeeze()[free]

            if(fileType=="chain"):
                recI=io.readRec(dir+"chainI_{:03d}.fits.gz".format(ii))
                recS=io.readRec(dir+"chainS_{:03d}.fits.gz".format(ii))
                recIS=io.readRec(dir+"chainIS_{:03d}.fits.gz".format(ii))

                chainI=io.recToPars(recI,labels=labels[free])
                chainS=io.recToPars(recS,labels=labels[free])
                chainIS=io.recToPars(recIS,labels=labels[free])

                obsI=fit.getMaxProb(chainI,recI['lnprob'])
                obsS=fit.getMaxProb(chainS,recS['lnprob'])
                obsIS=fit.getMaxProb(chainIS,recIS['lnprob'])
            
                obsIkde=fit.getPeakKDE(chainI,obsI)
                obsSkde=fit.getPeakKDE(chainS,obsS)
                obsISkde=fit.getPeakKDE(chainIS,obsIS)

                hwI[ii,:]=fit.get68(chainI,opt="hw")
                hwS[ii,:]=fit.get68(chainS,opt="hw")
                hwIS[ii,:]=fit.get68(chainIS,opt="hw")
            elif(fileType=="stats"):
                statsI=io.readRec(dir+"statsI_{:03d}.fits.gz".format(ii))
                statsS=io.readRec(dir+"statsS_{:03d}.fits.gz".format(ii))
                statsIS=io.readRec(dir+"statsIS_{:03d}.fits.gz".format(ii))

                obsI=statsI['mp']
                obsS=statsS['mp']
                obsIS=statsIS['mp']

                obsIkde=statsI['kde']
                obsSkde=statsS['kde']
                obsISkde=statsIS['kde']

                hwI[ii,:]=statsI['hw']
                hwS[ii,:]=statsS['hw']
                hwIS[ii,:]=statsIS['hw']

            dI[ii,:]=obsI-inputPars[ii,:]
            dS[ii,:]=obsS-inputPars[ii,:]
            dIS[ii,:]=obsIS-inputPars[ii,:]
            dIkde[ii,:]=obsIkde-inputPars[ii,:]
            dSkde[ii,:]=obsSkde-inputPars[ii,:]
            dISkde[ii,:]=obsISkde-inputPars[ii,:]
        else:
            dI[ii,:]=np.repeat(np.nan,len(free))
            dS[ii,:]=np.repeat(np.nan,len(free))
            dIS[ii,:]=np.repeat(np.nan,len(free))
            hwI[ii,:]=np.repeat(np.nan,len(free))
            hwS[ii,:]=np.repeat(np.nan,len(free))
            hwIS[ii,:]=np.repeat(np.nan,len(free))

    good=~np.isnan(dI[:,0])

    print "STD Max"
    print np.std(dI[good,:],axis=0)
    print np.std(dS[good,:],axis=0)
    print np.std(dIS[good,:],axis=0)

    return (dI[good],dS[good],dIS[good],dIkde[good],dSkde[good],dISkde[good],hwI[good],hwS[good],hwIS[good],inputPars[good])

def getScatterAll():
    dataDir="/data/mgeorge/speclens/data/"

    nGal=100

    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]

    convOpt=np.append(np.repeat(None,6),np.repeat("pixel",12))
    nEnsemble=len(convOpt)
    atmos_fwhm=np.append(np.repeat(None,6),np.repeat([0.5,1.4],6))
    numFib=np.tile([6,10,100],6)
    fibRad=np.tile([1.,0.5,0.5],6)
    fibConvolve=np.append(np.repeat(False,6),np.repeat(True,12))
    fibConfig=np.tile(["hexNoCen","slit","ifu"],6)
    sigma=np.tile(np.repeat([5.,30.],3),3)
    ellErr=np.tile(np.array([10.,0.1]),nEnsemble).reshape((nEnsemble,2))

    for ii in range(nEnsemble):
        subDir="opt_{}_{}_{}_{}_{}_{:d}_{}".format(fibConfig[ii],numFib[ii],fibRad[ii],atmos_fwhm[ii],sigma[ii],bool(fibConvolve[ii]),convOpt[ii])

        print subDir
        getScatter(dataDir+subDir+"/",nGal)


if __name__ == "__main__":
# main creates a list of control pars and then calls create_qsub_galArr to make an ensemble of galaxies for each set of control pars
# this can be used to test different observing configurations (fiber sizes, shapes, PSFs, velocity errors, etc)

    figExt="pdf"
    dataDir="/data/mgeorge/speclens/data/"
    batch="-q batch"

    nGal=100
    #    disk_r=np.repeat(1.,nGal)

    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]

    convOpt=np.append(np.repeat(None,6),np.repeat("pixel",12))
    nEnsemble=len(convOpt)
    atmos_fwhm=np.append(np.repeat(None,6),np.repeat([0.5,1.4],6))
    numFib=np.tile([6,10,100],6)
    fibRad=np.tile([1.,0.5,0.5],6)
    fibConvolve=np.append(np.repeat(False,6),np.repeat(True,12))
    fibConfig=np.tile(["hexNoCen","slit","ifu"],6)
    sigma=np.tile(np.repeat([5.,30.],3),3)
    ellErr=np.tile(np.array([10.,0.1]),nEnsemble).reshape((nEnsemble,2))

    origcwd=os.getcwd()
    for ii in range(nEnsemble):
        subDir="opt_{}_{}_{}_{}_{}_{:d}_{}".format(fibConfig[ii],numFib[ii],fibRad[ii],atmos_fwhm[ii],sigma[ii],bool(fibConvolve[ii]),convOpt[ii])
        if(not(os.path.exists(dataDir+subDir))):
            os.makedirs(dataDir+subDir)
            os.makedirs(dataDir+subDir+"/plots")

        # move to job dir so log files are stored there
        os.chdir(dataDir+subDir)

        #        jobFile=create_qsub_ensemble(dataDir+subDir,nGal,disk_r,convOpt[ii],atmos_fwhm[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii],seed)
        #        os.system("qsub -VX {} {}".format(batch,jobFile))
        jobFile=create_qsub_galArr(dataDir+subDir,nGal,inputPriors,convOpt[ii],atmos_fwhm[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii],sigma[ii],ellErr[ii])
        os.system("qsub -VX -t 0-{} {} {}".format(nGal-1,batch,jobFile))

    os.chdir(origcwd)
    

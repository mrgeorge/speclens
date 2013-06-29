#! env python
import sim
import numpy as np
import os

def makeObs(inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],disk_r=None,convOpt=None,atmos_fwhm=None,numFib=6,fibRad=1,fibConvolve=False,fibConfig="hexNoCen",fibPA=None,sigma=30.,ellErr=np.array([10.,0.1]),seed=None):
# Generate input pars and observed values (w/o noise added) for a galaxy

    inputPars=sim.generateEnsemble(1,inputPriors,shearOpt=None,seed=seed).squeeze()

    # first get imaging observables (with noise) to get PA for slit/ifu alignment
    ellObs=sim.ellModel(inputPars)
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

    chains,lnprobs=sim.fitObs(vvals,sigma,ellObs,ellErr,obsPriors,**kwargs)
    sim.writeRec(sim.chainToRec(chains[0],lnprobs[0],labels=labels),outDir+"/chainI_{:03d}.fits".format(galID))
    sim.writeRec(sim.chainToRec(chains[1],lnprobs[1],labels=labels),outDir+"/chainS_{:03d}.fits".format(galID))
    sim.writeRec(sim.chainToRec(chains[2],lnprobs[2],labels=labels),outDir+"/chainIS_{:03d}.fits".format(galID))
    sim.contourPlotAll(chains,inputPars=inputPars,smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

def create_qsub_galArr(outDir,nGal,inputPriors,convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig,sigma,ellErr,seed):
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


if __name__ == "__main__":
# main creates a list of control pars and then calls create_qsub_galArr to make an ensemble of galaxies for each set of control pars
# this can be used to test different observing configurations (fiber sizes, shapes, PSFs, velocity errors, etc)

    figExt="pdf"
    dataDir="/data/mgeorge/speclens/data/"
    batch="-q batch"

    nGal=50
    #    disk_r=np.repeat(1.,nGal)
    seed=7

    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]

    convOpt=np.array([None,"pixel","pixel","pixel",None])
    nEnsemble=len(convOpt)
    atmos_fwhm=np.array([None,1.5,1.5,0.5,None])
    numFib=np.array([6,6,10,100,100])
    fibRad=np.array([1.,1.,0.5,0.5,0.5])
    fibConvolve=np.array([False,True,True,True,False])
    fibConfig=np.array(["hexNoCen","hexNoCen","slit","ifu","ifu"])
    sigma=np.array([30.,30.,30.,5.,5.])
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
        jobFile=create_qsub_galArr(dataDir+subDir,nGal,inputPriors,convOpt[ii],atmos_fwhm[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii],sigma[ii],ellErr[ii],seed)
        os.system("qsub -VX -t 0-{} {} {}".format(nGal-1,batch,jobFile))

    os.chdir(origcwd)
    

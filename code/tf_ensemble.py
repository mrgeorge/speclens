#! env python
import sim
import numpy as np
import os
import glob

# try to reproduce Eric's noise estimates from Tully-Fisher scatter
# fix PA, g1, g2 and estimate shape from sin(i) given slit observables
# for an appropriate prior on vmax, we should get out a similar scatter in sin(i) and thus on the intrinsic shape


def makeObs(inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],disk_r=None,convOpt=None,atmos_fwhm=None,numFib=6,fibRad=1,fibConvolve=False,fibConfig="hexNoCen",fibPA=None,sigma=30.,ellErr=np.array([10.,0.1]),seed=None):
# Generate input pars and observed values (w/o noise added) for a galaxy

    inputPars=sim.generateEnsemble(1,inputPriors,shearOpt=None,seed=seed).squeeze()

    # first get imaging observables (with noise) to get PA for slit/ifu alignment
    ellObs=sim.ellModel(inputPars)
    np.random.seed(100*seed)

# NOTE - imNoise IS TURNED OFF!!
    #    imNoise=np.random.randn(ellObs.size)*ellErr
    #    ellObs+=imNoise

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
    sim.writeRec(sim.chainToRec(chains[0],lnprobs[0],labels=labels),outDir+"/chainI_{:03d}.fits.gz".format(galID),compress="GZIP")
    sim.writeRec(sim.chainToRec(chains[1],lnprobs[1],labels=labels),outDir+"/chainS_{:03d}.fits.gz".format(galID),compress="GZIP")
    sim.writeRec(sim.chainToRec(chains[2],lnprobs[2],labels=labels),outDir+"/chainIS_{:03d}.fits.gz".format(galID),compress="GZIP")
    sim.contourPlotAll(chains,inputPars=inputPars,smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

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
               "import tf_ensemble\n"
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
              "labels=np.array(['PA','b/a','vmax','g1','g2'])\n"
              "xvals,yvals,vvals,ellObs,inputPars=tf_ensemble.makeObs(inputPriors={inputPriors},disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},numFib={numFib},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",sigma={sigma},ellErr={ellErr},seed=thisGal)\n"
              "obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]\n"
              "free=np.array([0,1,2,3,4])\n"
              "tf_ensemble.runGal(\"{outDir}\",thisGal,inputPars[free],labels[free],vvals,{sigma},ellObs,{ellErr},obsPriors,figExt=\"{figExt}\",disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",fibPA=ellObs[0],addNoise=False,seed=thisGal)\n\n".format(nGal=nGal,inputPriors=inputPriors,convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr.tolist(),outDir=outDir,figExt=figExt)
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

    nGal=100

    inputPriors=[[0,360],[0,1],150,(0.,0.02),(0.,0.02)]

    convOpt=np.array([None,None,"pixel"])
    nEnsemble=len(convOpt)
    atmos_fwhm=np.array([None,None,1.4])
    numFib=np.array([10,6,6])
    fibRad=np.array([0.5,1,1])
    fibConvolve=np.array([False,False,True])
    fibConfig=np.array(["slit","hexNoCen","hexNoCen"])
    sigma=np.array([5.,5.,30])
    ellErr=np.tile(np.array([3.,0.01]),nEnsemble).reshape((nEnsemble,2))

    origcwd=os.getcwd()
    for ii in range(nEnsemble):
        subDir="tfshearpa_opt_{}_{}_{}_{}_{}_{:d}_{}".format(fibConfig[ii],numFib[ii],fibRad[ii],atmos_fwhm[ii],sigma[ii],bool(fibConvolve[ii]),convOpt[ii])
        if(not(os.path.exists(dataDir+subDir))):
            os.makedirs(dataDir+subDir)
            os.makedirs(dataDir+subDir+"/plots")

        # move to job dir so log files are stored there
        os.chdir(dataDir+subDir)

        jobFile=create_qsub_galArr(dataDir+subDir,nGal,inputPriors,convOpt[ii],atmos_fwhm[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii],sigma[ii],ellErr[ii])
        os.system("qsub -VX -t 0-{} {} {}".format(nGal-1,batch,jobFile))

    os.chdir(origcwd)
    

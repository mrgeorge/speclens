#! env python
import sim
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# try to reproduce Eric's noise estimates from Tully-Fisher scatter
# fix PA, g1, g2 and estimate shape from sin(i) given slit observables
# for an appropriate prior on vmax, we should get out a similar scatter in sin(i) and thus on the intrinsic shape


def makeObs(inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],disk_r=None,convOpt=None,atmos_fwhm=None,atmosNoise=0.,posNoise=0.,numFib=6,fibRad=1,fibConvolve=False,fibConfig="hexNoCen",fibPA=None,sigma=30.,ellErr=np.array([10.,0.1]),seed=None):
# Generate input pars and observed values (w/o noise added) for a galaxy

    inputPars=sim.generateEnsemble(1,inputPriors,shearOpt=None,seed=seed).squeeze()

    # first get imaging observables (with noise) to get PA for slit/ifu alignment
    ellObs=sim.ellModel(inputPars)
    if(seed is not None):
        np.random.seed(100*seed)

    imNoise=np.random.randn(ellObs.size)*ellErr
    ellObs+=imNoise

    fibPA=ellObs[0] # align fibers to observed PA (no effect for circular fibers)

    pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig,fibPA)
    xvals,yvals=pos
    if(convOpt is not None):
        if((atmosNoise==0.) & (posNoise==0.)):
            kernel=sim.makeConvolutionKernel(xvals,yvals,atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA)
            vvals=sim.vmapObs(inputPars,xvals,yvals,disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
        elif((atmosNoise!=0.) & (posNoise==0.)): # generate a different kernel for each observation
            vvals=np.zeros(numFib)
            for ii in range(numFib):
                thisPSF=atmos_fwhm+np.random.randn(1)*atmosNoise
                kernel=sim.makeConvolutionKernel(np.array([xvals[ii]]),np.array([yvals[ii]]),thisPSF,fibRad,fibConvolve,fibShape,fibPA)
                vvals[ii]=sim.vmapObs(inputPars,np.array([xvals[ii]]),np.array([yvals[ii]]),disk_r,convOpt=convOpt,atmos_fwhm=thisPSF,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
        elif((atmosNoise==0.) & (posNoise!=0.)): # generate a different kernel for each observation
            vvals=np.zeros(numFib)
            for ii in range(numFib):
                thisX=xvals[ii]+np.random.randn(1)*posNoise/np.sqrt(2)
                thisY=yvals[ii]+np.random.randn(1)*posNoise/np.sqrt(2)
                kernel=sim.makeConvolutionKernel(np.array([thisX]),np.array([thisY]),atmos_fwhm,fibRad,fibConvolve,fibShape,fibPA)
                vvals[ii]=sim.vmapObs(inputPars,np.array([thisX]),np.array([thisY]),disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,kernel=kernel)
                
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
    sim.contourPlotAll(chains,lnprobs=lnprobs,inputPars=inputPars,showMax=True,showPeakKDE=True,show68=True,smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

def create_qsub_galArr(outDir,inputPriors,convOpt,atmos_fwhm,atmosNoise,posNoise,numFib,fibRad,fibConvolve,fibConfig,sigma,ellErr):
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
              "disk_r=2.9\n"
              "labels=np.array(['PA','b/a','vmax','g1','g2'])\n"
              "xvals,yvals,vvals,ellObs,inputPars=tf_ensemble.makeObs(inputPriors={inputPriors},disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},atmosNoise={atmosNoise},posNoise={posNoise},numFib={numFib},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",sigma={sigma},ellErr={ellErr},seed=thisGal)\n"
#              "obsPriors=[[0,360],[0,1],(150,20),[-0.5,0.5],[-0.5,0.5]]\n"
              "obsPriors=[[0,360],[0,1],[0,300],0,0]\n"
#              "obsPriors=[[0,360],[0,1],[50,250],[-0.5,0.5],[-0.5,0.5]]\n"
              "free=np.array([0,1,2])\n"
              "tf_ensemble.runGal(\"{outDir}\",thisGal,inputPars[free],labels[free],vvals,{sigma},ellObs,{ellErr},obsPriors,figExt=\"{figExt}\",disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",fibPA=ellObs[0],addNoise=False,seed=thisGal,nSteps=100)\n\n".format(inputPriors=inputPriors,convOpt=convOpt,atmos_fwhm=atmos_fwhm,atmosNoise=atmosNoise,posNoise=posNoise,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr.tolist(),outDir=outDir,figExt=figExt)
        )

    # create qsub file
    jf=open(jobFile,'w')
    jf.write(jobHeader)
    jf.write(commands)
    jf.write(jobTail)
    jf.close()

    return jobFile

def getVSini(dir="/home/mgeorge/speclens/data/data/tf_vmax_hexNoCen_6_1.0_1.4_30.0_1_pixel/",nGal=100):
    inputPriors=[[0,360],[0,1],150,0,0]
    unc=np.zeros(nGal)
    resid=np.zeros_like(unc)
    inputQ=np.zeros_like(unc)
    for ii in range(nGal):
        inputPars=sim.generateEnsemble(1,inputPriors,shearOpt=None,seed=ii).squeeze()
        inputQ[ii]=inputPars[1]
        vsiniTrue=inputPars[2]*np.sin(sim.getInclination(inputPars[1]))

        filename=dir+"chainIS_{:03d}.fits.gz".format(ii)
        if(os.path.exists(filename)):
            chain=sim.readRec(filename)
            vsini=chain['vmax']*np.sin(sim.getInclination(chain['b/a']))

            unc[ii]=np.std(vsini)
            #            resid[ii]=np.mean(vsini)-vsiniTrue
            resid[ii]=sim.getPeakKDE(vsini,vsiniTrue)-vsiniTrue

        else:
            unc[ii]=np.nan
            resid[ii]=np.nan

    good=~np.isnan(unc)

    print np.std(unc[good]), np.median(unc[good])
    print np.std(resid[good]), np.median(resid[good])
    
    plt.hist(unc[good],bins=10)
    plt.show()
    plt.hist(resid[good],bins=10)
    plt.show()

if __name__ == "__main__":
# main creates a list of control pars and then calls create_qsub_galArr to make an ensemble of galaxies for each set of control pars
# this can be used to test different observing configurations (fiber sizes, shapes, PSFs, velocity errors, etc)

    figExt="pdf"
    dataDir="/data/mgeorge/speclens/data/"
    batch="-q batch -l walltime=24:00:00"
    #    batch="-q big"

    galStart=0
    nGal=100

    #    inputPriors=[[0,360],[0,1],150,(0.,0.02),(0.,0.02)]
    inputPriors=[[0,360],[0,1],150,0,0]

    convOpt=np.array(["pixel"])
    nEnsemble=len(convOpt)
    atmos_fwhm=np.array([1.4])
    atmosNoise=np.array([0.])
    posNoise=np.array([0.])
    numFib=np.array([5])
    fibRad=np.array([1.])
    fibConvolve=np.array([True])
    fibConfig=np.array(["hexNoCen"])
    sigma=np.array([30.])
    ellErr=np.tile(np.array([10.,0.06]),nEnsemble).reshape((nEnsemble,2))

    origcwd=os.getcwd()
    for ii in range(nEnsemble):
        subDir="tf_vmax_{}_{}_{}_{}_{}_{:d}_{}".format(fibConfig[ii],numFib[ii],fibRad[ii],atmos_fwhm[ii],sigma[ii],bool(fibConvolve[ii]),convOpt[ii])
        if(not(os.path.exists(dataDir+subDir))):
            os.makedirs(dataDir+subDir)
            os.makedirs(dataDir+subDir+"/plots")

        # move to job dir so log files are stored there
        os.chdir(dataDir+subDir)

        jobFile=create_qsub_galArr(dataDir+subDir,inputPriors,convOpt[ii],atmos_fwhm[ii],atmosNoise[ii],posNoise[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii],sigma[ii],ellErr[ii])
        os.system("qsub -VX -t {}-{} {} {}".format(galStart,galStart+nGal-1,batch,jobFile))

    os.chdir(origcwd)
    

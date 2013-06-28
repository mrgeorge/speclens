#! env python
import sim
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

def runGal(outDir,galID,inputPars,labels,vvals,sigma,ellObs,ellErr,obsPriors,figExt="pdf",**kwargs):
    chains,lnprobs=sim.fitObs(vvals,sigma,ellObs,ellErr,obsPriors,**kwargs)
    sim.writeRec(sim.chainToRec(chains[0],lnprobs[0],labels=labels),outDir+"/chainI_{:03d}.fits".format(galID))
    sim.writeRec(sim.chainToRec(chains[1],lnprobs[1],labels=labels),outDir+"/chainS_{:03d}.fits".format(galID))
    sim.writeRec(sim.chainToRec(chains[2],lnprobs[2],labels=labels),outDir+"/chainIS_{:03d}.fits".format(galID))
    sim.contourPlotAll(chains,inputPars=inputPars[ii],smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

def runEnsemble(outDir,nGal,inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)],disk_r=None,convOpt=None,atmos_fwhm=None,numFib=6,fibRad=1,fibConvolve=False,fibConfig="hexNoCen",sigma=30.,ellErr=np.array([10.,0.1]),seed=None,figExt="pdf"):
    obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    nPars=len(obsPriors)
    obsParsI=np.zeros((nGal,nPars))
    obsParsS=np.zeros_like(obsParsI)
    obsParsIS=np.zeros_like(obsParsI)

    xvals,yvals,vvals,ellObs,inputPars=makeObs(nGal,inputPriors=inputPriors,disk_r=disk_r,convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr,seed=seed)
    sim.writeRec(sim.parsToRec(inputPars,labels=labels),outDir+"/inputPars.fits")

    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)
        runGal(outDir,ii,inputPars[ii],labels,vvals[ii,:],sigma,ellObs[ii,:],ellErr,obsPriors,figExt=figExt,disk_r=disk_r[ii],convOpt=convOpt,atmos_fwhm=atmos_fwhm,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,addNoise=True,seed=ii)

def create_qsub_galArr(outDir,nGal,inputPriors,convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig,sigma,ellErr,seed):
    
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
              "disk_r=np.repeat(1.,{nGal})\n"
              "obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]\n"
              "labels=np.array(['PA','b/a','vmax','g1','g2'])\n"
              "xvals,yvals,vvals,ellObs,inputPars=ensemble.makeObs({nGal:d},inputPriors={inputPriors},disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},numFib={numFib},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",sigma={sigma},ellErr={ellErr},seed={seed})\n"
              "ensemble.runGal(\"{outDir}\",thisGal,inputPars[thisGal],labels,vvals[thisGal,:],{sigma},ellObs[thisGal,:],{ellErr},obsPriors,figExt=\"{figExt}\",disk_r=disk_r[thisGal],convOpt={convOpt},atmos_fwhm={atmos_fwhm},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",addNoise=True,seed=thisGal)\n\n".format(nGal=nGal,inputPriors=inputPriors,convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr.tolist(),seed=seed,outDir=outDir,figExt=figExt)
        )

    # create qsub file
    jf=open(jobFile,'w')
    jf.write(jobHeader)
    jf.write(commands)
    jf.write(jobTail)
    jf.close()

    return jobFile



def create_qsub_ensemble(outDir,nGal,disk_r,convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig,seed):
    
    # text for qsub file
    jobHeader=("#!/clusterfs/riemann/software/Python/2.7.1/bin/python\n"
               "#PBS -j oe\n"
               "#PBS -m bea\n"
               "#PBS -M mgeorge@astro.berkeley.edu\n"
               "#PBS -V\n"
               "\n"
               "import os\n"
               "import ensemble\n"
               "\n"
               "os.system('date')\n"
               "os.system('echo `hostname`')\n"
        )
    jobTail="os.system('date')\n"
    
    jobFile="{}/qsub".format(outDir)

    if(convOpt is not None):
        convOpt="\"{}\"".format(convOpt)

    command="ensemble.runEnsemble(\"{}\",{},disk_r={},convOpt={},atmos_fwhm={},numFib={},fibRad={},fibConvolve={},fibConfig=\"{}\",seed={})\n".format(outDir,nGal,disk_r.tolist(),convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig,seed)

    # create qsub file
    jf=open(jobFile,'w')
    jf.write(jobHeader)
    jf.write(command)
    jf.write(jobTail)
    jf.close()

    return jobFile

if __name__ == "__main__":

    figExt="pdf"
    dataDir="/data/mgeorge/speclens/data/"
    batch="-q batch"

    nGal=3
    #    disk_r=np.repeat(1.,nGal)
    seed=7

    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]

    convOpt=np.array([None,"pixel"])
    nEnsemble=len(convOpt)
    atmos_fwhm=np.array([None,1.5])
    numFib=np.array([6,6])
    fibRad=np.array([1.,1.])
    fibConvolve=np.array([False,True])
    fibConfig=np.array(["hexNoCen","hexNoCen"])
    sigma=np.repeat(30.,nEnsemble)
    ellErr=np.tile(np.array([10.,0.1]),nEnsemble).reshape((nEnsemble,2))

    origcwd=os.getcwd()
    for ii in range(nEnsemble):
        subDir="opt_{}_{}_{}_{}_{}_{}".format(convOpt[ii],atmos_fwhm[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii])
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
    

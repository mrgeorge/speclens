#! env python
import sim
import numpy as np
import os
import fitsio
import esutil
import galsim
import galsim.integ

# select galaxies behind Coma
# generate ensemble of disks with shears from their positions
# measure noisy shears and recover tangential shear profile vs distance

def selectComaTargets():
    cat=fitsio.read("/data/mgeorge/speclens/data/coma_sdss_cas.fits")
    comaRA=194.898779 # J200 for NGC 4874 from NED
    comaDec=27.959268 # Note - Kubo gives different (wrong?) coords
    plateRad=1.5 # deg
    sel=((cat['z'] > 0.1) &
         (cat['z'] < 0.8) &
         (cat['zErr'] < 0.4) &
         (cat['dered_r'] > 18.) &
         (cat['dered_r'] < 21.) &
         (cat['dered_g'] < 21.) &
         (cat['expRad_r'] > 2) &
         (cat['expRad_r'] < 10.) &
         (esutil.coords.sphdist(comaRA,comaDec,cat['ra'],cat['dec']) < plateRad)
        )

    # TO DO - use circularized radii (sqrt(ab)), and use known-z catalog

    cat=cat[sel]

    m200c=1.88e15 # Kubo
    conc=3.84 # Kubo
    redshift=0.0239 # NED NGC 4874
    nfw=galsim.NFWHalo(mass=m200c,conc=conc,redshift=redshift,halo_pos=(0,0),omega_m=0.3,omega_lam=0.7)

    dtype=[(label,float) for label in ("disk_r","g1","g2")]
    rec=np.recarray(len(cat),dtype=dtype)
    for ii in range(len(cat)):
        gal=galsim.Sersic(n=1,scale_radius=float(cat[ii]['expRad_r']))
        rec[ii]['disk_r']=gal.getHalfLightRadius()
        rec[ii]['g1'],rec[ii]['g2']=nfw.getShear(pos=(cat[ii]['ra']-comaRA,cat[ii]['dec']-comaDec),units=galsim.degrees,z_s=cat[ii]['z'])

    return rec


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

    chains,lnprobs=sim.fitObs(vvals,sigma,ellObs,ellErr,obsPriors,**kwargs)
    sim.writeRec(sim.chainToRec(chains[0],lnprobs[0],labels=labels),outDir+"/chainI_{:03d}.fits.gz".format(galID),compress="GZIP")
    sim.writeRec(sim.chainToRec(chains[1],lnprobs[1],labels=labels),outDir+"/chainS_{:03d}.fits.gz".format(galID),compress="GZIP")
    sim.writeRec(sim.chainToRec(chains[2],lnprobs[2],labels=labels),outDir+"/chainIS_{:03d}.fits.gz".format(galID),compress="GZIP")
    sim.contourPlotAll(chains,inputPars=inputPars,smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

def create_qsub_galArr(outDir,convOpt,atmos_fwhm,numFib,fibRad,fibConvolve,fibConfig,sigma,ellErr):
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
               "import coma\n"
               "\n"
               "os.system('date')\n"
               "os.system('echo `hostname`')\n"
        )
    jobTail="os.system('date')\n"
    
    jobFile="{}/qsub".format(outDir)

    if(convOpt is not None):
        convOpt="\"{}\"".format(convOpt)

    commands=("\n"
              "gals=coma.selectComaTargets()\n"
              "thisGal=int(os.environ['PBS_ARRAYID'])\n"
              "disk_r=gals[thisGal]['disk_r']\n"
              "inputPriors=[[0,360],[0,1],150,gals[thisGal]['g1'],gals[thisGal]['g2']]\n"
              "labels=np.array(['PA','b/a','vmax','g1','g2'])\n"
              "xvals,yvals,vvals,ellObs,inputPars=coma.makeObs(inputPriors=inputPriors,disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},numFib={numFib},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",sigma={sigma},ellErr={ellErr},seed=thisGal)\n"
              "obsPriors=[[0,360],[0,1],(150,20),[-0.5,0.5],[-0.5,0.5]]\n"
              "free=np.array([0,1,2,3,4])\n"
              "coma.runGal(\"{outDir}\",thisGal,inputPars[free],labels[free],vvals,{sigma},ellObs,{ellErr},obsPriors,figExt=\"{figExt}\",disk_r=disk_r,convOpt={convOpt},atmos_fwhm={atmos_fwhm},fibRad={fibRad},fibConvolve={fibConvolve},fibConfig=\"{fibConfig}\",fibPA=ellObs[0],addNoise=False,seed=thisGal)\n\n".format(convOpt=convOpt,atmos_fwhm=atmos_fwhm,numFib=numFib,fibRad=fibRad,fibConvolve=fibConvolve,fibConfig=fibConfig,sigma=sigma,ellErr=ellErr.tolist(),outDir=outDir,figExt=figExt)
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

    gals=selectComaTargets()

    galStart=0
    nGal=len(gals)

    convOpt=np.array(["pixel"])
    nEnsemble=len(convOpt)
    atmos_fwhm=np.array([1.4])
    numFib=np.array([6])
    fibRad=np.array([1.])
    fibConvolve=np.array([True])
    fibConfig=np.array(["hexNoCen"])
    sigma=np.array([30.])
    ellErr=np.tile(np.array([3.,0.01]),nEnsemble).reshape((nEnsemble,2))

    origcwd=os.getcwd()
    for ii in range(nEnsemble):
        subDir="coma_{}_{}_{}_{}_{}_{:d}_{}".format(fibConfig[ii],numFib[ii],fibRad[ii],atmos_fwhm[ii],sigma[ii],bool(fibConvolve[ii]),convOpt[ii])
        if(not(os.path.exists(dataDir+subDir))):
            os.makedirs(dataDir+subDir)
            os.makedirs(dataDir+subDir+"/plots")

        # move to job dir so log files are stored there
        os.chdir(dataDir+subDir)

        jobFile=create_qsub_galArr(dataDir+subDir,convOpt[ii],atmos_fwhm[ii],numFib[ii],fibRad[ii],fibConvolve[ii],fibConfig[ii],sigma[ii],ellErr[ii])
        os.system("qsub -VX -t {}-{} {} {}".format(galStart,galStart+nGal-1,batch,jobFile))

    os.chdir(origcwd)
    

#! env python
import matplotlib.pyplot as plt
import sim
import numpy as np
import os
import fitsio
import esutil
import galsim
import galsim.integ
import re
import astropy.io.ascii
import ensemble

cosmo=esutil.cosmology.Cosmo()

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
plt.rc('text', usetex=True)
plt.rc('axes',linewidth=1.5)

# select galaxies behind Coma
# generate ensemble of disks with shears from their positions
# measure noisy shears and recover tangential shear profile vs distance

def parseRAStr(raStr):
    hms=re.split('h|m|s',raStr)
    hh,mm,ss=float(hms[0]),float(hms[1]),float(hms[2])
    raDeg=(hh + mm/60. + ss/3600.)*15
    return raDeg

def parseDecStr(decStr):
    dms=re.split('d|m|s',decStr)
    dd,mm,ss=float(dms[0]),float(dms[1]),float(dms[2])
    decDeg=(dd + mm/60. + ss/3600.)
    return decDeg
    
def getNedZ():
    rec=astropy.io.ascii.read("/data/mgeorge/speclens/data/coma_z_ned.clean")
    ra=np.zeros(len(rec))
    dec=np.zeros_like(ra)
    z=np.zeros_like(ra)
    for ii in range(len(rec)):
        ra[ii]=parseRAStr(rec['col1'][ii])
        dec[ii]=parseDecStr(rec['col2'][ii])
        z[ii]=rec['col3'][ii]

    good=(z>0)
    return (ra[good],dec[good],z[good])
    
def selectComaTargets():
    cat=fitsio.read("/data/mgeorge/speclens/data/coma_sdss_cas.fits")
    comaRA=194.898779 # J200 for NGC 4874 from NED
    comaDec=27.959268 # Note - Kubo gives different (wrong?) coords
    plateRad=1.5 # deg
    sel=((cat['z'] > 0.1) &
         (cat['z'] < 0.8) &
         (cat['zErr'] > 0.) &
         (cat['zErr'] < 0.1) &
         (cat['dered_r'] > 18.) &
         (cat['dered_r'] < 21.) &
         (cat['dered_g'] < 21.) &
         (cat['expRad_r']*np.sqrt(cat['expAB_r']) > 1.4) &
         (cat['expRad_r']*np.sqrt(cat['expAB_r']) < 10.) &
         (esutil.coords.sphdist(comaRA,comaDec,cat['ra'],cat['dec']) < plateRad)
        )

    # TO DO - use known-z catalog

    cat=cat[sel]

    m200c=1.88e15 # Kubo
    conc=3.84 # Kubo
    redshift=0.0239 # NED NGC 4874
    nfw=galsim.NFWHalo(mass=m200c,conc=conc,redshift=redshift,halo_pos=(0,0),omega_m=0.3,omega_lam=0.7)

    dtype=[(label,float) for label in ("ra","dec","z","disk_r","g1","g2")]
    rec=np.recarray(len(cat),dtype=dtype)
    rec['ra']=cat['ra']
    rec['dec']=cat['dec']
    rec['z']=cat['z']
    for ii in range(len(cat)):
        gal=galsim.Sersic(n=1,scale_radius=float(cat[ii]['expRad_r']*np.sqrt(cat[ii]['expAB_r'])))
        rec[ii]['disk_r']=gal.getHalfLightRadius()
        rec[ii]['g1'],rec[ii]['g2']=nfw.getShear(pos=(cat[ii]['ra']-comaRA,cat[ii]['dec']-comaDec),units=galsim.degrees,z_s=cat[ii]['z'])

    return rec

def plotTargetDistr(gals):
    plt.hist(gals['disk_r'],bins=20,histtype="step")
    plt.xlim((0,10))
    plt.xlabel(r"$\theta_{1/2}$ (arcsec)")
    plt.ylabel("N")
    plt.savefig("coma_size.png")
    plt.show()

    plt.hist(gals['z'],bins=20,histtype="step")
    plt.xlim((0,0.5))
    plt.xlabel("$z_{phot}$")
    plt.ylabel("N")
    plt.savefig("coma_redshifts.png")
    plt.show()

def shearProfile(ra,dec,g1,g2,weight):
    comaRA=194.898779 # J200 for NGC 4874 from NED
    comaDec=27.959268 # Note - Kubo gives different (wrong?) coords

    phi=np.arctan2(dec-comaDec,ra-comaRA)
    dist=esutil.coords.sphdist(ra,dec,comaRA,comaDec) # degrees
    gtan=-(g1*np.cos(2.*phi) + g2*np.sin(2.*phi))

    bins=np.array([0,0.4,0.8,1.2,1.6]) # degrees
    nBins=len(bins)-1
    gtanAvg=np.zeros(nBins)
    gtanErr=np.zeros_like(gtanAvg)
    distAvg=np.zeros_like(gtanAvg)

    for ii in range(nBins):
        sel=((dist > bins[ii]) &
             (dist < bins[ii+1]))
        nSel=len(sel.nonzero()[0])
        gtanAvg[ii]=np.sum(gtan[sel]*weight[sel])/np.sum(weight[sel])
        gtanErr[ii]=np.std(gtan[sel]*weight[sel])/np.sum(weight[sel])
        distAvg[ii]=np.mean(dist[sel])

    return (distAvg,gtanAvg,gtanErr)

def plotShearProfile(distAvgs,gtanAvgs,gtanErrs,colors=['black','red','blue']):
    for ii in range(len(distAvgs)):
        plt.errorbar(distAvgs[ii],gtanAvgs[ii],yerr=gtanErrs[ii],c=colors[ii],ecolor=colors[ii])
    plt.show()

def plotModelShear():
    bins=np.array([0,0.5,1.0,1.5]) # degrees
    nBins=len(bins)-1
    distAvg=0.5*(bins[:-1]+bins[1:])
    binArea=np.pi*(bins[1:]**2 - bins[:-1]**2) # deg^2

    distModel=np.linspace(np.min(bins),np.max(bins),num=100)

    m200c=1.88e15 # Kubo
    conc=3.84 # Kubo
    redshift=0.0239 # NED NGC 4874
    nfw=galsim.NFWHalo(mass=m200c,conc=conc,redshift=redshift,halo_pos=(0,0),omega_m=0.3,omega_lam=0.7)
    z_s=0.1

    nKubo=2.7e5/200. # N/deg^2
    nBOSS=120 # N/deg^2

    sigmaKubo=0.37
    sigmaI=0.37
    sigmaIS=0.15

    gtanKubo=np.zeros(nBins)
    for ii in range(nBins):
        gtanKubo[ii],zero=nfw.getShear(pos=(distAvg[ii],0),units=galsim.degrees,z_s=z_s)
    gtanKubo*=-1

    gtanModel=np.zeros(len(distModel))
    for ii in range(len(distModel)):
        gtanModel[ii],zero=nfw.getShear(pos=(distModel[ii],0),units=galsim.degrees,z_s=z_s)
    gtanModel*=-1

    gerrKubo=sigmaKubo/np.sqrt(2.*(1.-sigmaKubo**2) * nKubo * binArea)
    gerrI=sigmaI/np.sqrt(2.*(1.-sigmaI**2) * nBOSS * binArea)
    gerrIS=sigmaIS/np.sqrt(2.*(1.-sigmaIS**2) * nBOSS * binArea)

    offset=0.08*(bins[1]-bins[0])
    colors=["black","red","blue"]
    plt.plot(distModel,gtanModel,color="black")
    plt.errorbar(distAvg-offset,gtanKubo,yerr=gerrKubo,ecolor=colors[0],mfc=colors[0],label=r"Kubo $\sigma_{SN}=0.37$, n=1350 deg$^{-2}$",fmt='o')
    plt.errorbar(distAvg,gtanKubo,yerr=gerrI,ecolor=colors[1],mfc=colors[1],label=r"Imaging $\sigma_{SN}=0.37$, n=120 deg$^{-2}$",fmt='s')
    plt.errorbar(distAvg+offset,gtanKubo,yerr=gerrIS,ecolor=colors[2],mfc=colors[2],label=r"Im+Spec $\sigma_{SN}=0.15$, n=120 deg$^{-2}$",fmt='^')

    plt.legend(prop={'size':14},numpoints=1)
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel(r"$\gamma_{t}$")
    plt.xlim((bins[0],bins[-1]))
    plt.ylim((-0.019,0.049))
    plt.gcf().subplots_adjust(left=0.15)

    radTicks=np.array([0,0.5,1.,1.5])
    ax2=plt.gca().twiny()
    ax2.set_xlim(bins[0],bins[-1])
    ax2.set_xticklabels(radTicks)
    ax2.set_xticks(np.rad2deg(radTicks/cosmo.Da(0,redshift)))
    ax2.set_xlabel(r"R (Mpc)")

    plt.savefig("coma_radial.pdf")
    plt.show()

def plotZDep():
    gals=selectComaTargets()
    zbins=[0.1,0.2,0.3,0.4]
    nzBins=len(zbins)-1

    zNorm=0.1
    redshift=0.0239 # NED NGC 4874
    sciNorm=cosmo.sigmacritinv(redshift,0.1)
    zModel=np.linspace(zbins[0],zbins[-1],num=50)
    sciModel=cosmo.sigmacritinv(redshift,zModel)/sciNorm

    zAvg=np.zeros(nzBins)
    sciAvg=np.zeros(nzBins)
    sciErr=np.zeros(nzBins)
    sigmaIS=0.12
    for ii in range(nzBins):
        sel=((gals['z']>zbins[ii]) & (gals['z']<zbins[ii+1]))
        zAvg[ii]=np.mean(gals[sel]['z'])
        sciAvg[ii]=np.mean(cosmo.sigmacritinv(redshift,gals[sel]['z']))
        sciErr[ii]=sigmaIS/np.sqrt(2.*(1.-sigmaIS**2) * len(sel.nonzero()[0]))
    sciAvg/=sciNorm
    sciErr/=sciNorm

    plt.plot(zModel,sciModel)
    plt.errorbar(zAvg,sciAvg-1,yerr=sciErr,fmt='o')
    plt.xlabel(r"$z_s$")
    plt.ylabel(r"$<\Sigma_c(z=0.1)>/<\Sigma_c(z)>-1$")
    plt.xlim(zbins[0],zbins[-1])
    plt.gcf().subplots_adjust(left=0.2)
    plt.savefig("coma_zdep.png")
    plt.show()

def parsToProfile():
    gals=selectComaTargets()
    inputPriors=[[[0,360],[0,1],150,float(gals[thisGal]['g1']),float(gals[thisGal]['g2'])] for thisGal in
range(len(gals))]
    dI,dS,dIS,dIkde,dSkde,dISkde,hwI,hwS,hwIS,inputPars=ensemble.getScatter("/data/mgeorge/speclens/data/coma_hexNoCen_6_1.0_1.4_30.0_1_pixel/",len(gals),inputPriors=inputPriors,fileType="stats")
    obsI=dIkde+inputPars
    obsS=dSkde+inputPars
    obsIS=dISkde+inputPars

    weightTrue=np.ones(len(gals))
    #    varI=hwI[:,3]**2 + hwI[:,4]**2
    #    varS=hwS[:,3]**2 + hwS[:,4]**2
    #    varIS=hwIS[:,3]**2 + hwIS[:,4]**2

    varI=((1.-inputPars[:,1]**2)/inputPars[:,1])**2
    varS=varI
    varIS=varI

    weightI=(1/varI)/np.sum(1./varI)
    weightS=(1/varS)/np.sum(1./varS)
    weightIS=(1/varIS)/np.sum(1./varIS)

    distTrue,gtanTrue,gerrTrue=shearProfile(gals['ra'],gals['dec'],gals['g1'],gals['g2'],weightTrue)
    distI,gtanI,gerrI=shearProfile(gals['ra'],gals['dec'],obsI[:,3],obsI[:,4],weightI)
    distS,gtanS,gerrS=shearProfile(gals['ra'],gals['dec'],obsS[:,3],obsS[:,4],weightS)
    distIS,gtanIS,gerrIS=shearProfile(gals['ra'],gals['dec'],obsIS[:,3],obsIS[:,4],weightIS)

    plotShearProfile((distTrue,distI,distS,distIS),(gtanTrue,gtanI,gtanS,gtanIS),(gerrTrue,gerrI,gerrS,gerrIS),colors=["black","red","green","blue"])


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
    sim.contourPlotAll(chains,lnprobs=lnprobs,inputPars=inputPars,showMax=True,showPeakKDE=True,show68=True,smooth=3,percentiles=[0.68,0.95],labels=labels,showPlot=False,filename=outDir+"/plots/gal_{:03d}.{}".format(galID,figExt))

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
              "inputPriors=[[0,360],[0,1],150,float(gals[thisGal]['g1']),float(gals[thisGal]['g2'])]\n"
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
    

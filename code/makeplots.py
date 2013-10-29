#! env python
import sim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

# This is a driver for functions in sim.py to create some figures for a proposal

if __name__ == "__main__":

    figExt="pdf" # pdf or png
    plotDir="/data/mgeorge/speclens/plots"
    dataDir="/data/mgeorge/speclens/data"
    showPlot=False

    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':30})
    plt.rc('text', usetex=True)
    plt.rc('axes',linewidth=1.5)

    # Fig 1
    # Mimic Morales Fig. 1
    # unweighted velocity map with no PSF, unsheared then sheared

    bulge_n=4.
    bulge_r=1.
    disk_n=1.
    disk_r=2.
    bulge_frac=0.
    gal_q=0.75
    gal_beta=0.1
    gal_flux=1.
    atmos_fwhm=1.
    rotCurveOpt='flat'
    vmax=100.
    rotCurvePars=np.array([vmax])
    g1=0.
    g2=0.
    
    ell=[disk_r,gal_q,gal_beta]
    lines=sim.getEllipseAxes(ell)
    lw=5
    
    vmap,fluxVMap,gal = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,g1,g2)
    trim=2.5
    plt.clf()
    sim.showImage(vmap,None,None,None,trim=trim,colorbar=True,ellipse=ell,lines=lines,lw=lw,filename="{}/fig1a.{}".format(plotDir,figExt),title=r"q$_{\rm int}=0.75$, $\gamma_+=0, \gamma_{\times}=0$",showPlot=showPlot)
    
    g1=0.2
    g2=0.
    ellSheared=sim.shearEllipse(ell,g1,g2)
    linesSheared=sim.shearLines(lines,g1,g2)
    linesObs=sim.getEllipseAxes(ellSheared)
    
    vmapInc,fluxVMapInc,gal = sim.makeGalVMap(bulge_n,bulge_r,disk_n,ellSheared[0],bulge_frac,ellSheared[1],ellSheared[2],gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,0.,0.)
    plt.clf()
    sim.showImage(vmapInc,None,None,None,trim=trim,ellipse=ellSheared,lines=linesSheared,lw=lw,filename="{}/fig1b.{}".format(plotDir,figExt),title=r"q$_{\rm int}=0.5$",showPlot=showPlot)

    vmapSheared,fluxVMapSheared,galSheared = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,g1,g2)
    plt.clf()
    sim.showImage(vmapSheared,None,None,None,trim=trim,ellipse=ellSheared,lines=linesSheared,lw=lw,filename="{}/fig1c.{}".format(plotDir,figExt),title=r"$\gamma_+=0.2$",showPlot=showPlot)
    

    g1=0.0
    g2=0.2
    ellSheared=sim.shearEllipse(ell,g1,g2)
    linesSheared=sim.shearLines(lines,g1,g2)
    linesObs=sim.getEllipseAxes(ellSheared)
    
    vmapSheared,fluxVMapSheared,galSheared = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,g1,g2)
    plt.clf()
    sim.showImage(vmapSheared,None,None,None,trim=trim,ellipse=ellSheared,lines=np.array([linesSheared,linesObs]).reshape(4,4),lcolors=['w','w',"gray","gray"],lstyles=["--","--","-","-"],lw=lw,filename="{}/fig1d.{}".format(plotDir,figExt),title=r"$\gamma_{\times}=0.2$",showPlot=showPlot)

    print "Finished Fig 1"
    # Fig 2
    # galaxy image, velocity map, flux-weighted velocity map with PSF and fiber positions
    # (unsheared)
    
    plt.clf()
    numFib=7
    fibRad=1.
    fibPA=None
    fibConfig="hex"
    pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig)
    xfib,yfib=pos
    sim.showImage(gal,xfib,yfib,fibRad,fibShape=fibShape,fibPA=fibPA,trim=trim,cmap=matplotlib.cm.gray,colorbar=False,filename="{}/fig2a.{}".format(plotDir,figExt),showPlot=showPlot)
    plt.clf()
    sim.showImage(fluxVMap,xfib,yfib,fibRad,fibShape=fibShape,fibPA=fibPA,trim=trim,colorbar=False,filename="{}/fig2b.{}".format(plotDir,figExt),showPlot=showPlot)
    
    
    # Fig 3
    # vobs vs theta for a few sets of parameters
    plt.clf()
    
    sigma=30. # velocity unc in km/s
    
    gal_beta=300.
    gal_q=0.5
    vmax=200.
    g1=0.
    g2=0.
    pars = np.array([gal_beta, gal_q, vmax, g1, g2])
    numFib=7
    fibRad=1.
    fibConfig="hex"
    pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig)
    xfib,yfib=pos
    
    theta=np.linspace(0,2.*np.pi,num=200)
    xvals=2.*fibRad*np.cos(theta)
    yvals=2.*fibRad*np.sin(theta)
    vvals=sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="blue",linestyle='-',lw=3,label="Fiducial: PA={}, b/a={}".format(pars[0],pars[1])+r", v$_{max}$"+"={}, g1={}, g2={}".format(pars[2],pars[3],pars[4]))
    
    thetasamp=np.linspace(0,2.*np.pi,num=6,endpoint=False)
    xsamp=2.*fibRad*np.cos(thetasamp)
    ysamp=2.*fibRad*np.sin(thetasamp)
    vsamp=sim.vmapModel(pars, xsamp, ysamp)
    verr=np.repeat(sigma,xsamp.size)
    
    plt.errorbar(np.rad2deg(thetasamp),vsamp,yerr=verr,fmt=None,lw=2,ecolor='black',elinewidth=5,capsize=7)
    
    pars = np.array([gal_beta, gal_q+0.2, vmax, g1, g2])
    vvals=sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="green",linestyle="--",lw=2,label="b/a={}".format(pars[1]))
    
    pars = np.array([gal_beta+20, gal_q, vmax, g1, g2])
    vvals=sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="orange",linestyle="-.",lw=2,label="PA={}".format(pars[0]))
    
    pars = np.array([gal_beta, gal_q, vmax+50, g1, g2])
    vvals=sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="red",linestyle=":",lw=2,label=r"v$_{max}$"+"={}".format(pars[2]))
    
    pars = np.array([gal_beta, gal_q, vmax, g1+0.1, g2])
    vvals=sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="yellow",linestyle="-",lw=2,label="g1"+"={}".format(pars[3]))
    
    pars = np.array([gal_beta, gal_q, vmax+50, g1, g2+0.1])
    vvals=sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="magenta",linestyle="--",lw=2,label="g2"+"={}".format(pars[4]))
    
    plt.legend(loc="upper right",prop={'size':14},frameon=False)
    
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel('v(r$_{proj}$=2\") (km/s)')
    plt.xlim((-5,365))
    plt.ylim(np.array([-250,250]))
    
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig("{}/fig3.{}".format(plotDir,figExt))
    if(showPlot):
        plt.show()
    
    # Fig 4
    # parameter constraints from a number of noise realizations
    
    pars = np.array([105, 0.5, 200, 0, 0.15])
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    sigma=30.
    numFib=6
    fibRad=1
    fibPA=0.
    fibConfig="hexNoCen"
    pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig,fibPA=fibPA)
    xvals,yvals=pos
    vvals=sim.vmapModel(pars, xvals, yvals)
    ellObs=sim.ellModel(pars)
    ellErr=np.array([10.,0.1])
    priors=[None,[0.1,1],(pars[2],10),[-0.5,0.5],[-0.5,0.5]]

    # compare imaging vs spectro vs combined
    chains,lnprobs=sim.fitObs(vvals,sigma,ellObs,ellErr,priors,fibRad=fibRad,fibConfig=fibConfig,fibPA=fibPA,addNoise=False,nSteps=250)
    smooth=3
    plt.clf()
    sim.contourPlotAll(chains,inputPars=pars,smooth=smooth,percentiles=[0.68,0.95],labels=labels,filename="{}/fig4a.{}".format(plotDir,figExt),showPlot=showPlot)

    # now try with PSF and fiber convolution
    atmos_fwhm=1.5
    disk_r=1.
    fibConvolve=True
    convOpt="pixel"

    chains,lnprobs=sim.fitObs(vvals,sigma,ellObs,ellErr,priors,fibRad=fibRad,disk_r=disk_r,atmos_fwhm=atmos_fwhm,fibConvolve=fibConvolve,addNoise=True,convOpt=convOpt)
    smooth=3
    plt.clf()
    sim.contourPlotAll(chains,inputPars=pars,smooth=smooth,percentiles=[0.68,0.95],labels=labels,filename="{}/fig4b.{}".format(plotDir,figExt),showPlot=showPlot)


    # Fig 5
    # shear error distribution for ensemble
    nGal=100
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    inputPriors=[[0,360],[0,1],150,(0,0.05),(0,0.05)]
    obsPriors=[[0,360],[0,1],(150,15),[-0.5,0.5],[-0.5,0.5]]
    inputPars=sim.generateEnsemble(nGal,inputPriors,shearOpt=None)
    numFib=6
    fibRad=1
    fibConfig="hexNoCen"
    pos,fibShape=sim.getFiberPos(numFib,fibRad,fibConfig)
    xvals,yvals=pos
    sigma=30.
    ellErr=np.array([10.,0.1])
    smooth=3

    obsParsI=np.zeros_like(inputPars)
    obsParsS=np.zeros_like(inputPars)
    obsParsIS=np.zeros_like(inputPars)
    
    for ii in range(nGal):
        print "************Running Galaxy {}".format(ii)
        vvals=sim.vmapModel(inputPars[ii,:], xvals, yvals)
        ellObs=sim.ellModel(inputPars[ii,:])
        chains,lnprobs=sim.fitObs(vvals,sigma,ellObs,ellErr,obsPriors,fibRad=fibRad,addNoise=True)
        obsParsI[ii,:]=sim.getMaxProb(chains[0],lnprobs[0])
        obsParsS[ii,:]=sim.getMaxProb(chains[1],lnprobs[1])
        obsParsIS[ii,:]=sim.getMaxProb(chains[2],lnprobs[2])
        print inputPars[ii,:]
        print obsParsI[ii,:]
        print obsParsS[ii,:]
        print obsParsIS[ii,:]
        sim.contourPlotAll(chains,inputPars=inputPars[ii,:],smooth=smooth,percentiles=[0.68,0.95],labels=labels,filename="{}/fig5_gal{}.{}".format(plotDir,ii,figExt),showPlot=showPlot)

    sim.writeRec(sim.parsToRec(inputPars),"{}/fig5_inputPars.fits".format(dataDir))
    sim.writeRec(sim.parsToRec(obsParsI),"{}/fig5_obsParsI.fits".format(dataDir))
    sim.writeRec(sim.parsToRec(obsParsS),"{}/fig5_obsParsS.fits".format(dataDir))
    sim.writeRec(sim.parsToRec(obsParsIS),"{}/fig5_obsParsIS.fits".format(dataDir))

#! env python

import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import os
import sys

try:
    import speclens
except ImportError: # add parent dir to python search path
    path, filename = os.path.split(__file__)
    sys.path.append(os.path.abspath(os.path.join(path,"../")))
    import speclens
    
# This is a driver for functions in sim.py to create some figures for a proposal

def getShapes(model):
    ell=speclens.sim.defineEllipse(model)
    lines=speclens.sim.getEllipseAxes(ell)
    return (ell,lines)

def shearVMapPlot(plotDir, figExt="pdf", showPlot=False):
    """Create plots illustrating shear effect on velocity map.

    Similar to Fig. 1 of Morales 2006, make simple velocity maps
    and plot 4 separate figures:
        a) unlensed velocity map
        b) velocity map with different inclination
        c) velocity map with shear applied at 0 deg to mimic 
            inclination in image
        d) velocity map with shear at 45 deg to show misalignment
    """

    lw=5
    trim=2.5

    # Start with basic model
    model=speclens.Model("B",galName="default")
    model.diskCA=0.
    model.rotCurveOpt="flat"
    model.rotCurvePars=[model.vCirc]
    model.cosi=0.75
    model.diskBA=speclens.sim.convertInclination(inc=np.arccos(model.cosi),diskCA=model.diskCA)
    
    ell,lines=getShapes(model)
    vmap,fluxVMap,thinImg,img = speclens.sim.makeGalVMap2(model)

    plt.clf()
    speclens.plot.showImage(vmap, model, None, None,
        filename="{}/fig1a.{}".format(plotDir,figExt), trim=trim,
        ellipse=ell, lines=lines, lw=lw, 
        title=r"cos$(i)=0.75$, $\gamma_+=0, \gamma_{\times}=0$",
        showPlot=showPlot)

    # Now change the inclination
    model.cosi=0.5
    model.diskBA=speclens.sim.convertInclination(inc=np.arccos(model.cosi),diskCA=model.diskCA)

    ellInc,linesInc=getShapes(model)
    vmapInc,fluxVMapInc,thinImgInc,imgInc = speclens.sim.makeGalVMap2(model)

    plt.clf()
    speclens.plot.showImage(vmapInc, model, None, None,
        filename="{}/fig1b.{}".format(plotDir,figExt), trim=trim,
        ellipse=ellInc, lines=linesInc, lw=lw, 
        title=r"cos$(i)=0.5$, $\gamma_+=0, \gamma_{\times}=0$",
        showPlot=showPlot)
    
    # Now undo inclination and apply a shear that mimics it
    model.cosi=0.75
    model.diskBA=speclens.sim.convertInclination(inc=np.arccos(model.cosi),diskCA=model.diskCA)
    model.g1=0.2

    ellG1,linesG1=getShapes(model) # note these are the unsheared shapes
    ellSheared=speclens.sim.shearEllipse(ellG1,model.g1,model.g2)
    linesSheared=speclens.sim.shearLines(linesG1,model.g1,model.g2)
    linesObs=speclens.sim.getEllipseAxes(ellSheared)

    vmapG1,fluxVMapG1,thinImgG1,imgG1 = speclens.sim.makeGalVMap2(model)

    plt.clf()
    speclens.plot.showImage(vmapG1, model, None, None,
        filename="{}/fig1c.{}".format(plotDir,figExt), trim=trim,
        ellipse=ellSheared, lines=linesObs, lw=lw, 
        title=r"cos$(i)=0.75$, $\gamma_+=0.2, \gamma_{\times}=0$",
        showPlot=showPlot)
    

    # Finally show the cross shear
    model.g1=0.
    model.g2=0.2
    
    ellG2,linesG2=getShapes(model) # note these are the unsheared shapes
    ellSheared=speclens.sim.shearEllipse(ellG2,model.g1,model.g2)
    linesSheared=speclens.sim.shearLines(linesG2,model.g1,model.g2)
    linesObs=speclens.sim.getEllipseAxes(ellSheared)
    
    vmapG2,fluxVMapG2,thinImgG2,imgG2 = speclens.sim.makeGalVMap2(model)

    plt.clf()
    
    speclens.plot.showImage(vmapG2, model, None, None,
        filename="{}/fig1d.{}".format(plotDir,figExt), trim=trim,
        ellipse=ellSheared,
        lines=np.array([linesSheared,linesObs]).reshape(4,4),
        lcolors=['w','w',"gray","gray"], lstyles=["--","--","-","-"],
        lw=lw, title=r"cos$(i)=0.75$, $\gamma_+=0, \gamma_{\times}=0.2$",
        showPlot=showPlot)

    print "Finished Fig 1"
    return

def samplingPlot(plotDir, figExt="pdf", showPlot=False):
    """Create plots of fibers overlaid on image and vmap.

    Show 2 arcsec diameter fibers in hex patter overlaid
    on galaxy image and flux-weighted velocity map to 
    illustrate spatial sampling of the velocity field
    including the effects of seeing.
    """
        
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
    
    vmap,fluxVMap,gal = speclens.sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,rotCurveOpt,rotCurvePars,g1,g2)
    trim=1.

    plt.clf()
    numFib=7
    fibRad=1.
    fibPA=None
    fibConfig="hex"
    pos,fibShape=speclens.sim.getFiberPos(numFib,fibRad,fibConfig)
    xfib,yfib=pos
    speclens.plot.showImage(gal,xfib,yfib,fibRad,fibShape=fibShape,fibPA=fibPA,trim=trim,cmap=matplotlib.cm.gray,colorbar=False,filename="{}/fig2a.{}".format(plotDir,figExt),showPlot=showPlot)
    plt.clf()
    speclens.plot.showImage(fluxVMap,xfib,yfib,fibRad,fibShape=fibShape,fibPA=fibPA,trim=trim,colorbar=False,filename="{}/fig2b.{}".format(plotDir,figExt),showPlot=showPlot)

    print "Finished Fig 2"
    return

def vThetaPlot(plotDir, figExt="pdf", showPlot=False):
    """Create plot of velocity vs azimuthal angle

    Illustrate the effects of changing a few parameters
    on the observed velocity as a function of azimuthal angle.
    Assumes a hex fiber sampling pattern.
    """

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
    pos,fibShape=speclens.sim.getFiberPos(numFib,fibRad,fibConfig)
    xfib,yfib=pos
    
    theta=np.linspace(0,2.*np.pi,num=200)
    xvals=2.*fibRad*np.cos(theta)
    yvals=2.*fibRad*np.sin(theta)
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="blue",linestyle='-',lw=3,label="Fiducial: PA={}, b/a={}".format(pars[0],pars[1])+r", v$_{max}$"+"={}, g1={}, g2={}".format(pars[2],pars[3],pars[4]))
    
    thetasamp=np.linspace(0,2.*np.pi,num=6,endpoint=False)
    xsamp=2.*fibRad*np.cos(thetasamp)
    ysamp=2.*fibRad*np.sin(thetasamp)
    vsamp=speclens.sim.vmapModel(pars, xsamp, ysamp)
    verr=np.repeat(sigma,xsamp.size)
    
    plt.errorbar(np.rad2deg(thetasamp),vsamp,yerr=verr,fmt=None,lw=2,ecolor='black',elinewidth=5,capsize=7)
    
    pars = np.array([gal_beta, gal_q+0.2, vmax, g1, g2])
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="green",linestyle="--",lw=2,label="b/a={}".format(pars[1]))
    
    pars = np.array([gal_beta+20, gal_q, vmax, g1, g2])
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="orange",linestyle="-.",lw=2,label="PA={}".format(pars[0]))
    
    pars = np.array([gal_beta, gal_q, vmax+50, g1, g2])
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="red",linestyle=":",lw=2,label=r"v$_{max}$"+"={}".format(pars[2]))
    
    pars = np.array([gal_beta, gal_q, vmax, g1+0.1, g2])
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
    plt.plot(np.rad2deg(theta),vvals,color="yellow",linestyle="-",lw=2,label="g1"+"={}".format(pars[3]))
    
    pars = np.array([gal_beta, gal_q, vmax+50, g1, g2+0.1])
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
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

    print "Finished Fig 3"
    return

def modelConstraintPlot(plotDir, figExt="pdf", showPlot=False):
    """Plot parameter constraints given a measurement with errors

    Take the imaging and velocity observables for a single sheared
    galaxy and fit a 5 parameter model using MCMC to maximize the
    fit likelihood. Plot joint posterior constraints.
    """

    pars = np.array([105, 0.5, 200, 0, 0.15])
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    sigma=30.
    numFib=6
    fibRad=1
    fibPA=0.
    fibConfig="hexNoCen"
    pos,fibShape=speclens.sim.getFiberPos(numFib,fibRad,fibConfig,fibPA=fibPA)
    xvals,yvals=pos
    vvals=speclens.sim.vmapModel(pars, xvals, yvals)
    ellObs=speclens.sim.ellModel(pars)
    ellErr=np.array([10.,0.1])
    priors=[None,[0.1,1],(pars[2],10),[-0.5,0.5],[-0.5,0.5]]

    # compare imaging vs spectro vs combined
    chains,lnprobs=speclens.fit.fitObs(vvals,sigma,ellObs,ellErr,priors,fibRad=fibRad,fibConfig=fibConfig,fibPA=fibPA,addNoise=False,nSteps=250)
    smooth=3
    plt.clf()
    speclens.plot.contourPlotAll(chains,lnprobs=lnprobs,inputPars=pars,showMax=True,showPeakKDE=True,show68=True,smooth=smooth,percentiles=[0.68,0.95],labels=labels,filename="{}/fig4a.{}".format(plotDir,figExt),showPlot=showPlot)

    # now try with PSF and fiber convolution
    atmos_fwhm=1.5
    disk_r=1.
    fibConvolve=True
    convOpt="pixel"

    chains,lnprobs=speclens.fit.fitObs(vvals,sigma,ellObs,ellErr,priors,fibRad=fibRad,disk_r=disk_r,atmos_fwhm=atmos_fwhm,fibConvolve=fibConvolve,addNoise=True,convOpt=convOpt)
    smooth=3
    plt.clf()
    speclens.plot.contourPlotAll(chains,lnprobs=lnprobs,inputPars=pars,showMax=True,showPeakKDE=True,show68=True,smooth=smooth,percentiles=[0.68,0.95],labels=labels,filename="{}/fig4b.{}".format(plotDir,figExt),showPlot=showPlot)

    print "Finished Fig 4"
    return

    
if __name__ == "__main__":

    # set up paths for output dirs
    speclensDir="../"
    if not os.path.isdir(speclensDir):
        raise NameError(speclensDir)
    plotDir=speclensDir+"/plots"
    dataDir=speclensDir+"/data"
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)

    figExt="pdf" # pdf or png
    showPlot=False

    # Fig 1
    # Mimic Morales 2006, Fig. 1 - illustrate effect of shear on velocity map
    shearVMapPlot(plotDir, figExt=figExt, showPlot=showPlot)

    # Fig 2
    # galaxy image and flux-weighted velocity map with PSF and fiber positions
    # (unsheared)
    samplingPlot(plotDir, figExt=figExt, showPlot=showPlot)
    
    # Fig 3
    # vobs vs theta for a few sets of parameters
    vThetaPlot(plotDir, figExt=figExt, showPlot=showPlot)
    
    # Fig 4
    # parameter constraints from a number of noise realizations
    modelConstraintPlot(plotDir, figExt=figExt, showPlot=showPlot)

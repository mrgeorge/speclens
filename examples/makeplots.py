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
    model=speclens.Model("A",galName="default")
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
    """Plot sampling configuration overlaid on image and fvmap

    Illustrate spatial sampling of the velocity field
    including the effects of seeing. Uses default sampling
    configuration from Model class, can be changed for
    fibers/slits/ifu etc.
    """

    trim=1.

    model=speclens.Model("A")
    model.rotCurveOpt="flat"
    model.rotCurvePars=np.array([model.vCirc])

    vmap,fluxVMap,thinImg,img = speclens.sim.makeGalVMap2(model)

    pos,sampShape=speclens.sim.getSamplePos(model.nVSamp,
        model.vSampSize, model.vSampConfig, sampPA=model.vSampPA)

    xpos,ypos=pos
    model.vSampShape=sampShape

    plt.clf()
    speclens.plot.showImage(img, model, xpos, ypos,
        filename="{}/fig2a".format(plotDir,figExt), trim=trim,
        colorbar=True, cmap=matplotlib.cm.gray, colorbarLabel=None,
        showPlot=showPlot)
    plt.clf()
    speclens.plot.showImage(fluxVMap, model, xpos, ypos,
        filename="{}/fig2b".format(plotDir,figExt), trim=trim,
        colorbar=True, cmap=matplotlib.cm.jet, colorbarLabel=None,
        showPlot=showPlot)

    print "Finished Fig 2"
    return

def vThetaPlot(plotDir, figExt="pdf", showPlot=False):
    """Create plot of velocity vs azimuthal angle

    Illustrate the effects of changing a few parameters
    on the observed velocity as a function of azimuthal angle.
    Assumes a hex fiber sampling pattern.
    """

    plt.clf()
    
    # Define galaxy model    
    model=speclens.Model("A")
    model.vSampConfig="hexNoCen"
    model.rotCurveOpt="flat"
    model.rotCurvePars=[model.vCirc]
    model.cosi=0.8
    model.nVSamp=6
    model.vSampSize=1.
    model.atmosFWHM=None
    model.vSampConvolve=False

    # Get velocity sampling positions
    pos,sampShape=speclens.sim.getSamplePos(model.nVSamp,model.vSampSize,model.vSampConfig)
    sigma=30. # velocity unc in km/s
    xpos,ypos=pos
    model.vSampShape=sampShape
    
    # Evaluate model as smooth function of azimuthal angle
    theta=np.linspace(0,2.*np.pi,num=200)
    xvals=2.*model.vSampSize*np.cos(theta)
    yvals=2.*model.vSampSize*np.sin(theta)
    vvals=speclens.sim.vmapModel(model, xvals, yvals)

    plt.plot(np.rad2deg(theta), vvals, color="blue", linestyle='-',
             lw=3, label="Fiducial: PA={}, cos(i)={:0.2}".format(
                model.diskPA, model.cosi) + 
             r", v$_{max}$"+"={}, g1={}, g2={}".format(
                 model.vCirc, model.g1, model.g2))

    thetasamp=np.linspace(0,2.*np.pi,num=6,endpoint=False)
    xsamp=2.*model.vSampSize*np.cos(thetasamp)
    ysamp=2.*model.vSampSize*np.sin(thetasamp)
    vsamp=speclens.sim.vmapModel(model, xsamp, ysamp)
    verr=np.repeat(sigma,xsamp.size)
    
    plt.errorbar(np.rad2deg(thetasamp), vsamp, yerr=verr, fmt=None,
        lw=2, ecolor='black', elinewidth=5, capsize=7)
    
    model.cosi+=0.2
    vvals=speclens.sim.vmapModel(model, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="green", linestyle="--",
        lw=2, label="cos(i)={:0.2}".format(model.cosi))
    
    model.cosi-=0.2
    model.diskPA+=20.
    vvals=speclens.sim.vmapModel(model, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="orange", linestyle="-.",
        lw=2, label="PA={}".format(model.diskPA))
    
    model.diskPA-=20.
    model.vCirc+=50.
    model.rotCurvPars=[model.vCirc]
    vvals=speclens.sim.vmapModel(model, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="red", linestyle=":",
        lw=2, label=r"v$_{max}$"+"={}".format(model.vCirc))
    
    model.vCirc-=50.
    model.rotCurvPars=[model.vCirc]
    model.g1+=0.1
    vvals=speclens.sim.vmapModel(model, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="yellow", linestyle="-",
        lw=2, label="g1"+"={}".format(model.g1))
    
    model.g1-=0.1
    model.g2+=0.1
    vvals=speclens.sim.vmapModel(model, xvals, yvals)

    plt.plot(np.rad2deg(theta), vvals, color="magenta",
        linestyle="--", lw=2, label="g2"+"={}".format(model.g2))
    
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
    galaxy and fit a 6 parameter model using MCMC to maximize the
    fit likelihood. Plot joint posterior constraints.
    """

    nThreads=8
    
    model=speclens.Model("A")

    sigma=10.
    ellErr=np.array([10.,0.1])

    # first try w/o PSF and fiber convolution
    model.atmosFWHM=None
    model.vSampConvolve=False
    model.convOpt=None

    xvals,yvals,vvals,ellObs,inputPars = speclens.ensemble.makeObs(
        model, sigma=sigma, ellErr=ellErr, randomPars=False)
    
    # compare imaging vs spectro vs combined
    chains,lnprobs=speclens.fit.fitObs(vvals, sigma, ellObs, ellErr,
        model, addNoise=True, nThreads=nThreads)
    smooth=3
    plt.clf()
    speclens.plot.contourPlotAll(chains, lnprobs=lnprobs,
        inputPars=model.origPars, showMax=True, showPeakKDE=True,
        show68=True, smooth=smooth, percentiles=[0.68,0.95],
        labels=model.labels,
        filename="{}/fig4a.{}".format(plotDir,figExt),
        showPlot=showPlot)

    # now try with PSF and fiber convolution
    model.atmosFWHM=1.
    model.vSampConvolve=True
    model.convOpt="pixel"

    xvals,yvals,vvals,ellObs,inputPars = speclens.ensemble.makeObs(
        model, sigma=sigma, ellErr=ellErr, randomPars=False)

    chains,lnprobs=speclens.fit.fitObs(vvals, sigma, ellObs, ellErr,
        model, addNoise=True, nThreads=nThreads)
    smooth=3
    plt.clf()
    speclens.plot.contourPlotAll(chains, lnprobs=lnprobs,
        inputPars=model.origPars, showMax=True, showPeakKDE=True,
        show68=True, smooth=smooth, percentiles=[0.68,0.95],
        labels=model.labels, filename="{}/fig4b.{}".format(plotDir,
        figExt), showPlot=showPlot)

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

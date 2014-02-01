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

def getShapes(galaxy):
    ell=speclens.sim.defineEllipse(galaxy)
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
    galaxy = speclens.Galaxy()
    galaxy.diskCA=0.
    galaxy.rotCurveOpt="flat"
    galaxy.cosi=0.75

    detector = speclens.Detector()

    ell,lines=getShapes(galaxy)
    vmap,fluxVMap,thinImg,img = speclens.sim.makeGalVMap2(galaxy, detector)

    plt.clf()
    speclens.plot.showImage(vmap, detector, None, None,
        filename="{}/fig1a.{}".format(plotDir,figExt), trim=trim,
        ellipse=ell, lines=lines, lw=lw, 
        title=r"cos$(i)=0.75$, $\gamma_+=0, \gamma_{\times}=0$",
        showPlot=showPlot)

    # Now change the inclination
    galaxy.cosi=0.5

    ellInc,linesInc=getShapes(galaxy)
    vmapInc,fluxVMapInc,thinImgInc,imgInc = speclens.sim.makeGalVMap2(galaxy, detector)

    plt.clf()
    speclens.plot.showImage(vmapInc, detector, None, None,
        filename="{}/fig1b.{}".format(plotDir,figExt), trim=trim,
        ellipse=ellInc, lines=linesInc, lw=lw, 
        title=r"cos$(i)=0.5$, $\gamma_+=0, \gamma_{\times}=0$",
        showPlot=showPlot)
    
    # Now undo inclination and apply a shear that mimics it
    galaxy.cosi=0.75
    galaxy.g1=0.2

    ellG1,linesG1=getShapes(galaxy) # note these are the unsheared shapes
    ellSheared=speclens.sim.shearEllipse(ellG1,galaxy.g1,galaxy.g2)
    linesSheared=speclens.sim.shearLines(linesG1,galaxy.g1,galaxy.g2)
    linesObs=speclens.sim.getEllipseAxes(ellSheared)

    vmapG1,fluxVMapG1,thinImgG1,imgG1 = speclens.sim.makeGalVMap2(galaxy, detector)

    plt.clf()
    speclens.plot.showImage(vmapG1, detector, None, None,
        filename="{}/fig1c.{}".format(plotDir,figExt), trim=trim,
        ellipse=ellSheared, lines=linesObs, lw=lw, 
        title=r"cos$(i)=0.75$, $\gamma_+=0.2, \gamma_{\times}=0$",
        showPlot=showPlot)
    

    # Finally show the cross shear
    galaxy.g1=0.
    galaxy.g2=0.2
    
    ellG2,linesG2=getShapes(galaxy) # note these are the unsheared shapes
    ellSheared=speclens.sim.shearEllipse(ellG2,galaxy.g1,galaxy.g2)
    linesSheared=speclens.sim.shearLines(linesG2,galaxy.g1,galaxy.g2)
    linesObs=speclens.sim.getEllipseAxes(ellSheared)
    
    vmapG2,fluxVMapG2,thinImgG2,imgG2 = speclens.sim.makeGalVMap2(galaxy, detector)

    plt.clf()
    
    speclens.plot.showImage(vmapG2, detector, None, None,
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
    configuration from Detector class, can be changed for
    fibers/slits/ifu etc.
    """

    trim=1.

    galaxy = speclens.Galaxy()
    galaxy.rotCurveOpt="flat"

    detector = speclens.Detector()
    detector.vSampPA = galaxy.diskPA
    
    vmap,fluxVMap,thinImg,img = speclens.sim.makeGalVMap2(galaxy, detector)

    pos=speclens.sim.getSamplePos(detector.nVSamp,
        detector.vSampSize, detector.vSampConfig, sampPA=detector.vSampPA)

    xpos,ypos=pos

    plt.clf()
    speclens.plot.showImage(img, detector, xpos, ypos,
        filename="{}/fig2a".format(plotDir,figExt), trim=trim,
        colorbar=True, cmap=matplotlib.cm.gray, colorbarLabel=None,
        showPlot=showPlot)
    plt.clf()
    speclens.plot.showImage(fluxVMap, detector, xpos, ypos,
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

    # Define galaxy and detector
    galaxy = speclens.Galaxy()
    galaxy.rotCurveOpt="flat"
    galaxy.cosi=0.8

    detector = speclens.Detector()
    detector.vSampConfig="hexNoCen"
    detector.nVSamp=6
    detector.vSampSize=1.
    detector.vSampConvolve=False
    detector.vSampPA=galaxy.diskPA
    
    psf = speclens.PSF()
    psf.atmosFWHM=None

    # Get velocity sampling positions
    pos=speclens.sim.getSamplePos(detector.nVSamp, detector.vSampSize,
        detector.vSampConfig, sampPA=detector.vSampPA)
    sigma=30. # velocity unc in km/s
    xpos,ypos=pos
    
    # Evaluate model as smooth function of azimuthal angle
    theta=np.linspace(0,2.*np.pi,num=200)
    xvals=2.*detector.vSampSize*np.cos(theta)
    yvals=2.*detector.vSampSize*np.sin(theta)
    vvals=speclens.sim.vmapModel(galaxy, xvals, yvals)

    plt.plot(np.rad2deg(theta), vvals, color="blue", linestyle='-',
             lw=3, label="Fiducial: PA={}, cos(i)={:0.2}".format(
                galaxy.diskPA, galaxy.cosi) + 
             r", v$_{max}$"+"={}, g1={}, g2={}".format(
                 galaxy.vCirc, galaxy.g1, galaxy.g2))

    thetasamp=np.linspace(0,2.*np.pi,num=6,endpoint=False)
    xsamp=2.*detector.vSampSize*np.cos(thetasamp)
    ysamp=2.*detector.vSampSize*np.sin(thetasamp)
    vsamp=speclens.sim.vmapModel(galaxy, xsamp, ysamp)
    verr=np.repeat(sigma,xsamp.size)

    plt.errorbar(np.rad2deg(thetasamp), vsamp, yerr=verr, fmt=None,
        lw=2, ecolor='black', elinewidth=5, capsize=7)

    galaxy.cosi+=0.2
    vvals=speclens.sim.vmapModel(galaxy, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="green", linestyle="--",
        lw=2, label="cos(i)={:0.2}".format(galaxy.cosi))

    galaxy.cosi-=0.2
    galaxy.diskPA+=20.
    vvals=speclens.sim.vmapModel(galaxy, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="orange", linestyle="-.",
        lw=2, label="PA={}".format(galaxy.diskPA))

    galaxy.diskPA-=20.
    galaxy.vCirc+=50.
    vvals=speclens.sim.vmapModel(galaxy, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="red", linestyle=":",
        lw=2, label=r"v$_{max}$"+"={}".format(galaxy.vCirc))

    galaxy.vCirc-=50.
    galaxy.g1+=0.1
    vvals=speclens.sim.vmapModel(galaxy, xvals, yvals)
    plt.plot(np.rad2deg(theta), vvals, color="yellow", linestyle="-",
        lw=2, label="g1"+"={}".format(galaxy.g1))
    
    galaxy.g1-=0.1
    galaxy.g2+=0.1
    vvals=speclens.sim.vmapModel(galaxy, xvals, yvals)

    plt.plot(np.rad2deg(theta), vvals, color="magenta",
        linestyle="--", lw=2, label="g2"+"={}".format(galaxy.g2))
    
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

def modelConstraintPlot(chainDir, plotDir, figExt="pdf", showPlot=False):
    """Get parameter constraints given a measurement with errors

    Take the imaging and velocity observables for a single sheared
    galaxy and fit a parametric model using MCMC to maximize the
    fit likelihood. Plot joint posterior constraints and store chain.
    """

    nThreads = 8

    model = speclens.Model("A")

    observation = speclens.Observable()
    observation.vObsErr = np.repeat(10., observation.detector.nVSamp)
    observation.diskPAErr = 10.
    observation.diskBAErr = 0.1
    observation.setPointing(vSampPA = observation.diskPA)

    # first try w/o PSF and fiber convolution
    galID=0
    model.convOpt=None
    observation.psf.atmosFWHM=None
    observation.detector.vSampConvolve=False

    xvals,yvals,vvals,ellObs,inputPars = speclens.ensemble.makeObs(
        model, sigma=sigma, ellErr=ellErr, randomPars=False)

    # compare imaging vs spectro vs combined
    # store chains and make contour plot
    speclens.ensemble.runGal(chainDir, plotDir, galID, inputPars, vvals,
        sigma, ellObs, ellErr, model, figExt=figExt, addNoise=True,
        nThreads=nThreads, seed=0)

    # now try with PSF and fiber convolution
    galID=1
    model.convOpt="pixel"
    observation.psf.atmosFWHM=1.
    observation.detector.vSampConvolve=True
    observation.makeConvolutionKernel(model.convOpt)

    xvals,yvals,vvals,ellObs,inputPars = speclens.ensemble.makeObs(
        model, sigma=sigma, ellErr=ellErr, randomPars=False)

    speclens.ensemble.runGal(chainDir, plotDir, galID, inputPars, vvals,
        sigma, ellObs, ellErr, model, figExt=figExt, addNoise=True,
        nThreads=nThreads, seed=0)

    print "Finished Fig 4 - gal 0 and 1"
    return

    
if __name__ == "__main__":

    # set up paths for output dirs
    speclensDir="../"
    if not os.path.isdir(speclensDir):
        raise NameError(speclensDir)
    plotDir=speclensDir+"/plots"
    chainDir=speclensDir+"/chains"
    if not os.path.isdir(plotDir):
        os.mkdir(plotDir)
    if not os.path.isdir(chainDir):
        os.mkdir(chainDir)

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
    modelConstraintPlot(chainDir, plotDir, figExt=figExt, showPlot=showPlot)

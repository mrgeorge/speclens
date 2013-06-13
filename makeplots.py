#! env python
import sim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

# This is a driver for functions in sim.py to create some figures for a proposal

if __name__ == "__main__":

    figExt="png" # pdf or png
    showPlot=False

    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
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
    gal_q=0.6
    gal_beta=20.
    gal_flux=1.
    atmos_fwhm=0.
    pixScale=0.1
    imgSizePix=100
    rotCurveOpt='flat'
    g1=0
    g2=0
    
    ell=[disk_r,gal_q,gal_beta]
    lines=sim.getEllipseAxes(ell)
    
    vmap,fluxVMap,galX,galK = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,g1,g2)
    trim=1
    plt.clf()
    sim.showImage(vmap,0,1,trim=trim,colorbar=False,ellipse=ell,lines=lines,filename="fig1a.{}".format(figExt),showPlot=showPlot)
    
    g1=0.2
    g2=0.
    
    vmapSheared,fluxVMapSheared,galXSheared,galKSheared = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,g1,g2)
    ellSheared=sim.shearEllipse(ell,g1,g2)
    linesSheared=sim.shearLines(lines,g1,g2)
    linesObs=sim.getEllipseAxes(ellSheared)
    plt.clf()
    sim.showImage(vmapSheared,0,1,trim=trim,ellipse=ellSheared,lines=np.array([linesSheared,linesObs]).reshape(4,4),lcolors=['w','w',"gray","gray"],lstyles=["--","--","-","-"],filename="fig1b.{}".format(figExt),showPlot=showPlot)
    
    # Fig 2
    # galaxy image, velocity map, flux-weighted velocity map with PSF and fiber positions
    # (unsheared)
    
    plt.clf()
    sim.showImage(galK,7,1,trim=trim,cmap=matplotlib.cm.gray,colorbar=False,filename="fig2a.{}".format(figExt),showPlot=showPlot)
    plt.clf()
    sim.showImage(fluxVMap,7,1,trim=trim,colorbar=False,filename="fig2b.{}".format(figExt),showPlot=showPlot)
    
    
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
    
    xvals=np.linspace(0,2.*np.pi,num=200)
    yvals,ellObs=sim.vmapModel(pars, xvals)
    plt.plot(np.rad2deg(xvals),yvals,color="blue",linestyle='-',lw=3,label="Fiducial: PA={}, b/a={}".format(pars[0],pars[1])+r", v$_{max}$"+"={}, g1={}, g2={}".format(pars[2],pars[3],pars[4]))
    
    xsamp=np.linspace(0,2.*np.pi,num=6,endpoint=False)
    ysamp,ellSamp=sim.vmapModel(pars, xsamp)
    yerr=np.repeat(sigma,xsamp.size)
    
    plt.errorbar(np.rad2deg(xsamp),ysamp,yerr=yerr,fmt=None,lw=2,ecolor='black',elinewidth=5,capsize=7)
    
    
    pars = np.array([gal_beta, gal_q+0.2, vmax, g1, g2])
    yvals,ellObs=sim.vmapModel(pars, xvals)
    plt.plot(np.rad2deg(xvals),yvals,color="green",linestyle="--",lw=2,label="b/a={}".format(pars[1]))
    
    pars = np.array([gal_beta+20, gal_q, vmax, g1, g2])
    yvals,ellObs=sim.vmapModel(pars, xvals)
    plt.plot(np.rad2deg(xvals),yvals,color="orange",linestyle="-.",lw=2,label="PA={}".format(pars[0]))
    
    pars = np.array([gal_beta, gal_q, vmax+50, g1, g2])
    yvals,ellObs=sim.vmapModel(pars, xvals)
    plt.plot(np.rad2deg(xvals),yvals,color="red",linestyle=":",lw=2,label=r"v$_{max}$"+"={}".format(pars[2]))
    
    pars = np.array([gal_beta, gal_q, vmax, g1+0.1, g2])
    yvals,ellObs=sim.vmapModel(pars, xvals)
    plt.plot(np.rad2deg(xvals),yvals,color="yellow",linestyle="-",lw=2,label="g1"+"={}".format(pars[3]))
    
    pars = np.array([gal_beta, gal_q, vmax+50, g1, g2+0.1])
    yvals,ellObs=sim.vmapModel(pars, xvals)
    plt.plot(np.rad2deg(xvals),yvals,color="magenta",linestyle="--",lw=2,label="g2"+"={}".format(pars[4]))
    
    plt.legend(loc="upper right",prop={'size':14},frameon=False)
    
    plt.xlabel(r"$\theta$ (deg)")
    plt.ylabel('v(r$_{proj}$=2\") (km/s)')
    plt.xlim((-5,365))
    plt.ylim(np.array([-250,250]))
    
    plt.gcf().subplots_adjust(left=0.15)
    plt.savefig("fig3.{}".format(figExt))
    if(showPlot):
        plt.show()
    
    # Fig 4
    # parameter constraints from a number of noise realizations
    
    pars = np.array([105, 0.5, 200, 0, 0.15])
    labels=np.array(["PA","b/a","vmax","g1","g2"])
    sigma=30.
    xvals=np.linspace(0,2.*np.pi,num=50,endpoint=False)
    yvals,ellObs=sim.vmapModel(pars, xvals)
    ellErr=np.array([0.1,10])
    priors=[None,[0,1],(pars[2],10),[-0.5,0.5],[-0.5,0.5]]
    sampler=sim.vmapFit(None,sigma,ellObs,ellErr,priors,addNoise=False)
    maxp=(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
    print sampler.flatchain[maxp,:]
    
    flatchain=sampler.flatchain
    flatlnprobability=sampler.flatlnprobability
    good=(sampler.flatlnprobability > -np.Inf)
    
    smooth=3
    plt.clf()
    sim.contourPlotAll([flatchain[good]],smooth=smooth,labels=labels,showPlot=showPlot)
    
    # compare imaging vs spectro vs combined
    samplerIS=sim.vmapFit(yvals,sigma,ellObs,ellErr,priors,addNoise=False)
    samplerI=sim.vmapFit(None,sigma,ellObs,ellErr,priors,addNoise=False)
    samplerS=sim.vmapFit(yvals,sigma,None,ellErr,priors,addNoise=False)
    
    flatchainIS=samplerIS.flatchain
    flatlnprobIS=samplerIS.flatlnprobability
    flatchainI=samplerI.flatchain
    flatlnprobI=samplerI.flatlnprobability
    flatchainS=samplerS.flatchain
    flatlnprobS=samplerS.flatlnprobability
    
    goodIS=(flatlnprobIS > -np.Inf)
    goodI=(flatlnprobI > -np.Inf)
    goodS=(flatlnprobS > -np.Inf)
    
    smooth=3
    plt.clf()
    sim.contourPlotAll([flatchainIS[goodIS],flatchainI[goodI],flatchainS[goodS]],smooth=smooth,percentiles=[0.68,0.95],labels=labels,filename="fig4_50fibers.{}".format(figExt),showPlot=showPlot)

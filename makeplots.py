#! env python
import sim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

# This is a driver for functions in sim.py to create some figures for a proposal


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
gal_q=0.3
gal_beta=20.
gal_flux=1.
atmos_fwhm=0.
pixScale=0.1
imgSizePix=100
rotCurveOpt='flat'
e1=0
e2=0

vmap,fluxVMap,galX,galK = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,e1,e2)
trim=1
sim.showImage(vmap,0,1,trim=trim,colorbar=False,filename="fig1a.pdf")

e1=0.
e2=0.3

vmapSheared,fluxVMapSheared,galXSheared,galKSheared = sim.makeGalVMap(bulge_n,bulge_r,disk_n,disk_r,bulge_frac,gal_q,gal_beta,gal_flux,atmos_fwhm,pixScale,imgSizePix,rotCurveOpt,e1,e2)
sim.showImage(vmapSheared,0,1,trim=trim,filename="fig1b.pdf")

# Fig 2
# galaxy image, velocity map, flux-weighted velocity map with PSF and fiber positions
# (unsheared)

sim.showImage(galK,7,1,trim=trim,cmap=matplotlib.cm.gray,colorbar=False,filename="fig2a.pdf")
sim.showImage(fluxVMap,7,1,trim=trim,colorbar=False,filename="fig2b.pdf")


# Fig 3
# vobs vs theta for a few sets of parameters

sigma=30. # velocity unc in km/s

pars = np.array([0, 0.5, 150])

xvals=np.linspace(0,2.*np.pi,num=200)
yvals=sim.vmapModel(pars, xvals)
plt.plot(np.rad2deg(xvals),yvals,color="blue",linestyle='-',lw=2,label="Fiducial: PA={}, b/a={}".format(pars[0],pars[1])+r", v$_{max}$"+"={}".format(pars[2]))

xsamp=np.linspace(0,2.*np.pi,num=6,endpoint=False)
ysamp=sim.vmapModel(pars, xsamp)
yerr=np.repeat(sigma,xsamp.size)

plt.errorbar(np.rad2deg(xsamp),ysamp,yerr=yerr,fmt=None,lw=2,ecolor='black',elinewidth=5,capsize=7)


pars = np.array([0, 0.8, 150])
yvals=sim.vmapModel(pars, xvals)
plt.plot(np.rad2deg(xvals),yvals,color="green",linestyle="--",lw=2,label="b/a={}".format(pars[1]))

pars = np.array([20, 0.5, 150])
yvals=sim.vmapModel(pars, xvals)
plt.plot(np.rad2deg(xvals),yvals,color="orange",linestyle="-.",lw=2,label="PA={}".format(pars[0]))

pars = np.array([0, 0.5, 200])
yvals=sim.vmapModel(pars, xvals)
plt.plot(np.rad2deg(xvals),yvals,color="red",linestyle=":",lw=2,label=r"v$_{max}$"+"={}".format(pars[2]))

plt.legend(loc=9,prop={'size':14},frameon=False)

plt.xlabel(r"$\theta$ (deg)")
plt.ylabel('v(r$_{proj}$=2\") (km/s)')
plt.xlim((-5,365))
plt.ylim(np.array([-200,200]))

plt.gcf().subplots_adjust(left=0.15)
plt.savefig("fig3.pdf")
plt.show()

# Fig 4
# parameter constraints from a number of noise realizations

pars = np.array([105, 0.5, 200])
sigma=30.
xvals=np.linspace(0,2.*np.pi,num=6,endpoint=False)
yvals=sim.vmapModel(pars, xvals)
priors=[None,[0,1],(pars[2],10)]
sampler=sim.vmapFit(yvals,sigma,priors,addNoise=False)
maxp=(sampler.flatlnprobability == np.max(sampler.flatlnprobability))
print sampler.flatchain[maxp,:]

xarr=np.linspace(0,2*np.pi,num=200)
yarr=sim.vmapModel(pars,xarr)
yarr2=sim.vmapModel(sampler.flatchain[maxp,:][0],xarr)
plt.errorbar(xvals,yvals,yerr=sigma,fmt=None,elinewidth=3)
plt.plot(xarr,yarr,'b-',xarr,yarr2,'g-')
plt.show()

flatchain=sampler.flatchain
flatlnprobability=sampler.flatlnprobability
good=(sampler.flatlnprobability > -np.Inf)

smooth=3
sim.contourPlot(flatchain[good,0],flatchain[good,1],smooth=smooth,xlabel="PA (deg)",ylabel="b/a")
sim.contourPlot(flatchain[good,0],flatchain[good,2],smooth=smooth,xlabel="PA (deg)",ylabel="vmax")
sim.contourPlot(flatchain[good,1],flatchain[good,2],smooth=smooth,xlabel="b/a",ylabel="vmax")

nSim=100
for ii in xrange(nSim):
    sampler=sim.vmapFit(yvals,sigma,priors,addNoise=True)
    if(ii == 0):
        flatchain=sampler.flatchain
        flatlnprobability=sampler.flatlnprobability
    else:
        flatchain=np.append(flatchain,sampler.flatchain,axis=0)
        flatlnprobability=np.append(flatlnprobability,sampler.flatlnprobability,axis=0)

good=(flatlnprobability > -np.Inf)

smooth=3
sim.contourPlot(flatchain[good,0],flatchain[good,1],smooth=smooth,xlabel="PA (deg)",ylabel="b/a",filename="fig4a_.pdf")
sim.contourPlot(flatchain[good,0],flatchain[good,2],smooth=smooth,xlabel="PA (deg)",ylabel="vmax",filename="fig4b_.pdf")
sim.contourPlot(flatchain[good,1],flatchain[good,2],smooth=smooth,xlabel="b/a",ylabel="vmax",filename="fig4c_.pdf")

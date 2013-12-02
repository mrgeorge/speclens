#! env python

import matplotlib
matplotlib.use('Agg') # must appear before importing pyplot to get plots w/o GUI
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches
import numpy as np
import scipy.stats
import scipy.signal
import galsim
import sim

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
plt.rc('text', usetex=True)
plt.rc('axes',linewidth=1.5)

def showImage(profile,xfib,yfib,fibRad,fibShape="circle",fibPA=None,filename=None,colorbar=True,colorbarLabel=r"v$_{LOS}$ (km/s)",cmap=matplotlib.cm.jet,plotScale="linear",trim=0,xlabel="x (arcsec)",ylabel="y (arcsec)",ellipse=None,lines=None,lcolors="white",lstyles="--",lw=2,title=None,showPlot=False):
    """Plot image given by galsim object with fiber pattern overlaid"""

    imgFrame=galsim.ImageF(sim.imgSizePix,sim.imgSizePix)
    img=profile.draw(image=imgFrame,dx=sim.pixScale)
    halfWidth=0.5*sim.imgSizePix*sim.pixScale # arcsec
    #    img.setCenter(0,0)

    if(plotScale=="linear"):
        plotArr=img.array
    elif(plotScale=="log"):
        plotArr=np.log(img.array)

    plt.imshow(plotArr,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest',cmap=cmap)

    if(xfib is not None):
        numFib=xfib.size
        for pos in zip(xfib,yfib):
            if(fibShape=="circle"):
                circ=plt.Circle((pos[0],pos[1]),radius=fibRad,fill=False,color='white',lw=lw)
                ax=plt.gca()
                ax.add_patch(circ)
            elif(fibShape=="square"):
                PArad=np.deg2rad(fibPA)
                corners=np.zeros((4,2))
                xx=np.array([-1,1,1,-1])
                yy=np.array([-1,-1,1,1])
                corners[:,0]=(xx*np.cos(PArad)-yy*np.sin(PArad))*0.5*fibRad+pos[0]
                corners[:,1]=(xx*np.sin(PArad)+yy*np.cos(PArad))*0.5*fibRad+pos[1]
                sq=plt.Polygon(corners,fill=False,color='white',lw=lw)
                ax=plt.gca()
                ax.add_patch(sq)

    if(colorbar):
        cbar=plt.colorbar(fraction=0.05)
        if(colorbarLabel is not None):
            cbar.set_label(colorbarLabel)

    if(ellipse is not None): # ellipse is either None or np.array([disk_r,gal_q,gal_beta])
        ax=plt.gca()
        rscale=2
        ell=matplotlib.patches.Ellipse(xy=(0,0),width=rscale*ellipse[0]*ellipse[1],height=rscale*ellipse[0],angle=ellipse[2]-90,fill=False,color="white",lw=lw)
        ax.add_artist(ell)
    
    if(lines is not None): # lines is either None or np.array([[x1,x2,y1,y2],...]) or np.array([x1,x2,y1,y2])
        if(lines.shape == (4,)): # only one line
            plt.plot(lines[0:2],lines[2:4],color=lcolors,lw=lw,ls=lstyles)
        else:
            if(type(lcolors) is str):
                lcolors=np.repeat(lcolors,len(lines))
            if(type(lstyles) is str):
                lstyles=np.repeat(lstyles,len(lines))
            for line,color,style in zip(lines,lcolors,lstyles):
                plt.plot(line[0:2],line[2:4],color=color,lw=lw,ls=style)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(trim>0): # trim all edges by this amount in arcsec
        plt.xlim((-halfWidth+trim,halfWidth-trim))
        plt.ylim((-halfWidth+trim,halfWidth-trim))

    if(title is not None):
        plt.title(title)

    if(filename):
        plt.savefig(filename,bbox_inches=matplotlib.transforms.Bbox(np.array(((0,-.4),(8.5,6))))) # annoying hardcode fudge to keep labels inside plot
    if(showPlot):
        plt.show()


def showArr(arr):
    """Pixel equivalent of showImage (with fewer features)"""
    
    halfWidth=0.5*sim.imgSizePix*sim.pixScale # arcsec
    plt.clf()
    plt.imshow(arr,origin='lower',extent=(-halfWidth,halfWidth,-halfWidth,halfWidth),interpolation='nearest',cmap=matplotlib.cm.jet)
    plt.show()


def contourPlot(xvals,yvals,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],xlabel=None,ylabel=None,xlim=None,ylim=None,filename=None,showPlot=False):
    """Make a 2d contour plot of parameter posteriors"""

    n2dbins=300

    # if it's a single ndarray wrapped in a list, convert to ndarray to use full color list
    if((type(xvals) is list) & (len(xvals) ==1)):
        xvals=xvals[0]
        yvals=yvals[0]

    if(type(xvals) is list):
        for ii in range(len(xvals)):
            zz,xx,yy=np.histogram2d(xvals[ii],yvals[ii],bins=n2dbins)
            xxbin=xx[1]-xx[0]
            yybin=yy[1]-yy[0]
            xx=xx[1:]+0.5*xxbin
            yy=yy[1:]+0.5*yybin

        if(smooth > 0):
            kernSize=int(10*smooth)
            sx,sy=scipy.mgrid[-kernSize:kernSize+1, -kernSize:kernSize+1]
            kern=np.exp(-(sx**2 + sy**2)/(2.*smooth**2))
            zz=scipy.signal.convolve2d(zz,kern/np.sum(kern),mode='same')

        hist,bins=np.histogram(zz.flatten(),bins=1000)
        sortzz=np.sort(zz.flatten())
        cumhist=np.cumsum(sortzz)*1./np.sum(zz)
        levels=np.array([sortzz[(cumhist>(1-pct)).nonzero()[0][0]] for pct in percentiles])

        plt.contour(xx,yy,zz.T,levels=levels,colors=colors[ii])
    else: #we just have single ndarrays for xvals and yvals
        zz,xx,yy=np.histogram2d(xvals,yvals,bins=n2dbins)
        xxbin=xx[1]-xx[0]
        yybin=yy[1]-yy[0]
        xx=xx[1:]+0.5*xxbin
        yy=yy[1:]+0.5*yybin

        if(smooth > 0):
            kernSize=int(10*smooth)
            sx,sy=scipy.mgrid[-kernSize:kernSize+1, -kernSize:kernSize+1]
            kern=np.exp(-(sx**2 + sy**2)/(2.*smooth**2))
            zz=scipy.signal.convolve2d(zz,kern/np.sum(kern),mode='same')

        hist,bins=np.histogram(zz.flatten(),bins=1000)
        sortzz=np.sort(zz.flatten())
        cumhist=np.cumsum(sortzz)*1./np.sum(zz)
        levels=np.array([sortzz[(cumhist>(1-pct)).nonzero()[0][0]] for pct in percentiles])

        plt.contour(xx,yy,zz.T,levels=levels,colors=colors)

    if(xlabel is not None):
        plt.xlabel(xlabel)
    if(ylabel is not None):
        plt.ylabel(ylabel)
    if(xlim is not None):
        plt.xlim(xlim)
    if(ylim is not None):
        plt.ylim(ylim)

    if(filename):
        plt.savefig(filename)
    if(showPlot):
        plt.show()
    
def contourPlotAll(chains,lnprobs=None,inputPars=None,showMax=True,showPeakKDE=True,show68=True,smooth=0,percentiles=[0.68,0.95,0.99],colors=["red","green","blue"],labels=None,figsize=(8,6),filename=None,showPlot=False):
    """Make a grid of contour plots for each pair of parameters

    Note - chain is actually a list of 1 or more chains from emcee sampler
    """
    
    nChains=len(chains)
    nPars=chains[0].shape[1]

    fig,axarr=plt.subplots(nPars,nPars,figsize=figsize)
    fig.subplots_adjust(hspace=0,wspace=0)

    if(labels is None):
        labels=np.repeat(None,nPars)

    # find max and min for all pars across chains
    limArr=np.tile((np.Inf,-np.Inf),nPars).reshape(nPars,2)
    for ch in chains:
        for par in range(nPars):
            lo,hi=np.min(ch[:,par]), np.max(ch[:,par])
            if(lo < limArr[par,0]):
                limArr[par,0]=lo.copy()
            if(hi > limArr[par,1]):
                limArr[par,1]=hi.copy()

    # handle colors
    if(len(colors) == len(chains)):
        histColors=colors
        contourColors=colors
    if((nChains == 1) & (len(colors) == len(percentiles))):
        histColors=colors[0]
        contourColors=colors
    
    # Get max posterior and width
    if((showMax) & (lnprobs is not None)):
        maxProbs=np.array([fit.getMaxProb(ch,lnp) for ch,lnp in zip(chains,lnprobs)])
    if((showPeakKDE) & (lnprobs is not None)):
        peakKDE=np.array([fit.getPeakKDE(ch,fit.getMaxProb(ch,lnp)) for ch,lnp in zip(chains,lnprobs)])
    if(show68):
        ranges=np.array([fit.get68(ch,opt="lowhigh") for ch in chains])
        
    # fill plot panels
    for row in range(nPars):
        for col in range(nPars):
            fig.sca(axarr[row,col])

            # setup axis labels
            if(row == nPars-1):
                xlabel=labels[col]
                plt.setp(axarr[row,col].get_xticklabels(), rotation="vertical", fontsize="xx-small")
            else:
                xlabel=None
                plt.setp(axarr[row,col].get_xticklabels(),visible=False)
            if(col == 0):
                ylabel=labels[row]
                plt.setp(axarr[row,col].get_yticklabels(), fontsize="xx-small")
            else:
                ylabel=None
                plt.setp(axarr[row,col].get_yticklabels(),visible=False)
    
            xarrs=[chain[:,col] for chain in chains]
            yarrs=[chain[:,row] for chain in chains]
            xlim=limArr[col]
            ylim=limArr[row]
            if(row == col):
                #histvals=axarr[row,col].hist(xarrs,bins=50,range=xlim,histtype="step",color=histColors)
                xKDE=np.linspace(xlim[0],xlim[1],num=50)
                for ii in range(nChains):
                    kern=scipy.stats.gaussian_kde(xarrs[ii])
                    yKDE=kern(xKDE)
                    axarr[row,col].plot(xKDE,yKDE,color=histColors[ii])
                    if(showMax):
                        # add vertical lines marking the maximum posterior value
                        plt.plot(np.repeat(maxProbs[ii][col],2),np.array([0,kern(maxProbs[ii][col])]),color=histColors[ii],ls="-.")
                    if(showPeakKDE):
                        # add vertical lines marking the maximum posterior density value
                        plt.plot(np.repeat(peakKDE[ii][col],2),np.array([0,kern(peakKDE[ii][col])]),color=histColors[ii],ls=":")
                    if(show68):
                        # fill band marking 68% width
                        plt.fill_between(xKDE,yKDE,where=((xKDE > ranges[ii][0][col]) & (xKDE < ranges[ii][1][col])),color=histColors[ii],alpha=0.5)
                if(inputPars is not None):
                    # add vertical lines marking the input value
                    plt.plot(np.repeat(inputPars[col],2),np.array(plt.gca().get_ylim()),color="yellow",ls="--")

                if(xlabel is not None):
                    axarr[row,col].set_xlabel(xlabel)
                if(ylabel is not None):
                    axarr[row,col].set_ylabel(ylabel)
                axarr[row,col].set_xlim(xlim)
                plt.setp(axarr[row,col].get_yticklabels(),visible=False)
            elif(col < row):
                contourPlot(xarrs,yarrs,smooth=smooth,percentiles=percentiles,colors=contourColors,xlabel=xlabel,ylabel=ylabel)
                axarr[row,col].set_xlim(xlim)
                axarr[row,col].set_ylim(ylim)
                if(inputPars is not None):
                    # add lines marking the input values
                    plt.plot(np.repeat(inputPars[col],2),ylim,color="yellow",ls="--")
                    plt.plot(xlim,np.repeat(inputPars[row],2),color="yellow",ls="--")
            else:
                axarr[row,col].axis("off")

    fig.subplots_adjust(bottom=0.15)
    if(filename):
        fig.savefig(filename)
    if(showPlot):
        fig.show()


#! env python

import astropy.io.ascii
import matplotlib
from astropy.coordinates.angle_utilities import angular_separation


# Target selection for Abell 1689
# given CFHT MegaCam g,r,z images

# Start with COSMOS Mock Catalogs
# Count successful targets as those with
# Flux_OII > limit
# type == 1 (galaxy)
# zmin < z < zmax


import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, neighbors, cross_validation, metrics
import scipy.interpolate

import fitsio

# read the data
# http://lamwws.oamp.fr/cosmowiki/RealisticSpectroPhotCat
#cmcfullclean = fitsio.read("../chains/CMC081211_all.fits", ext=1)
cmcfullnoise = fitsio.read("../chains/CFHT-CMC-matched.fits", ext=1)
asPerPix = 0.03

cmc = cmcfullnoise

features = ('RAN_B_SUBARU', 'RAN_G_SUBARU', 'RAN_R_SUBARU', 'RAN_Z_SUBARU')
cmcB,cmcg,cmcr,cmcz = [cmc[str] for str in features]


# CFHT data for A1689
catfull = fitsio.read("../chains/CFHT-A1689-merged_catalog.fits",ext=1)
catcut = fitsio.read("../chains/CFHT-A1689-candidates.fits",ext=1)
catext = fitsio.read("../chains/CFHT-A1689-candidates-extended.fits",ext=1)
cat = catext

B,g,r,z = [cat['MAG_AUTO'][:,ii] for ii in range(4)]


plt.scatter(r-z, g-r, color="gray", s=5)
plt.scatter(cmcr-cmcz, cmcg-cmcr, c=cmc['Z'],
            norm=matplotlib.colors.Normalize(0,1.5), s=5, linewidth=0)
cbar = plt.colorbar()
cbar.set_label("z")
plt.xlabel('r-z')
plt.ylabel('g-r')
plt.xlim((-0.5,2.5))
plt.ylim((-0.5,2.0))

a=1.1
b=1.0
c=0.1
xr = np.array(plt.xlim())
plt.plot(xr, b*xr-c,lw=5)
plt.plot(xr, np.repeat(a,2),lw=5)

# color selection
sel = ((g-r < a) & (g-r < b*(r-z)-c))
print sel.nonzero()[0].size

# proximity to cluster selection
raCen = 197.87292
decCen = -1.33806
maxSep = 2.8 # arcmins
near = (np.rad2deg(angular_separation(np.deg2rad(cat['ALPHAWIN_J2000']),
                                      np.deg2rad(cat['DELTAWIN_J2000']),
                                      np.deg2rad(raCen),
                                      np.deg2rad(decCen)
                                     ))*60. < maxSep)

morecuts = ((cat['FWHM_IMAGE'][:,2] > 4.) &
            (cat['FWHM_IMAGE'][:,2] < 30.) &
            (cat['MAG_AUTO'][:,2] > 18.) &
            (cat['MAG_AUTO'][:,2] < 23.5) &
            (cat['FLAGS'][:,2] < 4))

good = (sel & near & morecuts)

plt.scatter((r-z)[good],(g-r)[good], s=30, c="black")
plt.show()
            

f = open("targets.skycat",'w')
f.write("ID\tRA\t\tDec\n")
f.write("--\t------------\t-----------\n")
for ii,rec in enumerate(cat[good]):
    f.write("{:2d}\t{:0.8f}\t{:0.8f}\n".format(ii, rec['ALPHAWIN_J2000'], rec['DELTAWIN_J2000']))
f.close()


stars = fitsio.read("../chains/A1689-r-stars.fits",ext=1)
starMags = 22.5 - 2.5*np.log10(stars['PSFFLUX'][:,2])
starColors = np.repeat("     ",stars.size)
starColors[starMags < 15] = "green"
starColors[starMags > 15] = "red"
f = open("stars.reg",'w')
for ss,sc in zip(stars,starColors):
    f.write("wcs;j2000;circle {} {} 3\" # color = {}\n".format(ss['RA'],ss['DEC'],sc))
f.close()






# clean the data
magMin = 15.
magMax = 25.
minHLRpix = 0.5/asPerPix # half-light radius in pixels
good = (#(cmcfull['RAN_B_subaru'] > magMin) &
        #(cmcfull['RAN_B_subaru'] < magMax) &
        #(cmcfull['RAN_g_subaru'] > magMin) &
        #(cmcfull['RAN_g_subaru'] < magMax) &
        #(cmcfull['R_r_subaru'] > magMin) &
        #(cmcfull['Ran_r_subaru'] < magMax) &
        #(cmcfull['Ran_z_subaru'] > magMin) &
        #(cmcfull['Ran_z_subaru'] < magMax) &
        (cmcfull['HALF_LIGHT_RADIUS'] > minHLRpix))
cmc = cmcfull[good]
print "Initial cuts leave {} out of {} objects".format(len(cmc), len(cmcfull))


# Get typical errors in each band from CFHT data
def fitNoise(testMags, testMagErrs):
    magBins = np.linspace(magMin, magMax, 100)
    binIDs = np.digitize(testMags, magBins)
    
    
def addNoise(trainMags, testMags, testMagErrs):
    



# Set up design matrix
#features = ('Ran_B_subaru', 'Ran_g_subaru', 'Ran_r_subaru', 'Ran_i_subaru')
data = np.array([cmc[feat] for feat in features]).T

# Define training class labels, True = good target, False = bad target
lambdaMin = 6000.
lambdaMax = 8500.
target = (#(cmc['Flux_OII'] > lineFluxMin) &
          (cmc['type'] == 1) &
          (lambdaMin < cmc['Lambda_OII']) &
          (cmc['Lambda_OII'] < lambdaMax))
print "{} of {} are good targets".format(len(target.nonzero()[0]), len(target))

# Define classifiers
lin = svm.SVC(kernel="linear")
rbf = svm.SVC(kernel="rbf")
knn = neighbors.KNeighborsClassifier(n_neighbors=10)
classifiers = (lin, rbf, knn)

# Train classifiers
for est in classifiers:
    print est.__class__
    est.fit(data, target)

# Score classifiers
#for est in classifiers:
#    print est.__class__
#    est.score(data, target)

# Create custom scorer
def targetPurityScore(yTrue, yPred):
    return metrics.precision_score(yTrue, yPred, average=None)[1]
def targetCompletenessScore(yTrue, yPred):
    return metrics.recall_score(yTrue, yPred, average=None)[1]

targetPurityScorer = sklearn.metrics.make_scorer(targetPurityScore)
targetCompletenessScorer = sklearn.metrics.make_scorer(targetCompletenessScore)

# Cross-validate
nFolds = 10
skf = cross_validation.StratifiedKFold(target, nFolds)
for est in classifiers:
    print est.__class__
    print np.mean(cross_validation.cross_val_score(est, data, y=target, cv=skf,
                scoring=targetPurityScorer))
    print np.mean(cross_validation.cross_val_score(est, data, y=target, cv=skf,
                scoring=targetCompletenessScorer))

# TO DO: introduce a simple model to fit for linear combinations of color cuts


# Older CFHTLS stuff

def ccPlot(plt,xColor,yColor,zphot,xlabel,ylabel,title):
    norm=matplotlib.colors.Normalize(vmin=0,vmax=1.2)
    img=plt.scatter(xColor,yColor,c=zphot,s=4,lw=0,norm=norm)
    plt.set_xlim([-0.3,1.5])
    plt.set_ylim([-0.2,1.5])
#    plt.colorbar()
    plt.set_xlabel(xlabel)
    plt.set_ylabel(ylabel)
    plt.set_title(title)
 
    return img

print "reading CFHTLenS catalog"
dataDir="/data/mgeorge/cfhtls/"
cat=astropy.io.ascii.read(dataDir+"w3.tsv",header_start=0,data_start=1)
print "finished reading catalog"
sel=((cat['weight'] > 0) &
     (cat['MAG_r'] > 10.) &
     (cat['MAG_r'] < 23.))
#cat=cat[sel]


magMin=19.5
magMax=22.5
magBin=0.5
nMagBins=int((magMax-magMin)/magBin)

ug=cat['MAG_u']-cat['MAG_g']
gr=cat['MAG_g']-cat['MAG_r']
ri=cat['MAG_r']-cat['MAG_i']

fig=plt.figure()
ii=1

for maglim in np.arange(magMin,magMax,magBin):
    selG=((cat['MAG_g'] > maglim) & (cat['MAG_g'] < maglim+magBin) & (cat['weight'] > 0))
    selI=((cat['MAG_i'] > maglim) & (cat['MAG_i'] < maglim+magBin) & (cat['weight'] > 0))

    gfig=fig.add_subplot(nMagBins,2,ii)
    ifig=fig.add_subplot(nMagBins,2,ii+1)
    ii+=2

    imgG=ccPlot(gfig,gr[selG],ug[selG],cat[selG]['Z_B'],'g-r','u-g',"{}<g<{}".format(maglim,maglim+magBin))
    imgI=ccPlot(ifig,gr[selI],ri[selI],cat[selI]['Z_B'],'g-r','r-i',"{}<i<{}".format(maglim,maglim+magBin))

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(imgG, cax=cbar_ax)

fig.show()


# GRI selection
iMax=20.8
grMax=1.4
sel=((cat['weight'] > 0) &
    (cat['MAG_g'] > 10) &
    (cat['MAG_r'] > 10) &
    (cat['MAG_i'] > 10) &
    (cat['MAG_i'] < iMax) &
    (gr < grMax) &
    (ri>0.2*gr+0.3))
nSel=len(sel.nonzero()[0])
print nSel
plt.clf()
plt.hist(cat[sel]['Z_B'],bins=50)
plt.show()

plt.clf()
imgI=ccPlot(plt.gca(),gr[sel],ri[sel],cat[sel]['Z_B'],'g-r','r-i',"i<{}".format(iMax))
plt.colorbar(imgI)
plt.show()

plt.clf()
plt.hist((cat[sel]['MAG_g'],cat[sel]['MAG_i']),bins=50,color=('green','red'))
plt.show()

sel=((cat['weight'] > 0) & (cat['MAG_i'] > 10) & (cat['MAG_i'] < iMax) & (gr < grMax) & (ri>0.2*gr+0.3) & (cat['Z_B'] > 0.5) & (cat['Z_B'] < 0.6))
plt.clf()
plt.hist(gr[sel],bins=50,range=(-1,3))
plt.show()


# UGR selection
gMax=22.1
grMax=1.4
sel=((cat['weight'] > 0) &
    (cat['MAG_g'] > 10) &
    (cat['MAG_r'] > 10) &
    (cat['MAG_i'] > 10) &
    (cat['MAG_g'] < gMax) &
    (gr < grMax) &
    (ug<0.8*gr+0.1))
nSel=len(sel.nonzero()[0])
print nSel
plt.clf()
plt.hist(cat[sel]['Z_B'],bins=50)
plt.show()

plt.clf()
imgG=ccPlot(plt.gca(),gr[sel],ug[sel],cat[sel]['Z_B'],'g-r','u-g',"i<20.5")
plt.colorbar(imgG)
plt.show()

plt.clf()
plt.hist((cat[sel]['MAG_g'],cat[sel]['MAG_i']),bins=50,color=('green','red'))
plt.show()

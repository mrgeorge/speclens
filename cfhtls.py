#! env python

import astropy.io.ascii
import numpy as np
import matplotlib.pyplot as plt
import sim

print "reading CFHTLenS catalog"
dataDir="/data/mgeorge/cfhtls/"
cat=astropy.io.ascii.read(dataDir+"w3.tsv",header_start=0,data_start=1)
print "finished reading catalog"
sel=((cat['weight'] > 0) &
     (cat['MAG_r'] > 10.) &
     (cat['MAG_r'] < 23.))
cat=cat[sel]
print len(cat)

cat2=cat[0:10]

pixScale=0.187 # arcsec/pixel
disk_n=1
disk_r=cat2['scalelength']*pixScale
bulge_n=4
bulge_r=0.6*disk_r # Miller et al
bulge_frac=cat2['bulge_fraction']

# following Heymans Eq 2
# beta=1/gal_q
# phi=gal_beta
# e1 = (beta - 1)/(beta + 1) * cos(2phi)
# e2 = (beta - 1)/(beta + 1) * sin(2phi)
#-> phi = 0.5 * arctan(e2/e1)
#-> beta = (1 + e1/cos(2phi))/(1 - e1/cos(2phi))

gal_beta=0.5*np.arctan2(cat2['e2'],cat2['e1'])
gal_q=(1. - cat2['e1']/np.cos(2.*gal_beta))/(1. + cat2['e1']/np.cos(2.*gal_beta))


gal_flux=cat2['model_flux']


numFib=7
nGal=len(cat2)
fibFlux=np.zeros((nGal,numFib))

print "Calculating fiber fluxes"
for ii in range(nGal):
     print ii
     fibFlux[ii,:]=sim.main(bulge_n,bulge_r[ii],disk_n,disk_r[ii],bulge_frac[ii],gal_q[ii],gal_beta[ii],gal_flux[ii],numFib)

for ii in range(numFib):
     col=astropy.tables.Column(data=fibFlux[:,ii],name="fiberFlux{}".format(ii))
     cat2.add_column(col)

outFile=dataDir+"test.dat"
print "Writing to {}".format(outFile)
cat2.write(outFile,format='ascii')

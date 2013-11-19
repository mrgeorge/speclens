#! env python

import astropy.io.ascii
import numpy as np
import matplotlib.pyplot as plt
import sim
import fitsio
import esutil

print "reading DR9 spAll ELG file"
elg=fitsio.read("/data/mgeorge/cfhtls/spAll-v5_4_45_elg.fits",ext=1)
print "finished reading DR9 spAll elg"
sel=((elg['PLUG_DEC'] > 30) & (elg['ZWARNING']==0))
w3=elg[sel]

print "reading CFHTLenS catalog"
dataDir="/data/mgeorge/cfhtls/"
cat=astropy.io.ascii.read(dataDir+"w3.tsv",header_start=0,data_start=1)
print "finished reading catalog"
sel=((cat['weight'] > 0) &
     (cat['MAG_r'] > 10.) &
     (cat['MAG_r'] < 23.))
cat=cat[sel]
#print len(cat)

# match BOSS spectra with CFHTLenS targets
print "matching"
h=esutil.htm.HTM()
m1,m2,dist = h.match(w3['PLUG_RA'],w3['PLUG_DEC'],cat['ALPHA_J2000'],cat['DELTA_J2000'],1./3600)
print "N_boss={}, N_cfht={}, N_match={}".format(len(w3),len(cat),len(m1))
# note, there are a bunch of failed matches here!
bossMatch=w3[m1]
cfhtMatch=cat[m2]



pixScale=0.187 # arcsec/pixel
disk_n=1
disk_r=cfhtMatch['scalelength']*pixScale
bulge_n=4
bulge_r=0.6*disk_r # Miller et al
bulge_frac=cfhtMatch['bulge_fraction']

# following Heymans Eq 2
# beta=1/gal_q
# phi=gal_beta
# e1 = (beta - 1)/(beta + 1) * cos(2phi)
# e2 = (beta - 1)/(beta + 1) * sin(2phi)
#-> phi = 0.5 * arctan(e2/e1)
#-> beta = (1 + e1/cos(2phi))/(1 - e1/cos(2phi))

gal_beta=0.5*np.arctan2(cfhtMatch['e2'],cfhtMatch['e1'])
gal_q=(1. - cfhtMatch['e1']/np.cos(2.*gal_beta))/(1. + cfhtMatch['e1']/np.cos(2.*gal_beta))


gal_flux=cfhtMatch['model_flux']


numFib=7
nGal=len(cfhtMatch)
fibFlux=np.zeros((nGal,numFib))

print "Calculating fiber fluxes"
for ii in range(nGal):
     print ii
     fibFlux[ii,:]=sim.main(bulge_n,bulge_r[ii],disk_n,disk_r[ii],bulge_frac[ii],gal_q[ii],gal_beta[ii],gal_flux[ii],numFib)

for ii in range(numFib):
     col=astropy.table.Column(data=fibFlux[:,ii],name="fiberFlux{}".format(ii))
     cfhtMatch.add_column(col)

outFile=dataDir+"w3fib.dat"
print "Writing to {}".format(outFile)
cfhtMatch.write(outFile,format='ascii')

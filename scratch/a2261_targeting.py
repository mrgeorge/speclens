import astropy.io.ascii
from astropy.coordinates import ICRS
from astropy import units
import pandas as pd
import numpy as np

dataDir = "/Users/mgeorge/data/speclens/Keck/catalogs/"
catFile = dataDir + "A2261_Subaru_mario_photbpz.cat"
redFile = dataDir + "A2261_redcomb.asc"
blueFile = dataDir + "A2261_bluecomb.asc"

# use astropy to ingest file since it handles the header well
# then convert to pandas dataframe
df = pd.DataFrame(astropy.io.ascii.read(catFile)._data).set_index('id')

wlCols = ["ID", "RA", "Dec", "X", "Y", "g1", "g2", "weight", "rg", "mag",
          "htr_pg", "B", "V", "R", "zb", "odds"]
widths = [10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
red = pd.read_fwf(redFile, index_col=0, names=wlCols, widths=widths)
blue = pd.read_fwf(blueFile, index_col=0, names=wlCols, widths=widths)

copyCols = ["g1", "g2", "weight", "rg"]
fullCat = df.join(red[copyCols].join(blue[copyCols], how='outer',
                                     lsuffix='_red', rsuffix='_blue'))

# could try other ways to append columns like df['g1'] = red['g1'] but want
# to combine red and blue first ...


# once fullCat is made, slice on color/photoz/size etc for target selection

# Note that selection on shape or size requires Umetsu's red or blue catalog
# Cuts for those samples described in Umetsu++2014 Sec 4.4
# Color selection to identify background samples in Medezinski++2010 Sec 3.

# includes bright cut related to brightest cluster members,
# color cuts designed to cleanly pick out background sample
# Umetsu14 uses BVR for these cuts and R for shape measurements

minRg = 0. # gaussian size (units?)
minRC = 18. # R mag (used as primary band)
maxRC = 24. # R mag (used as primary band)
minZb = 0.6 # photoz
maxZb = 1.2 # photoz
minOdds = 0.8 # odds cut used by Umetsu

sel = ((fullCat.RC > minRC) &
       (fullCat.RC < maxRC) &
       (fullCat.zb > minZb) &
       (fullCat.zb < maxZb) &
       ((fullCat.rg_red > minRg) | (fullCat.rg_blue > minRg)) &
       (fullCat.odds > minOdds))

print "Sample size after cuts: ", len(fullCat[sel])

cat = fullCat[sel]


# Compute position angle
PA = np.empty(len(cat))
PA[:] = None
rad = np.empty_like(PA)
rad[:] = None
ba = np.empty_like(PA)
ba[:] = None
redSel = np.isfinite(cat['g1_red'].values)
blueSel = np.isfinite(cat['g1_blue'].values)
PA[redSel] = np.rad2deg(0.5 * np.arctan2(cat[redSel]['g2_red'],
                                         cat[redSel]['g1_red'])).values
PA[blueSel] = np.rad2deg(0.5 * np.arctan2(cat[blueSel]['g2_blue'],
                                          cat[blueSel]['g1_blue'])).values
rad[redSel] = cat[redSel]['rg_red']
rad[blueSel] = cat[blueSel]['rg_blue']

# TO DO - compute axis ratio ba

# Print target list for IRAF DSIMULATOR
coords = ICRS(cat.RA, cat.Dec, unit=(units.deg, units.deg))
raStr = coords.ra.to_string(unit=units.hour, sep=':')
decStr = coords.dec.to_string(unit=units.deg, sep=':')

maskList = ('a','b','c','d')
for mask in maskList:
    alignFile = dataDir + "align_{}.csv".format(mask)
    guideFile = dataDir + "guide_{}.csv".format(mask)
    alignCat = pd.read_csv(alignFile, names=["ra","dec","mag"]).dropna()
    guideCat = pd.read_csv(guideFile, names=["ra","dec","mag"]).dropna()

    alignCoords = ICRS(alignCat.ra, alignCat.dec, unit=(units.deg, units.deg))
    alignRaStr = alignCoords.ra.to_string(unit=units.hour, sep=':')
    alignDecStr = alignCoords.dec.to_string(unit=units.deg, sep=':')
    guideCoords = ICRS(guideCat.ra, guideCat.dec, unit=(units.deg, units.deg))
    guideRaStr = guideCoords.ra.to_string(unit=units.hour, sep=':')
    guideDecStr = guideCoords.dec.to_string(unit=units.deg, sep=':')

    objFilename = dataDir + "a2261_targets_{}.dat".format(mask)
    with open(objFilename, 'w') as ff:
        ff.write("# OBJNAME         RA          DEC        EQX   MAG band PCODE "
                 "LIST SEL? PA L1 L2\n")
        for ii in range(len(alignCat)):
            ff.write("{name:10} {ra:16} {dec:16} {eqx:8.1f} {mag:6.2f} {band:4} "
                    "{pcode:4} {sample:4} {presel:4} {pa:8s} {len1} {len2}\n".format(
                        name=str(alignCat.iloc[ii].name + 98000).zfill(5),
                        ra=alignRaStr[ii],
                        dec=alignDecStr[ii],
                        eqx=2000.0,
                        mag=alignCat['mag'].iloc[ii],
                        band='r',
                        pcode=-2,
                        sample=1,
                        presel=0,
                        pa="INDEF",
                        len1='',
                        len2=''))
        for ii in range(len(guideCat)):
            ff.write("{name:10} {ra:16} {dec:16} {eqx:8.1f} {mag:6.2f} {band:4} "
                    "{pcode:4} {sample:4} {presel:4} {pa:8s} {len1} {len2}\n".format(
                        name=str(guideCat.iloc[ii].name + 99000).zfill(5),
                        ra=guideRaStr[ii],
                        dec=guideDecStr[ii],
                        eqx=2000.0,
                        mag=guideCat['mag'].iloc[ii],
                        band='r',
                        pcode=-1,
                        sample=2,
                        presel=1,
                        pa="INDEF",
                        len1='',
                        len2=''))
        for ii in range(len(cat)):
            ff.write("{name:10} {ra:16} {dec:16} {eqx:8.1f} {mag:6.2f} {band:4} "
#                 "{pcode:4} {sample:4} {presel:4} {pa:6.1f} {len1} {len2}\n".format(
                     "{pcode:4} {sample:4} {presel:4} {pa:8s} {len1} {len2}\n".format(
                        name=str(cat.iloc[ii].name).zfill(5),
                        ra=raStr[ii],
                        dec=decStr[ii],
                        eqx=2000.0,
                        mag=cat['RC'].iloc[ii],
                        band='R',
                        pcode=1,
                        sample=3,
                        presel=0,
#                     pa=PA[ii],
                        pa="INDEF",
                        len1='',
                        len2=''))

# Print target list for DS9 region file
regFilename = dataDir + "a2261_targets.reg"
with open(regFilename, 'w') as ff:
    for ii in range(len(cat)):
        ff.write("wcs;ellipse({ra:14.6f},{dec:14.6f},{arad:6.1f}p,{brad:6.1f}p,"
                 "{pa:6.1f}) #\n".format(
                    ra=cat.iloc[ii]['RA'],
                    dec=cat.iloc[ii]['Dec'],
                    arad=rad[ii]*5,
                    brad=0.5*rad[ii]*5,
                    pa=PA[ii]))

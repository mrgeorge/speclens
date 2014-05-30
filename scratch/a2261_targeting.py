import astropy.io.ascii
import astropy.io.fits
from astropy.coordinates import ICRS
from astropy import units
import pandas as pd
import numpy as np
import pydl.pydlutils.spheregroup.spherematch as spherematch

dataDir = "/Users/mgeorge/data/speclens/Keck/catalogs/"
catFile = dataDir + "A2261_Subaru_mario_photbpz.cat"
redFile = dataDir + "A2261_redcomb.asc"
blueFile = dataDir + "A2261_bluecomb.asc"
sbpFile = dataDir + "a2261_selected.fits"

# Read CLASH + Keiichi Umetsu's catalogs
# use astropy to ingest file since it handles the header well
# then convert to pandas dataframe
df = pd.DataFrame(astropy.io.ascii.read(catFile)._data).set_index('id')

wlCols = ["ID", "RA", "Dec", "X", "Y", "g1", "g2", "weight", "rg", "mag",
          "htr_pg", "B", "V", "R", "zb", "odds"]
widths = [10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
red = pd.read_fwf(redFile, index_col=0, names=wlCols, widths=widths)
blue = pd.read_fwf(blueFile, index_col=0, names=wlCols, widths=widths)

# Combine CLASH + Umetsu's catalog
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
maxRC = 23. # R mag (used as primary band)
maxZ = 22.5 # z band cut used by Eric
minZb = 0.5 # photoz
maxZb = 1.2 # photoz
minOdds = 0.8 # odds cut used by Umetsu

sel = (
#       (fullCat.RC > minRC) &
#       (fullCat.RC < maxRC) &
#       (fullCat.zb > minZb) &
#       (fullCat.zb < maxZb)
#       ((fullCat.rg_red > minRg) | (fullCat.rg_blue > minRg)) &
#       (fullCat.odds > minOdds)
        (fullCat.Z < maxZ) &
        (fullCat.tb > 8) &
        (fullCat.zb > 0.324)
      )

print "Sample size after cuts: ", len(fullCat[sel])

cat = fullCat[sel]



# Rmead Eric Huff's catalog for SBP fits
sbpCols = ["NUMBER", "MAG_AUTO", "ALPHA_J2000", "DELTA_J2000", "FLAGS",
           "FLUX_POINTSOURCE", "FLUX_SPHEROID", "FLUX_DISK",
           "DISK_SCALE_IMAGE", "DISK_INCLINATION", "DISK_THETA_J2000"]

sbpRec = astropy.io.fits.getdata(sbpFile, 1)

# fits tables are stored as big endian, but pandas prefers little
# must swap endianness in each column in recarray before DataFrame construction
# endian fix from
# https://github.com/aringlis/sunpy/commit/7bc5e222023c8a661c47ce718dc3cd092684fd6f
sbpTable = {}
for i, col in enumerate(sbpRec.columns[1:-1]):
    #temporary patch for big-endian data bug on pandas 0.13
    if sbpRec.field(i+1).dtype.byteorder == '>':
        sbpTable[col.name] = sbpRec.field(i + 1).byteswap().newbyteorder()
    else:
        sbpTable[col.name] = sbpRec.field(i + 1)

sbpNames = sbpRec.dtype.names
sbp = pd.DataFrame.from_records(sbpTable,
                                exclude=set(sbpNames) - set(sbpCols), coerce_float=True)

sep = 1./3600 # 1 arcsecond
m1, m2, d12 = spherematch(cat.RA.values, cat.Dec.values,
                          sbp.ALPHA_J2000.values, sbp.DELTA_J2000.values,
                          sep, maxmatch=1)

# Keep only the matches
print "Sample size after matching: ", len(m1)
cat = cat.iloc[m1]
sbp = sbp.iloc[m2]

# Further cuts
sel = (
#       (sbp.FLUX_DISK > sbp.FLUX_SPHEROID) &
#       (sbp.FLUX_DISK > sbp.FLUX_POINTSOURCE)
       (sbp.FLAGS == 0) &
       (sbp.DISK_SCALE_IMAGE * 1.68 * 0.2 > 1.0)
      )
print "Sample size after morph cuts: ", len(sbp[sel])

sbp = sbp[sel]
cat = cat[sel.values] # kludge since indices aren't joined


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









################################################################################
################################################################################
# The following catalog is a combination of the above files with cuts from Eric Huff
# so the above code can be ignored.
################################################################################
################################################################################


catRec = astropy.io.fits.getdata(dataDir + "mcat_disk_flux_cut_slightly_relaxed.fits", 1)
catRecSwap = {}
for i, col in enumerate(catRec.columns[1:-1]):
    #temporary patch for big-endian data bug on pandas 0.13
    if catRec.field(i+1).dtype.byteorder == '>':
        catRecSwap[col.name] = catRec.field(i + 1).byteswap().newbyteorder()
    else:
        catRecSwap[col.name] = catRec.field(i + 1)
cat = pd.DataFrame.from_records(catRecSwap)



# Print target list for IRAF DSIMULATOR
coords = ICRS(cat.ALPHA_J2000, cat.DELTA_J2000, unit=(units.deg, units.deg))
raStr = coords.ra.to_string(unit=units.hour, sep=':')
decStr = coords.dec.to_string(unit=units.deg, sep=':')

maskList = ('a','b','c','d')
#maskPAs = (60., -120., 150., -30.) # for use in dsim since masks can fill -180 to 180
maskPAs = (60., 60., -30., -30.) # wrapped over -90 to 90 for simpler comparison to galaxy PAs
for mask, maskPA in zip(maskList, maskPAs):
    alignFile = dataDir + "align_{}.csv".format(mask)
    guideFile = dataDir + "guide_{}.csv".format(mask)
    alignCat = pd.read_csv(alignFile, names=["ra","dec","mag"], comment='#').dropna()
    guideCat = pd.read_csv(guideFile, names=["ra","dec","mag"], comment='#').dropna()

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
                        presel=1,
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
        pixscale = 0.2
        extralen = 1. # extra length in each direction in arcsec
        qz = 0.2 # typical disk thickness c/a

        nMajor = 0
        nMinor = 0
        nOther = 0
        for ii in range(len(cat)):

            # start with WCS position angle
            # if |pa - mask PA| < 30, we can get major axis
            # else if |(pa + 90) - mask PA| < 30 we can get minor axis
            # else lower priority

            pa = cat['DISK_THETA_J2000'].iloc[ii]
            if (np.abs(pa - maskPA) < 30.): # use major axis
                halfslit = cat['DISK_SCALE_IMAGE'].iloc[ii] * 0.2 * 2.2 + extralen
                halfslit = np.min([halfslit, 10.])
                halfslit = np.max([halfslit, 4.])
                nMajor += 1
            elif (np.abs(pa - ((maskPA + 90 + 90) % 180 - 90)) < 30.): # use minor axis
                pa = pa + 90
                halfslit = cat['DISK_SCALE_IMAGE'].iloc[ii] * 0.2 * 2.2 * np.sqrt(1. - np.sin(cat['DISK_INCLINATION'].iloc[ii])**2 * (1. - qz**2)) + extralen
                halfslit = np.min([halfslit, 6.])
                halfslit = np.max([halfslit, 3.])
                nMinor += 1
            else:
                pa = maskPA + 5. # small offset from mask PA for better lsf sampling
                halfslit = cat['DISK_SCALE_IMAGE'].iloc[ii] * 0.2 * 2.2 + extralen
                halfslit = np.min([halfslit, 10.])
                halfslit = np.max([halfslit, 4.])
                nOther += 1

            # assign priorities
            # goal is to get bright objects near z ~ 0.8
            # but not to prioritize any individual object too heavily
            pcode = (30. - cat['RCMAG'].iloc[ii]) - 2.*np.abs(cat['ZPHOT'].iloc[ii] - 0.8)

            ff.write("{name:10} {ra:16} {dec:16} {eqx:8.1f} {mag:6.2f} {band:4} "
                     "{pcode:4} {sample:4} {presel:4} {pa:6.1f} {len1:6.1f} {len2:6.1f}\n".format(
#                     "{pcode:4} {sample:4} {presel:4} {pa:8s} {len1} {len2}\n".format(
                        name=str(cat.iloc[ii].name).zfill(5),
                        ra=raStr[ii],
                        dec=decStr[ii],
                        eqx=2000.0,
                        mag=cat['RCMAG'].iloc[ii],
                        band='R',
                        pcode=pcode,
                        sample=3,
                        presel=0,
                        pa=pa,
#                        pa=maskPA,
#                        pa="INDEF",
                        len1=halfslit,
                        len2=halfslit))

        print nMajor, nMinor, nOther


# Print target list for DS9 region file
regFilename = dataDir + "a2261_targets.reg"
with open(regFilename, 'w') as ff:
    for ii in range(len(cat)):
        ff.write("wcs;ellipse({ra:14.6f},{dec:14.6f},{arad:6.1f}p,{brad:6.1f}p,"
                 "{pa:6.1f}) #\n".format(
                    ra=cat['ALPHA_J2000'].iloc[ii],
                    dec=cat['DELTA_J2000'].iloc[ii],
                    arad=cat['DISK_SCALE_IMAGE'].iloc[ii]*5,
                    brad=0.5*cat['DISK_SCALE_IMAGE'].iloc[ii]*5,
                    pa=cat['DISK_THETA_J2000'].iloc[ii] + 90.)
                )

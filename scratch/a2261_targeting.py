import astropy.io.ascii
import pandas as pd

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

minRg = 1. # gaussian size (units?)
minRC = 18. # R mag (used as primary band)
maxRC = 23.7 # R mag (used as primary band)
minZb = 0.6 # photoz
maxZb = 1.2 # photoz
minOdds = 0.8 # odds cut used by Umetsu

sel = ((fullCat.RC < maxRC) &
       (fullCat.zb > minZb) &
       (fullCat.zb < maxZb) &
       ((fullCat.rg_red > minRg) | (fullCat.rg_blue > minRg)) &
       (fullCat.odds > minOdds))

print "Sample size after cuts: ", len(fullCat[sel])

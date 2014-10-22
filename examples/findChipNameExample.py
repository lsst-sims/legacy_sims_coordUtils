import numpy
from collections import OrderedDict
import lsst.afw.cameraGeom.testUtils as testUtils
from lsst.sims.catalogs.generation.db import ObservationMetaData, \
                                             calcObsDefaults, getRotTelPos, \
                                             altAzToRaDec, Site
from lsst.sims.coordUtils import CameraCoords

"""
Using cameraGeomExample.py in afw, we ought to be able to make an LSST-like
camera object

However, most of the methods behind findChipName assume that you are calling it
from a catalog object (with all of the attendant MJD, Site, dbobj.epoch
variables available.

Something must be done to make the method more general.

split the method up

one method will perform the calculation.  It will accept an ObservationMetaData
object, a Site object, and a reference epoch

the other method will be a getter that will actually perform the calculations

I suspect we should just split all of the astrometry methods

"""

def makeObservationMetaData():
    #create the ObservationMetaData object
    mjd = 52000.0
    alt = numpy.pi/2.0
    az = 0.0
    band = 'r'
    testSite = Site()
    centerRA, centerDec = altAzToRaDec(alt,az,testSite.longitude,testSite.latitude,mjd)
    rotTel = getRotTelPos(az, centerDec, testSite.latitude, 0.0)

    obsDict = calcObsDefaults(centerRA, centerDec, alt, az, rotTel, mjd, band, 
                 testSite.longitude, testSite.latitude)

    obsDict['Opsim_expmjd'] = mjd
    radius = 0.1
    phoSimMetadata = OrderedDict([
                      (k, (obsDict[k],numpy.dtype(type(obsDict[k])))) for k in obsDict])

    obs_metadata = ObservationMetaData(boundType = 'circle', unrefractedRA = numpy.degrees(centerRA),
                                       unrefractedDec = numpy.degrees(centerDec), boundLength = 2.0*radius,
                                       phoSimMetadata=phoSimMetadata)

    return obs_metadata

camera = testUtils.CameraWrapper(isLsstLike=True).camera
obs_metadata = makeObservationMetaData()

import numpy
import os
from collections import OrderedDict
import lsst.afw.cameraGeom.testUtils as testUtils
from lsst.sims.catalogs.generation.db import ObservationMetaData, \
                                             calcObsDefaults, getRotTelPos, \
                                             altAzToRaDec, Site
from lsst.sims.coordUtils import CameraCoords

import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE
from lsst.afw.cameraGeom.cameraConfig import CameraConfig
from lsst.afw.cameraGeom.cameraFactory import makeCameraFromPath
from lsst.afw.table import AmpInfoCatalog
from astropy.io import fits

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

def convertAmpNames(rawName):
   name = rawName.replace(':','')
   name = name.replace(',','')
   name = name.replace(' ','')
   name = name.replace('S','_S')
   return name

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
                                       phoSimMetadata=phoSimMetadata, site=testSite)

    return obs_metadata

epoch = 2000.0
obs_metadata = makeObservationMetaData()

filedir = os.path.join(os.getenv('OBS_LSSTSIM_DIR'), 'description/camera/')
filename = os.path.join(filedir, 'camera.py')
myCameraConfig = CameraConfig()
myCameraConfig.load(filename)

camera = makeCameraFromPath(myCameraConfig, filedir, convertAmpNames)

myCamCoords = CameraCoords()

rap, decp = myCamCoords.applyMeanApparentPlace(numpy.array([numpy.radians(obs_metadata.unrefractedRA)]),
                                               numpy.array([numpy.radians(obs_metadata.unrefractedDec)]),
                                               Epoch0=epoch, MJD=obs_metadata.mjd)

ra, dec = myCamCoords.applyMeanObservedPlace(rap,
                                            decp,
                                            obs_metadata=obs_metadata)

print myCamCoords.findChipName(ra=ra, dec=dec, epoch=epoch, camera=camera, obs_metadata=obs_metadata)

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

def convertAmpNames(rawName):
   #This method takes the names of detectors as stored in a CameraConfig object
   #and converts them into the roots of the fits files storing amplifier information

   name = rawName.replace(':','')
   name = name.replace(',','')
   name = name.replace(' ','')
   name = name.replace('S','_S')
   return name

def makeLSSTcamera():
    filedir = os.path.join(os.getenv('OBS_LSSTSIM_DIR'), 'description/camera/')
    #this directory contains camera.py, which describes the lay out of
    #detectors on the LSST camera, as well as fits files that describe
    #the amplifiers on those detectors

    filename = os.path.join(filedir, 'camera.py')
    myCameraConfig = CameraConfig()
    myCameraConfig.load(filename)
    #converts the camera.py file into a CameraConfig object

    camera = makeCameraFromPath(myCameraConfig, filedir, convertAmpNames)
    #reads in the CameraConfig file as well as the fits files describing the
    #amplifiers and makes an afwCameraGeom.Camera object out of them

    return camera

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

myCamCoords = CameraCoords()

camera = makeLSSTcamera()

nsamples = 10
numpy.random.seed(32)
rr = numpy.radians(2.0)*numpy.random.sample(nsamples)
theta = 2.0*numpy.pi*numpy.random.sample(nsamples)

ra = numpy.radians(obs_metadata.unrefractedRA) + rr*numpy.cos(theta)
dec = numpy.radians(obs_metadata.unrefractedDec) + rr*numpy.sin(theta)

chipNames = myCamCoords.findChipName(ra=ra, dec=dec, epoch=epoch, camera=camera, obs_metadata=obs_metadata)

for (rr,dd,nn) in zip(ra,dec,chipNames):
    print rr,dd,nn

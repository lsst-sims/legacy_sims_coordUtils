"""
This file is just makeLsstCameraRepository.py from obs_lsstSim/bin/

the method ReturnCamera() is the __main__ method from that script
modified so that it just returns a camera, rather than writing out a
repository of camera data
"""

from __future__ import absolute_import, division
from builtins import range
import argparse
import os
import re
import shutil

import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
from lsst.afw.cameraGeom import SCIENCE
from lsst.afw.cameraGeom import (DetectorConfig, CameraConfig, makeCameraFromCatalogs,
                                 FIELD_ANGLE, FOCAL_PLANE, PIXELS)

__all__ = ["ReturnCamera"]

def expandDetectorName(abbrevName):
    return abbrevName

def detectorIdFromAbbrevName(abbrevName):
    """Compute detector ID from an abbreviated detector name of the form Rxy_Sxy_Ci

    value = digits in this order: ci+1 rx ry sx sy
    """
    idNum=int(abbrevName[-2:])
    return idNum

def makeAmpTables(segmentsFile, gainFile):
    """
    Read the segments file from a PhoSim release and produce the appropriate AmpInfo
    @param segmentsFile -- String indicating where the file is located
    """
    gainDict = {}
    """
    with open(gainFile) as fh:
        for l in fh:
            els = l.rstrip().split()
            gainDict[els[0]] = {'gain':float(els[1]), 'saturation':int(els[2])}
    """
    returnDict = {}
    #TODO currently there is no linearity provided, but we should identify
    #how to get this information.
    linearityCoeffs = (0.,1.,0.,0.)
    linearityType = "Polynomial"
    readoutMap = {'LL':afwTable.LL, 'LR':afwTable.LR, 'UR':afwTable.UR, 'UL':afwTable.UL}
    ampCatalog = None
    detectorName = [] # set to a value that is an invalid dict key, to catch bugs
    correctY0 = False
    with open(segmentsFile) as fh:
        for l in fh:
            if l.startswith("#"):
                continue

            els = l.rstrip().split()
            if len(els) == 4:
                if ampCatalog is not None:
                    returnDict[detectorName] = ampCatalog
                detectorName = expandDetectorName(els[0])
                numy = int(els[2])
                schema = afwTable.AmpInfoTable.makeMinimalSchema()
                ampCatalog = afwTable.AmpInfoCatalog(schema)
                if len(els[0].split('_')) == 3:   #wavefront sensor
                    correctY0 = True
                else:
                    correctY0 = False
                continue
            record = ampCatalog.addNew()
            name = els[0].split("_")[-1]
            name = '%s,%s'%(name[1], name[2])
            #Because of the camera coordinate system, we choose an
            #image coordinate system that requires a -90 rotation to get
            #the correct pixel positions from the
            #phosim segments file
            y0 = numy - 1 - int(els[2])
            y1 = numy - 1 - int(els[1])
            #Another quirk of the phosim file is that one of the wavefront sensor
            #chips has an offset of 2000 pix in y.  It's always the 'C1' chip.
            if correctY0:
                if y0 > 0:
                    y1 -= y0
                    y0 = 0
            x0 = int(els[3])
            x1 = int(els[4])
            try:
                saturation = gainDict[els[0]]['saturation']
                gain = gainDict[els[0]]['gain']
            except KeyError:
                # Set default if no gain exists
                saturation = 65535
                gain = float(els[7])
            readnoise = float(els[11])
            bbox = afwGeom.Box2I(afwGeom.Point2I(x0, y0), afwGeom.Point2I(x1, y1))

            if int(els[5]) == -1:
                flipx = False
            else:
                flipx = True
            if int(els[6]) == 1:
                flipy = False
            else:
                flipy = True

            #Since the amps are stored in amp coordinates, the readout is the same
            #for all amps
            readCorner = readoutMap['LL']

            ndatax = x1 - x0 + 1
            ndatay = y1 - y0 + 1
            #Because in versions v3.3.2 and earlier there was no overscan, we use the extended register as the overscan region
            prescan = 1
            hoverscan = 0
            extended = 4
            voverscan = 0
            rawBBox = afwGeom.Box2I(afwGeom.Point2I(0,0), afwGeom.Extent2I(extended+ndatax+hoverscan, prescan+ndatay+voverscan))
            rawDataBBox = afwGeom.Box2I(afwGeom.Point2I(extended, prescan), afwGeom.Extent2I(ndatax, ndatay))
            rawHorizontalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(0, prescan), afwGeom.Extent2I(extended, ndatay))
            rawVerticalOverscanBBox = afwGeom.Box2I(afwGeom.Point2I(extended, prescan+ndatay), afwGeom.Extent2I(ndatax, voverscan))
            rawPrescanBBox = afwGeom.Box2I(afwGeom.Point2I(extended, 0), afwGeom.Extent2I(ndatax, prescan))

            extraRawX = extended + hoverscan
            extraRawY = prescan + voverscan
            rawx0 = x0 + extraRawX*(x0//ndatax)
            rawy0 = y0 + extraRawY*(y0//ndatay)
            #Set the elements of the record for this amp
            record.setBBox(bbox)
            record.setName(name)
            record.setReadoutCorner(readCorner)
            record.setGain(gain)
            record.setSaturation(saturation)
            record.setReadNoise(readnoise)
            record.setLinearityCoeffs(linearityCoeffs)
            record.setLinearityType(linearityType)
            record.setHasRawInfo(True)
            record.setRawFlipX(flipx)
            record.setRawFlipY(flipy)
            record.setRawBBox(rawBBox)
            record.setRawXYOffset(afwGeom.Extent2I(rawx0, rawy0))
            record.setRawDataBBox(rawDataBBox)
            record.setRawHorizontalOverscanBBox(rawHorizontalOverscanBBox)
            record.setRawVerticalOverscanBBox(rawVerticalOverscanBBox)
            record.setRawPrescanBBox(rawPrescanBBox)
    returnDict[detectorName] = ampCatalog
    return returnDict

def makeLongName(shortName):
    """
    Make the long name from the PhoSim short name
    @param shortName -- string name like R??_S??[_C??] to parse
    """
    parts = shortName.split("_")
    if len(parts) == 2:
        return " ".join(["%s:%s"%(el[0], ",".join(el[1:])) for el in parts])
    elif len(parts) == 3:
        #This must be a wavefront sensor
        wsPartMap = {'S':{'C0':'A', 'C1':'B'},
                     'R':{'C0':'', 'C1':''}}
        return " ".join(["%s:%s"%(el[0], ",".join(el[1:]+wsPartMap[el[0]][parts[-1]])) for el in parts[:-1]])
    else:
        raise ValueError("Could not parse %s: has %i parts"%(shortName, len(parts)))

def makeDetectorConfigs(detectorLayoutFile, phosimVersion):
    """
    Create the detector configs to use in building the Camera
    @param detectorLayoutFile -- String describing where the focalplanelayout.txt file is located.

    @todo:
    * set serial to something other than name (e.g. include git sha)
    * deal with the extra orientation angles (not that they really matter)
    """
    detectorConfigs = []
    detType = int(SCIENCE)
    #We know we need to rotate 3 times and also apply the yaw perturbation
    nQuarter = 1
    with open(detectorLayoutFile) as fh:
        for l in fh:
            if l.startswith("#"):
                continue
            detConfig = DetectorConfig()
            els = l.rstrip().split()
            detConfig.name = expandDetectorName(els[0])
            detConfig.id = detectorIdFromAbbrevName(els[0])
            detConfig.bbox_x0 = 0
            detConfig.bbox_y0 = 0
            detConfig.bbox_x1 = int(els[5]) - 1
            detConfig.bbox_y1 = int(els[4]) - 1
            detConfig.detectorType = detType
            detConfig.serial = els[0]+"_"+phosimVersion

            # Convert from microns to mm.
            detConfig.offset_x = float(els[1])/1000. + float(els[12])
            detConfig.offset_y = float(els[2])/1000. + float(els[13])

            detConfig.refpos_x = (int(els[5]) - 1.)/2.
            detConfig.refpos_y = (int(els[4]) - 1.)/2.
            # TODO translate between John's angles and Orientation angles.
            # It's not an issue now because there is no rotation except about z in John's model.
            detConfig.yawDeg = 90.*nQuarter + float(els[9])
            detConfig.pitchDeg = float(els[10])
            detConfig.rollDeg = float(els[11])
            detConfig.pixelSize_x = float(els[3])/1000.
            detConfig.pixelSize_y = float(els[3])/1000.
            detConfig.transposeDetector = False
            detConfig.transformDict.nativeSys = PIXELS.getSysName()
            # The FOCAL_PLANE and TAN_PIXEL transforms are generated by the Camera maker,
            # based on orientaiton and other data.
            # Any additional transforms (such as ACTUAL_PIXELS) should be inserted here.
            detectorConfigs.append(detConfig)
    return detectorConfigs

def ReturnCamera(baseDir):
    """
    This method reads in the files

    baseDir/focalplanelayout.txt
    baseDir/segmentation.txt

    and returns an afw.cameraGeom object

    Below is the original documentation of the function this code was copied from:

    Create the configs for building a camera.  This runs on the files distributed with PhoSim.  Currently gain and
    saturation need to be supplied as well.  The file should have three columns: on disk amp id (R22_S11_C00), gain, saturation.
    For example:
    DetectorLayoutFile -- https://dev.lsstcorp.org/cgit/LSST/sims/phosim.git/plain/data/lsst/focalplanelayout.txt?h=dev
    SegmentsFile -- https://dev.lsstcorp.org/cgit/LSST/sims/phosim.git/plain/data/lsst/segmentation.txt?h=dev
    """
    defaultOutDir = 'scratch'

    DetectorLayoutFile = os.path.join(baseDir, 'focalplanelayout.txt')
    SegmentsFile = os.path.join(baseDir, 'segmentation.txt')
    GainFile = None
    phosimVersion='1.0'

    ampTableDict = makeAmpTables(SegmentsFile, GainFile)
    detectorConfigList = makeDetectorConfigs(DetectorLayoutFile, phosimVersion)

    #Build the camera config.
    camConfig = CameraConfig()
    camConfig.detectorList = dict([(i,detectorConfigList[i]) for i in range(len(detectorConfigList))])
    camConfig.name = 'LSST'
    camConfig.plateScale = 2.0 #arcsec per mm
    pScaleRad = afwGeom.arcsecToRad(camConfig.plateScale)
    pincushion = 0.925
    # Don't have this yet ticket/3155
    #camConfig.boresiteOffset_x = 0.
    #camConfig.boresiteOffset_y = 0.
    tConfig = afwGeom.TransformConfig()
    tConfig.transform.name = 'inverted'
    radialClass = afwGeom.xyTransformRegistry['radial']
    tConfig.transform.active.transform.retarget(radialClass)
    # According to Dave M. the simulated LSST transform is well approximated (1/3 pix)
    # by a scale and a pincusion.

    #this is ultimately used to convert from focal plane coordinates to pupil coordinates
    #see the asgnment below to tmc.transforms
    tConfig.transform.active.transform.coeffs = [0., 1./pScaleRad, 0., pincushion/pScaleRad]

    #tConfig.transform.active.boresiteOffset_x = camConfig.boresiteOffset_x
    #tConfig.transform.active.boresiteOffset_y = camConfig.boresiteOffset_y
    tmc = afwGeom.TransformMapConfig()
    tmc.nativeSys = FOCAL_PLANE.getSysName()
    tmc.transforms = {FIELD_ANGLE.getSysName():tConfig}
    camConfig.transformDict = tmc

    myCamera = makeCameraFromCatalogs(camConfig, ampTableDict)
    return myCamera



import numpy as np
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, TAN_PIXELS, FOCAL_PLANE
from lsst.sims.utils import _pupilCoordsFromRaDec, _raDecFromPupilCoords

__all__ = ["getCornerPixels", "_getCornerRaDec", "getCornerRaDec",
           "chipNameFromPupilCoords", "chipNameFromRaDec", "_chipNameFromRaDec",
           "pixelCoordsFromPupilCoords", "pixelCoordsFromRaDec", "_pixelCoordsFromRaDec",
           "focalPlaneCoordsFromPupilCoords", "focalPlaneCoordsFromRaDec", "_focalPlaneCoordsFromRaDec",
           "pupilCoordsFromPixelCoords",
           "raDecFromPixelCoords", "_raDecFromPixelCoords"]


def getCornerPixels(detector_name, camera):
    """
    Return the pixel coordinates of the corners of a detector.

    @param [in] detector_name is the name of the detector in question

    @param [in] camera is the afwCameraGeom camera object containing
    that detector

    @param [out] a list of tuples representing the (x,y) pixel coordinates
    of the corners of the detector.  Order will be

    [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    """

    det = camera[detector_name]
    bbox = det.getBBox()
    xmin = bbox.getMinX()
    xmax = bbox.getMaxX()
    ymin = bbox.getMinY()
    ymax = bbox.getMaxY()
    return [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]


def getCornerRaDec(detector_name, camera, obs_metadata, epoch=2000.0,
                   includeDistortion=True):
    """
    Return the ICRS RA, Dec values of the corners of the specified
    detector in degrees.

    @param [in] detector_name is the name of the detector in question

    @param [in] camera is the afwCameraGeom camera object containing
    that detector

    @param [in] obs_metadata is an ObservationMetaData characterizing
    the pointing (and orientation) of the telescope.

    @param [in] epoch is the mean Julian epoch of the coordinate system
    (default is 2000)

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    convert from pixel coordinates to RA, Dec with optical distortion included.  If False, this
    method will use TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a list of tuples representing the (RA, Dec) coordinates
    of the corners of the detector in degrees.  The corners will be
    returned in the order

    [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

    where (x, y) are pixel coordinates.  This will not necessarily
    correspond to any order in RAmin, RAmax, DecMin, DecMax, because
    of the ambiguity imposed by the rotator angle.
    """

    cc = _getCornerRaDec(detector_name, camera, obs_metadata,
                         epoch=epoch, includeDistortion=includeDistortion)
    return [(np.degrees(cc[0][0]), np.degrees(cc[0][1])),
            (np.degrees(cc[1][0]), np.degrees(cc[1][1])),
            (np.degrees(cc[2][0]), np.degrees(cc[2][1])),
            (np.degrees(cc[3][0]), np.degrees(cc[3][1]))]


def _getCornerRaDec(detector_name, camera, obs_metadata,
                    epoch=2000.0, includeDistortion=True):
    """
    Return the ICRS RA, Dec values of the corners of the specified
    detector in radians.

    @param [in] detector_name is the name of the detector in question

    @param [in] camera is the afwCameraGeom camera object containing
    that detector

    @param [in] obs_metadata is an ObservationMetaData characterizing
    the pointing (and orientation) of the telescope.

    @param [in] epoch is the mean Julian epoch of the coordinate system
    (default is 2000)

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    convert from pixel coordinates to RA, Dec with optical distortion included.  If False, this
    method will use TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a list of tuples representing the (RA, Dec) coordinates
    of the corners of the detector in radians.  The corners will be
    returned in the order

    [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

    where (x, y) are pixel coordinates.  This will not necessarily
    correspond to any order in RAmin, RAmax, DecMin, DecMax, because
    of the ambiguity imposed by the rotator angle.
    """

    cc_pix = getCornerPixels(detector_name, camera)

    ra, dec = _raDecFromPixelCoords(np.array([cc[0] for cc in cc_pix]),
                                    np.array([cc[1] for cc in cc_pix]),
                                    [detector_name]*len(cc_pix),
                                    camera=camera, obs_metadata=obs_metadata,
                                    epoch=epoch, includeDistortion=True)

    return [(ra[0], dec[0]), (ra[1], dec[1]), (ra[2], dec[2]), (ra[3], dec[3])]



def chipNameFromRaDec(ra, dec, obs_metadata=None, epoch=None, camera=None,
                           allow_multiple_chips=False):
    """
    Return the names of detectors that see the object specified by
    either (xPupil, yPupil).

    @param [in] ra in degrees (a numpy array).
    In the International Celestial Reference System.

    @param [in] dec in degrees (a numpy array).
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope pointing

    @param [in] epoch is the epoch in Julian years of the equinox against which RA and Dec are
    measured

    @param [in] camera is an afw.cameraGeom camera instance characterizing the camera

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, an exception will be raised.  If it is 'True'
    and an object falls on more than one chip, it will still only return the first chip in the
    list of chips returned. THIS BEHAVIOR SHOULD BE FIXED IN A FUTURE TICKET.

    @param [out] a numpy array of chip names
    """

    return _chipNameFromRaDec(np.radians(ra), np.radians(dec),
                                  obs_metadata=obs_metadata, epoch=epoch,
                                  camera=camera, allow_multiple_chips=allow_multiple_chips)


def _chipNameFromRaDec(ra, dec, obs_metadata=None, epoch=None, camera=None,
                           allow_multiple_chips=False):
    """
    Return the names of detectors that see the object specified by
    either (xPupil, yPupil).

    @param [in] ra in radians (a numpy array).
    In the International Celestial Reference System.

    @param [in] dec in radians (a numpy array).
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope pointing

    @param [in] epoch is the epoch in Julian years of the equinox against which RA and Dec are
    measured

    @param [in] camera is an afw.cameraGeom camera instance characterizing the camera

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, an exception will be raised.  If it is 'True'
    and an object falls on more than one chip, it will still only return the first chip in the
    list of chips returned. THIS BEHAVIOR SHOULD BE FIXED IN A FUTURE TICKET.

    @param [out] a numpy array of chip names
    """

    if not isinstance(ra, np.ndarray) or not isinstance(dec, np.ndarray):
        raise RuntimeError("You need to pass numpy arrays of RA and Dec to chipName")

    if len(ra) != len(dec):
        raise RuntimeError("You passed %d RAs and %d Decs " % (len(ra), len(dec)) +
                           "to chipName.")

    if epoch is None:
        raise RuntimeError("You need to pass an epoch into chipName")

    if obs_metadata is None:
        raise RuntimeError("You need to pass an ObservationMetaData into chipName")

    if obs_metadata.mjd is None:
        raise RuntimeError("You need to pass an ObservationMetaData with an mjd into chipName")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("You need to pass an ObservationMetaData with a rotSkyPos into chipName")

    xp, yp = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata, epoch=epoch)
    return chipNameFromPupilCoords(xp, yp, camera=camera, allow_multiple_chips=allow_multiple_chips)


def chipNameFromPupilCoords(xPupil, yPupil, camera=None, allow_multiple_chips=False):
    """
    Return the names of detectors that see the object specified by
    either (xPupil, yPupil).

    @param [in] xPupil a numpy array of x pupil coordinates in radians

    @param [in] yPupil a numpy array of y pupil coordinates in radians

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, an exception will be raised.  If it is 'True'
    and an object falls on more than one chip, it will still only return the first chip in the
    list of chips returned. THIS BEHAVIOR SHOULD BE FIXED IN A FUTURE TICKET.

    @param [in] camera is an afwCameraGeom object that specifies the attributes of the camera.

    @param [out] a numpy array of chip names

    """

    if not isinstance(xPupil, np.ndarray) or not isinstance(yPupil, np.ndarray):
        raise RuntimeError("You need to pass numpy arrays of xPupil and yPupil to chipNameFromPupilCoords")

    if len(xPupil) != len(yPupil):
        raise RuntimeError("You passed %d xPupils and %d yPupils " % (len(xPupil), len(yPupil)) +
                           "to chipName.")

    if camera is None:
        raise RuntimeError("No camera defined.  Cannot run chipName.")

    chipNames = []

    cameraPointList = [afwGeom.Point2D(x,y) for x,y in zip(xPupil, yPupil)]

    detList = camera.findDetectorsList(cameraPointList, PUPIL)

    for pt, det in zip(cameraPointList, detList):
        if len(det)==0 or np.isnan(pt.getX()) or np.isnan(pt.getY()):
            chipNames.append(None)
        else:
            names = [dd.getName() for dd in det]
            if len(names)>1 and not allow_multiple_chips:
                raise RuntimeError("This method does not know how to deal with cameras " +
                                   "where points can be on multiple detectors.  " +
                                   "Override CameraCoords.get_chipName to add this.\n" +
                                   "If you were only asking for the chip name (as opposed " +
                                   "to pixel coordinates) you can try re-running with " +
                                   "the kwarg allow_multiple_chips=True.")
            elif len(names)==0:
                chipNames.append(None)
            else:
                chipNames.append(names[0])

    return np.array(chipNames)


def pixelCoordsFromRaDec(ra, dec, obs_metadata=None, epoch=None,
                          chipNames=None, camera=None, includeDistortion=True):
    """
    Get the pixel positions (or nan if not on a chip) for objects based
    on their RA, and Dec (in degrees)

    @param [in] ra is a numpy array containing the RA of the objects in degrees.
    In the International Celestial Reference System.

    @param [in] dec is a numpy array containing the Dec of the objects in degrees.
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope
    pointing.

    @param [in] epoch is the epoch in Julian years of the equinox against which
    RA is measured.

    @param [in] chipNames a numpy array of chipNames.  If it is None, this method will call chipName
    to find the array.  The option exists for the user to specify chipNames, just in case the user
    has already called chipName for some reason.

    @param [in] camera is an afwCameraGeom object specifying the attributes of the camera.
    This is an optional argument to be passed to chipName.

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    return the true pixel coordinates with optical distortion included.  If False, this
    method will return TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the x pixel coordinate
    and the second row is the y pixel coordinate
    """

    return _pixelCoordsFromRaDec(np.radians(ra), np.radians(dec),
                                 chipNames=chipNames, camera=camera,
                                 includeDistortion=includeDistortion,
                                 obs_metadata=obs_metadata, epoch=epoch)


def _pixelCoordsFromRaDec(ra, dec, obs_metadata=None, epoch=None,
                          chipNames=None, camera=None, includeDistortion=True):
    """
    Get the pixel positions (or nan if not on a chip) for objects based
    on their RA, and Dec (in radians)

    @param [in] ra is a numpy array containing the RA of the objects in radians.
    In the International Celestial Reference System.

    @param [in] dec is a numpy array containing the Dec of the objects in radians.
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope
    pointing.

    @param [in] epoch is the epoch in Julian years of the equinox against which
    RA is measured.

    @param [in] chipNames a numpy array of chipNames.  If it is None, this method will call chipName
    to find the array.  The option exists for the user to specify chipNames, just in case the user
    has already called chipName for some reason.

    @param [in] camera is an afwCameraGeom object specifying the attributes of the camera.
    This is an optional argument to be passed to chipName.

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    return the true pixel coordinates with optical distortion included.  If False, this
    method will return TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the x pixel coordinate
    and the second row is the y pixel coordinate
    """

    if epoch is None:
        raise RuntimeError("You need to pass an epoch into pixelCoordsFromRaDec")

    if obs_metadata is None:
        raise RuntimeError("You need to pass an ObservationMetaData into pixelCoordsFromRaDec")

    if obs_metadata.mjd is None:
        raise RuntimeError("You need to pass an ObservationMetaData with an mjd into " \
                           + "pixelCoordsFromRaDec")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("You need to pass an ObservationMetaData with a rotSkyPos into " \
                           + "pixelCoordsFromRaDec")

    if not isinstance(ra, np.ndarray) or not isinstance(dec, np.ndarray):
        raise RuntimeError("You need to pass numpy arrays of RA and Dec to pixelCoordsFromRaDec")

    if len(ra) != len(dec):
        raise RuntimeError("You passed %d RA and %d Dec coordinates " % (len(ra), len(dec)) +
                           "to pixelCoordsFromRaDec")

    if chipNames is not None:
        if len(ra) != len(chipNames):
            raise RuntimeError("You passed %d points but %d chipNames to pixelCoordsFromRaDec" %
                               (len(ra), len(chipNames)))

    xPupil, yPupil = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata, epoch=epoch)
    return pixelCoordsFromPupilCoords(xPupil, yPupil, chipNames=chipNames, camera=camera,
                                      includeDistortion=includeDistortion)


def pixelCoordsFromPupilCoords(xPupil, yPupil, chipNames=None,
                               camera=None, includeDistortion=True):
    """
    Get the pixel positions (or nan if not on a chip) for objects based
    on their pupil coordinates.

    @param [in] xPupil a numpy array containing x pupil coordinates in radians

    @param [in] yPupil a numpy array containing y pupil coordinates in radians

    @param [in] chipNames a numpy array of chipNames.  If it is None, this method will call chipName
    to find the array.  The option exists for the user to specify chipNames, just in case the user
    has already called chipName for some reason.

    @param [in] camera is an afwCameraGeom object specifying the attributes of the camera.
    This is an optional argument to be passed to chipName.

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    return the true pixel coordinates with optical distortion included.  If False, this
    method will return TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the x pixel coordinate
    and the second row is the y pixel coordinate
    """

    if includeDistortion:
        pixelType = PIXELS
    else:
        pixelType = TAN_PIXELS

    if not camera:
        raise RuntimeError("Camera not specified.  Cannot calculate pixel coordinates.")

    if not isinstance(xPupil, np.ndarray) or not isinstance(yPupil, np.ndarray):
        raise RuntimeError("You need to pass numpy arrays of xPupil and yPupil to pixelCoordsFromPupilCoords")

    if len(xPupil) != len(yPupil):
        raise RuntimeError("You passed %d xPupil and %d yPupil coordinates " % (len(xPupil), len(yPupil)) +
                           "to pixelCoordsFromPupilCoords")

    if chipNames is not None:
        if len(xPupil) != len(chipNames):
            raise RuntimeError("You passed %d points but %d chipNames to pixelCoordsFromPupilCoords" %
                               (len(xPupil), len(chipNames)))

    if chipNames is None:
        chipNames = chipNameFromPupilCoords(xPupil, yPupil, camera=camera)

    xPix = []
    yPix = []
    for name, x, y in zip(chipNames, xPupil, yPupil):
        if not name:
            xPix.append(np.nan)
            yPix.append(np.nan)
            continue
        cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
        det = camera[name]
        cs = det.makeCameraSys(pixelType)
        detPoint = camera.transform(cp, cs)
        xPix.append(detPoint.getPoint().getX())
        yPix.append(detPoint.getPoint().getY())
    return np.array([xPix, yPix])


def pupilCoordsFromPixelCoords(xPixList, yPixList, chipNameList, camera=None,
                               includeDistortion=True):

    """
    Convert pixel coordinates into pupil coordinates

    @param [in] xPixList is a numpy array of x pixel coordinates

    @param [in] yPixList is a numpy array of y pixel coordinates

    @param [in] chipNameList is a numpy array of chip names (corresponding to the points
    in xPixList and yPixList)

    @param [in] camera is an afw.CameraGeom.camera object defining the camera

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    expect the true pixel coordinates with optical distortion included.  If False, this
    method will expect TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the x pupil coordinate
    and the second row is the y pupil coordinate (both in radians)
    """

    if camera is None:
        raise RuntimeError("You cannot call pupilCoordsFromPixelCoords without specifying a camera")

    if includeDistortion:
        pixelType = PIXELS
    else:
        pixelType = TAN_PIXELS

    pixelSystemDict = {}
    pupilSystemDict = {}
    detectorDict = {}
    for name in chipNameList:
        if name not in pixelSystemDict and name is not None and name!='None':
                pixelSystemDict[name] = camera[name].makeCameraSys(pixelType)
                pupilSystemDict[name] = camera[name].makeCameraSys(PUPIL)



    xPupilList = []
    yPupilList = []

    for xPix, yPix, chipName in zip(xPixList, yPixList, chipNameList):
        if chipName is None or chipName=='None':
            xPupilList.append(np.NaN)
            yPupilList.append(np.NaN)
        else:
            pixPoint = camera.makeCameraPoint(afwGeom.Point2D(xPix, yPix), pixelSystemDict[chipName])
            pupilPoint =  camera.transform(pixPoint, pupilSystemDict[chipName]).getPoint()
            xPupilList.append(pupilPoint.getX())
            yPupilList.append(pupilPoint.getY())

    xPupilList = np.array(xPupilList)
    yPupilList = np.array(yPupilList)

    return np.array([xPupilList, yPupilList])


def raDecFromPixelCoords(xPixList, yPixList, chipNameList, camera=None,
                         obs_metadata=None, epoch=None, includeDistortion=True):
    """
    Convert pixel coordinates into RA, Dec

    @param [in] xPixList is a numpy array of x pixel coordinates

    @param [in] yPixList is a numpy array of y pixel coordinates

    @param [in] chipNameList is a numpy array of chip names (corresponding to the points
    in xPixList and yPixList)

    @param [in] camera is an afw.CameraGeom.camera object defining the camera

    @param [in] obs_metadata is an ObservationMetaData defining the pointing

    @param [in] epoch is the mean epoch in years of the celestial coordinate system

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    expect the true pixel coordinates with optical distortion included.  If False, this
    method will expect TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the RA coordinate
    and the second row is the Dec coordinate (both in degrees; in the
    International Celestial Reference System)
    """
    output = _raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                   camera=camera, obs_metadata=obs_metadata,
                                   epoch=epoch, includeDistortion=includeDistortion)

    return np.degrees(output)


def _raDecFromPixelCoords(xPixList, yPixList, chipNameList, camera=None,
                          obs_metadata=None, epoch=None, includeDistortion=True):
    """
    Convert pixel coordinates into RA, Dec

    @param [in] xPixList is a numpy array of x pixel coordinates

    @param [in] yPixList is a numpy array of y pixel coordinates

    @param [in] chipNameList is a numpy array of chip names (corresponding to the points
    in xPixList and yPixList)

    @param [in] camera is an afw.CameraGeom.camera object defining the camera

    @param [in] obs_metadata is an ObservationMetaData defining the pointing

    @param [in] epoch is the mean epoch in years of the celestial coordinate system

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    expect the true pixel coordinates with optical distortion included.  If False, this
    method will expect TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the RA coordinate
    and the second row is the Dec coordinate (both in radians; in the International
    Celestial Reference System)
    """

    if camera is None:
        raise RuntimeError("You cannot call raDecFromPixelCoords without specifying a camera")

    if epoch is None:
        raise RuntimeError("You cannot call raDecFromPixelCoords without specifying an epoch")

    if obs_metadata is None:
        raise RuntimeError("You cannot call raDecFromPixelCoords without an ObservationMetaData")

    if obs_metadata.mjd is None:
        raise RuntimeError("The ObservationMetaData in raDecFromPixelCoords must have an mjd")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("The ObservationMetaData in raDecFromPixelCoords must have a rotSkyPos")

    if not isinstance(xPixList, np.ndarray) or not isinstance(yPixList, np.ndarray):
        raise RuntimeError("You must pass numpy arrays of xPix and yPix to raDecFromPixelCoords")

    if len(xPixList)!=len(yPixList):
        raise RuntimeError("You passed %d xPix coordinates but %d yPix coordinates " \
                           % (len(xPixList), len(yPixList)) \
                           +"to raDecFromPixelCoords")

    if len(xPixList)!=len(chipNameList):
        raise RuntimeError("You passed %d pixel coordinate pairs but %d chip names " \
                          % (len(xPixList), len(chipNameList)) \
                          +"to raDecFromPixelCoords")


    xPupilList, yPupilList = pupilCoordsFromPixelCoords(xPixList, yPixList, chipNameList,
                                                        camera=camera, includeDistortion=includeDistortion)

    raOut, decOut = _raDecFromPupilCoords(xPupilList, yPupilList,
                                  obs_metadata=obs_metadata, epoch=epoch)

    return np.array([raOut, decOut])


def focalPlaneCoordsFromRaDec(ra, dec, obs_metadata=None, epoch=None, camera=None):
    """
    Get the focal plane coordinates for all objects in the catalog.

    @param [in] ra is a numpy array in degrees.
    In the International Celestial Reference System.

    @param [in] dec is a numpy array in degrees.
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData object describing the telescope
    pointing (only if specifying RA and Dec rather than pupil coordinates)

    @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
    (in years; only if specifying RA and Dec rather than pupil coordinates)

    @param [in] camera is an afw.cameraGeom camera object

    @param [out] a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    return _focalPlaneCoordsFromRaDec(np.radians(ra), np.radians(dec),
                                      obs_metadata=obs_metadata, epoch=epoch,
                                      camera=camera)


def _focalPlaneCoordsFromRaDec(ra, dec, obs_metadata=None, epoch=None, camera=None):
    """
    Get the focal plane coordinates for all objects in the catalog.

    @param [in] ra is a numpy array in radians.
    In the International Celestial Reference System.

    @param [in] dec is a numpy array in radians.
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData object describing the telescope
    pointing (only if specifying RA and Dec rather than pupil coordinates)

    @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
    (in years; only if specifying RA and Dec rather than pupil coordinates)

    @param [in] camera is an afw.cameraGeom camera object

    @param [out] a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    if not isinstance(ra, np.ndarray) or not isinstance(dec, np.ndarray):
        raise RuntimeError("You must pass numpy arrays of RA and Dec to focalPlaneCoordsFromRaDec")

    if len(ra) != len(dec):
        raise RuntimeError("You specified %d RAs and %d Decs in focalPlaneCoordsFromRaDec" %
                           (len(ra), len(dec)))

    if epoch is None:
        raise RuntimeError("You have to specify an epoch to run " + \
                            "focalPlaneCoordsFromRaDec")

    if obs_metadata is None:
        raise RuntimeError("You have to specify an ObservationMetaData to run " + \
                               "focalPlaneCoordsFromRaDec")


    if obs_metadata.mjd is None:
        raise RuntimeError("You need to pass an ObservationMetaData with an " \
                           + "mjd into focalPlaneCoordsFromRaDec")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("You need to pass an ObservationMetaData with a " \
                           + "rotSkyPos into focalPlaneCoordsFromRaDec")


    xPupil, yPupil = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata,
                                                epoch=epoch)

    return focalPlaneCoordsFromPupilCoords(xPupil, yPupil, camera=camera)


def focalPlaneCoordsFromPupilCoords(xPupil, yPupil, camera=None):
    """
    Get the focal plane coordinates for all objects in the catalog.

    @param [in] xPupil a numpy array of x pupil coordinates in radians

    @param [in] yPupil a numpy array of y pupil coordinates in radians

    @param [in] camera is an afw.cameraGeom camera object

    @param [out] a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    if not isinstance(xPupil, np.ndarray) or not isinstance(yPupil, np.ndarray):
        raise RuntimeError("You must pass numpy arrays of xPupil and yPupil to " +
                               "focalPlaneCoordsFromPupilCoords")

    if len(xPupil) != len(yPupil):
        raise RuntimeError("You specified %d xPupil and %d yPupil coordinates " % (len(xPupil), len(yPupil)) +
                           "in focalPlaneCoordsFromPupilCoords")

    if camera is None:
        raise RuntimeError("You cannot calculate focal plane coordinates without specifying a camera")

    xPix = []
    yPix = []
    for x, y in zip(xPupil, yPupil):
        cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
        fpPoint = camera.transform(cp, FOCAL_PLANE)
        xPix.append(fpPoint.getPoint().getX())
        yPix.append(fpPoint.getPoint().getY())

    return np.array([xPix, yPix])

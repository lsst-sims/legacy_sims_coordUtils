import numpy as np
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, TAN_PIXELS, FOCAL_PLANE
from lsst.afw.cameraGeom import WAVEFRONT
from lsst.sims.utils.CodeUtilities import _validate_inputs
from lsst.sims.utils import _pupilCoordsFromRaDec, _raDecFromPupilCoords

__all__ = ["getCornerPixels", "_getCornerRaDec", "getCornerRaDec",
           "chipNameFromPupilCoords", "chipNameFromRaDec", "_chipNameFromRaDec",
           "pixelCoordsFromPupilCoords", "pixelCoordsFromRaDec", "_pixelCoordsFromRaDec",
           "focalPlaneCoordsFromPupilCoords", "focalPlaneCoordsFromRaDec", "_focalPlaneCoordsFromRaDec",
           "pupilCoordsFromPixelCoords",
           "raDecFromPixelCoords", "_raDecFromPixelCoords"]


def _validate_inputs_and_chipname(input_list, input_names, method_name,
                                  chip_name, chipname_can_be_none = True):
    """
    This will wrap _validate_inputs, but also reformat chip_name if necessary.

    input_list is a list of the inputs passed to a method.

    input_name is a list of the variable names associated with
    input_list

    method_name is the name of the method whose input is being validated.

    chip_name is the chip_name variable passed into the calling method.

    chipname_can_be_none is a boolean that controls whether or not
    chip_name is allowed to be None.

    This method will raise a RuntimeError if:

    1) the contents of input_list are not all of the same type
    2) the contents of input_list are not all floats or numpy arrays
    3) the contents of input_list are different lengths (if numpy arrays)
    4) chip_name is None and chipname_can_be_none is False
    5) chip_name is a list or array of different length than input_list[0]
       (if input_list[0] is a list or array) and len(chip_name)>1

    This method returns a boolean indicating whether input_list[0]
    is a numpy array and a re-casting of chip_name as a list
    of length equal to input_list[0] (unless chip_name is None;
    then it will leave chip_name untouched)
    """

    are_arrays = _validate_inputs(input_list, input_names, method_name)

    if chip_name is None and not chipname_can_be_none:
        raise RuntimeError("You passed chipName=None to %s" % method_name)

    if are_arrays:
        n_pts = len(input_list[0])
    else:
        n_pts = 1

    if isinstance(chip_name, list) or isinstance(chip_name, np.ndarray):
        if len(chip_name) > 1 and len(chip_name) != n_pts:
            raise RuntimeError("You passed %d chipNames to %s.\n" % (len(chip_name), method_name) +
                               "You passed %d %s values." % (len(input_list[0]), input_names[0]))

        if len(chip_name) == 1 and n_pts > 1:
            chip_name_out = [chip_name[0]]*n_pts
        else:
            chip_name_out = chip_name

        return are_arrays, chip_name_out

    elif chip_name is None:
        return are_arrays, chip_name
    else:
        return are_arrays, [chip_name]*n_pts


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
    return [tuple(np.degrees(row)) for row in cc]


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


def chipNameFromRaDec(ra, dec, obs_metadata=None, camera=None,
                      epoch=2000.0, allow_multiple_chips=False):
    """
    Return the names of detectors that see the object specified by
    either (xPupil, yPupil).

    @param [in] ra in degrees (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] dec in degrees (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope pointing

    @param [in] epoch is the epoch in Julian years of the equinox against which RA and Dec are
    measured.  Default is 2000.

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


def _chipNameFromRaDec(ra, dec, obs_metadata=None, camera=None,
                       epoch=2000.0, allow_multiple_chips=False):
    """
    Return the names of detectors that see the object specified by
    either (xPupil, yPupil).

    @param [in] ra in radians (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] dec in radians (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope pointing

    @param [in] epoch is the epoch in Julian years of the equinox against which RA and Dec are
    measured.  Default is 2000.

    @param [in] camera is an afw.cameraGeom camera instance characterizing the camera

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, an exception will be raised.  If it is 'True'
    and an object falls on more than one chip, it will still only return the first chip in the
    list of chips returned. THIS BEHAVIOR SHOULD BE FIXED IN A FUTURE TICKET.

    @param [out] a numpy array of chip names
    """

    _validate_inputs([ra, dec], ['ra', 'dec'], "chipNameFromRaDec")

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

    @param [in] xPupil is the x pupil coordinate in radians.
    Can be either a float or a numpy array.

    @param [in] yPupil is the y pupil coordinate in radians.
    Can be either a float or a numpy array.

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, an exception will be raised.  If it is 'True'
    and an object falls on more than one chip, it will still only return the first chip in the
    list of chips returned. THIS BEHAVIOR SHOULD BE FIXED IN A FUTURE TICKET.

    @param [in] camera is an afwCameraGeom object that specifies the attributes of the camera.

    @param [out] a numpy array of chip names

    """

    are_arrays = _validate_inputs([xPupil, yPupil], ['xPupil', 'yPupil'], "chipNameFromPupilCoords")

    if camera is None:
        raise RuntimeError("No camera defined.  Cannot run chipName.")

    chipNames = []

    if are_arrays:
        cameraPointList = [afwGeom.Point2D(x, y) for x, y in zip(xPupil, yPupil)]
    else:
        cameraPointList = [afwGeom.Point2D(xPupil, yPupil)]

    detList = camera.findDetectorsList(cameraPointList, PUPIL)

    for pt, det in zip(cameraPointList, detList):
        if len(det) == 0 or np.isnan(pt.getX()) or np.isnan(pt.getY()):
            chipNames.append(None)
        else:
            name_list = [dd.getName() for dd in det]
            if len(name_list) > 1 and not allow_multiple_chips:
                for dd in det:
                    # Because each A, B pair of wavefront sensors is positioned so that
                    # one is in focus and one is out of focus, it is possible that a particular
                    # RA, Dec could land on both wavefront sensors.  If that is what happened,
                    # we will permit it.
                    #
                    # See figure 2 of arXiv:1506.04839v2
                    if dd.getType() != WAVEFRONT:
                        raise RuntimeError("This method does not know how to deal with cameras " +
                                           "where points can be on multiple detectors.  " +
                                           "Override CameraCoords.get_chipName to add this.\n" +
                                           "If you were only asking for the chip name (as opposed " +
                                           "to pixel coordinates) you can try re-running with " +
                                           "the kwarg allow_multiple_chips=True.\n" +
                                           "Chip names were %s\n" % str(name_list) +
                                           "Pupil coordinat point was %.12f %.12f\n" % (pt[0], pt[1]))

                chipNames.append('%s' % str(name_list))

            elif len(name_list) == 0:
                chipNames.append(None)
            else:
                chipNames.append(name_list[0])

    if not are_arrays:
        return chipNames[0]

    return np.array(chipNames)


def pixelCoordsFromRaDec(ra, dec, obs_metadata=None,
                         chipName=None, camera=None,
                         epoch=2000.0, includeDistortion=True):
    """
    Get the pixel positions (or nan if not on a chip) for objects based
    on their RA, and Dec (in degrees)

    @param [in] ra is in degrees in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] dec is in degrees in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope
    pointing.

    @param [in] epoch is the epoch in Julian years of the equinox against which
    RA is measured.  Default is 2000.

    @param [in] chipName designates the names of the chips on which the pixel
    coordinates will be reckoned.  Can be either single value, an array, or None.
    If an array, there must be as many chipNames as there are (RA, Dec) pairs.
    If a single value, all of the pixel coordinates will be reckoned on the same
    chip.  If None, this method will calculate which chip each(RA, Dec) pair actually
    falls on, and return pixel coordinates for each (RA, Dec) pair on the appropriate
    chip.  Default is None.

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
                                 chipName=chipName, camera=camera,
                                 includeDistortion=includeDistortion,
                                 obs_metadata=obs_metadata, epoch=epoch)


def _pixelCoordsFromRaDec(ra, dec, obs_metadata=None,
                          chipName=None, camera=None,
                          epoch=2000.0, includeDistortion=True):
    """
    Get the pixel positions (or nan if not on a chip) for objects based
    on their RA, and Dec (in radians)

    @param [in] ra is in radians in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] dec is in radians in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope
    pointing.

    @param [in] epoch is the epoch in Julian years of the equinox against which
    RA is measured.  Default is 2000.

    @param [in] chipName designates the names of the chips on which the pixel
    coordinates will be reckoned.  Can be either single value, an array, or None.
    If an array, there must be as many chipNames as there are (RA, Dec) pairs.
    If a single value, all of the pixel coordinates will be reckoned on the same
    chip.  If None, this method will calculate which chip each(RA, Dec) pair actually
    falls on, and return pixel coordinates for each (RA, Dec) pair on the appropriate
    chip.  Default is None.

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

    are_arrays, \
    chipNameList = _validate_inputs_and_chipname([ra, dec], ['ra', 'dec'],
                                                 'pixelCoordsFromRaDec',
                                                 chipName)

    if epoch is None:
        raise RuntimeError("You need to pass an epoch into pixelCoordsFromRaDec")

    if obs_metadata is None:
        raise RuntimeError("You need to pass an ObservationMetaData into pixelCoordsFromRaDec")

    if obs_metadata.mjd is None:
        raise RuntimeError("You need to pass an ObservationMetaData with an mjd into "
                           "pixelCoordsFromRaDec")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("You need to pass an ObservationMetaData with a rotSkyPos into "
                           "pixelCoordsFromRaDec")

    xPupil, yPupil = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata, epoch=epoch)
    return pixelCoordsFromPupilCoords(xPupil, yPupil, chipName=chipNameList, camera=camera,
                                      includeDistortion=includeDistortion)


def pixelCoordsFromPupilCoords(xPupil, yPupil, chipName=None,
                               camera=None, includeDistortion=True):
    """
    Get the pixel positions (or nan if not on a chip) for objects based
    on their pupil coordinates.

    @param [in] xPupil is the x pupil coordinates in radians.
    Can be either a float or a numpy array.

    @param [in] yPupil is the y pupil coordinates in radians.
    Can be either a float or a numpy array.

    @param [in] chipName designates the names of the chips on which the pixel
    coordinates will be reckoned.  Can be either single value, an array, or None.
    If an array, there must be as many chipNames as there are (RA, Dec) pairs.
    If a single value, all of the pixel coordinates will be reckoned on the same
    chip.  If None, this method will calculate which chip each(RA, Dec) pair actually
    falls on, and return pixel coordinates for each (RA, Dec) pair on the appropriate
    chip.  Default is None.

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

    are_arrays, \
    chipNameList = _validate_inputs_and_chipname([xPupil, yPupil], ["xPupil", "yPupil"],
                                                 "pixelCoordsFromPupilCoords",
                                                 chipName)
    if includeDistortion:
        pixelType = PIXELS
    else:
        pixelType = TAN_PIXELS

    if not camera:
        raise RuntimeError("Camera not specified.  Cannot calculate pixel coordinates.")

    if chipNameList is None:
        chipNameList = chipNameFromPupilCoords(xPupil, yPupil, camera=camera)
        if not isinstance(chipNameList, list) and not isinstance(chipNameList, np.ndarray):
            chipNameList = [chipNameList]

    if are_arrays:
        xPix = []
        yPix = []
        for name, x, y in zip(chipNameList, xPupil, yPupil):
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
    else:
        if not chipNameList[0]:
            return np.array([np.NaN, np.NaN])

        cp = camera.makeCameraPoint(afwGeom.Point2D(xPupil, yPupil), PUPIL)
        det = camera[chipNameList[0]]
        cs = det.makeCameraSys(pixelType)
        detPoint = camera.transform(cp, cs)
        return np.array([detPoint.getPoint().getX(), detPoint.getPoint().getY()])


def pupilCoordsFromPixelCoords(xPix, yPix, chipName, camera=None,
                               includeDistortion=True):

    """
    Convert pixel coordinates into pupil coordinates

    @param [in] xPix is the x pixel coordinate of the point.
    Can be either a float or a numpy array.

    @param [in] yPix is the y pixel coordinate of the point.
    Can be either a float or a numpy array.

    @param [in] chipName is the name of the chip(s) on which the pixel coordinates
    are defined.  This can be a list (in which case there should be one chip name
    for each (xPix, yPix) coordinate pair), or a single value (in which case, all
    of the (xPix, yPix) points will be reckoned on that chip).

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

    are_arrays, \
    chipNameList = _validate_inputs_and_chipname([xPix, yPix], ['xPix', 'yPix'],
                                                 "pupilCoordsFromPixelCoords",
                                                 chipName,
                                                 chipname_can_be_none=False)

    if includeDistortion:
        pixelType = PIXELS
    else:
        pixelType = TAN_PIXELS

    pixelSystemDict = {}
    pupilSystemDict = {}
    for name in chipNameList:
        if name not in pixelSystemDict and name is not None and name != 'None':
                pixelSystemDict[name] = camera[name].makeCameraSys(pixelType)
                pupilSystemDict[name] = camera[name].makeCameraSys(PUPIL)

    if are_arrays:
        xPupilList = []
        yPupilList = []

        for xx, yy, name in zip(xPix, yPix, chipNameList):
            if name is None or name == 'None':
                xPupilList.append(np.NaN)
                yPupilList.append(np.NaN)
            else:
                pixPoint = camera.makeCameraPoint(afwGeom.Point2D(xx, yy), pixelSystemDict[name])
                pupilPoint = camera.transform(pixPoint, pupilSystemDict[name]).getPoint()
                xPupilList.append(pupilPoint.getX())
                yPupilList.append(pupilPoint.getY())

        xPupilList = np.array(xPupilList)
        yPupilList = np.array(yPupilList)

        return np.array([xPupilList, yPupilList])

    # if not are_arrays
    if chipNameList[0] is None or chipNameList[0] == 'None':
        return np.array([np.NaN, np.NaN])

    pixPoint = camera.makeCameraPoint(afwGeom.Point2D(xPix, yPix), pixelSystemDict[chipNameList[0]])
    pupilPoint = camera.transform(pixPoint, pupilSystemDict[chipNameList[0]]).getPoint()
    return np.array([pupilPoint.getX(), pupilPoint.getY()])


def raDecFromPixelCoords(xPix, yPix, chipName, camera=None,
                         obs_metadata=None, epoch=2000.0, includeDistortion=True):
    """
    Convert pixel coordinates into RA, Dec

    @param [in] xPix is the x pixel coordinate.  It can be either
    a float or a numpy array.

    @param [in] yPix is the y pixel coordinate.  It can be either
    a float or a numpy array.

    @param [in] chipName is the name of the chip(s) on which the pixel coordinates
    are defined.  This can be a list (in which case there should be one chip name
    for each (xPix, yPix) coordinate pair), or a single value (in which case, all
    of the (xPix, yPix) points will be reckoned on that chip).

    @param [in] camera is an afw.CameraGeom.camera object defining the camera

    @param [in] obs_metadata is an ObservationMetaData defining the pointing

    @param [in] epoch is the mean epoch in years of the celestial coordinate system.
    Default is 2000.

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    expect the true pixel coordinates with optical distortion included.  If False, this
    method will expect TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the RA coordinate
    and the second row is the Dec coordinate (both in degrees; in the
    International Celestial Reference System)
    """
    output = _raDecFromPixelCoords(xPix, yPix, chipName,
                                   camera=camera, obs_metadata=obs_metadata,
                                   epoch=epoch, includeDistortion=includeDistortion)

    return np.degrees(output)


def _raDecFromPixelCoords(xPix, yPix, chipName, camera=None,
                          obs_metadata=None, epoch=2000.0, includeDistortion=True):
    """
    Convert pixel coordinates into RA, Dec

    @param [in] xPix is the x pixel coordinate.  It can be either
    a float or a numpy array.

    @param [in] yPix is the y pixel coordinate.  It can be either
    a float or a numpy array.

    @param [in] chipName is the name of the chip(s) on which the pixel coordinates
    are defined.  This can be a list (in which case there should be one chip name
    for each (xPix, yPix) coordinate pair), or a single value (in which case, all
    of the (xPix, yPix) points will be reckoned on that chip).

    @param [in] camera is an afw.CameraGeom.camera object defining the camera

    @param [in] obs_metadata is an ObservationMetaData defining the pointing

    @param [in] epoch is the mean epoch in years of the celestial coordinate system.
    Default is 2000.

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    expect the true pixel coordinates with optical distortion included.  If False, this
    method will expect TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [out] a 2-D numpy array in which the first row is the RA coordinate
    and the second row is the Dec coordinate (both in radians; in the International
    Celestial Reference System)
    """

    are_arrays, \
    chipNameList = _validate_inputs_and_chipname([xPix, yPix],
                                                 ['xPix', 'yPix'],
                                                 'raDecFromPixelCoords',
                                                 chipName,
                                                 chipname_can_be_none=False)

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

    xPupilList, yPupilList = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList,
                                                        camera=camera, includeDistortion=includeDistortion)

    raOut, decOut = _raDecFromPupilCoords(xPupilList, yPupilList,
                                          obs_metadata=obs_metadata, epoch=epoch)

    return np.array([raOut, decOut])


def focalPlaneCoordsFromRaDec(ra, dec, obs_metadata=None, epoch=2000.0, camera=None):
    """
    Get the focal plane coordinates for all objects in the catalog.

    @param [in] ra is in degrees in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] dec is in degrees in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] obs_metadata is an ObservationMetaData object describing the telescope
    pointing (only if specifying RA and Dec rather than pupil coordinates)

    @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
    (in years; only if specifying RA and Dec rather than pupil coordinates; default is 2000)

    @param [in] camera is an afw.cameraGeom camera object

    @param [out] a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    return _focalPlaneCoordsFromRaDec(np.radians(ra), np.radians(dec),
                                      obs_metadata=obs_metadata, epoch=epoch,
                                      camera=camera)


def _focalPlaneCoordsFromRaDec(ra, dec, obs_metadata=None, epoch=2000.0, camera=None):
    """
    Get the focal plane coordinates for all objects in the catalog.

    @param [in] ra is in radians in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] dec is in radians in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] obs_metadata is an ObservationMetaData object describing the telescope
    pointing (only if specifying RA and Dec rather than pupil coordinates)

    @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
    (in years; only if specifying RA and Dec rather than pupil coordinates; default is 2000)

    @param [in] camera is an afw.cameraGeom camera object

    @param [out] a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    _validate_inputs([ra, dec], ['ra', 'dec'], 'focalPlaneCoordsFromRaDec')

    if epoch is None:
        raise RuntimeError("You have to specify an epoch to run "
                           "focalPlaneCoordsFromRaDec")

    if obs_metadata is None:
        raise RuntimeError("You have to specify an ObservationMetaData to run "
                           "focalPlaneCoordsFromRaDec")

    if obs_metadata.mjd is None:
        raise RuntimeError("You need to pass an ObservationMetaData with an "
                           "mjd into focalPlaneCoordsFromRaDec")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("You need to pass an ObservationMetaData with a "
                           "rotSkyPos into focalPlaneCoordsFromRaDec")

    xPupil, yPupil = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata,
                                           epoch=epoch)

    return focalPlaneCoordsFromPupilCoords(xPupil, yPupil, camera=camera)


def focalPlaneCoordsFromPupilCoords(xPupil, yPupil, camera=None):
    """
    Get the focal plane coordinates for all objects in the catalog.

    @param [in] xPupil the x pupil coordinates in radians.
    Can be a float or a numpy array.

    @param [in] yPupil the y pupil coordinates in radians.
    Can be a float or a numpy array.

    @param [in] camera is an afw.cameraGeom camera object

    @param [out] a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    are_arrays = _validate_inputs([xPupil, yPupil],
                                  ['xPupil', 'yPupil'], 'focalPlaneCoordsFromPupilCoords')

    if camera is None:
        raise RuntimeError("You cannot calculate focal plane coordinates without specifying a camera")

    if are_arrays:
        xPix = []
        yPix = []
        for x, y in zip(xPupil, yPupil):
            cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            fpPoint = camera.transform(cp, FOCAL_PLANE).getPoint()
            xPix.append(fpPoint.getX())
            yPix.append(fpPoint.getY())

        return np.array([xPix, yPix])

    # if not are_arrays
    cp = camera.makeCameraPoint(afwGeom.Point2D(xPupil, yPupil), PUPIL)
    fpPoint = camera.transform(cp, FOCAL_PLANE).getPoint()
    return np.array([fpPoint.getX(), fpPoint.getY()])

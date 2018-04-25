from __future__ import division
from builtins import zip
from builtins import range
import numpy as np
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS, WAVEFRONT
from lsst.afw.geom import Box2D
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import focalPlaneCoordsFromPupilCoords
from lsst.sims.coordUtils import LsstZernikeFitter
from lsst.sims.coordUtils import pupilCoordsFromPixelCoords, pixelCoordsFromPupilCoords
from lsst.sims.coordUtils import pupilCoordsFromFocalPlaneCoords
from lsst.sims.coordUtils import pupilCoordsFromPixelCoords
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.coordUtils import getCornerPixels, _validate_inputs_and_chipname
from lsst.sims.utils.CodeUtilities import _validate_inputs
from lsst.sims.utils import radiansFromArcsec


__all__ = ["focalPlaneCoordsFromPupilCoordsLSST",
           "pupilCoordsFromFocalPlaneCoordsLSST",
           "chipNameFromPupilCoordsLSST",
           "_chipNameFromRaDecLSST", "chipNameFromRaDecLSST",
           "pixelCoordsFromPupilCoordsLSST",
           "pupilCoordsFromPixelCoordsLSST",
           "_pixelCoordsFromRaDecLSST", "pixelCoordsFromRaDecLSST"]


def focalPlaneCoordsFromPupilCoordsLSST(xPupil, yPupil, band='r'):
    """
    Get the focal plane coordinates for all objects in the catalog.

    Parameters
    ----------
    xPupil -- the x pupil coordinates in radians.
    Can be a float or a numpy array.

    yPupil -- the y pupil coordinates in radians.
    Can be a float or a numpy array.

    band -- the filter being simulated (default='r')

    Returns
    --------
    a 2-D numpy array in which the first row is the x
    focal plane coordinate and the second row is the y focal plane
    coordinate (both in millimeters)
    """

    if not hasattr(focalPlaneCoordsFromPupilCoordsLSST, '_z_fitter'):
        focalPlaneCoordsFromPupilCoordsLSST._z_fitter = LsstZernikeFitter()

    z_fitter = focalPlaneCoordsFromPupilCoordsLSST._z_fitter
    x_f0, y_f0 = focalPlaneCoordsFromPupilCoords(xPupil, yPupil, camera=lsst_camera())
    dx, dy = z_fitter.dxdy(x_f0, y_f0, band)
    return np.array([x_f0+dx, y_f0+dy])


def pupilCoordsFromFocalPlaneCoordsLSST(xmm, ymm, band='r'):
    """
    Convert radians on the pupil into mm on the focal plane.

    Note: round-tripping through focalPlaneCoordsFromPupilCoordsLSST
    and pupilCoordsFromFocalPlaneCoordsLSST introduces a residual
    of up to 2.18e-6 mm that accumulates with each round trip.

    Parameters
    ----------
    xmm -- x coordinate in millimeters on the focal plane

    ymm -- y coordinate in millimeters on the focal plane

    band -- the filter we are simulating (default='r')

    Returns
    -------
    a 2-D numpy array in which the first row is the x
    pupil coordinate and the second row is the y pupil
    coordinate (both in radians)
    """
    if not hasattr(pupilCoordsFromFocalPlaneCoordsLSST, '_z_fitter'):
        pupilCoordsFromFocalPlaneCoordsLSST._z_fitter = LsstZernikeFitter()

    z_fitter = pupilCoordsFromFocalPlaneCoordsLSST._z_fitter
    dx, dy = z_fitter.dxdy_inverse(xmm, ymm, band)
    x_f1 = xmm + dx
    y_f1 = ymm + dy
    xp, yp = pupilCoordsFromFocalPlaneCoords(x_f1, y_f1, camera=lsst_camera())
    return np.array([xp, yp])


def _build_lsst_focal_coord_map():
    """
    Build a map of focal plane coordinates on the LSST focal plane.
    Returns _lsst_focal_coord_map, which is a dict.
    _lsst_focal_coord_map['name'] contains a list of the names of each chip in the lsst camera
    _lsst_focal_coord_map['xx'] contains the x focal plane coordinate of the center of each chip (mm)
    _lsst_focal_coord_map['yy'] contains the y focal plane coordinate of the center of each chip (mm)
    _lsst_focal_coord_map['dp'] contains the radius (in mm) of the circle containing each chip
    """

    camera = lsst_camera()

    name_list = []
    x_pix_list = []
    y_pix_list = []
    x_mm_list = []
    y_mm_list = []
    n_chips = 0
    for chip in camera:
        chip_name = chip.getName()
        pixels_to_focal = chip.getTransform(PIXELS, FOCAL_PLANE)
        n_chips += 1
        corner_list = getCornerPixels(chip_name, lsst_camera())
        for corner in corner_list:
            x_pix_list.append(corner[0])
            y_pix_list.append(corner[1])
            pixel_pt = afwGeom.Point2D(corner[0], corner[1])
            focal_pt = pixels_to_focal.applyForward(pixel_pt)
            x_mm_list.append(focal_pt.getX())
            y_mm_list.append(focal_pt.getY())
            name_list.append(chip_name)

    x_pix_list = np.array(x_pix_list)
    y_pix_list = np.array(y_pix_list)
    x_mm_list = np.array(x_mm_list)
    y_mm_list = np.array(y_mm_list)

    center_x = np.zeros(n_chips, dtype=float)
    center_y = np.zeros(n_chips, dtype=float)
    extent = np.zeros(n_chips, dtype=float)
    final_name = []
    for ix_ct in range(n_chips):
        ix = ix_ct*4
        chip_name = name_list[ix]
        xx = 0.25*(x_mm_list[ix] + x_mm_list[ix+1] +
                   x_mm_list[ix+2] + x_mm_list[ix+3])

        yy = 0.25*(y_mm_list[ix] + y_mm_list[ix+1] +
                   y_mm_list[ix+2] + y_mm_list[ix+3])

        dx = 0.25*np.array([np.sqrt(np.power(xx-x_mm_list[ix+ii], 2) +
                                    np.power(yy-y_mm_list[ix+ii], 2)) for ii in range(4)]).sum()

        center_x[ix_ct] = xx
        center_y[ix_ct] = yy
        extent[ix_ct] = dx
        final_name.append(chip_name)

    final_name = np.array(final_name)

    lsst_focal_coord_map = {}
    lsst_focal_coord_map['name'] = final_name
    lsst_focal_coord_map['xx'] = center_x
    lsst_focal_coord_map['yy'] = center_y
    lsst_focal_coord_map['dp'] = extent
    return lsst_focal_coord_map


def _findDetectorsListLSST(focalPointList, detectorList, possible_points,
                           allow_multiple_chips=False):
    """!Find the detectors that cover a list of points specified by x and y coordinates in any system

    This is based one afw.camerGeom.camera.findDetectorsList.  It has been optimized for the LSST
    camera in the following way:

        - it accepts a limited list of detectors to check in advance (this list should be
          constructed by comparing the pupil coordinates in question and comparing to the
          pupil coordinates of the center of each detector)

       - it will stop looping through detectors one it has found one that is correct (the LSST
         camera does not allow an object to fall on more than one detector)

    @param[in] focalPointList  a list of points in FOCAL_PLANE coordinates

    @param[in] detectorList is a list of the afwCameraGeom detector objects being considered

    @param[in] possible_points is a list of lists.  possible_points[ii] is a list of integers
    corresponding to the indices in focalPointList of the pupilPoints that may be on detectorList[ii].

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, only the first chip will appear in the list of
    chipNames but NO WARNING WILL BE EMITTED.  If it is 'True' and an object falls on more than one
    chip, a list of chipNames will appear for that object.

    @return outputNameList is a numpy array of the names of the detectors
    """
    # transform the points to the native coordinate system
    #
    # The conversion to a numpy array looks a little clunky.
    # The problem, if you do the naive thing (nativePointList = np.array(lsst_camera().....),
    # the conversion to a numpy array gets passed down to the contents of nativePointList
    # and they end up in a form that the afwCameraGeom code does not know how to handle
    nativePointList = np.zeros(len(focalPointList), dtype=object)
    for ii in range(len(focalPointList)):
        nativePointList[ii] = focalPointList[ii]

    # initialize output and some caching lists
    outputNameList = [None]*len(focalPointList)
    chip_has_found = np.array([-1]*len(focalPointList))
    unfound_pts = len(chip_has_found)

    # Figure out if any of these (RA, Dec) pairs could be
    # on more than one chip.  This is possible on the
    # wavefront sensors, since adjoining wavefront sensors
    # are kept one in focus, one out of focus.
    # See figure 2 of arXiv:1506.04839v2
    # (This might actually be a bug in obs_lsstSim
    # I opened DM-8075 on 25 October 2016 to investigate)
    could_be_multiple = [False]*len(focalPointList)
    if allow_multiple_chips:
        for ipt in range(len(focalPointList)):
            for det in detectorList[ipt]:
                if det.getType() == WAVEFRONT:
                    could_be_multiple[ipt] = True

    # t_assemble_list = 0.0
    # loop over detectors
    for i_detector, detector in enumerate(detectorList):
        if len(possible_points[i_detector]) == 0:
            continue

        if unfound_pts <= 0:
            if unfound_pts<0:
                raise RuntimeError("Somehow, unfound_pts = %d in _findDetectorsListLSST" % unfound_pts)
            # we have already found all of the (RA, Dec) pairs
            for ix, name in enumerate(outputNameList):
                if isinstance(name, list):
                    outputNameList[ix] = str(name)
            return np.array(outputNameList)

        # find all of the pupil points that could be on this detector
        valid_pt_dexes = possible_points[i_detector][np.where(chip_has_found[possible_points[i_detector]]<0)]

        if len(valid_pt_dexes) > 0:
            valid_pt_list = nativePointList[valid_pt_dexes]
            transform = detector.getTransform(lsst_camera()._nativeCameraSys, PIXELS)
            detectorPointList = transform.applyForward(valid_pt_list)

            box = afwGeom.Box2D(detector.getBBox())
            for ix, pt in zip(valid_pt_dexes, detectorPointList):
                if box.contains(pt):
                    if not could_be_multiple[ix]:
                        # because this (RA, Dec) pair is not marked as could_be_multiple,
                        # the fact that this (RA, Dec) pair is on the current chip
                        # means this (RA, Dec) pair no longer needs to be considered.
                        # You can set chip_has_found[ix] to unity.
                        outputNameList[ix] = detector.getName()
                        chip_has_found[ix] = 1
                        unfound_pts -= 1
                    else:
                        # Since this (RA, Dec) pair has been makred could_be_multiple,
                        # finding this (RA, Dec) pair on the chip does not remove the
                        # (RA, Dec) pair from contention.
                        if outputNameList[ix] is None:
                            outputNameList[ix] = detector.getName()
                        elif isinstance(outputNameList[ix], list):
                            outputNameList[ix].append(detector.getName())
                        else:
                            outputNameList[ix] = [outputNameList[ix], detector.getName()]

    # convert entries corresponding to multiple chips into strings
    # (i.e. [R:2,2 S:0,0, R:2,2 S:0,1] becomes `[R:2,2 S:0,0, R:2,2 S:0,1]`)
    for ix, name in enumerate(outputNameList):
        if isinstance(name, list):
            outputNameList[ix] = str(name)

    # print('t_assemble %.2e' % t_assemble_list)

    return np.array(outputNameList)


def chipNameFromPupilCoordsLSST(xPupil_in, yPupil_in, allow_multiple_chips=False, band='r'):
    """
    Return the names of LSST detectors that see the object specified by
    either (xPupil, yPupil).

    @param [in] xPupil_in is the x pupil coordinate in radians.
    Must be a numpy array.

    @param [in] yPupil_in is the y pupil coordinate in radians.
    Must be a numpy array.

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, only the first chip will appear in the list of
    chipNames and warning will be emitted.  If it is 'True' and an object falls on more than one
    chip, a list of chipNames will appear for that object.

    @param[in] band is the bandpass being simulated (default='r')

    @param [out] a numpy array of chip names

    """
    if (not hasattr(chipNameFromPupilCoordsLSST, '_pupil_map') or
    not hasattr(chipNameFromPupilCoordsLSST, '_detector_arr') or
    len(chipNameFromPupilCoordsLSST._detector_arr) == 0):
        focal_map = _build_lsst_focal_coord_map()
        chipNameFromPupilCoordsLSST._focal_map = focal_map
        camera = lsst_camera()
        detector_arr = np.zeros(len(focal_map['name']), dtype=object)
        for ii in range(len(focal_map['name'])):
            detector_arr[ii] = camera[focal_map['name'][ii]]

        chipNameFromPupilCoordsLSST._detector_arr = detector_arr

        # build a Box2D that contains all of the detectors in the camera
        focal_to_field = camera.getTransformMap().getTransform(FOCAL_PLANE, FIELD_ANGLE)
        focal_bbox = camera.getFpBBox()
        focal_corners = focal_bbox.getCorners()
        camera_bbox = Box2D()
        x_focal_max = None
        x_focal_min = None
        y_focal_max = None
        y_focal_min = None
        for cc in focal_corners:
            xx = cc.getX()
            yy = cc.getY()
            if x_focal_max is None or xx > x_focal_max:
                x_focal_max = xx
            if x_focal_min is None or xx < x_focal_min:
                x_focal_min = xx
            if y_focal_max is None or yy > y_focal_max:
                y_focal_max = yy
            if y_focal_min is None or yy < y_focal_min:
                y_focal_min = yy

        chipNameFromPupilCoordsLSST._x_focal_center = 0.5*(x_focal_max+x_focal_min)
        chipNameFromPupilCoordsLSST._y_focal_center = 0.5*(y_focal_max+y_focal_min)

        radius_sq_max = None
        for cc in focal_corners:
            xx = cc.getX()
            yy = cc.getY()
            radius_sq = ((xx-chipNameFromPupilCoordsLSST._x_focal_center)**2 +
                         (yy-chipNameFromPupilCoordsLSST._y_focal_center)**2)
            if radius_sq_max is None or radius_sq > radius_sq_max:
                radius_sq_max = radius_sq

        chipNameFromPupilCoordsLSST._camera_focal_radius_sq = radius_sq_max*1.1

    are_arrays = _validate_inputs([xPupil_in, yPupil_in], ['xPupil_in', 'yPupil_in'],
                                  "chipNameFromPupilCoordsLSST")

    if not are_arrays:
        xPupil_in = np.array([xPupil_in])
        yPupil_in = np.array([yPupil_in])

    xFocal, yFocal = focalPlaneCoordsFromPupilCoordsLSST(xPupil_in, yPupil_in, band=band)

    radius_sq_list = ((xFocal-chipNameFromPupilCoordsLSST._x_focal_center)**2 +
                      (yFocal-chipNameFromPupilCoordsLSST._y_focal_center)**2)

    good_radii = np.where(radius_sq_list<chipNameFromPupilCoordsLSST._camera_focal_radius_sq)

    if len(good_radii[0]) == 0:
        return np.array([None]*len(xPupil_in))

    xFocal_good = xFocal[good_radii]
    yFocal_good = yFocal[good_radii]

    ############################################################
    # in the code below, we will only consider those points which
    # passed the 'good_radii' test above; the other points will
    # be added in with chipName == None at the end
    #
    focalPointList = [afwGeom.Point2D(xFocal[i_pt], yFocal[i_pt])
                      for i_pt in good_radii[0]]

    # Loop through every detector on the camera.  For each detector, assemble a list of points
    # whose centers are within 1.1 detector radii of the center of the detector.

    x_cam_list = chipNameFromPupilCoordsLSST._focal_map['xx']
    y_cam_list = chipNameFromPupilCoordsLSST._focal_map['yy']
    rrsq_lim_list = (1.1*chipNameFromPupilCoordsLSST._focal_map['dp'])**2

    possible_points = []
    for i_chip, (x_cam, y_cam, rrsq_lim) in \
    enumerate(zip(x_cam_list, y_cam_list, rrsq_lim_list)):

        local_possible_pts = np.where(((xFocal_good - x_cam)**2 +
                                       (yFocal_good - y_cam)**2) < rrsq_lim)[0]

        possible_points.append(local_possible_pts)

    nameList_good = _findDetectorsListLSST(focalPointList,
                                           chipNameFromPupilCoordsLSST._detector_arr,
                                           possible_points,
                                           allow_multiple_chips=allow_multiple_chips)

    ####################################################################
    # initialize output as an array of Nones, effectively adding back in
    # the points which failed the initial radius cut
    nameList = np.array([None]*len(xPupil_in))

    nameList[good_radii] = nameList_good

    if not are_arrays:
        return nameList[0]

    return nameList


def _chipNameFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                           obs_metadata=None, epoch=2000.0, allow_multiple_chips=False,
                           band='r'):
    """
    Return the names of detectors on the LSST camera that see the object specified by
    (RA, Dec) in radians.

    @param [in] ra in radians (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] dec in radians (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] pm_ra is proper motion in RA multiplied by cos(Dec) (radians/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] pm_dec is proper motion in dec (radians/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] parallax is parallax in radians
    Can be a numpy array or a number or None (default=None).

    @param [in] v_rad is radial velocity (km/s)
    Can be a numpy array or a number or None (default=None).

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope pointing

    @param [in] epoch is the epoch in Julian years of the equinox against which RA and Dec are
    measured.  Default is 2000.

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, only the first chip will appear in the list of
    chipNames but NO WARNING WILL BE EMITTED.  If it is 'True' and an object falls on more than one
    chip, a list of chipNames will appear for that object.

    @param [in] band is the filter we are simulating (Default=r)

    @param [out] the name(s) of the chips on which ra, dec fall (will be a numpy
    array if more than one)
    """

    are_arrays = _validate_inputs([ra, dec], ['ra', 'dec'], "chipNameFromRaDecLSST")

    if epoch is None:
        raise RuntimeError("You need to pass an epoch into chipName")

    if obs_metadata is None:
        raise RuntimeError("You need to pass an ObservationMetaData into chipName")

    if obs_metadata.mjd is None:
        raise RuntimeError("You need to pass an ObservationMetaData with an mjd into chipName")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("You need to pass an ObservationMetaData with a rotSkyPos into chipName")

    xp, yp = _pupilCoordsFromRaDec(ra, dec,
                                   pm_ra=pm_ra, pm_dec=pm_dec,
                                   parallax=parallax, v_rad=v_rad,
                                   obs_metadata=obs_metadata, epoch=epoch)

    return chipNameFromPupilCoordsLSST(xp, yp, allow_multiple_chips=allow_multiple_chips,
                                       band=band)


def chipNameFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                          obs_metadata=None, epoch=2000.0, allow_multiple_chips=False,
                          band='r'):
    """
    Return the names of detectors on the LSST camera that see the object specified by
    (RA, Dec) in degrees.

    @param [in] ra in degrees (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] dec in degrees (a numpy array or a float).
    In the International Celestial Reference System.

    @param [in] pm_ra is proper motion in RA multiplied by cos(Dec) (arcsec/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] pm_dec is proper motion in dec (arcsec/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] parallax is parallax in arcsec
    Can be a numpy array or a number or None (default=None).

    @param [in] v_rad is radial velocity (km/s)
    Can be a numpy array or a number or None (default=None).

    @param [in] obs_metadata is an ObservationMetaData characterizing the telescope pointing

    @param [in] epoch is the epoch in Julian years of the equinox against which RA and Dec are
    measured.  Default is 2000.

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, only the first chip will appear in the list of
    chipNames but NO WARNING WILL BE EMITTED.  If it is 'True' and an object falls on more than one
    chip, a list of chipNames will appear for that object.

    @param [in] band is the filter that we are simulating (Default=r)

    @param [out] the name(s) of the chips on which ra, dec fall (will be a numpy
    array if more than one)
    """
    if pm_ra is not None:
        pm_ra_out = radiansFromArcsec(pm_ra)
    else:
        pm_ra_out = None

    if pm_dec is not None:
        pm_dec_out = radiansFromArcsec(pm_dec)
    else:
        pm_dec_out = None

    if parallax is not None:
        parallax_out = radiansFromArcsec(parallax)
    else:
        parallax_out = None

    return _chipNameFromRaDecLSST(np.radians(ra), np.radians(dec),
                                  pm_ra=pm_ra_out, pm_dec=pm_dec_out,
                                  parallax=parallax_out, v_rad=v_rad,
                                  obs_metadata=obs_metadata, epoch=epoch,
                                  allow_multiple_chips=allow_multiple_chips,
                                  band=band)


def pupilCoordsFromPixelCoordsLSST(xPix, yPix, chipName=None, band="r",
                                   includeDistortion=True):
    """
    Convert pixel coordinates into radians on the pupil

    Parameters
    ----------
    xPix -- the x pixel coordinate

    yPix -- the y pixel coordinate

    chipName -- the name(s) of the chips on which xPix, yPix are reckoned

    band -- the filter we are simulating (default=r)

    includeDistortion -- a boolean which turns on or off optical
    distortions (default=True)

    Returns
    -------
    a 2-D numpy array in which the first row is the x
    pupil coordinate and the second row is the y pupil
    coordinate (both in radians)
    """

    if not includeDistortion:
        return pupilCoordsFromPixelCoords(xPix, yPix, chipName=chipName,
                                          camera=lsst_camera(),
                                          includeDistortion=includeDistortion)

    are_arrays, \
    chipNameList = _validate_inputs_and_chipname([xPix, yPix], ['xPix', 'yPix'],
                                                 "pupilCoordsFromPixelCoords",
                                                 chipName,
                                                 chipname_can_be_none=False)

    pixel_to_focal_dict = {}
    camera = lsst_camera()
    for name in chipNameList:
        if name not in pixel_to_focal_dict and name is not None and name != 'None':
            pixel_to_focal_dict[name] = camera[name].getTransform(PIXELS, FOCAL_PLANE)

    if are_arrays:
        x_f = np.zeros(len(xPix), dtype=float)
        y_f = np.zeros(len(yPix), dtype=float)
        for ii in range(len(xPix)):
            if chipNameList[ii] is None or chipNameList[ii] == 'None':
                x_f[ii] = np.NaN
                y_f[ii] = np.NaN
                continue

            pixel_pt = afwGeom.Point2D(xPix[ii], yPix[ii])
            focal_pt = pixel_to_focal_dict[chipNameList[ii]].applyForward(pixel_pt)
            x_f[ii] = focal_pt.getX()
            y_f[ii] = focal_pt.getY()

    else:
        if chipNameList[0] is None or chipNameList[ii] == 'None':
            x_f = np.NaN
            y_f = np.NaN
        else:
            pixel_pt = afwGeom.Point2D(xPix, yPix)
            focal_pt = pixel_to_focal_dict[chipNameList[0]].applyForward(pixel_pt)
            x_f = focal_pt.getX()
            y_f = focal_pt.getY()

    return pupilCoordsFromFocalPlaneCoordsLSST(x_f, y_f, band=band)


def pixelCoordsFromPupilCoordsLSST(xPupil, yPupil, chipName=None, band="r",
                                   includeDistortion=True):
    """
    Convert radians on the pupil into pixel coordinates.

    Parameters
    ----------
    xPupil -- is the x coordinate on the pupil in radians

    yPupil -- is the y coordinate on the pupil in radians

    chipName -- designates the names of the chips on which the pixel
    coordinates will be reckoned.  Can be either single value, an array, or None.
    If an array, there must be as many chipNames as there are (xPupil, yPupil) pairs.
    If a single value, all of the pixel coordinates will be reckoned on the same
    chip.  If None, this method will calculate which chip each(xPupil, yPupil) pair
    actually falls on, and return pixel coordinates for each (xPupil, yPupil) pair on
    the appropriate chip.  Default is None.

    band -- the filter we are simulating (default=r)

    includeDistortion -- a boolean which turns on and off optical distortions
    (default=True)

    Returns
    -------
    a 2-D numpy array in which the first row is the x pixel coordinate
    and the second row is the y pixel coordinate
    """

    if not includeDistortion:
        return pixelCoordsFromPupilCoords(xPupil, yPupil, chipName=chipName,
                                          camera=lsst_camera(),
                                          includeDistortion=includeDistortion)

    are_arrays, \
    chipNameList = _validate_inputs_and_chipname([xPupil, yPupil],
                                                 ['xPupil', 'yPupil'],
                                                 'pixelCoordsFromPupilCoordsLSST',
                                                 chipName)

    if chipNameList is None:
        chipNameList = chipNameFromPupilCoordsLSST(xPupil, yPupil)
        if not isinstance(chipNameList, np.ndarray):
            chipNameList = np.array([chipNameList])
    else:
        if not isinstance(chipNameList, list) and not isinstance(chipNameList, np.ndarray):
            chipNameList = np.array([chipNameList])
        elif isinstance(chipNameList, list):
            chipNameList = np.array(chipNameList)

    x_f, y_f = focalPlaneCoordsFromPupilCoordsLSST(xPupil, yPupil, band=band)

    if are_arrays:
        x_pix = np.NaN*np.ones(len(x_f), dtype=float)
        y_pix = np.NaN*np.ones(len(x_f), dtype=float)

        chipNameList_str = chipNameList.astype(str)
        for chip_name in np.unique(chipNameList_str):
            if chip_name == 'None':
                continue
            det = lsst_camera()[chip_name]
            focal_to_pixels = det.getTransform(FOCAL_PLANE, PIXELS)

            valid = np.where(np.char.find(chipNameList_str, chip_name)==0)
            for ii in valid[0]:
                focal_pt = afwGeom.Point2D(x_f[ii], y_f[ii])
                pixel_pt = focal_to_pixels.applyForward(focal_pt)
                x_pix[ii] = pixel_pt.getX()
                y_pix[ii] = pixel_pt.getY()
    else:
        chip_name = chipNameList[0]
        if chip_name is None:
            x_pix = np.NaN
            y_pix = np.NaN
        else:
            det = lsst_camera()[chip_name]
            focal_to_pixels = det.getTransform(FOCAL_PLANE, PIXELS)
            focal_pt = afwGeom.Point2D(x_f, y_f)
            pixel_pt = focal_to_pixels.applyForward(focal_pt)
            x_pix= pixel_pt.getX()
            y_pix = pixel_pt.getY()

    return np.array([x_pix, y_pix])


def _pixelCoordsFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                              obs_metadata=None,
                              chipName=None, camera=None,
                              epoch=2000.0, includeDistortion=True,
                              band='r'):
    """
    Get the pixel positions on the LSST camera (or nan if not on a chip) for objects based
    on their RA, and Dec (in radians)

    @param [in] ra is in radians in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] dec is in radians in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] pm_ra is proper motion in RA multiplied by cos(Dec) (radians/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] pm_dec is proper motion in dec (radians/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] parallax is parallax in radians
    Can be a numpy array or a number or None (default=None).

    @param [in] v_rad is radial velocity (km/s)
    Can be a numpy array or a number or None (default=None).

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

    @param [in] band is the filter we are simulating ('u', 'g', 'r', etc.) Default='r'

    @param [out] a 2-D numpy array in which the first row is the x pixel coordinate
    and the second row is the y pixel coordinate
    """

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

    xPupil, yPupil = _pupilCoordsFromRaDec(ra, dec,
                                           pm_ra=pm_ra, pm_dec=pm_dec,
                                           parallax=parallax, v_rad=v_rad,
                                           obs_metadata=obs_metadata, epoch=epoch)

    return pixelCoordsFromPupilCoordsLSST(xPupil, yPupil, chipName=chipName, band=band,
                                          includeDistortion=includeDistortion)


def pixelCoordsFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                             obs_metadata=None, chipName=None,
                             epoch=2000.0, includeDistortion=True,
                             band='r'):
    """
    Get the pixel positions on the LSST camera (or nan if not on a chip) for objects based
    on their RA, and Dec (in degrees)

    @param [in] ra is in degrees in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] dec is in degrees in the International Celestial Reference System.
    Can be either a float or a numpy array.

    @param [in] pm_ra is proper motion in RA multiplied by cos(Dec) (arcsec/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] pm_dec is proper motion in dec (arcsec/yr)
    Can be a numpy array or a number or None (default=None).

    @param [in] parallax is parallax in arcsec
    Can be a numpy array or a number or None (default=None).

    @param [in] v_rad is radial velocity (km/s)
    Can be a numpy array or a number or None (default=None).

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

    @param [in] includeDistortion is a boolean.  If True (default), then this method will
    return the true pixel coordinates with optical distortion included.  If False, this
    method will return TAN_PIXEL coordinates, which are the pixel coordinates with
    estimated optical distortion removed.  See the documentation in afw.cameraGeom for more
    details.

    @param [in] band is the filter we are simulating ('u', 'g', 'r', etc.) Default='r'

    @param [out] a 2-D numpy array in which the first row is the x pixel coordinate
    and the second row is the y pixel coordinate
    """

    if pm_ra is not None:
        pm_ra_out = radiansFromArcsec(pm_ra)
    else:
        pm_ra_out = None

    if pm_dec is not None:
        pm_dec_out = radiansFromArcsec(pm_dec)
    else:
        pm_dec_out = None

    if parallax is not None:
        parallax_out = radiansFromArcsec(parallax)
    else:
        parallax_out = None

    return _pixelCoordsFromRaDecLSST(np.radians(ra), np.radians(dec),
                                     pm_ra=pm_ra_out, pm_dec=pm_dec_out,
                                     parallax=parallax_out, v_rad=v_rad,
                                     chipName=chipName, obs_metadata=obs_metadata,
                                     epoch=2000.0, includeDistortion=includeDistortion,
                                     band=band)

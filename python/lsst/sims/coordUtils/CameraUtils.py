import numpy
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE
from lsst.afw.cameraGeom import SCIENCE
from lsst.sims.coordUtils.AstrometryUtils import calculatePupilCoordinates

__all__ = ["findChipName"]

def findChipName(xPupil=None, yPupil=None, ra=None, dec=None,
                 obs_metadata=None, epoch=None, camera=None,
                 allow_multiple_chips=False):
    """
    Return the names of science detectors that see the object specified by
    either (xPupil, yPupil) or (ra, dec).  Note: this method does not return
    the name of guide, focus, or wavefront detectors.

    @param [in] xPupil a numpy array of x pupil coordinates

    @param [in] yPupil a numpy array of y pupil coordinates

    @param [in] ra in radians (optional; should not specify both ra/dec and pupil coordinates)

    @param [in] dec in radians (optional; should not specify both ra/dec and pupil coordinates)

    WARNING: if you are passing in RA and Dec, you should make sure they are in
    the observed reference system, as opposed to the mean, International Celestial Reference
    System (ICRS).  You can transform from the ICRS to the observed reference system using
    the method observedFromICRS in AstrometryUtils.py.  The bore site will be in the observed
    reference system when calculating where on the focal plane your RA and Dec fall.  Thus, the result will
    be wrong if you do not transform your RA and Dec before passing them in.

    @param [in] obs_metadata is an ObservationMetaData object describing the telescope
    pointing (only if specifying RA and Dec rather than pupil coordinates)

    @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
    (in years; only if specifying RA and Dec rather than pupil coordinates)

    @param [in] allow_multiple_chips is a boolean (default False) indicating whether or not
    this method will allow objects to be visible on more than one chip.  If it is 'False'
    and an object appears on more than one chip, an exception will be raised.  If it is 'True'
    and an object falls on more than one chip, it will still only return the first chip in the
    list of chips returned. THIS BEHAVIOR SHOULD BE FIXED IN A FUTURE TICKET.

    @param [in] camera is an afwCameraGeom object that specifies the attributes of the camera.

    @param [out] a numpy array of chip names (science detectors only)

    """
    specifiedPupil = (xPupil is not None and yPupil is not None)
    specifiedRaDec = (ra is not None and dec is not None)

    if not specifiedPupil and not specifiedRaDec:
        raise RuntimeError("You must specifyeither pupil coordinates or equatorial coordinates in findChipName")

    if specifiedPupil and specifiedRaDec:
        raise RuntimeError("You cannot specify both pupil coordinates and equatorial coordinates in findChipName")

    if specifiedPupil and len(xPupil) != len(yPupil):
        raise RuntimeError("You did not pass in an equal number of xPupil and yPupil coordinates")

    if specifiedRaDec and len(ra) != len(dec):
        raise RuntimeError("You did not pass in an equal number of ra and dec coordinates")

    if specifiedRaDec:
        if epoch is None:
            raise RuntimeError("You have to specify an epoch to run " + \
                               "findChipName on these inputs")

        if obs_metadata is None:
            raise RuntimeError("You have to specifay an ObservationMetaData to run " + \
                               "findChipName on these inputs")

        xPupil, yPupil = calculatePupilCoordinates(ra, dec, epoch=epoch, obs_metadata=obs_metadata)

    if not camera:
        if not camera:
            raise RuntimeError("No camera defined.  Cannot rin findChipName.")

    chipNames = []

    cameraPointList = [afwGeom.Point2D(x,y) for x,y in zip(xPupil, yPupil)]

    detList = camera.findDetectorsList(cameraPointList, PUPIL)

    for det in detList:
        if len(det)==0:
            chipNames.append(None)
        else:
            names = [dd.getName() for dd in det if dd.getType()==SCIENCE]
            if len(names)>1 and not self.allow_multiple_chips:
                raise RuntimeError("This method does not know how to deal with cameras " +
                                   "where points can be on multiple detectors.  " +
                                   "Override CameraCoords.get_chipName to add this.")
            elif len(names)==0:
                chipNames.append(None)
            else:
                chipNames.append(names[0])

    return numpy.array(chipNames)

from __future__ import division
from builtins import zip
from builtins import range
import numpy as np
import numbers
import lsst.geom as geom
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
from lsst.afw.cameraGeom import DetectorType
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import focalPlaneCoordsFromPupilCoords
from lsst.sims.coordUtils import LsstZernikeFitter
from lsst.sims.coordUtils import pupilCoordsFromPixelCoords, pixelCoordsFromPupilCoords
from lsst.sims.coordUtils import pupilCoordsFromFocalPlaneCoords
from lsst.sims.coordUtils import pupilCoordsFromPixelCoords
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.utils import _raDecFromPupilCoords
from lsst.sims.coordUtils import getCornerPixels, _validate_inputs_and_chipname
from lsst.sims.utils.CodeUtilities import _validate_inputs
from lsst.sims.utils import radiansFromArcsec

__all__ = ["focalPlaneCoordsFromPupilCoordsLSST",
           "pupilCoordsFromFocalPlaneCoordsLSST",
           "chipNameFromPupilCoordsLSST",
           "_chipNameFromRaDecLSST", "chipNameFromRaDecLSST",
           "pixelCoordsFromPupilCoordsLSST",
           "pupilCoordsFromPixelCoordsLSST",
           "_pixelCoordsFromRaDecLSST", "pixelCoordsFromRaDecLSST",
           "_raDecFromPixelCoordsLSST", "raDecFromPixelCoordsLSST"]

def _lsstCoordUtilsError():
    msg = "\n\n"
    msg += "You are calling a method from sims_coordUtils.LsstCameraUtils.\n"
    msg += "This used to handle sky-to-focal-plane transformations based on "
    msg += "lsst.obs.lsstSim.\n"
    msg += "lsst.obs.lsstSim is no longer a supported part of the LSST stack.\n"
    msg += "It has been replaced with lsst.obs.lsst. Because lsst.obs.lsst uses\n"
    msg += "different coordinate conventions than lsst.obs.lsstSim, it was\n"
    msg += "decided not to port the LsstCameraUtils functionality over to\n"
    msg += "using lsst.obs.lsst. If you would still like to use the functionality\n"
    msg += "provided by sims_coordUtils for mapping sky coordinates to focal plane\n"
    msg += "coordinates, it is recommended that you use the methods in\n"
    msg += "lsst.sims.coordUtils.CameraUtils with a camera instantiated as\n\n"
    msg += "lsst.obs.lsst.phosim.PhosimMapper().camera\n\n"

    raise RuntimeError(msg)


def focalPlaneCoordsFromPupilCoordsLSST(xPupil, yPupil, band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.focalPlaneCoordsFromPupilCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def pupilCoordsFromFocalPlaneCoordsLSST(xmm, ymm, band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.pupilCoordsFromFocalPlaneCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """
    _lsstCoordUtilsError()
    return None


def chipNameFromPupilCoordsLSST(xPupil_in, yPupil_in, allow_multiple_chips=False, band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.chipNameFromPupilCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def _chipNameFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                           obs_metadata=None, epoch=2000.0, allow_multiple_chips=False,
                           band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils._chipNameFromRaDec()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def chipNameFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                          obs_metadata=None, epoch=2000.0, allow_multiple_chips=False,
                          band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.chipNameFromRaDec()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def pupilCoordsFromPixelCoordsLSST(xPix, yPix, chipName=None, band="r",
                                   includeDistortion=True):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.pupilCoordsFromPixelCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def pixelCoordsFromPupilCoordsLSST(xPupil, yPupil, chipName=None, band="r",
                                   includeDistortion=True):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.pixelCoordsFromPupilCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def _pixelCoordsFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                              obs_metadata=None,
                              chipName=None, camera=None,
                              epoch=2000.0, includeDistortion=True,
                              band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils._pixelCoordsFromRaDec()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def pixelCoordsFromRaDecLSST(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                             obs_metadata=None, chipName=None,
                             epoch=2000.0, includeDistortion=True,
                             band='r'):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.pixelCoordsFromRaDec()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def _raDecFromPixelCoordsLSST(xPix, yPix, chipName, band='r',
                              obs_metadata=None, epoch=2000.0,
                              includeDistortion=True):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils._raDecFromPixelCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None


def raDecFromPixelCoordsLSST(xPix, yPix, chipName, band='r',
                             obs_metadata=None, epoch=2000.0,
                             includeDistortion=True):
    """
    Deprecated in favor of
    lsst.sims.coordUtils.CameraUtils.raDecFromPixelCoords()
    with camera = lsst.obs.lsst.phosim.PhosimMapper().camera
    """

    _lsstCoordUtilsError()
    return None

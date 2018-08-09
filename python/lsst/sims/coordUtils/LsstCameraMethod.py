from lsst.obs.lsstCam import lsstCamMapper
import lsst.log as lsstLog


__all__ = ["lsst_camera"]


def lsst_camera():
    """
    Return a copy of the LSST Camera model as stored in obs_lsstSim.
    """
    if not hasattr(lsst_camera, '_lsst_camera'):
        lsstLog.setLevel('CameraMapper', lsstLog.WARN)
        lsst_camera._lsst_camera = lsstCamMapper().camera

    return lsst_camera._lsst_camera


__all__ = ["lsst_camera"]


def lsst_camera():
    """
    Return a copy of the LSST Camera model as stored in obs_lsstSim.
    """
    msg = "\n\n"
    msg += "You are calling sims_coordUtils.lsst_camera().\n"
    msg += "This used to return a model for the LSST camera based on "
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
    return None

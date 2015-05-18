import numpy
import palpy
from lsst.sims.utils import radiansToArcsec, sphericalToCartesian, cartesianToSpherical

def applyPrecession(ra, dec, EP0=2000.0, MJD=2000.0):
    """
    applyPrecession() applies precesion and nutation to coordinates between two epochs.
    Accepts RA and dec as inputs.  Returns corrected RA and dec (in radians).

    Assumes FK5 as the coordinate system
    units:  ra_in (radians), dec_in (radians)

    The precession-nutation matrix is calculated by the palpy.prenut method
    which uses the IAU 2006/2000A model

    @param [in] ra

    @param [in] dec

    @param [out] raOut is ra corrected for precession and nutation

    @param [out] decOut is dec corrected for precession and nutation

    """

    # Determine the precession and nutation
    #palpy.prenut takes the julian epoch for the mean coordinates
    #and the MJD for the the true coordinates
    #
    #TODO it is not specified what this MJD should be (i.e. in which
    #time system it should be reckoned)
    rmat=palpy.prenut(EP0, MJD)

    # Apply rotation matrix
    xyz = sphericalToCartesian(ra,dec)
    xyz =  numpy.dot(rmat,xyz)

    raOut,decOut = cartesianToSpherical(xyz)
    return raOut,decOut



def applyProperMotion(ra, dec, pm_ra, pm_dec, parallax, v_rad, \
                      EP0=2000.0, mjd=None):
    """Applies proper motion between two epochs.

    units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
    (radians/year), parallax (arcsec), v_rad (km/sec, positive if receding),
    EP0 (Julian years)

    Returns corrected ra and dec (in radians)

    The function palpy.pm does not work properly if the parallax is below
    0.00045 arcseconds

    @param [in] ra in radians

    @param [in] dec in radians

    @param [in] pm_ra is ra proper motion in radians/year

    @param [in] pm_dec is dec proper motion in radians/year

    @param [in] parallax in radians

    @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)

    @param [in] EP0 is epoch in Julian years

    @param [in] mjd is the MJD of the actual observation

    @param [out] raOut is corrected ra

    @param [out] decOut is corrected dec

    """

    if mjd is None:
        raise RuntimeError("in Astrometry.py cannot call applyProperMotion; mjd is None")

    parallaxArcsec=radiansToArcsec(parallax)
    #convert to Arcsec because that is what PALPY expects

    # Generate Julian epoch from MJD
    #
    #TODO do we actually want proper motion measured against
    #obs_metadata.mjd (it is unclear what time system we should
    #be using; just that the argument passed to palpy.pmVector should be in julian years)
    julianEpoch = palpy.epj(mjd)

    #The pm_dec argument passed to pmVector used to be pm_dec/cos(dec).
    #I have done away with that, since PAL expects the user to pass in
    #proper motion in radians/per year.  I leave it to the user to perform
    #whatever coordinate transformations are appropriate to the data.
    raOut, decOut = palpy.pmVector(ra,dec,pm_ra,pm_dec,parallaxArcsec,v_rad, EP0, julianEpoch)

    return raOut,decOut

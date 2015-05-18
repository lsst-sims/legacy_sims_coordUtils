import numpy
import palpy
from lsst.sims.utils import radiansToArcsec, sphericalToCartesian, cartesianToSpherical

__all__ = ["applyPrecession", "applyProperMotion", "appGeoFromICRS"]

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



def appGeoFromICRS(ra, dec, pm_ra=None, pm_dec=None, parallax=None,
                   v_rad=None, Epoch0=2000.0, MJD = None):
    """
    Convert the mean position (RA, Dec) in the International Celestial Reference
    System (ICRS) to the mean apparent geocentric position in
    (Ra, Dec)-like coordinates

    Uses PAL mappa routine, which computes precession and nutation

    units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
    (radians/year), parallax (arcsec), v_rad (km/sec; positive if receding),
    EP0 (Julian years)

    Returns corrected RA and Dec

    This calls palpy.mapqk(z) which accounts for proper motion, parallax,
    radial velocity, aberration, precession-nutation

    @param [in] ra in radians (ICRS).  Must be a numpy array.

    @param [in] dec in radians (ICRS).  Must be a numpy array.

    @param [in] pm_ra is ra proper motion in radians/year

    @param [in] pm_dec is dec proper motion in radians/year

    @param [in] parallax in radians

    @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)

    @param [in] Epoch0 is the julian epoch (in years) of the equinox against which to
    measure RA

    @param[in] MJD is the date of the observation (optional; if None, the code will
    try to set it from self.obs_metadata assuming that this method is being called
    by an InstanceCatalog daughter class.  If that is not the case, an exception
    will be raised.)

    @param [out] raOut is apparent geocentric RA-like coordinate in radians

    @param [out] decOut is apparent geocentric Dec-like coordinate in radians

    """

    if MJD is None:
        raise RuntimeError("in Astrometry.py cannot call applyMeanApparentPlace; mjd is None")

    if len(ra) != len(dec):
        raise RuntimeError('in Astrometry.py:applyMeanApparentPlace len(ra) %d len(dec) %d '
                        % (len(ra),len(dec)))

    if pm_ra is None:
        pm_ra=numpy.zeros(len(ra))

    if pm_dec is None:
        pm_dec=numpy.zeros(len(ra))

    if v_rad is None:
        v_rad=numpy.zeros(len(ra))

    if parallax is None:
        parallax=numpy.zeros(len(ra))

    # Define star independent mean to apparent place parameters
    #palpy.mappa calculates the star-independent parameters
    #needed to correct RA and Dec
    #e.g the Earth barycentric and heliocentric position and velocity,
    #the precession-nutation matrix, etc.
    #
    #arguments of palpy.mappa are:
    #epoch of mean equinox to be used (Julian)
    #
    #date (MJD)
    #
    #TODO This mjd should be the Barycentric Dynamical Time
    prms=palpy.mappa(Epoch0, MJD)

    #palpy.mapqk does a quick mean to apparent place calculation using
    #the output of palpy.mappa
    #
    #Taken from the palpy source code (palMap.c which calls both palMappa and palMapqk):
    #The accuracy is sub-milliarcsecond, limited by the
    #precession-nutation model (see palPrenut for details).

    raOut,decOut = palpy.mapqkVector(ra,dec,pm_ra,pm_dec,radiansToArcsec(parallax),v_rad,prms)

    return raOut,decOut

import numpy
import palpy
from lsst.sims.utils import radiansToArcsec, sphericalToCartesian, cartesianToSpherical

__all__ = ["applyPrecession", "applyProperMotion", "appGeoFromICRS", "observedFromAppGeo",
           "observedFromICRS"]

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
        raise RuntimeError("cannot call applyProperMotion; mjd is None")

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

    @param[in] MJD is the date of the observation

    @param [out] raOut is apparent geocentric RA-like coordinate in radians

    @param [out] decOut is apparent geocentric Dec-like coordinate in radians

    """

    if MJD is None:
        raise RuntimeError("cannot call appGeoFromICRS; mjd is None")

    if len(ra) != len(dec):
        raise RuntimeError('appGeoFromICRS: len(ra) %d len(dec) %d '
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



def observedFromAppGeo(ra, dec, includeRefraction = True,
                       altAzHr=False, wavelength=0.5, obs_metadata = None):
    """
    Convert apparent geocentric (RA, Dec)-like coordinates to observed
    (RA, Dec)-like coordinates.  More specifically, apply refraction and
    diurnal aberration.

    Uses PAL aoppa routines

    This will call palpy.aopqk

    @param [in] ra is geocentric apparent RA (radians).  Must be a numpy array.

    @param [in] dec is geocentric apparent Dec (radians).  Must be a numpy array.

    @param [in] includeRefraction is a boolean to turn refraction on and off

    @param [in] altAzHr is a boolean indicating whether or not to return altitude
    and azimuth

    @param [in] wavelength is effective wavelength in microns

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    observation (optional; if not included, the code will try to set it from
    self assuming it is in an InstanceCatalog daughter class.  If that is not
    the case, an exception will be raised.)

    @param [out] raOut is oberved RA (radians)

    @param [out] decOut is observed Dec (radians)

    @param [out] alt is altitude angle (only returned if altAzHr == True) (radians)

    @param [out] az is azimuth angle (only returned if altAzHr == True) (radians)

    """

    if obs_metadata is None:

        if obs_metadata is None:
            raise RuntimeError("Cannot call observedFromAppGeo without an obs_metadata")

    if not hasattr(obs_metadata, 'site') or obs_metadata.site is None:
        raise RuntimeError("Cannot call observedFromAppGeo: obs_metadata has no site info")

    # Correct site longitude for polar motion slaPolmo
    #
    #17 October 2014
    #  palAop.c (which calls Aoppa and Aopqk, as we do here) says
    #  *     - The azimuths etc produced by the present routine are with
    #  *       respect to the celestial pole.  Corrections to the terrestrial
    #  *       pole can be computed using palPolmo.
    #
    #currently, palPolmo is not implemented in PAL
    #I have filed an issue with the PAL team to change that.

    # TODO NEED UT1 - UTC to be kept as a function of date.
    # Requires a look up of the IERS tables (-0.9<dut1<0.9)
    # Assume dut = 0.3 (seconds)
    dut = 0.3

    #
    #palpy.aoppa computes star-independent parameters necessary for
    #converting apparent place into observed place
    #i.e. it calculates geodetic latitude, magnitude of diurnal aberration,
    #refraction coefficients and the like based on data about the observation site
    #
    #TODO: palpy.aoppa requires as its first argument
    #the UTC time expressed as an MJD.  It is not clear to me
    #how to actually calculate that.
    if (includeRefraction == True):
        obsPrms=palpy.aoppa(obs_metadata.mjd, dut,
                          obs_metadata.site.longitude,
                          obs_metadata.site.latitude,
                          obs_metadata.site.height,
                          obs_metadata.site.xPolar,
                          obs_metadata.site.yPolar,
                          obs_metadata.site.meanTemperature,
                          obs_metadata.site.meanPressure,
                          obs_metadata.site.meanHumidity,
                          wavelength ,
                          obs_metadata.site.lapseRate)
    else:
        #we can discard refraction by setting pressure and humidity to zero
        obsPrms=palpy.aoppa(obs_metadata.mjd, dut,
                          obs_metadata.site.longitude,
                          obs_metadata.site.latitude,
                          obs_metadata.site.height,
                          obs_metadata.site.xPolar,
                          obs_metadata.site.yPolar,
                          obs_metadata.site.meanTemperature,
                          0.0,
                          0.0,
                          wavelength ,
                          obs_metadata.site.lapseRate)

    #palpy.aopqk does an apparent to observed place
    #correction
    #
    #it corrects for diurnal aberration and refraction
    #(using a fast algorithm for refraction in the case of
    #a small zenith distance and a more rigorous algorithm
    #for a large zenith distance)
    #

    azimuth, zenith, hourAngle, decOut, raOut = palpy.aopqkVector(ra,dec,obsPrms)

    #
    #Note: this is a choke point.  Even the vectorized version of aopqk
    #is expensive (it takes about 0.006 seconds per call)
    #
    #Actually, this is only a choke point if you are dealing with zenith
    #distances of greater than about 70 degrees

    if altAzHr == True:
        #
        #palpy.de2h converts equatorial to horizon coordinates
        #
        az, alt = palpy.de2hVector(hourAngle,decOut,obs_metadata.site.latitude)
        return raOut, decOut, alt, az
    return raOut, decOut


def observedFromICRS(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                     obs_metadata=None, epoch=None, includeRefraction=True):
    """
    Convert mean position (RA, Dec) in the International Celestial Reference Frame
    to observed (RA, Dec)-like coordinates

    included are precession-nutation, aberration, proper motion, parallax, refraction,
    radial velocity, diurnal aberration,

    @param [in] ra is the unrefracted RA in radians (ICRS).  Must be a numpy array.

    @param [in] dec is the unrefracted Dec in radians (ICRS).  Must be a numpy array.

    @param [in] pm_ra is proper motion in RA (radians/yr)

    @param [in] pm_dec is proper motion in dec (radians/yr)

    @param [in] parallax is parallax in radians

    @param [in] v_rad is radial velocity (km/s)

    @param [in] obs_metadata is an ObservationMetaData object describing the
    telescope pointing.  If it is None, the code will try to set it from self
    assuming that this method is being called from within an InstanceCatalog
    daughter class.  If that is not the case, an exception will be raised

    @param [in] epoch is the julian epoch (in years) against which the mean
    equinoxes are measured.

    @param [in] includeRefraction toggles whether or not to correct for refraction

    @param [out] ra_out RA corrected for all included effects

    @param [out] dec_out Dec corrected for all included effects

    """

    if obs_metadata is None:
        raise RuntimeError("cannot call observedFromICRS; obs_metadata is none")

    if obs_metadata.mjd is None:
        raise RuntimeError("cannot call observedFromICRS; obs_metadata.mjd is none")

    if epoch is None:
        raise RuntimeError("cannot call observedFromICRS; you have not specified an epoch")

    ra_apparent, dec_apparent = appGeoFromICRS(ra, dec, pm_ra = pm_ra,
             pm_dec = pm_dec, parallax = parallax, v_rad = v_rad, Epoch0 = epoch, MJD=obs_metadata.mjd)

    ra_out, dec_out = observedFromAppGeo(ra_apparent, dec_apparent, obs_metadata=obs_metadata,
                                               includeRefraction = includeRefraction)

    return numpy.array([ra_out,dec_out])
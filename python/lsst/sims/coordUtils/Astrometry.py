import numpy
import ctypes
import math
import palpy as pal
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE
from lsst.afw.cameraGeom import SCIENCE
from lsst.sims.catalogs.measures.instance import compound
from lsst.sims.utils import haversine, radiansToArcsec, arcsecToRadians
from lsst.sims.utils import equatorialToGalactic, cartesianToSpherical, sphericalToCartesian

from lsst.sims.coordUtils.AstrometryUtils import appGeoFromICRS, observedFromAppGeo

__all__ = ["AstrometryBase", "AstrometryStars", "AstrometryGalaxies",
           "CameraCoords"]

class AstrometryBase(object):
    """Collection of astrometry routines that operate on numpy arrays"""

    @compound('glon','glat')
    def get_galactic_coords(self):
        """
        Getter for galactic coordinates, in case the catalog class does not provide that

        Reads in the ra and dec from the data base and returns columns with galactic
        longitude and latitude.

        All angles are in radians
        """
        ra=self.column_by_name('raJ2000')
        dec=self.column_by_name('decJ2000')

        glon, glat = self.equatorialToGalactic(ra,dec)

        return numpy.array([glon,glat])


    def correctCoordinates(self, ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
             obs_metadata=None, epoch=None, includeRefraction=True):
        """
        correct coordinates for all possible effects.

        included are precession-nutation, aberration, proper motion, parallax, refraction,
        radial velocity, diurnal aberration,

        @param [in] ra is the unrefracted RA in radians

        @param [in] dec is the unrefracted Dec in radians

        @param [in] pm_ra is proper motion in RA (radians/yr)

        @param [in] pm_dec is proper motion in dec (radians/yr)

        @param [in] parallax is parallax in radians

        @param [in] v_rad is radial velocity (km/s)

        @param [in] obs_metadata is an ObservationMetaData object describing the
        telescope pointing.  If it is None, the code will try to set it from self
        assuming that this method is being called from within an InstanceCatalog
        daughter class.  If that is not the case, an exception will be raised

        @param [in] epoch is the julian epoch (in years) against which the mean
        equinoxes are measured.  If it is None, the code will try to set it from
        self.db_obj, assuming that the code is being called from an InstanceCatalog
        daughter class.  If that is not the case, an exception will be raised.

        @param [in] includeRefraction toggles whether or not to correct for refraction

        @param [out] ra_out RA corrected for all included effects

        @param [out] dec_out Dec corrected for all included effects

        """

        if obs_metadata is None:
            if hasattr(self, 'obs_metadata'):
                obs_metadata = self.obs_metadata

            if obs_metadata is None:
                raise RuntimeError("in Astrometry.py cannot call correctCoordinates; obs_metadata is none")

        if obs_metadata.mjd is None:
            raise RuntimeError("in Astrometry.py cannot call correctCoordinates; obs_metadata.mjd is none")

        if epoch is None:
            if hasattr(self, 'db_obj'):
                epoch = self.db_obj.epoch

            if epoch is None:
                raise RuntimeError("in Astrometry.py cannot call correctCoordinates; you have not specified an epoch")

        ra_apparent, dec_apparent = appGeoFromICRS(ra, dec, pm_ra = pm_ra,
                 pm_dec = pm_dec, parallax = parallax, v_rad = v_rad, Epoch0 = epoch, MJD=obs_metadata.mjd)

        ra_out, dec_out = observedFromAppGeo(ra_apparent, dec_apparent, obs_metadata=obs_metadata,
                                                   includeRefraction = includeRefraction)

        return numpy.array([ra_out,dec_out])

    def refractionCoefficients(self, wavelength=0.5, site=None):
        """ Calculate the refraction using PAL's refco routine

        This calculates the refraction at 2 angles and derives a tanz and tan^3z
        coefficient for subsequent quick calculations. Good for zenith distances < 76 degrees

        @param [in] wavelength is effective wavelength in microns

        @param [in] site is an instantiation of the Site class defined in
        sims_utils/../Site.py; (optional; if not provided,
        this routine will use the site member variable provided by the
        InstanceCatalog this method is being called from, if one exists)

        One should call PAL refz to apply the coefficients calculated here

        """
        precision = 1.e-10

        hasSite = (site is not None)

        if site is None:
            if hasattr(self,'obs_metadata'):
                if hasattr(self.obs_metadata, 'site'):
                    hasSite = True
                    site = self.obs_metadata.site

        if not hasSite:
            raise RuntimeError("Cannot call refractionCoefficients; no site information")

        #TODO the latitude in refco needs to be astronomical latitude,
        #not geodetic latitude
        _refcoOutput=pal.refco(site.height,
                        site.meanTemperature,
                        site.meanPressure,
                        site.meanHumidity,
                        wavelength ,
                        site.latitude,
                        site.lapseRate,
                        precision)

        return _refcoOutput[0], _refcoOutput[1]

    def applyRefraction(self, zenithDistance, tanzCoeff, tan3zCoeff):
        """ Calculted refracted Zenith Distance

        uses the quick PAL refco routine which approximates the refractin calculation

        @param [in] zenithDistance is unrefracted zenith distance of the source in radians

        @param [in] tanzCoeff is the first output from refractionCoefficients (above)

        @param [in] tan3zCoeff is the second output from refractionCoefficients (above)

        @param [out] refractedZenith is the refracted zenith distance in radians

        """

        refractedZenith=pal.refz(zenithDistance, tanzCoeff, tan3zCoeff)

        return refractedZenith

    def calcLast(self, mjd, long):
        """
        Converts the date mjd+long into Greenwhich Mean Sidereal Time (in radians)

        Note that mjd is the UT1 time expressed as an MJD
        """

        #TODO the arguments of palGmsta are
        # - the UT1 date expressed as an MJD
        # - the UT1 time (fraction of a day)
        D = pal.gmsta(mjd, 0.)
        D += long
        D = D%(2.*math.pi)
        return D

    def equatorialToHorizontal(self, ra, dec, mjd):
        """
        Converts from equatorial to horizon coordinates

        @param [in] ra is in radians

        @param [in] dec is declination in radians

        @param [in] mjd is the date

        @param [out] returns elevation angle and azimuth in that order (radians)

        """

        hourAngle = self.calcLast(mjd, self.obs_metadata.site.longitude) - ra

        _de2hOutput=pal.de2h(hourAngle, dec,  self.obs_metadata.site.latitude)

        #return (altitude, azimuth)
        return _de2hOutput[1], _de2hOutput[0]

    def paralacticAngle(self, az, dec):
        """
        This returns the paralactic angle between the zenith and the pole that is up.
        I need to check this, but this should be +ve in the East and -ve in the West if
        Az is measured from North through East.
        """

        sinpa = math.sin(az)*math.cos(self.obs_metadata.site.latitude)/math.cos(dec)
        return math.asin(sinpa)

    def calculatePupilCoordinates(self, raObj, decObj,
                                  obs_metadata=None, epoch=None):
        """
        @param [in] raObj is a numpy array of RAs in radians

        @param [in] decObj is a numpy array of Decs in radians

        @param [in] obs_metadata is an ObservationMetaData object containing information
        about the telescope pointing (optional; if None, the code will try to set it
        from self assuming that this method is being called from an InstanceCatalog
        daughter class.  If that is not the case, an exception will be raised.)

        @param [in] epoch is the julian epoch of the mean equinox used for the coordinate
        transforations (in years; optional)

        @param [out] a numpy array in which the first row is the x pupil coordinate and the second
        row is the y pupil coordinate

        Take an input RA and dec from the sky and convert it to coordinates
        in the pupil.

        This routine will use the haversine formula to calculate the arc distance h
        between the bore sight and the object.  It will convert this into pupil coordinates
        by assuming that the y-coordinate is identically the declination of the object.
        It will find the x-coordinate by demanding that

        h^2 = (y_bore - y_obj)^2 + (x_bore - x_obj)^2
        """

        if obs_metadata is None:
            if hasattr(self, 'obs_metadata'):
                obs_metadata = self.obs_metadata

            if obs_metadata is None:
                raise RuntimeError("in Astrometry.py cannot calculate x_pupil, y_pupil without obs_metadata")

        if epoch is None:
            if hasattr(self, 'db_obj'):
                epoch = self.db_obj.epoch

            if epoch is None:
                raise RuntimeError("in Astrometry.py cannot call get_skyToPupil; epoch is None")

        if obs_metadata.rotSkyPos is None:
            raise RuntimeError("Cannot calculate x_pupil or y_pupil without rotSkyPos")

        if obs_metadata.unrefractedRA is None or obs_metadata.unrefractedDec is None:
            raise RuntimeError("Cannot calculate x_pupil, y_pupil without unrefractedRA, unrefractedDec")

        if obs_metadata.mjd is None:
            raise RuntimeError("Cannot calculate x_pupil, y_pupil without mjd")

        theta = -1.0*obs_metadata._rotSkyPos

        #correct for precession and nutation

        pointingRA=numpy.array([obs_metadata._unrefractedRA])
        pointingDec=numpy.array([obs_metadata._unrefractedDec])

        x, y = appGeoFromICRS(pointingRA, pointingDec, Epoch0=epoch, MJD=obs_metadata.mjd)

        #correct for refraction
        boreRA, boreDec = observedFromAppGeo(x, y, obs_metadata=obs_metadata)

        #we should now have the true tangent point for the gnomonic projection
        dPhi = decObj - boreDec
        dLambda = raObj - boreRA

        #see en.wikipedia.org/wiki/Haversine_formula
        #Phi is latitude on the sphere (declination)
        #Lambda is longitude on the sphere (RA)
        #
        #Now that there is a haversine method in
        #sims_catalogs_generation/.../db/obsMetadataUtils.py
        #I am using that function so that we only have one
        #haversine formula floating around the stack
        h = haversine(raObj, decObj, boreRA, boreDec)

        #demand that the Euclidean distance on the pupil matches
        #the haversine distance on the sphere
        dx = numpy.sign(dLambda)*numpy.sqrt(h**2 - dPhi**2)
        #correct for rotation of the telescope
        x_out = dx*numpy.cos(theta) - dPhi*numpy.sin(theta)
        y_out = dx*numpy.sin(theta) + dPhi*numpy.cos(theta)

        return numpy.array([x_out, y_out])

    def calculateGnomonicProjection(self, ra_in, dec_in, obs_metadata=None, epoch=None):
        """
        Take an input RA and dec from the sky and convert it to coordinates
        on the focal plane.

        This uses PAL's gnonomonic projection routine which assumes that the focal
        plane is perfectly flat.  The output is in Cartesian coordinates, assuming
        that the Celestial Sphere is a unit sphere.

        @param [in] ra_in is a numpy array of RAs in radians

        @param [in] dec_in in radians

        @param [in] obs_metadata is an ObservationMetaData instantiation characterizing the
        telescope location and pointing (optional; if not provided, the method will try to
        get it from the InstanceCatalog member variable, assuming this is part of an
        InstanceCatalog)

        @param [in] epoch is the epoch of mean ra and dec in julian years (optional; if not
        provided, this method will try to get it from the db_obj member variable, assuming this
        method is part of an InstanceCatalog)

        @param [out] returns a numpy array whose first row is the x coordinate according to a naive
        gnomonic projection and whose second row is the y coordinate
        """

        if obs_metadata is None:
            if hasattr(self, 'obs_metadata'):
                obs_metadata = self.obs_metadata
            else:
                raise RuntimeError("in Astrometry.py cannot call calculateGnomonicProjection without obs_metadata")

        if obs_metadata.mjd is None:
            raise RuntimeError("in Astrometry.py cannot call calculatGnomonicProjection; obs_metadata.mjd is None")

        if epoch is None:
            if hasattr(self, 'db_obj'):
                epoch = self.db_obj.epoch
            else:
                raise RuntimeError("in Astrometry.py cannot call calculateGnomonicProjection without epoch")

        if epoch is None:
            raise RuntimeError("in Astrometry.py cannot call calculateGnomonicProjection; epoch is None")

        x_out=numpy.zeros(len(ra_in))
        y_out=numpy.zeros(len(ra_in))

        if obs_metadata.rotSkyPos is None:
            #there is no observation meta data on which to base astrometry
            raise RuntimeError("Cannot calculate [x,y]_focal_nominal without obs_metadata.rotSkyPos")

        if obs_metadata.unrefractedRA is None or obs_metadata.unrefractedDec is None:
            raise RuntimeError("Cannot calculate [x,y]_focal_nominal without unrefracted RA and Dec in obs_metadata")

        theta = -1.0*obs_metadata._rotSkyPos

        #correct RA and Dec for refraction, precession and nutation
        #
        #correct for precession and nutation
        inRA=numpy.array([obs_metadata._unrefractedRA])
        inDec=numpy.array([obs_metadata._unrefractedDec])

        x, y = appGeoFromICRS(inRA, inDec, Epoch0=epoch, MJD=obs_metadata.mjd)

        #correct for refraction
        trueRA, trueDec = observedFromAppGeo(x, y, obs_metadata=obs_metadata)
        #we should now have the true tangent point for the gnomonic projection

        #pal.ds2tp performs the gnomonic projection on ra_in and dec_in
        #with a tangent point at (trueRA, trueDec)
        #
        x, y = pal.ds2tpVector(ra_in,dec_in,trueRA[0],trueDec[0])

        #rotate the result by rotskypos (rotskypos being "the angle of the sky relative to
        #camera cooridnates" according to phoSim documentation) to account for
        #the rotation of the focal plane about the telescope pointing

        x_out = x*numpy.cos(theta) - y*numpy.sin(theta)
        y_out = x*numpy.sin(theta) + y*numpy.cos(theta)

        return numpy.array([x_out, y_out])

    @compound('x_focal_nominal', 'y_focal_nominal')
    def get_gnomonicProjection(self):
        ra = self.column_by_name('raObserved')
        dec = self.column_by_name('decObserved')
        return self.calculateGnomonicProjection(ra, dec)

    @compound('x_pupil','y_pupil')
    def get_skyToPupil(self):
        """
        Take an input RA and dec from the sky and convert it to coordinates
        in the pupil.
        """

        raObj = self.column_by_name('raObserved')
        decObj = self.column_by_name('decObserved')

        return self.calculatePupilCoordinates(raObj, decObj)

class CameraCoords(AstrometryBase):
    """Methods for getting coordinates from the camera object"""
    camera = None
    allow_multiple_chips = False #this is a flag which, if true, would allow
                                 #findChipName to return objects that land on
                                 #multiple chips; only the first chip would be
                                 #written to the catalog

    def findChipName(self, xPupil=None, yPupil=None, ra=None, dec=None,
                     obs_metadata=None, epoch=None, camera=None):
        """
        Return the names of science detectors that see the object specified by
        either (xPupil, yPupil) or (ra, dec).  Note: this method does not return
        the name of guide, focus, or wavefront detectors

        @param [in] xPupil a numpy array of x pupil coordinates

        @param [in] yPupil a numpy array of y pupil coordinates

        @param [in] ra in radians (optional; should not specify both ra/dec and pupil coordinates)

        @param [in] dec in radians (optional; should not specify both ra/dec and pupil coordinates)

        WARNING: if you are passing in RA and Dec, you should make sure they are corrected
        for all of the astrometric quantities (precession, nutation, refraction, proper motion,
        etc.) relevant to your problem.  The bore site will be corrected for these quantities
        when calculating where on the focal plane your RA and Dec fall.  Thus, the result will
        be wrong if you do not correct your RA and Dec before passing them in.  Consider using
        the method AstrometryBase.correctCoordinates()

        @param [in] obs_metadata is an ObservationMetaData object describing the telescope
        pointing (optional)

        @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
        (in years; optional)

                The optional arguments are there to be passed to calculatePupilCoordinates if you choose
                to call this routine specifying ra and dec, rather than xPupil and yPupil.  If they are
                not set, calculatePupilCoordinates will try to set them from self and db_obj,
                assuming that this routine is being called from an InstanceCatalog daughter class.
                If that is not the case, an exception will be raised.

        @param [in] camera is an afwCameraGeom object that specifies the attributes of the camera.
        If it is not set, this routine will try to set it from self.camera assuming that the code
        is in an InstanceCatalog daughter class.  If that is not the case, an exception will be
        raised

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
            xPupil, yPupil = self.calculatePupilCoordinates(ra, dec, epoch=epoch, obs_metadata=obs_metadata)

        if not camera:
            if hasattr(self, 'camera'):
                camera = self.camera

            if not camera:
                raise RuntimeError("No camera defined.  Cannot retrieve detector name.")

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

    def calculatePixelCoordinates(self, xPupil=None, yPupil=None, ra=None, dec=None, chipNames=None,
                                  obs_metadata=None, epoch=None, camera=None):
        """
        Get the pixel positions (or nan if not on a chip) for all objects in the catalog

        @param [in] xPupil a numpy array containing x pupil coordinates

        @param [in] yPupil a numpy array containing y pupil coordinates

        @param [in] ra one could alternatively provide a numpy array of ra and...

        @param [in] ...dec (both in radians)

        @param [in] chipNames a numpy array of chipNames.  If it is None, this method will call findChipName
        to find the array.  The option exists for the user to specify chipNames, just in case the user
        has already called findChipName for some reason.

        @param [in] obs_metadata is an ObservationMetaData object describing the telescope pointing
        (optional)

        @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
        (in years; optional)

            The optional arguments above are there to be passed to calculatePupilCoordinates if you choose
            to call this routine specifying ra and dec, rather than xPupil and yPupil.  If they are
            not set, calculatePupilCoordinates will try to set them from self and db_obj,
            assuming that this ro;utine is being called from an InstanceCatalog daughter class.
            If that is not the case, an exception will be raised.

        @param [in] camera is an afwCameraGeom object specifying the attributes of the camera.
        This is an optional argument to be passed to findChipName; if it is None, findChipName
        will try to set it from self.camera, assuming that the code is in an InstanceCatalog
        daughter class.  If this is not so, an exception will be raised.

        @param [out] a numpy array of pixel coordinates

        """

        if not camera:
            if hasattr(self, 'camera'):
                camera = self.camera

            if not camera:
                raise RuntimeError("No camera defined.  Cannot calculate pixel coordinates")

        specifiedPupil = False
        specifiedRaDec = False

        specifiedPupil = (xPupil is not None and yPupil is not None)
        specifiedRaDec = (ra is not None and dec is not None)

        if not specifiedPupil and not specifiedRaDec:
            raise RuntimeError("You need to specifiy either pupil coordinates or equatorial coordinates in calculatePixelCoordinates")

        if specifiedPupil and specifiedRaDec:
            raise RuntimeError("You cannot specify both pupil coordinates and equatorial coordinates in calculatePixelCoordinates")

        if specifiedRaDec:
            xPupil, yPupil = self.calculatePupilCoordinates(ra, dec,
                                                            obs_metadata=obs_metadata,
                                                            epoch=epoch)

        if chipNames is None:
            chipNames = self.findChipName(xPupil = xPupil, yPupil = yPupil, camera=camera)

        xPix = []
        yPix = []
        for name, x, y in zip(chipNames, xPupil, yPupil):
            if not name:
                xPix.append(numpy.nan)
                yPix.append(numpy.nan)
                continue
            cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            det = camera[name]
            cs = det.makeCameraSys(PIXELS)
            detPoint = camera.transform(cp, cs)
            xPix.append(detPoint.getPoint().getX())
            yPix.append(detPoint.getPoint().getY())
        return numpy.array([xPix, yPix])

    def calculateFocalPlaneCoordinates(self, xPupil=None, yPupil=None, ra=None, dec=None,
                                       obs_metadata=None, epoch=None, camera=None):
        """
        Get the focal plane coordinates for all objects in the catalog.

        @param [in] xPupil a numpy array of x pupil coordinates

        @param [in] yPupil a numpy array of y pupil coordinates

        @param [in] alternatively, one can specify numpy arrays of ra and dec (in radians)

        @param [in] obs_metadata is an ObservationMetaData object describing the telescope
        pointing (optional)

        @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
        (in years; optional)

            The optional arguments are there to be passed to calculatePupilCoordinates if you choose
            to call this routine specifying ra and dec, rather than xPupil and yPupil.  If they are
            not set, calculatePupilCoordinates will try to set them from self and db_obj,
            assuming that this routine is being called from an InstanceCatalog daughter class.
            If that is not the case, an exception will be raised.

        @param [out] a numpy array in which the first row is the x pixel coordinates
        and the second row is the y pixel coordinates

        """

        specifiedPupil = (xPupil is not None and yPupil is not None)
        specifiedRaDec = (ra is not None and dec is not None)

        if not specifiedPupil and not specifiedRaDec:
            raise RuntimeError("You must specify either pupil coordinates or equatorial coordinates to call calculateFocalPlaneCoordinates")

        if specifiedPupil and specifiedRaDec:
            raise RuntimeError("You cannot specify both pupil and equaltorial coordinates when calling calculateFocalPlaneCoordinates")

        if not camera:
            if hasattr(self, 'camera'):
                camera = self.camera

            if not camera:
                raise RuntimeError("No camera defined.  Cannot calculate focalplane coordinates")

        if specifiedRaDec:
            xPupil, yPupil = self.calculatePupilCoordinates(ra ,dec,
                                                            obs_metadata=obs_metadata, epoch=epoch)

        xPix = []
        yPix = []
        for x, y in zip(xPupil, yPupil):
            cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            fpPoint = camera.transform(cp, FOCAL_PLANE)
            xPix.append(fpPoint.getPoint().getX())
            yPix.append(fpPoint.getPoint().getY())
        return numpy.array([xPix, yPix])

    def get_chipName(self):
        """Get the chip name if there is one for each catalog entry"""
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))
        return self.findChipName(xPupil=xPupil, yPupil=yPupil)

    @compound('xPix', 'yPix')
    def get_pixelCoordinates(self):
        """Get the pixel positions (or nan if not on a chip) for all objects in the catalog"""
        if not self.camera:
            raise RuntimeError("No camera defined.  Cannot calculate pixel coordinates")
        chipNames = self.column_by_name('chipName')
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))

        return self.calculatePixelCoordinates(xPupil = xPupil, yPupil = yPupil, chipNames=chipNames)

    @compound('xFocalPlane', 'yFocalPlane')
    def get_focalPlaneCoordinates(self):
        """Get the focal plane coordinates for all objects in the catalog."""
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))

        return self.calculateFocalPlaneCoordinates(xPupil = xPupil, yPupil = yPupil)

class AstrometryGalaxies(AstrometryBase):
    """
    This mixin contains a getter for the corrected RA and dec which ignores parallax and proper motion
    """

    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')
        return self.correctCoordinates(ra, dec, includeRefraction = False)


    @compound('raObserved','decObserved')
    def get_observedCoordinates(self):
        """
        get coordinates corrected for everything
        """
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')
        return self.correctCoordinates(ra, dec)


class AstrometryStars(AstrometryBase):
    """
    This mixin contains a getter for the corrected RA and dec which takes account of proper motion and parallax
    """

    def correctStellarCoordinates(self, includeRefraction = True):
        """
        Getter which coorrects RA and Dec for propermotion, radial velocity, and parallax

        """

        #TODO
        #are we going to store proper motion in raw radians per year
        #or in sky motion = cos(dec) * (radians per year)
        #PAL asks for radians per year inputs

        pr = self.column_by_name('properMotionRa') #in radians per year
        pd = self.column_by_name('properMotionDec') #in radians per year
        px = self.column_by_name('parallax') #in radians
        rv = self.column_by_name('radialVelocity') #in km/s; positive if receding
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')

        return self.correctCoordinates(ra, dec, pm_ra = pr, pm_dec = pd, parallax = px, v_rad = rv,
                     includeRefraction = includeRefraction)


    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        return self.correctStellarCoordinates(includeRefraction = False)

    @compound('raObserved','decObserved')
    def get_observedCoordinates(self):
        return self.correctStellarCoordinates()

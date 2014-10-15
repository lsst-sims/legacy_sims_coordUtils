import numpy
import ctypes
import math
import palpy as pal
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE
from lsst.sims.catalogs.measures.instance import compound

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

    @staticmethod
    def sphericalToCartesian(longitude, latitude):
        """
        Transforms between spherical and Cartesian coordinates.

        @param [in] longitude is the input longitudinal coordinate

        @param [in] latitude is the input latitudinal coordinate

        @param [out] a list of the (three-dimensional) cartesian coordinates on a unit sphere

        All angles are in radians
        """

        cosDec = numpy.cos(latitude)
        return numpy.array([numpy.cos(longitude)*cosDec,
                          numpy.sin(longitude)*cosDec,
                          numpy.sin(latitude)])

    @staticmethod
    def cartesianToSpherical(xyz):
        """
        Transforms between Cartesian and spherical coordinates

        @param [in] xyz is a list of the three-dimensional Cartesian coordinates

        @param [out] returns longitude and latitude

        All angles are in radians
        """

        rad = numpy.sqrt(xyz[:][0]*xyz[:][0] + xyz[:][1]*xyz[:][1] + xyz[:][2]*xyz[:][2])

        longitude = numpy.arctan2( xyz[:][1], xyz[:][0])
        latitude = numpy.arcsin( xyz[:][2] / rad)

        return longitude, latitude

    @staticmethod
    def angularSeparation(long1, lat1, long2, lat2):
        ''' Given two spherical points in radians, calculate the angular
        separation between them.

        @param [in] long1 is the longitudinal coordinate of one point
        (long2 is the longitude of the other point)

        @param [in] lat1 is the latitudinal coordinate of one point
        (lat2 is the latitude of the other point)

        @param [out] D the angular separation in radians

        All angles are in radians
        '''
        D = pal.dsep (long1, lat1, long2, lat2)
        return D

    @staticmethod
    def rotationMatrixFromVectors(v1, v2):
        '''
        Given two vectors v1,v2 calculate the rotation matrix for v1->v2 using the axis-angle approach

        @param [in] v1, v2 are two Cartesian vectors (in three dimensions)

        @param [out] rot is the rotation matrix that rotates from one to the other

        '''

        # Calculate the axis of rotation by the cross product of v1 and v2
        cross = numpy.cross(v1,v2)
        cross = cross / math.sqrt(numpy.dot(cross,cross))

        # calculate the angle of rotation via dot product
        angle  = numpy.arccos(numpy.dot(v1,v2))
        sinDot = math.sin(angle)
        cosDot = math.cos(angle)

        # calculate the corresponding rotation matrix
        # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        rot = [[cosDot + cross[0]*cross[0]*(1-cosDot), -cross[2]*sinDot+(1-cosDot)*cross[0]*cross[1], \
                cross[1]*sinDot + (1-cosDot)*cross[0]*cross[2]],\
                [cross[2]*sinDot+(1-cosDot)*cross[0]*cross[1], cosDot + (1-cosDot)*cross[1]*cross[1], \
                -cross[0]*sinDot+(1-cosDot)*cross[1]*cross[2]], \
                [-cross[1]*sinDot+(1-cosDot)*cross[0]*cross[2], \
                cross[0]*sinDot+(1-cosDot)*cross[1]*cross[2], \
                cosDot + (1-cosDot)*(cross[2]*cross[2])]]

        return rot


    def applyPrecession(self, ra, dec, EP0=2000.0, MJD=2000.0):
        """
        applyPrecession() applies precesion and nutation to coordinates between two epochs.
        Accepts RA and dec as inputs.  Returns corrected RA and dec (in radians).

        Assumes FK5 as the coordinate system
        units:  ra_in (radians), dec_in (radians)

        The precession-nutation matrix is calculated by the pal.prenut method
        which uses the IAU 2006A/2000 model

        @param [in] ra

        @param [in] dec

        @param [out] raOut is ra corrected for precession and nutation

        @param [out] decOut is dec corrected for precession and nutation

        """

        # Determine the precession and nutation
        #pal.prenut takes the julian epoch for the mean coordinates
        #and the MJD for the the true coordinates
        rmat=pal.prenut(EP0, MJD)

        # Apply rotation matrix
        xyz = self.sphericalToCartesian(ra,dec)
        xyz =  numpy.dot(rmat,xyz)

        raOut,decOut = self.cartesianToSpherical(xyz)
        return raOut,decOut

    def applyProperMotion(self, ra, dec, pm_ra, pm_dec, parallax, v_rad, \
                          EP0=2000.0, MJD=2015.0):
        """Applies proper motion between two epochs.

        Note pm_ra is measured in sky velocity (cos(dec)*dRa/dt).
        PAL assumes dRa/dt

        units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
        (radians/year), parallax (arcsec), v_rad (km/sec, positive if receding),
        EP0 (Julian years)

        Returns corrected ra and dec (in radians)

        The function pal.pm does not work properly if the parallax is below
        0.00045 arcseconds

        @param [in] ra in radians

        @param [in] dec in radians

        @param [in] pm_ra is ra proper motion in radians/year

        @param [in] pm_dec is dec proper motion in radians/year

        @param [in] parallax in arcseconds

        @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)

        @param [in] EP0 is epoch in Julian years

        @param [in] MJD is modified Julian date

        @param [out] raOut is corrected ra

        @param [out] decOut is corrected dec

        """

        if self.obs_metadata.mjd is None:
            raise ValueError("in Astrometry.py cannot call applyProperMotion; self.obs_metadata.mjd is None")

        px = numpy.where(parallax<0.00045, 0.00045, parallax)
        #so that pal.Pm returns meaningful values

        # Generate Julian epoch from MJD
        julianEpoch = pal.epj(self.obs_metadata.mjd)
        raOut, decOut = pal.pmVector(ra,dec,pm_ra,pm_dec/numpy.cos(dec),parallax,v_rad, EP0, julianEpoch)

        return raOut,decOut

    @staticmethod
    def equatorialToGalactic(ra, dec):
        '''Convert RA,Dec (J2000) to Galactic Coordinates

        All angles are in radians
        '''
        gLong = numpy.zeros(len(ra))
        gLat = numpy.zeros(len(ra))

        for i in range(len(ra)):
            _eqgalOutput=pal.eqgal(ra[i], dec[i])
            gLong[i] = _eqgalOutput[0]
            gLat[i] = _eqgalOutput[1]

        return gLong, gLat

    @staticmethod
    def galacticToEquatorial(gLong, gLat):
        '''Convert Galactic Coordinates to RA, dec (J2000)'''
        ra = numpy.zeros(len(gLong))
        dec = numpy.zeros(len(gLong))

        for i in range(len(ra)):
            _galeqOutput=pal.galeq(gLong[i], gLat[i])
            ra[i] = _galeqOutput[0]
            dec[i] = _galeqOutput[1]

        return ra, dec

    def applyMeanApparentPlace(self, ra, dec, pm_ra=None, pm_dec=None, parallax=None,
         v_rad=None, Epoch0=2000.0, MJD=2015.0):
        """Calculate the Mean Apparent Place given an Ra and Dec

        Uses PAL mappa routines
        Recomputers precession and nutation

        units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
        (radians/year), parallax (arcsec), v_rad (km/sec; positive if receding),
        EP0 (Julian years)

        Returns corrected RA and Dec

        This calls pal.mapqk(z) which accounts for proper motion, parallax,
        radial velocity, aberration, precession-nutation

        @param [in] ra in radians

        @param [in] dec in radians

        @param [in] pm_ra is ra proper motion in radians/year

        @param [in] pm_dec is dec proper motoin in radians/year

        @param [in] parallax in arcseconds

        @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)

        @param [in] Epoch0 is epoch in Julian years

        @param [in] MJD is modified Julian date in Julian years

        @param [out] raOut is corrected ra

        @param [out] decOut is corrected dec

        """

        if len(ra) != len(dec):
            raise ValueError('in Astrometry.py:applyMeanApparentPlace len(ra) %d len(dec) %d '
                            % (len(ra),len(dec)))

        if pm_ra == None:
            pm_ra=numpy.zeros(len(ra))

        if pm_dec == None:
            pm_dec=numpy.zeros(len(ra))

        if v_rad == None:
            v_rad=numpy.zeros(len(ra))

        if parallax == None:
            parallax=numpy.zeros(len(ra))

        # Define star independent mean to apparent place parameters
        #pal.mappa calculates the star-independent parameters
        #needed to correct RA and Dec
        #e.g the Earth barycentric and heliocentric position and velocity,
        #the precession-nutation matrix, etc.
        prms=pal.mappa(Epoch0, MJD)

        #pal.mapqk does a quick mean to apparent place calculation using
        #the output of pal.mappa
        #
        #Assuming that everything has propermotion, radial velocity, and parallax
        #is accurate to 10^-5 radians

        raOut,decOut = pal.mapqkVector(ra,dec,pm_ra,pm_dec,parallax,v_rad,prms)

        return raOut,decOut

    def applyMeanObservedPlace(self, ra, dec, MJD = 2015., includeRefraction = True,  \
                               altAzHr=False, wavelength=5000.):
        """Calculate the Mean Observed Place

        Uses PAL aoppa routines

        accepts RA and Dec.

        Returns corrected RA and Dec

        This will call pal.aopqk which accounts for refraction and diurnal aberration

        @param [out] raOut is corrected ra

        @param [out] decOut is corrected dec

        @param [out] alt is altitude angle (only returned if altAzHr == True)

        @param [out] az is azimuth angle (only returned if altAzHr == True)

        """

        # Correct site longitude for polar motion slaPolmo
        #
        # As of 4 February 2014, I am not sure what to make of this comment.
        # It appears that aoppa corrects for polar motion.
        # This could mean that the call to slaPolmo is unnecessary...


        # TODO NEED UT1 - UTC to be kept as a function of date.
        # Requires a look up of the IERS tables (-0.9<dut1<0.9)
        # Assume dut = 0.3 (seconds)
        dut = 0.3

        #
        #pal.aoppa computes star-independent parameters necessary for
        #converting apparent place into observed place
        #i.e. it calculates geodetic latitude, magnitude of diurnal aberration,
        #refraction coefficients and the like based on data about the observation site
        #
        if (includeRefraction == True):
            obsPrms=pal.aoppa(MJD, dut,
                            self.site.longitude,
                            self.site.latitude,
                            self.site.height,
                            self.site.xPolar,
                            self.site.yPolar,
                            self.site.meanTemperature,
                            self.site.meanPressure,
                            self.site.meanHumidity,
                            wavelength ,
                            self.site.lapseRate)
        else:
            #we can discard refraction by setting pressure and humidity to zero
            obsPrms=pal.aoppa(MJD, dut,
                            self.site.longitude,
                            self.site.latitude,
                            self.site.height,
                            self.site.xPolar,
                            self.site.yPolar,
                            self.site.meanTemperature,
                            0.0,
                            0.0,
                            wavelength ,
                            self.site.lapseRate)

        #pal.aopqk does an apparent to observed place
        #correction
        #
        #it corrects for diurnal aberration and refraction
        #(using a fast algorithm for refraction in the case of
        #a small zenith distance and a more rigorous algorithm
        #for a large zenith distance)
        #

        azimuth, zenith, hourAngle, decOut, raOut = pal.aopqkVector(ra,dec,obsPrms)
        #
        #Note: this is a choke point.  Even the vectorized version of aopqk
        #is expensive (it takes about 0.006 seconds per call)
        #
        #Actually, this is only a choke point if you are dealing with zenith
        #distances of greater than about 70 degrees

        if altAzHr == True:
            #
            #pal.de2h converts equatorial to horizon coordinates
            #
            az, alt = pal.de2hVector(hourAngle,decOut,self.site.latitude)
            return raOut, decOut, alt, az
        return raOut, decOut

    def correctCoordinates(self, pm_ra = None, pm_dec = None, parallax = None, v_rad = None,
             includeRefraction = True):
        """
        correct coordinates for all possible effects.

        included are precession-nutation, aberration, proper motion, parallax, refraction,
        radial velocity, diurnal aberration,

        @param [in] pm_ra is proper motion in RA (radians)

        @param [in] pm_dec is proper motion in dec (radians)

        @param [in] parallax is parallax (arcseconds)

        @param [in] v_rad is radial velocity (km/s)

        @param [in] includeRefraction toggles whether or not to correct for refraction

        @param [out] ra_out RA corrected for all included effects

        @param [out] dec_out Dec corrected for all included effects

        """

        if self.obs_metadata.mjd is None:
            raise ValueError("in Astrometry.py cannot call correctCoordinates; self.obs_metadata.mjd is none")

        if self.db_obj.epoch is None:
            raise ValueError("in Astrometry.py cannot call correctCoordinates; you have no db_obj")

        ra=self.column_by_name('raJ2000') #in radians
        dec=self.column_by_name('decJ2000') #in radians

        ep0 = self.db_obj.epoch
        mjd = self.obs_metadata.mjd

        ra_apparent, dec_apparent = self.applyMeanApparentPlace(ra, dec, pm_ra = pm_ra,
                 pm_dec = pm_dec, parallax = parallax, v_rad = v_rad, Epoch0 = ep0, MJD = mjd)

        ra_out, dec_out = self.applyMeanObservedPlace(ra_apparent, dec_apparent, MJD = mjd,
                                                   includeRefraction = includeRefraction)

        return numpy.array([ra_out,dec_out])

    def refractionCoefficients(self):
        """ Calculate the refraction using PAL's refco routine

        This calculates the refraction at 2 angles and derives a tanz and tan^3z
        coefficient for subsequent quick calculations. Good for zenith distances < 76 degrees

        One should call PAL refz to apply the coefficients calculated here

        """

        wavelength = 5000.
        precision = 1.e-10
        _refcoOutput=pal.refco(self.site.height,
                        self.site.meanTemperature,
                        self.site.meanPressure,
                        self.site.meanHumidity,
                        wavelength ,
                        self.site.latitude,
                        self.site.lapseRate,
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
        """
        D = pal.gmsta(mjd, 0.)
        D += long
        D = D%(2.*math.pi)
        return D

    def equatorialToHorizontal(self, ra, dec, mjd):
        """
        Converts from equatorial to horizon coordinates

        @param [in] ra is hour angle in radians

        @param [in] dec is declination in radians

        @param [out] returns elevation angle and azimuth in that order

        """

        hourAngle = self.calcLast(mjd, self.site.longitude) - ra

        _de2hOutput=pal.de2h(hourAngle, dec,  self.site.latitude)

        #return (altitude, azimuth)
        return _de2hOutput[1], _de2hOutput[0]

    def paralacticAngle(self, az, dec):
        """
        This returns the paralactic angle between the zenith and the pole that is up.
        I need to check this, but this should be +ve in the East and -ve in the West if
        Az is measured from North through East.
        """

        sinpa = math.sin(az)*math.cos(self.site.latitude)/math.cos(dec)
        return math.asin(sinpa)

    def calculatePupilCoordinates(self, ra_obj, dec_obj):
        """
        @param [in] ra_obj is a numpy array of RAs in radians

        @param [in] dec_obj is a numpy array of Decs in radians

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

        if self.obs_metadata.mjd is None:
            raise ValueError("in Astrometry.py cannot call get_skyToPupil; obs_metadata.mjd is None")

        if self.db_obj.epoch is None:
            raise ValueError("in Astrometry.py cannot call get_skyToPupil; db_obj epoch is None")

        if self.rotSkyPos is None:
            #there is no observation data on which to base astrometry
            raise ValueError("Cannot calculate x_pupil, y_pupil without rotSkyPos in obs_metadata")

        if self.unrefractedRA is None or self.unrefractedDec is None:
            raise ValueError("Cannot calculate x_pupil, y_pupil without unrefracted RA, Dec in obs_metadata")

        theta = -numpy.radians(self.rotSkyPos)

        #correct for precession and nutation

        inRA=numpy.array([numpy.radians(self.unrefractedRA)])
        inDec=numpy.array([numpy.radians(self.unrefractedDec)])

        x, y = self.applyMeanApparentPlace(inRA, inDec,
                   Epoch0 = self.db_obj.epoch, MJD = self.obs_metadata.mjd)

        #correct for refraction
        boreRA, boreDec = self.applyMeanObservedPlace(x, y, MJD = self.obs_metadata.mjd)
        #we should now have the true tangent point for the gnomonic projection
        dPhi = dec_obj - boreDec
        dLambda = ra_obj - boreRA

        #see en.wikipedia.org/wiki/Haversine_formula
        #Phi is latitude on the sphere (declination)
        #Lambda is longitude on the sphere (RA)
        h = 2.0*numpy.arcsin(numpy.sqrt(numpy.sin(0.5 * dPhi)**2 + numpy.cos(boreDec) *
               numpy.cos(dec_obj) * (numpy.sin(0.5 * dLambda))**2))

        #demand that the Euclidean distance on the pupil matches
        #the haversine distance on the sphere
        dx = numpy.sign(dLambda)*numpy.sqrt(h**2 - dPhi**2)
        #correct for rotation of the telescope
        x_out = dx*numpy.cos(theta) - dPhi*numpy.sin(theta)
        y_out = dx*numpy.sin(theta) + dPhi*numpy.cos(theta)

        return numpy.array([x_out, y_out])

    def calculateGnomonicProjection(self, ra_in, dec_in):
        """
        Take an input RA and dec from the sky and convert it to coordinates
        on the focal plane.

        This uses PAL's gnonomonic projection routine which assumes that the focal
        plane is perfectly flat.  The output is in Cartesian coordinates, assuming
        that the Celestial Sphere is a unit sphere.

        @param [in] ra_in is a numpy array of RAs in radians

        @param [in] dec_in in radians

        @param [out] returns a numpy array whose first row is the x coordinate according to a naive
        gnomonic projection and whose second row is the y coordinate
        """

        if self.obs_metadata.mjd is None:
            raise ValueError("in Astrometry.py cannot call get_gnomonicProjection; obs_metadata.mjd is None")

        if self.db_obj.epoch is None:
            raise ValueError("in Astrometry.py cannot call get_gnomonicProjection; db_obj.epoch is None")

        x_out=numpy.zeros(len(ra_in))
        y_out=numpy.zeros(len(ra_in))

        if self.rotSkyPos is None:
            #there is no observation meta data on which to base astrometry
            raise ValueError("Cannot calculate [x,y]_focal_nominal without rotSkyPos obs_metadata")

        if self.unrefractedRA is None or self.unrefractedDec is None:
            raise ValueError("Cannot calculate [x,y]_focal_nominal without unrefracted RA and Dec in obs_metadata")

        theta = -numpy.radians(self.rotSkyPos)

        #correct RA and Dec for refraction, precession and nutation
        #
        #correct for precession and nutation
        inRA=numpy.array([numpy.radians(self.unrefractedRA)])
        inDec=numpy.array([numpy.radians(self.unrefractedDec)])

        x, y = self.applyMeanApparentPlace(inRA, inDec,
                   Epoch0 = self.db_obj.epoch, MJD = self.obs_metadata.mjd)

        #correct for refraction
        trueRA, trueDec = self.applyMeanObservedPlace(x, y, MJD = self.obs_metadata.mjd)
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

        ra_obj = self.column_by_name('raObserved')
        dec_obj = self.column_by_name('decObserved')

        return self.calculatePupilCoordinates(ra_obj, dec_obj)

class CameraCoords(AstrometryBase):
    """Methods for getting coordinates from the camera object"""
    camera = None

    def findChipName(self, xPupil=None, yPupil=None, ra=None, dec=None):
        """
        @param [in] xPupil a numpy array of x pupil coordinates

        @param [in] yPupil a numpy array of y pupil coordinates

        @param [out] a numpy array of chip names
        """
        specifiedPupil = (xPupil is not None and yPupil is not None)
        specifiedRaDec = (ra is not None and dec is not None)

        if not specifiedPupil and not specifiedRaDec:
            raise RuntimeError("You must specifyeither pupil coordinates or equatorial coordinates in findChipName")

        if specifiedPupil and specifiedRaDec:
            raise RuntimeError("You cannot specify both pupil coordinates and equatorial coordinates in findChipName")

        if specifiedRaDec:
            xPupil, yPupil = self.calculatePupilCoordinates(ra, dec)

        if not self.camera:
            raise RuntimeError("No camera defined.  Cannot retrieve detector name.")
        chipNames = []
        for x, y in zip(xPupil, yPupil):
            cp = self.camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            detList = self.camera.findDetectors(cp)
            if len(detList) > 1:
                raise RuntimeError("This method does not know how to deal with cameras where points can be"+
                                   " on multiple detectors.  Override CameraCoords.get_chipName to add this.")
            if not detList:
                chipNames.append(None)
            else:
                chipNames.append(detList[0].getName())

        return numpy.asarray(chipNames)

    def calculatePixelCoordinates(self, xPupil=None, yPupil=None, ra=None, dec=None, chipNames = None):
        """
        Get the pixel positions (or nan if not on a chip) for all objects in the catalog

        @param [in] xPupil a numpy array containing x pupil coordinates

        @param [in] yPupil a numpy array containing y pupil coordinates

        @param [in] ra one could alternatively provide a numpy array of ra and...

        @param [in] ...dec (both in radians)

        @param [in] chipNames a numpy array of chipNames.  If it is None, this method will call findChipName
        to find the array.  The option exists for the user to specify chipNames, just in case the user
        has already called findChipName for some reason.
        """

        if not self.camera:
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
            xPupil, yPupil = self.calculatePupilCoordinates(ra, dec)

        if chipNames is None:
            chipNames = self.findChipName(xPupil = xPupil, yPupil = yPupil)

        xPix = []
        yPix = []
        for name, x, y in zip(chipNames, xPupil, yPupil):
            if not name:
                xPix.append(numpy.nan)
                yPix.append(numpy.nan)
                continue
            cp = self.camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            det = self.camera[name]
            cs = det.makeCameraSys(PIXELS)
            detPoint = self.camera.transform(cp, cs)
            xPix.append(detPoint.getPoint().getX())
            yPix.append(detPoint.getPoint().getY())
        return numpy.array([xPix, yPix])

    def calculateFocalPlaneCoordinates(self, xPupil=None, yPupil=None, ra=None, dec=None):
        """
        Get the focal plane coordinates for all objects in the catalog.

        @param [in] xPupil a numpy array of x pupil coordinates

        @param [in] yPupil a numpy array of y pupil coordinates

        @param [in] alternatively, one can specify numpy arrays of ra and dec (in radians)

        @param [out] a numpy array in which the first row is the x pixel coordinates
        and the second row is the y pixel coordinates
        """

        specifiedPupil = (xPupil is not None and yPupil is not None)
        specifiedRaDec = (ra is not None and dec is not None)

        if not specifiedPupil and not specifiedRaDec:
            raise RuntimeError("You must specify either pupil coordinates or equatorial coordinates to call calculateFocalPlaneCoordinates")

        if specifiedPupil and specifiedRaDec:
            raise RuntimeError("You cannot specify both pupil and equaltorial coordinates when calling calculateFocalPlaneCoordinates")

        if not self.camera:
            raise RuntimeError("No camera defined.  Cannot calculate focalplane coordinates")

        if specifiedRaDec:
            xPupil, yPupil = self.calculatePupilCoordinates(ra,dec)

        xPix = []
        yPix = []
        for x, y in zip(xPupil, yPupil):
            cp = self.camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            fpPoint = self.camera.transform(cp, FOCAL_PLANE)
            xPix.append(fpPoint.getPoint().getX())
            yPix.append(fpPoint.getPoint().getY())
        return numpy.array([xPix, yPix])

    def get_chipName(self):
        """Get the chip name if there is one for each catalog entry"""
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))
        return self.findChipName(xPupil = xPupil, yPupil = yPupil)

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
        return self.correctCoordinates(includeRefraction = False)


    @compound('raObserved','decObserved')
    def get_observedCoordinates(self):
        """
        get coordinates corrected for everything
        """

        return self.correctCoordinates()


class AstrometryStars(AstrometryBase):
    """
    This mixin contains a getter for the corrected RA and dec which takes account of proper motion and parallax
    """

    def correctStellarCoordinates(self, includeRefraction = True):
        """
        Getter which coorrects RA and Dec for propermotion, radial velocity, and parallax

        """
        pr=self.column_by_name('properMotionRa') #in radians per year
        pd=self.column_by_name('properMotionDec') #in radians per year
        px=self.column_by_name('parallax') #in arcseconds
        rv=self.column_by_name('radialVelocity') #in km/s; positive if receding

        return self.correctCoordinates(pm_ra = pr, pm_dec = pd, parallax = px, v_rad = rv,
                     includeRefraction = includeRefraction)


    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        return self.correctStellarCoordinates(includeRefraction = False)

    @compound('raObserved','decObserved')
    def get_observedCoordinates(self):
        return self.correctStellarCoordinates()

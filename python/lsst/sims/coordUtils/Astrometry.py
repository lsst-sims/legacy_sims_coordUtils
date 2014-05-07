import numpy
import ctypes
import math
import palpy as pal
from .decorators import compound

class AstrometryBase(object):
    """Collection of astrometry routines that operate on numpy arrays"""
    
    @compound('glon','glat')
    def get_galactic_coords(self):
        """
        Getter for galactic coordinates, in case the catalog class does not provide that
        
        Reads in the ra and dec from the data base and returns columns with galactic
        longitude and latitude.
        
        """
        ra=self.column_by_name('raJ2000')
        dec=self.column_by_name('decJ2000')
        
        glon=numpy.zeros(len(ra))
        glat=numpy.zeros(len(ra))
        for i in range(len(ra)):
            gg=pal.eqgal(ra[i],dec[i])
            glon[i]=gg[0]
            glat[i]=gg[1]
        
        return numpy.array([glon,glat])
          
    def sphericalToCartesian(self, longitude, latitude):
        """
        Transforms between spherical and Cartesian coordinates.
        
        @param [in] longitude is the input longitudinal coordinate
        
        @param [in] latitutde is the input latitudinal coordinate
        
        @param [out] a list of the (three-dimensional) cartesian coordinates on a unit sphere
        """
        
        cosDec = numpy.cos(latitude) 
        return numpy.array([numpy.cos(longitude)*cosDec, 
                          numpy.sin(longitude)*cosDec, 
                          numpy.sin(latitude)])

    def cartesianToSpherical(self, xyz):
        """
        Transforms between Cartesian and spherical coordinates
        
        @param [in] xyz is a list of the three-dimensional Cartesian coordinates
        
        @param [out] returns longitude and latitude
        
        """
    
        rad = numpy.sqrt(xyz[:][0]*xyz[:][0] + xyz[:][1]*xyz[:][1] + xyz[:][2]*xyz[:][2])

        longitude = numpy.arctan2( xyz[:][1], xyz[:][0])
        latitude = numpy.arcsin( xyz[:][2] / rad)

        return longitude, latitude

    def angularSeparation(self, long1, lat1, long2, lat2):
        ''' Given two spherical points in radians, calculate the angular
        separation between them.
        
        @param [in] long1 is the longitudinal coordinate of one point 
        (long2 is the longitude of the other point)
        
        @param [in] lat1 is the latitudinal coordinate of one point
        (lat2 is the latitude of the other point)
        
        @param [out] D the angular separation in radians
        
        '''
        D = pal.dsep (long1, lat1, long2, lat2)
        return D

    def rotationMatrixFromVectors(self, v1, v2):
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

        # Generate Julian epoch from MJD
        julianEpoch = pal.epj(MJD)

        # Determine the precession and nutation
        rmat=pal.prenut(EP0, julianEpoch)

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
        
        @param [in] pm_dec is dec proper motoin in radians/year
        
        @param [in] parallax in arcseconds
        
        @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)
        
        @param [in] EP0 is epoch in Julian years
        
        @param [in] MJD is modified Julian date in Julian years
        
        @param [out] raOut is corrected ra
        
        @param [out] decOut is corrected dec
        
        """
        
        for i in range(len(parallax)):
            if parallax[i] < 0.00045:
                parallax[i]=0.00045 #so that pal.Pm returns meaningful values
        
        EPSILON = 1.e-10

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Generate Julian epoch from MJD
        julianEpoch = pal.epj(self.obs_metadata.mjd)
        
        for i in range(len(ra)):
            if ((math.fabs(pm_ra[i]) > EPSILON) or (math.fabs(pm_dec[i]) > EPSILON)):
                _raAndDec=pal.pm(ra[i], dec[i], pm_ra[i], pm_dec[i]/numpy.cos(dec[i]), parallax[i],
                                  v_rad[i] ,EP0, julianEpoch)
                raOut[i] = _raAndDec[0]
                decOut[i] = _raAndDec[1]
            else:
                raOut[i] = ra[i]
                decOut[i] = dec[i]
            
        return raOut,decOut


    def equatorialToGalactic(self, ra, dec):
        '''Convert RA,Dec (J2000) to Galactic Coordinates'''
        gLong = numpy.zeros(len(ra))
        gLat = numpy.zeros(len(ra))
        
        for i in range(len(ra)):
            _eqgalOutput=pal.eqgal(ra[i], dec[i])
            gLong[i] = _eqgalOutput[0]
            gLat[i] = _eqgalOutput[1]

        return gLong, gLat

    def galacticToEquatorial(self, gLong, gLat):
        '''Convert Galactic Coordinates to RA, dec (J2000)'''
        ra = numpy.zeros(len(gLong))
        dec = numpy.zeros(len(gLong))
       
        for i in range(len(ra)):
            _galeqOutput=pal.galeq(gLong[i], gLat[i])
            ra[i] = _galeqOutput[0]
            dec[i] = _galeqOutput[1]

        return ra, dec

    def applyMeanApparentPlace(self, ra, dec, pm_ra, pm_dec, parallax, v_rad, Epoch0=2000.0, MJD=2015.0):
        """Calculate the Mean Apparent Place given an Ra and Dec

        Uses PAL mappa routines
        Recomputers precession and nutation
        
        units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec 
        (radians/year), parallax (arcsec), v_rad (km/sec; positive if receding),
        EP0 (Julian years)
        
        Returns corrected RA and Dec
       
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
        # Define star independent mean to apparent place parameters
        prms=pal.mappa(Epoch0, MJD)
        
        #Apply source independent parameters
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Loop over postions and apply corrections
        for i in range(len(ra)):
            _mapqkOutput=pal.mapqk(ra[i], dec[i], pm_ra[i], (pm_dec[i]/numpy.cos(dec[i])),
                            parallax[i],v_rad[i], prms)
            raOut[i] = _mapqkOutput[0]
            decOut[i] = _mapqkOutput[1]

        return raOut,decOut

    def applyMeanObservedPlace(self, ra, dec, MJD = 2015., includeRefraction = True,  \
                               altAzHr=False, wavelength=5000.):
        """Calculate the Mean Observed Place

        Uses PAL aoppa routines
        
        accepts RA and Dec.
        
        Returns corrected RA and Dec
        
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

        
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))
        if (altAzHr == True):
            alt = numpy.zeros(len(ra))
            az = numpy.zeros(len(ra))
 
        for i in range(len(ra)):
            _aopqkOutput=pal.aopqk(ra[i], dec[i], obsPrms) 
            azimuth=_aopqkOutput[0]
            zenith=_aopqkOutput[1]
            hourAngle=_aopqkOutput[2]           
            raOut[i] = _aopqkOutput[4]
            decOut[i] = _aopqkOutput[3]
            if (altAzHr == True):
                _de2hOutput=pal.de2h(hourAngle, decOut[i],  self.site.latitude)
                alt[i] = _de2hOutput[1]
                az[i] = _de2hOutput[0]                   

        if (altAzHr == False):
            return raOut,decOut
        else:
            return raOut,decOut, alt, az


    def applyApparentToTrim(self, ra, dec, MJD = 2015., altAzHr=False):
        """ Generate TRIM coordinates

        From the observed coordinates generate the position of the
        source as observed from the telescope site (required for the
        trim files). This includes the hour angle, diurnal aberration,
        alt-az. This does NOT include refraction.

        Uses PAL aoppa routines (we turn off refractiony by
        artificially setting the pressure and humidity to zero)
        
        
        @param [out] raOut is corrected ra
        
        @param [out] decOut is corrected dec
        
        @param [out] _elevation is the elevation angle (only returned if altAzHr == True)
        
        @param [out] _azimuth (only returned if altAzHr == True)
        
        """

        # Correct site longitude for polar motion slaPolmo
        # see comment in applyMeanObservedPlace
        
        # wavelength is not used in this version as there is no refraction
        wavelength = 5000.

        # TODO NEED UT1 - UTC to be kept as a function of date.
        # Requires a look up of the IERS tables (-0.9<dut1<0.9)
        # Assume dut = 0.3 (seconds)
        dut = 0.3
        
        #as above, we deactivate refraction by artificially setting
        #pressure and humidity to zero
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
                             
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        for i in range(len(ra)):
            _aopqkOutput=pal.aopqk(ra[i], dec[i], obsPrms)   
            azimuth=_aopqkOutput[0]
            zenith=_aopqkOutput[1]
            hourAngle=_aopqkOutput[2]
            decOut[i]=_aopqkOutput[3]
            raOut[i]=_aopqkOutput[4]         
            if (altAzHr == True):
                _de2hOutput=pal.de2h(hourAngle, dec[i],  self.site.latitude)
                _azimuth=_de2hOutput[0]
                _elevation=_de2hOutput[1]

        if (altAzHr == False):
            return raOut,decOut
        else:
            return raOut,decOut, _elevation, _azimuth, 

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
    
    @compound('x_focal','y_focal')    
    def get_skyToFocalPlane(self):
        """
        Take an input RA and dec from the sky and convert it to coordinates
        on the focal plane
        """
        
        ra_in = self.column_by_name('ra_corr')
        dec_in = self.column_by_name('dec_corr')
        

        x_out=numpy.zeros(len(ra_in))
        y_out=numpy.zeros(len(ra_in))
        
        theta = -1.0 * self.obs_metadata.metadata['Opsim_rotskypos']
        
        #correct RA and Dec for refraction, precession and nutation
        #
        #correct for precession and nutation
        apparentRA=[]
        apparentDec=[]
        inRA=[self.obs_metadata.metadata['Unrefracted_RA']]
        inDec=[self.obs_metadata.metadata['Unrefracted_Dec']]
        motion=[0.0] #set parallax, proper motion, and radial velocity all to zero
                     #we only want to deal with precession and nutation
                     #(this may be inappropriate; I'm not sure what Unrefracted_RA and Unrefracted_Dec
                     #actually store)
        
        x, y = self.applyMeanApparentPlace(inRA, inDec, motion, motion, motion, motion,
                                 Epoch0 = self.db_obj.epoch, MJD = self.obs_metadata.mjd)
        
        apparentRA.append(x)
        apparentDec.append(y)
        #correct for refraction
        trueRA, trueDec = self.applyMeanObservedPlace(apparentRA, apparentDec, MJD = self.obs_metadata.mjd)
        #we should now have the true tangent point for the gnomonic projection
        
        for i in range(len(ra_in)):
 
            x, y = pal.ds2tp(ra_in[i], dec_in[i],trueRA[0],trueDec[0])

            #rotate the result by -1 * rotskypos (rotskypos being "the angle of the sky relative to
            #camera cooridnates" according to phoSim documentation) to account for
            #the rotation of the focal plane about the telescope pointing      
            x_out[i] = x*numpy.cos(theta) - y*numpy.sin(theta)
            y_out[i] = x*numpy.sin(theta) + y*numpy.cos(theta)

        return numpy.array([x_out,y_out])

class AstrometryGalaxies(AstrometryBase):
    """
    This mixin contains a getter for the corrected RA and dec which ignores parallax and proper motion
    """

    @compound('ra_corr','dec_corr')
    def get_correctedCoordinates(self):
        """
        Getter which coorrects RA and Dec for propermotion, radial velocity, and parallax
   
        """
        
        ra=self.column_by_name('raJ2000') #in radians
        dec=self.column_by_name('decJ2000') #in radians
        
        ep0 = self.db_obj.epoch
        mjd = self.obs_metadata.mjd
        
        ra_out=numpy.zeros(len(ra))
        dec_out=numpy.zeros(len(ra))
        
       
        for i in range(len(ra)):
            
            #first, apply proper motion and parallax
            if i == 0:
                aprms = pal.mappa(ep0,mjd)
            
            #use the transformation appripriate for zero proper motion and parallax
            yy=pal.mapqkz(ra[i],dec[i],aprms)
            
            #apply precession and nutation after applying parallax and proper motion
            output = self.applyPrecession(yy[0],yy[1],EP0 = self.db_obj.epoch, MJD = self.obs_metadata.mjd)
            ra_out[i] = output[0]
            dec_out[i] = output[1]
            
            
        return numpy.array([ra_out,dec_out])            
     


class AstrometryStars(AstrometryBase): 
    """
    This mixin contains a getter for the corrected RA and dec which takes account of proper motion and parallax
    """

    @compound('ra_corr','dec_corr')
    def get_correctedCoordinates(self):
        """
        Getter which coorrects RA and Dec for propermotion, radial velocity, and parallax
   
        """
        
        ra=self.column_by_name('raJ2000') #in radians
        dec=self.column_by_name('decJ2000') #in radians
        
        pr=self.column_by_name('properMotionRa') #in radians per year
        pd=self.column_by_name('properMotionDec') #in radians per year
        px=self.column_by_name('parallax') #in arcseconds
        
        rv=self.column_by_name('radialVelocity') #in km/s; positive if receding
  
        ep0 = self.db_obj.epoch
        mjd = self.obs_metadata.mjd
        
        ra_out=numpy.zeros(len(ra))
        dec_out=numpy.zeros(len(ra))
        
       
        for i in range(len(ra)):
            
            #first, apply proper motion and parallax
            if i == 0:
                aprms = pal.mappa(ep0,mjd)
            if px[i] != 0.0 or pr[i] != 0.0 or pd[i] != 0.0:
                yy=pal.mapqk(ra[i],dec[i],pr[i],pd[i],px[i],rv[i],aprms) 
            else:
                yy=pal.mapqkz(ra[i],dec[i],aprms)
            
            #apply precession and nutation after applying parallax and proper motion
            output = self.applyPrecession(yy[0],yy[1],EP0 = self.db_obj.epoch, MJD = self.obs_metadata.mjd)
            ra_out[i] = output[0]
            dec_out[i] = output[1]
            
            
        return numpy.array([ra_out,dec_out])            
     

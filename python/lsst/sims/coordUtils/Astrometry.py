import numpy
import ctypes
import math
import palpy as pal
from lsst.sims.catalogs.measures.instance import compound

class Astrometry(object):
    """Collection of astrometry routines that operate on numpy arrays"""
          
    @compound('ra_corr','dec_corr')
    def get_correctedCoordinates(self):
        
        ra=self.column_by_name('raJ2000') #in radians
        dec=self.column_by_name('decJ2000') #in radians
        
        pr=self.column_by_name('proper_motion_ra') #in radians per year
        pd=self.column_by_name('proper_motion_dec') #in radians per year
        px=self.column_by_name('parallax') #in arcseconds
        
        rv=self.column_by_name('radial_velocity') #in km/s; positive if receding
        
        ep0=self.column_by_name('epoch') #epoch of mean coordinates
        mjd=self.column_by_name('mjd') #modified julian date
        
        ra_out=numpy.zeros(len(ra))
        dec_out=numpy.zeros(len(ra))
        
        for i in range(len(mjd)):
            if i == 0 or (mjd[i]-mjd[i-1])*(mjd[i]-mjd[i-1])+(ep0[i]-ep0[i-1])*(ep0[i]-ep0[i-1]) > 0.000001:
                #only compute aprms again if the date or epoch have changed since the last time 
                #it was computed
                aprms=pal.mappa(ep0[i],mjd[i]) #returns a list with parameters needed to correct coordinates
            
            if px[i] != 0.0 or pr[i] != 0.0 or pd[i] != 0.0:
                output=pal.mapqk(ra[i],dec[i],pr[i],pd[i],px[i],rv[i],aprms)
                ra_out[i]=output[0]
                dec_out[i]=output[1]
            else:
                output=pal.mapqkz(ra[i],dec[i],aprms)
                ra_out[i]=output[0]
                dec_out[i]=output[1]
        
        return numpy.array([ra_out,dec_out])            
        
        
    def sphericalToCartesian(self, longitude, latitude):
        cosDec = numpy.cos(latitude) 
        return numpy.array([numpy.cos(longitude)*cosDec, 
                          numpy.sin(longitude)*cosDec, 
                          numpy.sin(latitude)])

    def cartesianToSpherical(self, xyz):
        rad = numpy.sqrt(xyz[:][0]*xyz[:][0] + xyz[:][1]*xyz[:][1] + xyz[:][2]*xyz[:][2])

        longitude = numpy.arctan2( xyz[:][1], xyz[:][0])
        latitude = numpy.arcsin( xyz[:][2] / rad)

        return longitude, latitude

    def angularSeparation(self, long1, lat1, long2, lat2):
        ''' Given two spherical points in radians, calculate the angular
        separation between them.
        '''
        D = pal.dsep (long1, lat1, long2, lat2)
        return D

    def rotationMatrixFromVectors(self, v1, v2):
        ''' Given two vectors v1,v2 calculate the rotation matrix for v1->v2 using the axis-angle approach'''

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
        """ applyPrecession() applies precesion and nutation to coordinates between two epochs.
        
        Assumes FK5 as the coordinate system
        units:  ra_in (radians), dec_in (radians)
        
        The precession-nutation matrix is calculated by the pal.prenut method
        which uses the IAU 2006A/2000 model
        """
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Generate Julian epoch from MJD
        julianEpoch = pal.epj(MJD)

        # Determine the precession and nutation
        rmat=pal.prenut(EP0, julianEpoch)

        # precession only
        #slalib.slaPrec.argtypes = [ctypes.c_double,ctypes.c_double, rmat_ptr]
        #slalib.slaPrec(EP0, julianEpoch, rmat)
        
        # Apply rotation matrix
        xyz = self.sphericalToCartesian(ra,dec)
        xyz =  numpy.dot(rmat,xyz)

        raOut,decOut = self.cartesianToSpherical(xyz)
        return raOut,decOut

    def applyProperMotion(self, ra, dec, pm_ra, pm_dec, parallax, v_rad, \
                          EP0=2000.0, MJD=2015.0):
        """Calculates proper motion between two epochs
        
        Note pm_ra is measured in sky velocity (cos(dec)*dRa/dt). 
        PAL assumes dRa/dt
        
        units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec 
        (radians/year), parallax (arcsec), v_rad (km/sec), EP0 (Julian years)
        
        The function pal.pm does not work properly if the parallax is below
        0.00045 arcseconds
        """
        
        for i in range(len(parallax)):
            if parallax[i] < 0.00045:
                parallax[i]=0.00045 #so that pal.Pm returns meaningful values
        
        EPSILON = 1.e-10

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Generate Julian epoch from MJD
        julianEpoch = pal.epj(self.obs_metadata.metadata['Opsim_expmjd'][0])
        
        for i,raVal in enumerate(ra):
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
        
        for i,raVal in enumerate(ra):
            _eqgalOutput=pal.eqgal(ra[i], dec[i])
            gLong[i] = _eqgalOutput[0]
            gLat[i] = _eqgalOutput[1]

        return gLong, gLat

    def galacticToEquatorial(self, gLong, gLat):
        '''Convert Galactic Coordinates to RA, dec (J2000)'''
        ra = numpy.zeros(len(gLong))
        dec = numpy.zeros(len(gLong))
       
        for i,raVal in enumerate(ra):
            _galeqOutput=pal.galeq(gLong[i], gLat[i])
            ra[i] = _galeqOutput[0]
            dec[i] = _galeqOutput[1]

        return ra, dec

    def applyMeanApparentPlace(self, ra, dec, pm_ra, pm_dec, parallax, v_rad, Epoch0=2000.0, MJD=2015.0):
        """Calculate the Mean Apparent Place given an Ra and Dec

        Optimized to use PAL mappa routines
        Recomputers precession and nutation
        """
        # Define star independent mean to apparent place parameters
        prms=pal.mappa(Epoch0, MJD)

        #Apply source independent parameters
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Loop over postions and apply corrections
        for i,raVal in enumerate(ra):
            _mapqkOutput=pal.mapqk(ra[i], dec[i], pm_ra[i], (pm_dec[i]/numpy.cos(dec[i])),
                            parallax[i],v_rad[i], prms)
            raOut[i] = _mapqkOutput[0]
            decOut[i] = _mapqkOutput[1]

        return raOut,decOut

    def applyMeanObservedPlace(self, ra, dec, MJD = 2015., includeRefraction = True,  \
                               altAzHr=False, wavelength=5000.):
        """Calculate the Mean Observed Place

        Optimized to use PAL aoppa routines
        """

        # Correct site longitude for polar motion slaPolmo
        # As of 4 February 2014, I am not sure what to make of this comment.
        # It appears taht aoppa corrects for polar motion.
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


        # slaaopqk to apply to sources 
        
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))
        if (altAzHr == True):
            alt = numpy.zeros(len(ra))
            az = numpy.zeros(len(ra))
 
        for i,raVal in enumerate(ra):
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

        Optimized to use PAL aoppa routines (we turn off refractiony by
        artificially setting the pressure and humidity to zero)
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

        for i,raVal in enumerate(ra):
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
        """
        
        refractedZenith=pal.refz(zenithDistance, tanzCoeff, tan3zCoeff)
        
        return refractedZenith

    def calcLast(self, mjd, long):
        D = pal.gmsta(mjd, 0.)
        D += long
        D = D%(2.*math.pi)
        return D
        
    def equatorialToHorizontal(self, ra, dec, mjd):
        hourAngle = self.calcLast(mjd, self.site.longitude) - ra

        _de2hOutput=pal.de2h(hourAngle, dec,  self.site.latitude)
        
        #return (altitude, azimuth)
        return _de2hOutput[1], _de2hOutput[0]

    def paralacticAngle(self, az, dec):
        #This returns the paralactic angle between the zenith and the pole that is up.  
        #I need to check this, but this should be +ve in the East and -ve in the West if 
        #Az is measured from North through East.
        sinpa = math.sin(az)*math.cos(self.site.latitude)/math.cos(dec)
        return math.asin(sinpa)

    

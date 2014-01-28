import numpy
import ctypes
import math
import palpy as pal

from lsst.sims.catalogs.measures.astrometry import Site

#slalib = numpy.ctypeslib.load_library("slalsst.so")
#slalib = ctypes.CDLL("slalsst.so")

class Astrometry(Site):
    """Collection of astrometry routines that operate on numpy arrays"""
    
    def sphericalToCartesian(self, longitude, latitude):
        cosDec = numpy.cos(latitude) 
        return numpy.array([numpy.cos(longitude)*cosDec, 
                          numpy.sin(longitude)*cosDec, 
                          numpy.sin(latitude)])

    def cartesianToSpherical(self, xyz):
        rad = numpy.sqrt(xyz[:][0]*xyz[:][0] + xyz[:][1]*xyz[:][1] + xyz[:][2]*xyz[:][2])

        longitude = numpy.arctan2( xyz[:][1], xyz[:][0])
        latitude = numpy.arcsin( xyz[:][2] / rad)

        # if rad == 0
        #latitude = numpy.zeros(len( xyz[0][:])

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
        rot = [[cosDot + cross[0]*cross[0]*(1-cosDot), -cross[2]*sinDot+(1-cosDot)*cross[0]*cross[1], cross[1]*sinDot + (1-cosDot)*cross[0]*cross[2]],[cross[2]*sinDot+(1-cosDot)*cross[0]*cross[1], cosDot + (1-cosDot)*cross[1]*cross[1], -cross[0]*sinDot+(1-cosDot)*cross[1]*cross[2]], [-cross[1]*sinDot+(1-cosDot)*cross[0]*cross[2], cross[0]*sinDot+(1-cosDot)*cross[1]*cross[2], cosDot + (1-cosDot)*(cross[2]*cross[2])]]

        return rot
    
    
    def applyPrecession(self, ra, dec, EP0=2000.0, MJD=2000.0):
        """ applyPrecession() applies precesion and nutation to coordinates between two epochs.
        
        Assumes FK5 as the coordinate system
        units:  ra_in (radians), dec_in (radians)

        This uses the Fricke IAU 1976 model for J2000 precession
        This uses the IAU 1980 nutation model
        """
        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))
        
        #        self.slalib.slaPreces.argtypes = [ctypes.c_char_p, ctypes.c_double,
        #                                     ctypes.c_double, ctypes.POINTER(ctypes.c_double),
        #                                     ctypes.POINTER(ctypes.c_double)]
        #        for i,raVal in enumerate(ra):
        #            raIn = ctypes.c_double(ra[i])
        #            decIn = ctypes.c_double(dec[i])
        #            self.slalib.slaPreces("FK5", EP0, EP1, ctypes.pointer(raIn),ctypes.pointer(decIn))
        #            raOut[i] = raIn.value
        #            decOut[i] = decIn.value

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

    def applyProperMotion(self, ra, dec, pm_ra, pm_dec, parallax, v_rad, EP0=2000.0, MJD=2015.0):
        """Calculates proper motion between two epochs
        
        Note pm_ra is measured in sky velocity (cos(dec)*dRa/dt). PAL assumes dRa/dt
        
        units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec 
        (radians/year), parallax (arcsec), v_rad (km/sec), EP0 (Julian years)
        
        The function pal.pm does not work properly if the parallax is below
        0.00045 arcseconds
        """

        EPSILON = 1.e-10

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Generate Julian epoch from MJD
        julianEpoch = pal.epj(self.metadata.parameters['Opsim_expmjd'])

        #Define proper motion interface
        
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

    def applyMeanObservedPlace(self, ra, dec, MJD = 2015., includeRefraction = True,  altAzHr=False, wavelength=5000.):
        """Calculate the Mean Observed Place

        Optimized to use PAL aoppa routines
        """

        # Correct site longitude for polar motion slaPolmo

        # TODO NEED UT1 - UTC to be kept as a function of date.
        # Requires a look up of the IERS tables (-0.9<dut1<0.9)
        # Assume dut = 0.3 (seconds)
        dut = 0.3


        if (includeRefraction == True):
            obsPrms=pal.aoppa(MJD, dut,
                            self.parameters["longitude"],
                            self.parameters["latitude"],
                            self.parameters["height"],
                            self.parameters["xPolar"],
                            self.parameters["yPolar"],
                            self.parameters["meanTemperature"],
                            self.parameters["meanPressure"],
                            self.parameters["meanHumidity"],
                            wavelength ,
                            self.parameters["lapseRate"])
        else:
            #we can discard refraction by setting pressure and humidity to zero
            obsPrms=pal.aoppa_nr(MJD, dut,
                            self.parameters["longitude"],
                            self.parameters["latitude"],
                            self.parameters["height"],
                            self.parameters["xPolar"],
                            self.parameters["yPolar"],
                            self.parameters["meanTemperature"],
                            0.0,
                            0.0,
                            wavelength ,
                            self.parameters["lapseRate"])


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
                _de2hOutput=pal.de2h(hourAngle, decOut[i],  self.parameters["latitude"])
                alt[i] = _de2hOutput[1]
                az[i] = _de2hOutput[0]                   


        #testing values
        #_azimuth = ctypes.c_double(0.)
        #_elevation = ctypes.c_double(0.)
        #print 360. + azimuth.value/0.017453293, zenith.value/0.017453293

        
        #print 360. + azimuth.value/0.017453293, hourAngle.value/0.017453293, decOut[i]/0.017453293, self.parameters["latitude"]/0.017453293,_azimuth.value/0.017453293, _elevation.value/0.017453293

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
        
        # wavelength is not used in this version as there is no refraction
        wavelength = 5000.

        # TODO NEED UT1 - UTC to be kept as a function of date.
        # Requires a look up of the IERS tables (-0.9<dut1<0.9)
        # Assume dut = 0.3 (seconds)
        dut = 0.3
        
        #as above, we deactivate refraction by artificially setting
        #pressure and humidity to zero
        obsPrms=pal.aoppa(MJD, dut,
                        self.parameters["longitude"],
                        self.parameters["latitude"],
                        self.parameters["height"],
                        self.parameters["xPolar"],
                        self.parameters["yPolar"],
                        self.parameters["meanTemperature"],
                        0.0,
                        0.0,
                        wavelength ,
                        self.parameters["lapseRate"])
                             
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
                _de2hOutput=pal.de2h(hourAngle, dec[i],  self.parameters["latitude"])
                _azimuth=_de2hOutput[0]
                _elevation=_de2hOutput[1]
#                print azimuth, hourAngle, dec[i], self.parameters["latitude"],_azimuth, _elevation


#        print azimuth.value/0.017453293, zenith.value/0.017453293
#        print 360. + azimuth.value/0.017453293, hourAngle.value/0.017453293, decOut[i]/0.017453293, self.parameters["latitude"]/0.017453293,_azimuth.value/0.017453293, _elevation.value/0.017453293
        if (altAzHr == False):
            return raOut,decOut
        else:
            return raOut,decOut, _elevation, _azimuth, 



    def refractionCoefficients():
        """ Calculate the refraction using PAL's refco routine

        This calculates the refraction at 2 angles and derives a tanz and tan^3z coefficient for subsequent quick
        calculations. Good for zenith distances < 76 degrees

        Call PAL refz to apply coefficients
        """

        wavelength = 5000.
        precison = 1.e-10
        _refcoOutput=pal.refco(self.parameters["height"],
                        self.parameters["meanTemperature"],
                        self.parameters["meanPressure"],
                        self.parameters["meanHumidity"],
                        wavelength ,
                        self.parameters["latitude"],
                        self.parameters["lapseRate"],
                        precision)
   
        return _refcoOutput[0], _refcoOutput[1]

    def applyRefraction(zenithDistance, tanzCoeff, tan3zCoeff):
        """ Calculted refracted Zenith Distance
        
        uses the quick PAL refco routine which approximates the refractin calculation
        """

        refractedZenith = 0.0
        refractedZenith=pal.refz(zenithDistance, tanzCoeff, tan3zCoeff)
        
        return refractedZenith

    def calcLast(self, mjd, long):
        D = pal.gmsta(mjd, 0.)
        D += long
        D = D%(2.*math.pi)
        return D
        
    def equatorialToHorizontal(self, ra, dec, mjd):
        hourAngle = self.calcLast(mjd, self.parameters["longitude"]) - ra

        _de2hOutput=pal.de2h(hourAngle, dec,  self.parameters["latitude"])
        
        #return (altitude, azimuth)
        return _de2hOutput[1], _de2hOutput[0]

    def paralacticAngle(self, az, dec):
        #This returns the paralactic angle between the zenith and the pole that is up.  
        #I need to check this, but this should be +ve in the East and -ve in the West if Az is measured from North through East.
        sinpa = math.sin(az)*math.cos(self.parameters["latitude"])/math.cos(dec)
        return math.asin(sinpa)

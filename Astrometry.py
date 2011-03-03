import numpy
import ctypes
import math

#slalib = numpy.ctypeslib.load_library("slalsst.so")
slalib = ctypes.CDLL("slalsst.so")

class Astrometry():
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
        slalib.slaEpj.argtypes = [ctypes.c_double]
        slalib.slaEpj.restype = ctypes.c_double
        julianEpoch = slalib.slaEpj(MJD)
#        print julianEpoch

        # Define rotation matrix and ctypes pointer
        rmat = numpy.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        rmat_ptr = numpy.ctypeslib.ndpointer(dtype=float, ndim=2, shape=(3,3))

        # Determine the precession and nutation
        slalib.slaPrenut.argtypes = [ctypes.c_double,ctypes.c_double, rmat_ptr]
        slalib.slaPrenut(EP0, julianEpoch, rmat)

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
        
        Note pm_ra is measured in sky velocity (cos(dec)*dRa/dt). Slalib assumes dRa/dt
        
        units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec 
        (radians/year), parallax (arcsec), v_rad (km/sec), EP0 (Julian years)
        """

        EPSILON = 1.e-10

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Generate Julian epoch from MJD
        slalib.slaEpj.argtypes = [ctypes.c_double]
        slalib.slaEpj.restype = ctypes.c_double
        julianEpoch = slalib.slaEpj(self.metadata.parameters['Opsim_expmjd'])

        #Define proper motion interface
        slalib.slaPm.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)] 

        _raOut = ctypes.c_double(0.)
        _decOut = ctypes.c_double(0.)
        for i,raVal in enumerate(ra):
            if ((math.fabs(pm_ra[i]) > EPSILON) or (math.fabs(pm_dec[i]) > EPSILON)):
                slalib.slaPm(ra[i], dec[i], pm_ra[i], pm_dec[i]/numpy.cos(dec[i]), parallax[i],
                                  v_rad[i] ,EP0, julianEpoch, _raOut, _decOut)
                raOut[i] = _raOut.value
                decOut[i] = _decOut.value
            else:
                raOut[i] = ra[i]
                decOut[i] = dec[i]
            
        return raOut,decOut

    def applyMeanApparentPlace(self, ra, dec, pm_ra, pm_dec, parallax, v_rad, Epoch0=2000.0, MJD=2015.0):
        """Calulate the Mean Apparent Place given an Ra and Dec

        Optimized to use slalib mappa routines
        Recomputers precession and nutation
        """
        # Define star independent mean to apparent place parameters
        prms = numpy.zeros(21)
        prms_ptr = numpy.ctypeslib.ndpointer(dtype=float, ndim=1, shape=(21))
        slalib.slaMappa.argtypes = [ctypes.c_double,ctypes.c_double, prms_ptr]
        slalib.slaMappa(Epoch0, MJD, prms)

        #Apply source independent parameters
        slalib.slaMapqk.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_double, ctypes.c_double, prms_ptr,
                                      ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)] 

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        # Loop over postions and apply corrections
        _raOut = ctypes.c_double(0.)
        _decOut = ctypes.c_double(0.)
        for i,raVal in enumerate(ra):
            slalib.slaMapqk(ra[i], dec[i], pm_ra[i], (pm_dec[i]/numpy.cos(dec[i])),
                            parallax[i],v_rad[i], prms, _raOut, _decOut)
            raOut[i] = _raOut.value
            decOut[i] = _decOut.value

        return raOut,decOut

    def applyMeanObservedPlace(self, ra, dec, MJD = 2015., includeRefraction = True,  altAzHr=False, wavelength=5000.):
        """Calculate the Mean Observed Place

        Optimized to use slalib aoppa routines
        """

        # Correct site longitude for polar motion slaPolmo
        obsPrms = numpy.zeros(14)
        obsPrms_ptr = numpy.ctypeslib.ndpointer(dtype=float, ndim=1, shape=(14))
        slalib.slaAoppa.argtypes= [ ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, obsPrms_ptr]
        slalib.slaAoppa_nr.argtypes= [ ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, obsPrms_ptr]

        # TODO NEED UT1 - UTC to be kept as a function of date.
        # Requires a look up of the IERS tables (-0.9<dut1<0.9)
        # Assume dut = 0.3 (seconds)
        dut = 0.3


        if (includeRefraction == True):
            slalib.slaAoppa(MJD, dut,
                            self.site.parameters["longitude"],
                            self.site.parameters["latitude"],
                            self.site.parameters["height"],
                            self.site.parameters["xPolar"],
                            self.site.parameters["yPolar"],
                            self.site.parameters["meanTemperature"],
                            self.site.parameters["meanPressure"],
                            self.site.parameters["meanHumidity"],
                            wavelength ,
                            self.site.parameters["lapseRate"],
                            obsPrms)
        else:
            slalib.slaAoppa_nr(MJD, dut,
                            self.site.parameters["longitude"],
                            self.site.parameters["latitude"],
                            self.site.parameters["height"],
                            self.site.parameters["xPolar"],
                            self.site.parameters["yPolar"],
                            self.site.parameters["meanTemperature"],
                            self.site.parameters["meanPressure"],
                            self.site.parameters["meanHumidity"],
                            wavelength ,
                            self.site.parameters["lapseRate"],
                            obsPrms)


        # slaaopqk to apply to sources self.slalib.slaAopqk.argtypes=
        slalib.slaAopqk.argtypes= [ctypes.c_double, ctypes.c_double, obsPrms_ptr, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]
        slalib.slaAopqk_nr.argtypes= [ctypes.c_double, ctypes.c_double, obsPrms_ptr, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]


        slalib.slaDe2h.argtypes= [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))
        if (altAzHr == True):
            alt = numpy.zeros(len(ra))
            az = numpy.zeros(len(ra))
            
        _raOut = ctypes.c_double(0.)
        _decOut = ctypes.c_double(0.)
        azimuth = ctypes.c_double(0.)
        zenith = ctypes.c_double(0.)
        hourAngle = ctypes.c_double(0.)
        _azimuth = ctypes.c_double(0.)
        _elevation = ctypes.c_double(0.)
        if (includeRefraction == True):
            for i,raVal in enumerate(ra):
                slalib.slaAopqk(ra[i], dec[i], obsPrms, azimuth, zenith, hourAngle, _decOut, _raOut)            
                raOut[i] = _raOut.value
                decOut[i] = _decOut.value
                if (altAzHr == True):
                    slalib.slaDe2h(hourAngle, decOut[i],  self.site.parameters["latitude"], _azimuth, _elevation)
                    alt[i] = _elevation.value
                    az[i] = _azimuth.value                    
        else:
            for i,raVal in enumerate(ra):
                slalib.slaAopqk_nr(ra[i], dec[i], obsPrms, azimuth, zenith, hourAngle, _decOut, _raOut)            
                raOut[i] = _raOut.value
                decOut[i] = _decOut.value
                if (altAzHr == True):
                    slalib.slaDe2h(hourAngle, decOut[i],  self.site.parameters["latitude"], _azimuth, _elevation)
                    alt[i] = _elevation.value
                    az[i] = _azimuth.value

        #testing values
        #_azimuth = ctypes.c_double(0.)
        #_elevation = ctypes.c_double(0.)
        #print 360. + azimuth.value/0.017453293, zenith.value/0.017453293

        
        #print 360. + azimuth.value/0.017453293, hourAngle.value/0.017453293, decOut[i]/0.017453293, self.site.parameters["latitude"]/0.017453293,_azimuth.value/0.017453293, _elevation.value/0.017453293

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

        Optimized to use slalib aoppa_nr routines (no refraction)
        """

        # Correct site longitude for polar motion slaPolmo
        obsPrms = numpy.zeros(14)
        obsPrms_ptr = numpy.ctypeslib.ndpointer(dtype=float, ndim=1, shape=(14))
        slalib.slaAoppa_nr.argtypes= [ ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                         ctypes.c_double, obsPrms_ptr]

        # wavelength is not used in this version as there is no refraction
        wavelength = 5000.

        # TODO NEED UT1 - UTC to be kept as a function of date.
        # Requires a look up of the IERS tables (-0.9<dut1<0.9)
        # Assume dut = 0.3 (seconds)
        dut = 0.3
        slalib.slaAoppa_nr(MJD, dut,
                        self.site.parameters["longitude"],
                        self.site.parameters["latitude"],
                        self.site.parameters["height"],
                        self.site.parameters["xPolar"],
                        self.site.parameters["yPolar"],
                        self.site.parameters["meanTemperature"],
                        self.site.parameters["meanPressure"],
                        self.site.parameters["meanHumidity"],
                        wavelength ,
                        self.site.parameters["lapseRate"],
                        obsPrms)
                             

        # slaaopqk to apply to sources self.slalib.slaAopqk.argtypes=
        slalib.slaAopqk_nr.argtypes= [ctypes.c_double, ctypes.c_double, obsPrms_ptr, ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double),
                                   ctypes.POINTER(ctypes.c_double)]

        slalib.slaDe2h.argtypes= [ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                  ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]

        raOut = numpy.zeros(len(ra))
        decOut = numpy.zeros(len(ra))

        _raOut = ctypes.c_double(0.)
        _decOut = ctypes.c_double(0.)
        _azimuth = ctypes.c_double(0.)
        _elevation = ctypes.c_double(0.)
        azimuth = ctypes.c_double(0.)
        zenith = ctypes.c_double(0.)
        hourAngle = ctypes.c_double(0.)
        for i,raVal in enumerate(ra):
            slalib.slaAopqk_nr(ra[i], dec[i], obsPrms, azimuth, zenith, hourAngle, _decOut, _raOut)            
            raOut[i] = _raOut.value
            decOut[i] = _decOut.value
            if (altAzHr == True):
                slalib.slaDe2h(hourAngle, dec[i],  self.site.parameters["latitude"], _azimuth, _elevation)
#                print azimuth, hourAngle, dec[i], self.site.parameters["latitude"],_azimuth, _elevation


#        print azimuth.value/0.017453293, zenith.value/0.017453293
#        print 360. + azimuth.value/0.017453293, hourAngle.value/0.017453293, decOut[i]/0.017453293, self.site.parameters["latitude"]/0.017453293,_azimuth.value/0.017453293, _elevation.value/0.017453293
        if (altAzHr == False):
            return raOut,decOut
        else:
            return raOut,decOut, _elevation.value, _azimuth.value, 



    def refractionCoefficients():
        """ Calculate the refraction using Slalib's refco routine

        This calculates the refraction at 2 angles and derives a tanz and tan^3z coefficient for subsequent quick
        calculations. Good for zenith distances < 76 degrees

        Call slalib refz to apply coefficients
        """

        slalib.slaRefco.argtypes= [ ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                    ctypes.c_double, ctypes.c_double, 
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        
        wavelength = 5000.
        precison = 1.e-10
        slalib.slaRefco(self.site.parameters["height"],
                        self.site.parameters["meanTemperature"],
                        self.site.parameters["meanPressure"],
                        self.site.parameters["meanHumidity"],
                        wavelength ,
                        self.site.parameters["longitude"],
                        self.site.parameters["latitude"],
                        self.site.parameters["lapseRate"],
                        precision,
                        tanzCoeff,
                        tan3zCoeff)

        return tanzCoeff, tan3zCoeff

    def applyRefraction(zenithDistance, tanzCoeff, tan3zCoeff):
        """ Calculted refracted Zenith Distance
        
        uses the quick slalib refco routine which approximates the refractin calculation
        """
        slalib.slaRefz.argtypes= [ ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                   ctypes.POINTER(ctypes.c_double)]
        refractedZenith = 0.0
        slalib.slaRefco(zenithDistance, tanzCoeff, tan3zCoeff, refractedZenith)
        
        return refractedZenith

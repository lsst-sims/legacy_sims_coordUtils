"""
Some of the data in this unit test will appear abitrary.  That is
because, in addition to testing the execution of all of the functionality
provided in the sims_coordUtils package, this unit test validates
the outputs of PALPY against the outputs of pySLALIB v 1.0.2
(it was written when we were making the transition from pySLALIB to PALPY).

There will be some difference, as the two libraries are based on slightly
different conventions (for example, the prenut routine which calculates
the matrix of precession and nutation is based on the IAU 2006/2000A
standard in PALPY and on SF2001 in pySLALIB; however, the two outputs
still agree to within one part in 10^5)

"""

import numpy

import os
import unittest
import warnings
import sys
import math
import palpy as pal
from collections import OrderedDict
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir

from lsst.sims.utils import ObservationMetaData
from lsst.sims.utils import _getRotTelPos, _raDecFromAltAz, calcObsDefaults, \
                            radiansFromArcsec, arcsecFromRadians, Site, \
                            raDecFromAltAz, haversine

from lsst.sims.coordUtils import _applyPrecession, _applyProperMotion
from lsst.sims.coordUtils import _appGeoFromICRS, _observedFromAppGeo
from lsst.sims.coordUtils import _observedFromICRS
from lsst.sims.coordUtils import _appGeoFromObserved
from lsst.sims.coordUtils import refractionCoefficients, applyRefraction

def makeObservationMetaData():
    #create a cartoon ObservationMetaData object
    mjd = 52000.0
    alt = numpy.pi/2.0
    az = 0.0
    band = 'r'
    testSite = Site(latitude=0.5, longitude=1.1, height=3000, meanTemperature=260.0,
                    meanPressure=725.0, lapseRate=0.005)
    centerRA, centerDec = _raDecFromAltAz(alt,az,testSite.longitude,testSite.latitude,mjd)
    rotTel = _getRotTelPos(centerRA, centerDec, testSite.longitude, testSite.latitude, mjd, 0.0)

    obsDict = calcObsDefaults(centerRA, centerDec, alt, az, rotTel, mjd, band,
                 testSite.longitude, testSite.latitude)

    obsDict['Opsim_expmjd'] = mjd
    radius = 0.1
    phoSimMetaData = OrderedDict([
                      (k, (obsDict[k],numpy.dtype(type(obsDict[k])))) for k in obsDict])

    obs_metadata = ObservationMetaData(boundType='circle', boundLength=2.0*radius,
                                       phoSimMetaData=phoSimMetaData, site=testSite)

    return obs_metadata

def makeRandomSample(raCenter=None, decCenter=None, radius=None):
    #create a random sample of object data

    nsamples=100
    numpy.random.seed(32)

    if raCenter is None or decCenter is None or radius is None:
        ra = numpy.random.sample(nsamples)*2.0*numpy.pi
        dec = (numpy.random.sample(nsamples)-0.5)*numpy.pi
    else:
        rr = numpy.random.sample(nsamples)*radius
        theta = numpy.random.sample(nsamples)*2.0*numpy.pi
        ra = raCenter + rr*numpy.cos(theta)
        dec = decCenter + rr*numpy.cos(theta)

    pm_ra = (numpy.random.sample(nsamples)-0.5)*0.1
    pm_dec = (numpy.random.sample(nsamples)-0.5)*0.1
    parallax = numpy.random.sample(nsamples)*0.01
    v_rad = numpy.random.sample(nsamples)*1000.0

    return ra, dec, pm_ra, pm_dec, parallax, v_rad


class astrometryUnitTest(unittest.TestCase):
    """
    The bulk of this unit test involves inputting a set list of input values
    and comparing the astrometric results to results derived from SLALIB run
    with the same input values.  We have to create a test catalog artificially (rather than
    querying the database) because SLALIB was originally run on values that did not correspond
    to any particular Opsim run.
    """

    def setUp(self):
        self.metadata={}

        #below are metadata values that need to be set in order for
        #get_getFocalPlaneCoordinates to work.  If we had been querying the database,
        #these would be set to meaningful values.  Because we are generating
        #an artificial set of inputs that must comport to the baseline SLALIB
        #inputs, these are set arbitrarily by hand
        self.metadata['Unrefracted_RA'] = (numpy.radians(200.0), float)
        self.metadata['Unrefracted_Dec'] = (numpy.radians(-30.0), float)
        self.metadata['Opsim_rotskypos'] = (1.0, float)

        self.obs_metadata=ObservationMetaData(mjd=50984.371741,
                                     boundType='circle',
                                     boundLength=0.05,
                                     phoSimMetaData=self.metadata)

        self.tol=1.0e-5

    def tearDown(self):
        del self.obs_metadata
        del self.metadata
        del self.tol


    def testAstrometryExceptions(self):
        """
        Test to make sure that stand-alone astrometry methods raise an exception when they are called without
        the necessary arguments
        """
        obs_metadata = makeObservationMetaData()
        ra, dec, pm_ra, pm_dec, parallax, v_rad = makeRandomSample()

        raShort = numpy.array([1.0])
        decShort = numpy.array([1.0])


        ##########test refractionCoefficients
        self.assertRaises(RuntimeError, refractionCoefficients)
        site = obs_metadata.site
        x, y = refractionCoefficients(site=site)

        ##########test applyRefraction
        zd = 0.1
        rzd = applyRefraction(zd, x, y)

        zd = [0.1, 0.2]
        self.assertRaises(RuntimeError, applyRefraction, zd, x, y)

        zd = numpy.array([0.1, 0.2])
        rzd = applyRefraction(zd, x, y)

        ##########test _applyPrecession
        #test without mjd
        self.assertRaises(RuntimeError, _applyPrecession, ra, dec)

        #test mismatches
        self.assertRaises(RuntimeError, _applyPrecession, raShort, dec, mjd=52000.0)
        self.assertRaises(RuntimeError, _applyPrecession, ra, decShort, mjd=52000.0)

        #test that it runs
        _applyPrecession(ra, dec, mjd=52000.0)

        ##########test _applyProperMotion
        raList = list(ra)
        decList = list(dec)
        pm_raList = list(pm_ra)
        pm_decList = list(pm_dec)
        parallaxList = list(parallax)
        v_radList = list(v_rad)

        pm_raShort = numpy.array([pm_ra[0]])
        pm_decShort = numpy.array([pm_dec[0]])
        parallaxShort = numpy.array([parallax[0]])
        v_radShort = numpy.array([v_rad[0]])

        #test without mjd
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallax, v_rad)

        #test passing lists
        self.assertRaises(RuntimeError, _applyProperMotion,
                          raList, dec, pm_ra, pm_dec, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, decList, pm_ra, pm_dec, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_raList, pm_dec, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_decList, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallaxList, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallax, v_radList,
                          mjd=52000.0)

        #test mismatches
        self.assertRaises(RuntimeError, _applyProperMotion,
                          raShort, dec, pm_ra, pm_dec, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, decShort, pm_ra, pm_dec, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_raShort, pm_dec, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_decShort, parallax, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallaxShort, v_rad,
                          mjd=52000.0)
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallax, v_radShort,
                          mjd=52000.0)

        #test that it actually runs
        _applyProperMotion(ra, dec, pm_ra, pm_dec, parallax, v_rad, mjd=52000.0)
        _applyProperMotion(ra[0], dec[0], pm_ra[0], pm_dec[0], parallax[0], v_rad[0],
                          mjd=52000.0)

        ##########test _appGeoFromICRS
        #test without mjd
        self.assertRaises(RuntimeError, _appGeoFromICRS, ra, dec)

        #test with mismatched ra, dec
        self.assertRaises(RuntimeError, _appGeoFromICRS, ra, decShort, mjd=52000.0)
        self.assertRaises(RuntimeError, _appGeoFromICRS, raShort, dec, mjd=52000.0)

        #test that it actually urns
        test=_appGeoFromICRS(ra, dec, mjd=obs_metadata.mjd)

        ##########test _observedFromAppGeo
        #test without obs_metadata
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec)

        #test without site
        dummy=ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                  unrefractedDec=obs_metadata.unrefractedDec,
                                  mjd=obs_metadata.mjd,
                                  site=None)
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec, obs_metadata=dummy)

        #test without mjd
        dummy=ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                  unrefractedDec=obs_metadata.unrefractedDec,
                                  site=Site())
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec, obs_metadata=dummy)

        #test mismatches
        dummy=ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                  unrefractedDec=obs_metadata.unrefractedDec,
                                  mjd=obs_metadata.mjd,
                                  site=Site())

        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, decShort, obs_metadata=dummy)
        self.assertRaises(RuntimeError, _observedFromAppGeo, raShort, dec, obs_metadata=dummy)

        #test that it actually runs
        test = _observedFromAppGeo(ra, dec, obs_metadata=dummy)

        ##########test _observedFromICRS
        #test without epoch
        self.assertRaises(RuntimeError, _observedFromICRS, ra, dec, obs_metadata=obs_metadata)

        #test without obs_metadata
        self.assertRaises(RuntimeError, _observedFromICRS, ra, dec, epoch=2000.0)

        #test without mjd
        dummy=ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                  unrefractedDec=obs_metadata.unrefractedDec,
                                  site=obs_metadata.site)
        self.assertRaises(RuntimeError, _observedFromICRS, ra, dec, epoch=2000.0, obs_metadata=dummy)

        #test that it actually runs
        dummy=ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                  unrefractedDec=obs_metadata.unrefractedDec,
                                  site=obs_metadata.site,
                                  mjd=obs_metadata.mjd)

        #test mismatches
        self.assertRaises(RuntimeError, _observedFromICRS, ra, decShort, epoch=2000.0, obs_metadata=dummy)
        self.assertRaises(RuntimeError, _observedFromICRS, raShort, dec, epoch=2000.0, obs_metadata=dummy)

        #test that it actually runs
        test = _observedFromICRS(ra, dec, obs_metadata=dummy, epoch=2000.0)



    def test_applyPrecession(self):

        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)

        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        ra[1]=8.693375673649429425e-01
        dec[1]=1.038086165642298164e+00
        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01

        self.assertRaises(RuntimeError, _applyPrecession, ra, dec)

        #just make sure it runs
        output=_applyPrecession(ra,dec, mjd=pal.epj(2000.0))


    def test_applyProperMotion(self):

        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)
        pm_ra=numpy.zeros((3),dtype=float)
        pm_dec=numpy.zeros((3),dtype=float)
        parallax=numpy.zeros((3),dtype=float)
        v_rad=numpy.zeros((3),dtype=float)

        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        pm_ra[0]=-8.472633255615005918e-05
        pm_dec[0]=-5.618517146980475171e-07
        parallax[0]=9.328946209650547383e-02
        v_rad[0]=3.060308412186171267e+02

        ra[1]=8.693375673649429425e-01
        dec[1]=1.038086165642298164e+00
        pm_ra[1]=-5.848962163813087908e-05
        pm_dec[1]=-3.000346282603337522e-05
        parallax[1]=5.392364722571952457e-02
        v_rad[1]=4.785834687356999098e+02

        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01
        pm_ra[2]=5.904070507320858615e-07
        pm_dec[2]=-2.958381482198743105e-05
        parallax[2]=2.172865273161764255e-02
        v_rad[2]=-3.225459751425886452e+02

        ep=2.001040286039033845e+03

        #The proper motion arguments in this function are weird
        #because there was a misunderstanding when the baseline
        #SLALIB data was made.
        output=_applyProperMotion(ra,dec,pm_ra*numpy.cos(dec),pm_dec/numpy.cos(dec),
                                 radiansFromArcsec(parallax),v_rad,epoch=ep,
                                 mjd=self.obs_metadata.mjd)

        self.assertAlmostEqual(output[0][0],2.549309127917495754e+00,6)
        self.assertAlmostEqual(output[1][0],5.198769294314042888e-01,6)
        self.assertAlmostEqual(output[0][1],8.694881589882680339e-01,6)
        self.assertAlmostEqual(output[1][1],1.038238225568303363e+00,6)
        self.assertAlmostEqual(output[0][2],7.740849573146946216e-01,6)
        self.assertAlmostEqual(output[1][2],2.758844356561930278e-01,6)


    def test_appGeoFromICRS(self):
        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)
        pm_ra=numpy.zeros((3),dtype=float)
        pm_dec=numpy.zeros((3),dtype=float)
        parallax=numpy.zeros((3),dtype=float)
        v_rad=numpy.zeros((3),dtype=float)


        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        pm_ra[0]=-8.472633255615005918e-05
        pm_dec[0]=-5.618517146980475171e-07
        parallax[0]=9.328946209650547383e-02
        v_rad[0]=3.060308412186171267e+02

        ra[1]=8.693375673649429425e-01
        dec[1]=1.038086165642298164e+00
        pm_ra[1]=-5.848962163813087908e-05
        pm_dec[1]=-3.000346282603337522e-05
        parallax[1]=5.392364722571952457e-02
        v_rad[1]=4.785834687356999098e+02

        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01
        pm_ra[2]=5.904070507320858615e-07
        pm_dec[2]=-2.958381482198743105e-05
        parallax[2]=2.172865273161764255e-02
        v_rad[2]=-3.225459751425886452e+02

        ep=2.001040286039033845e+03
        mjd=2.018749109074271473e+03

        #The proper motion arguments in this function are weird
        #because there was a misunderstanding when the baseline
        #SLALIB data was made.
        output=_appGeoFromICRS(ra,dec,pm_ra=pm_ra*numpy.cos(dec), pm_dec=pm_dec/numpy.cos(dec),
                              parallax=radiansFromArcsec(parallax),v_rad=v_rad, epoch=ep,
                              mjd=mjd)

        self.assertAlmostEqual(output[0][0],2.525858337335585180e+00,6)
        self.assertAlmostEqual(output[1][0],5.309044018653210628e-01,6)
        self.assertAlmostEqual(output[0][1],8.297492370691380570e-01,6)
        self.assertAlmostEqual(output[1][1],1.037400063009288331e+00,6)
        self.assertAlmostEqual(output[0][2],7.408639821342507537e-01,6)
        self.assertAlmostEqual(output[1][2],2.703229189890907214e-01,6)

    def test_observedFromAppGeo(self):
        """
        Note: this routine depends on Aopqk which fails if zenith distance
        is too great (or, at least, it won't warn you if the zenith distance
        is greater than pi/2, in which case answers won't make sense)
        """

        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)

        #we need to pass wv as the effective wavelength for methods that
        #calculate refraction because, when the control SLALIB runs were
        #done we misinterpreted the units of wavelength to be Angstroms
        #rather than microns.
        wv = 5000.0

        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        ra[1]=4.346687836824714712e-01
        dec[1]=-5.190430828211490821e-01
        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01

        mjd=2.018749109074271473e+03
        obs_metadata=ObservationMetaData(mjd=mjd,
                                     boundType='circle',
                                     boundLength=0.05,
                                     phoSimMetaData=self.metadata)

        output=_observedFromAppGeo(ra,dec, wavelength=wv, obs_metadata=obs_metadata)

        self.assertAlmostEqual(output[0][0],2.547475965605183745e+00,6)
        self.assertAlmostEqual(output[1][0],5.187045152602967057e-01,6)

        self.assertAlmostEqual(output[0][1],4.349858626308809040e-01,6)
        self.assertAlmostEqual(output[1][1],-5.191213875880701378e-01,6)

        self.assertAlmostEqual(output[0][2],7.743528611421227614e-01,6)
        self.assertAlmostEqual(output[1][2],2.755070101670137328e-01,6)

        output=_observedFromAppGeo(ra,dec,altAzHr=True, wavelength=wv, obs_metadata=obs_metadata)

        self.assertAlmostEqual(output[0][0][0],2.547475965605183745e+00,6)
        self.assertAlmostEqual(output[0][1][0],5.187045152602967057e-01,6)
        self.assertAlmostEqual(output[1][0][0],1.168920017932007643e-01,6)
        self.assertAlmostEqual(output[1][1][0],8.745379535264000692e-01,6)

        self.assertAlmostEqual(output[0][0][1],4.349858626308809040e-01,6)
        self.assertAlmostEqual(output[0][1][1],-5.191213875880701378e-01,6)
        self.assertAlmostEqual(output[1][0][1],6.766119585479937193e-01,6)
        self.assertAlmostEqual(output[1][1][1],4.433969998336554141e+00,6)

        self.assertAlmostEqual(output[0][0][2],7.743528611421227614e-01,6)
        self.assertAlmostEqual(output[0][1][2],2.755070101670137328e-01,6)
        self.assertAlmostEqual(output[1][0][2],5.275840601437552513e-01,6)
        self.assertAlmostEqual(output[1][1][2],5.479759580847959555e+00,6)

        output=_observedFromAppGeo(ra,dec,includeRefraction=False,
                                  wavelength=wv, obs_metadata=obs_metadata)

        self.assertAlmostEqual(output[0][0],2.549091783674975353e+00,6)
        self.assertAlmostEqual(output[1][0],5.198746844679964507e-01,6)

        self.assertAlmostEqual(output[0][1],4.346695674418772359e-01,6)
        self.assertAlmostEqual(output[1][1],-5.190436610150490626e-01,6)

        self.assertAlmostEqual(output[0][2],7.740875471580924705e-01,6)
        self.assertAlmostEqual(output[1][2],2.758055401087299296e-01,6)

        output=_observedFromAppGeo(ra,dec,includeRefraction=False,
                                  altAzHr=True, wavelength=wv, obs_metadata=obs_metadata)

        self.assertAlmostEqual(output[0][0][0],2.549091783674975353e+00,6)
        self.assertAlmostEqual(output[0][1][0],5.198746844679964507e-01,6)
        self.assertAlmostEqual(output[1][0][0],1.150652107618796299e-01,6)
        self.assertAlmostEqual(output[1][1][0],8.745379535264000692e-01,6)

        self.assertAlmostEqual(output[0][0][1],4.346695674418772359e-01,6)
        self.assertAlmostEqual(output[0][1][1],-5.190436610150490626e-01,6)
        self.assertAlmostEqual(output[1][0][1],6.763265401447272618e-01,6)
        self.assertAlmostEqual(output[1][1][1],4.433969998336554141e+00,6)

        self.assertAlmostEqual(output[0][0][2],7.740875471580924705e-01,6)
        self.assertAlmostEqual(output[0][1][2],2.758055401087299296e-01,6)
        self.assertAlmostEqual(output[1][0][2],5.271912536356709866e-01,6)
        self.assertAlmostEqual(output[1][1][2],5.479759580847959555e+00,6)

    def test_observedFromAppGeo_NoRefraction(self):

        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)

        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        ra[1]=4.346687836824714712e-01
        dec[1]=-5.190430828211490821e-01
        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01

        mjd=2.018749109074271473e+03
        obs_metadata=ObservationMetaData(mjd=mjd,
                                     boundType='circle',
                                     boundLength=0.05,
                                     phoSimMetaData=self.metadata)

        output=_observedFromAppGeo(ra,dec,altAzHr=True,
                                  includeRefraction=False, obs_metadata=obs_metadata)

        self.assertAlmostEqual(output[0][0][0],2.549091783674975353e+00,6)
        self.assertAlmostEqual(output[0][1][0],5.198746844679964507e-01,6)
        self.assertAlmostEqual(output[0][0][1],4.346695674418772359e-01,6)
        self.assertAlmostEqual(output[0][1][1],-5.190436610150490626e-01,6)
        self.assertAlmostEqual(output[0][0][2],7.740875471580924705e-01,6)
        self.assertAlmostEqual(output[0][1][2],2.758055401087299296e-01,6)
        self.assertAlmostEqual(output[1][0][2],5.271914342095551653e-01,6)
        self.assertAlmostEqual(output[1][1][2],5.479759402150099490e+00,6)


    def test_appGeoFromObserved(self):
        """
        Test that _appGeoFromObserved really does invert _observedFromAppGeo
        """
        mjd = 58350.0
        site = Site(longitude=0.235, latitude=-1.2)
        raCenter, decCenter = raDecFromAltAz(90.0, 0.0,
                                             numpy.degrees(site.longitude),
                                             numpy.degrees(site.latitude),
                                             mjd)

        obs = ObservationMetaData(unrefractedRA=raCenter, unrefractedDec=decCenter,
                                  mjd=58350.0,
                                  site=site)

        numpy.random.seed(125543)
        nSamples = 200

        # Note: the PALPY routines in question start to become inaccurate at
        # a zenith distance of about 75 degrees, so we restrict our test points
        # to be within 50 degrees of the telescope pointing, which is at zenith
        # in a flat sky approximation
        rr = numpy.random.random_sample(nSamples)*numpy.radians(50.0)
        theta = numpy.random.random_sample(nSamples)*2.0*numpy.pi
        ra_in = numpy.radians(raCenter) + rr*numpy.cos(theta)
        dec_in = numpy.radians(decCenter) + rr*numpy.sin(theta)

        xx_in = numpy.cos(dec_in)*numpy.cos(ra_in)
        yy_in = numpy.cos(dec_in)*numpy.sin(ra_in)
        zz_in = numpy.sin(dec_in)

        for includeRefraction in [True, False]:
            for wavelength in (0.5, 0.3, 0.7):
                ra_obs, dec_obs = _observedFromAppGeo(ra_in, dec_in, obs_metadata=obs,
                                                      wavelength=wavelength,
                                                      includeRefraction=includeRefraction)

                ra_out, dec_out = _appGeoFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                      wavelength=wavelength,
                                                      includeRefraction=includeRefraction)


                xx_out = numpy.cos(dec_out)*numpy.cos(ra_out)
                yy_out = numpy.cos(dec_out)*numpy.sin(ra_out)
                zz_out = numpy.sin(dec_out)

                distance = numpy.sqrt(numpy.power(xx_in-xx_out,2) +
                                      numpy.power(yy_in-yy_out,2) +
                                      numpy.power(zz_in-zz_out,2))

                self.assertLess(distance.max(), 1.0e-12)


    def testRefractionCoefficients(self):
        output=refractionCoefficients(wavelength=5000.0, site=self.obs_metadata.site)

        self.assertAlmostEqual(output[0],2.295817926320665320e-04,6)
        self.assertAlmostEqual(output[1],-2.385964632924575670e-07,6)

    def testApplyRefraction(self):
        coeffs=refractionCoefficients(wavelength=5000.0, site=self.obs_metadata.site)

        output=applyRefraction(0.25*numpy.pi,coeffs[0],coeffs[1])

        self.assertAlmostEqual(output,7.851689251070859132e-01,6)



def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(astrometryUnitTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

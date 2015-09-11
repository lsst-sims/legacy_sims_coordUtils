import unittest
import numpy
import lsst.utils.tests as utilsTests
from lsst.sims.utils import arcsecFromRadians, radiansFromArcsec
from lsst.sims.utils import ObservationMetaData
import lsst.sims.coordUtils as coordUtils

class AstrometryDegreesTest(unittest.TestCase):

    def setUp(self):
        self.nStars = 10
        numpy.random.seed(8273)
        self.raList = numpy.random.random_sample(self.nStars)*2.0*numpy.pi
        self.decList = (numpy.random.random_sample(self.nStars)-0.5)*numpy.pi
        self.mjdList = numpy.random.random_sample(10)*5000.0 + 52000.0
        self.pm_raList = radiansFromArcsec(numpy.random.random_sample(self.nStars)*10.0 - 5.0)
        self.pm_decList = radiansFromArcsec(numpy.random.random_sample(self.nStars)*10.0 - 5.0)
        self.pxList = radiansFromArcsec(numpy.random.random_sample(self.nStars)*2.0)
        self.v_radList = numpy.random.random_sample(self.nStars)*500.0 - 250.0


    def testApplyPrecession(self):
        for mjd in self.mjdList:
            raRad, decRad = coordUtils._applyPrecession(self.raList,
                                                        self.decList,
                                                        mjd=mjd)

            raDeg, decDeg = coordUtils.applyPrecession(numpy.degrees(self.raList),
                                                       numpy.degrees(self.decList),
                                                       mjd=mjd)

            dRa = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


    def testApplyProperMotion(self):
        for mjd in self.mjdList:
            raRad, decRad = coordUtils._applyProperMotion(self.raList, self.decList,
                                                          self.pm_raList, self.pm_decList,
                                                          self.pxList, self.v_radList, mjd=mjd)

            raDeg, decDeg = coordUtils.applyProperMotion(numpy.degrees(self.raList),
                                                         numpy.degrees(self.decList),
                                                         arcsecFromRadians(self.pm_raList),
                                                         arcsecFromRadians(self.pm_decList),
                                                         arcsecFromRadians(self.pxList),
                                                         self.v_radList, mjd=mjd)

            dRa = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


        for ra, dec, pm_ra, pm_dec, px, v_rad in \
        zip(self.raList, self.decList, self.pm_raList, self.pm_decList, \
        self.pxList, self.v_radList):

            raRad, decRad = coordUtils._applyProperMotion(ra, dec, pm_ra, pm_dec, px, v_rad,
                                                          mjd=self.mjdList[0])

            raDeg, decDeg = coordUtils.applyProperMotion(numpy.degrees(ra), numpy.degrees(dec),
                                                         arcsecFromRadians(pm_ra), arcsecFromRadians(pm_dec),
                                                         arcsecFromRadians(px), v_rad, mjd=self.mjdList[0])

            self.assertAlmostEqual(arcsecFromRadians(raRad-numpy.radians(raDeg)), 0.0, 9)
            self.assertAlmostEqual(arcsecFromRadians(decRad-numpy.radians(decDeg)), 0.0, 9)


    def testAppGeoFromICRS(self):
        mjd = 42350.0
        for pmRaList in [self.pm_raList, None]:
            for pmDecList in [self.pm_decList, None]:
                for pxList in [self.pxList, None]:
                    for vRadList in [self.v_radList, None]:
                        raRad, decRad = coordUtils._appGeoFromICRS(self.raList, self.decList,
                                                                   pmRaList, pmDecList,
                                                                   pxList, vRadList, mjd=mjd)

                        raDeg, decDeg = coordUtils.appGeoFromICRS(numpy.degrees(self.raList),
                                                                 numpy.degrees(self.decList),
                                                                 arcsecFromRadians(pmRaList),
                                                                 arcsecFromRadians(pmDecList),
                                                                 arcsecFromRadians(pxList),
                                                                 vRadList, mjd=mjd)

                        dRa = arcsecFromRadians(raRad-numpy.radians(raDeg))
                        numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

                        dDec = arcsecFromRadians(raRad-numpy.radians(raDeg))
                        numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)



    def testObservedFromAppGeo(self):
        obs = ObservationMetaData(unrefractedRA=35.0, unrefractedDec=-45.0,
                                  mjd=43572.0)

        for includeRefraction in [True, False]:
            raRad, decRad = coordUtils._observedFromAppGeo(self.raList, self.decList,
                                                           includeRefraction=includeRefraction,
                                                           altAzHr=False, obs_metadata=obs)

            raDeg, decDeg = coordUtils.observedFromAppGeo(numpy.degrees(self.raList),
                                                          numpy.degrees(self.decList),
                                                          includeRefraction=includeRefraction,
                                                          altAzHr=False, obs_metadata=obs)

            dRa = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


            raRad, decRad, altRad, azRad = coordUtils._observedFromAppGeo(self.raList, self.decList,
                                                                          includeRefraction=includeRefraction,
                                                                          altAzHr=True, obs_metadata=obs)

            raDeg, decDeg, altDeg, azDeg = coordUtils.observedFromAppGeo(numpy.degrees(self.raList),
                                                                         numpy.degrees(self.decList),
                                                                         includeRefraction=includeRefraction,
                                                                         altAzHr=True, obs_metadata=obs)

            dRa = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)

            dAz = arcsecFromRadians(azRad-numpy.radians(azDeg))
            numpy.testing.assert_array_almost_equal(dAz, numpy.zeros(self.nStars), 9)

            dAlt = arcsecFromRadians(altRad-numpy.radians(altDeg))
            numpy.testing.assert_array_almost_equal(dAlt, numpy.zeros(self.nStars), 9)


    def testObservedFromICRS(self):
        obs = ObservationMetaData(unrefractedRA=35.0, unrefractedDec=-45.0,
                                  mjd=43572.0)
        for pmRaList in [self.pm_raList, None]:
            for pmDecList in [self.pm_decList, None]:
                for pxList in [self.pxList, None]:
                    for vRadList in [self.v_radList, None]:
                        for includeRefraction in [True, False]:


                            raRad, decRad = coordUtils._observedFromICRS(self.raList, self.decList,
                                                                         pm_ra=pmRaList, pm_dec=pmDecList,
                                                                         parallax=pxList, v_rad=vRadList,
                                                                         obs_metadata=obs, epoch=2000.0,
                                                                         includeRefraction=includeRefraction)

                            raDeg, decDeg = coordUtils.observedFromICRS(numpy.degrees(self.raList), numpy.degrees(self.decList),
                                                                         pm_ra=arcsecFromRadians(pmRaList),
                                                                         pm_dec=arcsecFromRadians(pmDecList),
                                                                         parallax=arcsecFromRadians(pxList),
                                                                         v_rad=vRadList,
                                                                         obs_metadata=obs, epoch=2000.0,
                                                                     includeRefraction=includeRefraction)


                            dRa = arcsecFromRadians(raRad-numpy.radians(raDeg))
                            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

                            dDec = arcsecFromRadians(decRad-numpy.radians(decDeg))
                            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)



def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(AstrometryDegreesTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

import unittest
import numpy
import lsst.utils.tests as utilsTests
from lsst.sims.utils import arcsecFromRadians, radiansFromArcsec
import lsst.sims.coordUtils as coordUtils

class AstrometryDegreesTest(unittest.TestCase):

    def setUp(self):
        self.nStars = 100
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

            numpy.testing.assert_array_almost_equal((raRad-self.raList)/(numpy.radians(raDeg)-self.raList),
                                                    numpy.ones(self.nStars), 9)

            numpy.testing.assert_array_almost_equal((decRad-self.decList)/(numpy.radians(decDeg)-self.decList),
                                                    numpy.ones(self.nStars), 9)


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

            numpy.testing.assert_array_almost_equal((raRad-self.raList)/(numpy.radians(raDeg)-self.raList),
                                                    numpy.ones(self.nStars), 9)

            numpy.testing.assert_array_almost_equal((decRad-self.decList)/(numpy.radians(decDeg)-self.decList),
                                                    numpy.ones(self.nStars), 9)


        for ra, dec, pm_ra, pm_dec, px, v_rad in \
        zip(self.raList, self.decList, self.pm_raList, self.pm_decList, \
        self.pxList, self.v_radList):

            raRad, decRad = coordUtils._applyProperMotion(ra, dec, pm_ra, pm_dec, px, v_rad,
                                                          mjd=self.mjdList[0])

            raDeg, decDeg = coordUtils.applyProperMotion(numpy.degrees(ra), numpy.degrees(dec),
                                                         arcsecFromRadians(pm_ra), arcsecFromRadians(pm_dec),
                                                         arcsecFromRadians(px), v_rad, mjd=self.mjdList[0])

            self.assertAlmostEqual((ra-raRad)/(ra-numpy.radians(raDeg)), 1.0, 9)
            self.assertAlmostEqual((dec-decRad)/(dec-numpy.radians(decDeg)), 1.0, 9)


    def testAppGeoFromICRS(self):
        for mjd in self.mjdList:
            raRad, decRad = coordUtils._appGeoFromICRS(self.raList, self.decList,
                                                          self.pm_raList, self.pm_decList,
                                                          self.pxList, self.v_radList, mjd=mjd)

            raDeg, decDeg = coordUtils.appGeoFromICRS(numpy.degrees(self.raList),
                                                         numpy.degrees(self.decList),
                                                         arcsecFromRadians(self.pm_raList),
                                                         arcsecFromRadians(self.pm_decList),
                                                         arcsecFromRadians(self.pxList),
                                                         self.v_radList, mjd=mjd)

            numpy.testing.assert_array_almost_equal((raRad-self.raList)/(numpy.radians(raDeg)-self.raList),
                                                    numpy.ones(self.nStars), 9)

            numpy.testing.assert_array_almost_equal((decRad-self.decList)/(numpy.radians(decDeg)-self.decList),
                                                    numpy.ones(self.nStars), 9)




def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(AstrometryDegreesTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

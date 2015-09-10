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
        pm_ra = radiansFromArcsec(numpy.random.random_sample(self.nStars)*10.0 - 5.0)
        pm_dec = radiansFromArcsec(numpy.random.random_sample(self.nStars)*10.0 - 5.0)
        px = radiansFromArcsec(numpy.random.random_sample(self.nStars)*2.0)
        v_rad = numpy.random.random_sample(self.nStars)*500.0 - 250.0

        for mjd in self.mjdList:
            raRad, decRad = coordUtils._applyProperMotion(self.raList, self.decList,
                                                          pm_ra, pm_dec, px, v_rad,
                                                          mjd=mjd)

            raDeg, decDeg = coordUtils.applyProperMotion(numpy.degrees(self.raList),
                                                         numpy.degrees(self.decList),
                                                         arcsecFromRadians(pm_ra),
                                                         arcsecFromRadians(pm_dec),
                                                         arcsecFromRadians(px),
                                                         v_rad, mjd=mjd)

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

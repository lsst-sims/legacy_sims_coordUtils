import unittest
import numpy
import lsst.utils.tests as utilsTests
import lsst.sims.coordUtils as coordUtils

class AstrometryDegreesTest(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(8273)
        self.raList = numpy.random.random_sample(100)*2.0*numpy.pi
        self.decList = (numpy.random.random_sample(100)-0.5)*numpy.pi
        self.mjdList = numpy.random.random_sample(10)*5000.0 + 52000.0


    def testApplyPrecession(self):
        for mjd in self.mjdList:
            raRad, decRad = coordUtils._applyPrecession(self.raList,
                                                        self.decList,
                                                        mjd=mjd)

            raDeg, decDeg = coordUtils.applyPrecession(numpy.degrees(self.raList),
                                                       numpy.degrees(self.decList),
                                                       mjd=mjd)

            numpy.testing.assert_array_almost_equal(raRad, numpy.radians(raDeg), 10)
            numpy.testing.assert_array_almost_equal(decRad, numpy.radians(decDeg), 10)


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(AstrometryDegreesTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

import numpy
import unittest
import lsst.utils.tests as utilsTests

from lsst.sims.utils import ObservationMetaData, _nativeLonLatFromRaDec
from lsst.sims.coordUtils import calculatePupilCoordinates, observedFromICRS
from lsst.sims.coordUtils import raDecFromPupilCoordinates

class PupilCoordinateUnitTest(unittest.TestCase):

    def testExceptions(self):
        """
        Test that exceptions are raised when they ought to be
        """
        obs_metadata = ObservationMetaData(unrefractedRA=25.0,
                                           unrefractedDec=25.0,
                                           rotSkyPos=25.0,
                                           mjd=52000.0)

        numpy.random.seed(42)
        ra = numpy.random.random_sample(10)*numpy.radians(1.0) + numpy.radians(obs_metadata.unrefractedRA)
        dec = numpy.random.random_sample(10)*numpy.radians(1.0) + numpy.radians(obs_metadata.unrefractedDec)
        raShort = numpy.array([1.0])
        decShort = numpy.array([1.0])

        #test without epoch
        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, dec,
                          obs_metadata=obs_metadata)

        #test without obs_metadata
        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, dec,
                          epoch=2000.0)

        #test without unrefractedRA
        dummy = ObservationMetaData(unrefractedDec=obs_metadata.unrefractedDec,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        #test without unrefractedDec
        dummy = ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        #test without rotSkyPos
        dummy = ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                    unrefractedDec=obs_metadata.unrefractedDec,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        #test without mjd
        dummy = ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                    unrefractedDec=obs_metadata.unrefractedDec,
                                    rotSkyPos=obs_metadata.rotSkyPos)
        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)


        #test for mismatches
        dummy = ObservationMetaData(unrefractedRA=obs_metadata.unrefractedRA,
                                    unrefractedDec=obs_metadata.unrefractedDec,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)

        self.assertRaises(RuntimeError, calculatePupilCoordinates, ra, decShort, epoch=2000.0,
                          obs_metadata=dummy)

        self.assertRaises(RuntimeError, calculatePupilCoordinates, raShort, dec, epoch=2000.0,
                          obs_metadata=dummy)

        #test that it actually runs
        test = calculatePupilCoordinates(ra, dec, obs_metadata=obs_metadata, epoch=2000.0)


    def testCardinalDirections(self):
        """
        This unit test verifies that the following conventions hold:

        if rotSkyPos = 0, then north is +y the camera and east is -x

        if rotSkyPos = -90, then north is +x on the camera and east is +y

        if rotSkyPos = 90, then north is -x on the camera and east is -y

        if rotSkyPos = 180, then north is -y on the camera and east is +x

        This is consistent with rotSkyPos = rotTelPos - parallacticAngle

        parallacticAngle is negative when the pointing is east of the meridian.
        http://www.petermeadows.com/html/parallactic.html

        rotTelPos is the angle between up on the telescope and up on
        the camera, where positive rotTelPos goes from north to west
        (from an email sent to me by LynneJones)

        I have verified that OpSim follows the rotSkyPos = rotTelPos - paralacticAngle
        convention.

        I have verified that altAzPaFromRaDec follows the convention that objects
        east of the meridian have a negative parallactic angle.  (altAzPaFromRaDec
        uses PALPY under the hood, so it can probably be taken as correct)

        It will verify this convention for multiple random pointings.
        """

        epoch = 2000.0
        mjd = 42350.0
        numpy.random.seed(42)
        raList = numpy.random.random_sample(10)*360.0
        decList = numpy.random.random_sample(10)*180.0 - 90.0


        for rotSkyPos in numpy.arange(-90.0, 181.0, 90.0):
            for ra, dec in zip(raList, decList):
                obs = ObservationMetaData(unrefractedRA=ra,
                                          unrefractedDec=dec,
                                          mjd=mjd,
                                          rotSkyPos=rotSkyPos)

                centerRA, centerDec = observedFromICRS(numpy.array([numpy.radians(obs.unrefractedRA)]),
                                                       numpy.array([numpy.radians(obs.unrefractedDec)]),
                                                       obs_metadata=obs, epoch=epoch)

                #test order E, W, N, S
                raTest = centerRA[0] + numpy.array([0.01, -0.01, 0.0, 0.0])
                decTest = centerDec[0] + numpy.array([0.0, 0.0, 0.01, -0.01])
                x, y = calculatePupilCoordinates(raTest, decTest, obs_metadata=obs, epoch=epoch)

                lon, lat = _nativeLonLatFromRaDec(raTest, decTest, centerRA[0], centerDec[0])
                rr = numpy.abs(numpy.cos(lat)/numpy.sin(lat))

                if numpy.abs(rotSkyPos)<0.01:
                    control_x = numpy.array([-1.0*rr[0], 1.0*rr[1], 0.0, 0.0])
                    control_y = numpy.array([0.0, 0.0, 1.0*rr[2], -1.0*rr[3]])
                elif numpy.abs(rotSkyPos+90.0)<0.01:
                    control_x = numpy.array([0.0, 0.0, 1.0*rr[2], -1.0*rr[3]])
                    control_y = numpy.array([1.0*rr[0], -1.0*rr[1], 0.0, 0.0])
                elif numpy.abs(rotSkyPos-90.0)<0.01:
                    control_x = numpy.array([0.0, 0.0, -1.0*rr[2], 1.0*rr[3]])
                    control_y = numpy.array([-1.0*rr[0], 1.0*rr[1], 0.0, 0.0])
                elif numpy.abs(rotSkyPos-180.0)<0.01:
                    control_x = numpy.array([1.0*rr[0], -1.0*rr[1], 0.0, 0.0])
                    control_y = numpy.array([0.0, 0.0, -1.0*rr[2], 1.0*rr[3]])

                dx = numpy.array([xx/cc if numpy.abs(cc)>1.0e-10 else 1.0-xx for xx, cc in zip(x, control_x)])
                dy = numpy.array([yy/cc if numpy.abs(cc)>1.0e-10 else 1.0-yy for yy, cc in zip(y, control_y)])
                numpy.testing.assert_array_almost_equal(dx, numpy.ones(4), decimal=4)
                numpy.testing.assert_array_almost_equal(dy, numpy.ones(4), decimal=4)



    def testRaDecFromPupil(self):
        """
        Test conversion from pupil coordinates back to Ra, Dec
        """
        raCenter = 25.0
        decCenter = -10.0
        obs = ObservationMetaData(unrefractedRA=raCenter,
                                  unrefractedDec=decCenter,
                                  boundType='circle',
                                  boundLength=0.1,
                                  rotSkyPos=23.0,
                                  mjd=52000.0)

        nSamples = 100
        numpy.random.seed(42)
        ra = (numpy.random.random_sample(nSamples)*0.1-0.2) + numpy.radians(raCenter)
        dec = (numpy.random.random_sample(nSamples)*0.1-0.2) + numpy.radians(decCenter)
        xp, yp = calculatePupilCoordinates(ra, dec, obs_metadata=obs, epoch=2000.0)
        raTest, decTest = raDecFromPupilCoordinates(xp, yp, obs_metadata=obs, epoch=2000.0)
        numpy.testing.assert_array_almost_equal(raTest, ra, decimal=10)
        numpy.testing.assert_array_almost_equal(decTest, dec, decimal=10)





def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(PupilCoordinateUnitTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

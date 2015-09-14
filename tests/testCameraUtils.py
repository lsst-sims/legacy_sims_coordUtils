from __future__ import with_statement
import os
import numpy
import unittest
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir

from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import pupilCoordsFromRaDec
from lsst.sims.coordUtils import observedFromICRS
from lsst.sims.coordUtils import chipNameFromRaDec, \
                                 chipNameFromPupilCoords, \
                                 _chipNameFromRaDec

from lsst.sims.coordUtils import pixelCoordsFromPupilCoords, \
                                 pixelCoordsFromRaDec, \
                                 _pixelCoordsFromRaDec

class ChipNameTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    def setUp(self):
        numpy.random.seed(45532)


    def testRuns(self):
        """
        Test that chipName runs, and that the various iterations of that method
        are all self-consistent
        """
        nStars = 100
        ra0 = 45.0
        dec0 = -112.0
        rotSkyPos=135.0
        mjd = 42350.0
        obs = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                  mjd=mjd, rotSkyPos=rotSkyPos)

        raListRaw = (numpy.random.random_sample(nStars)-0.5)*1000.0/3600.0 + ra0
        decListRaw = (numpy.random.random_sample(nStars)-0.5)*1000.0/3600.0 + dec0

        raList, decList = observedFromICRS(raListRaw, decListRaw, obs_metadata=obs,
                                           epoch=2000.0)

        xpList, ypList = pupilCoordsFromRaDec(raList, decList,
                                                   obs_metadata=obs,
                                                   epoch=2000.0)

        names1 = chipNameFromRaDec(raList, decList,
                                   obs_metadata=obs,
                                   epoch=2000.0,
                                   camera=self.camera)

        names2 = _chipNameFromRaDec(numpy.radians(raList), numpy.radians(decList),
                                    obs_metadata=obs,
                                    epoch=2000.0,
                                    camera=self.camera)

        names3 = chipNameFromPupilCoords(xpList, ypList, camera=self.camera)

        numpy.testing.assert_array_equal(names1, names2)
        numpy.testing.assert_array_equal(names1, names3)

        isNone = 0
        isNotNone = 0
        for name in names1:
            if name is None:
                isNone += 1
            else:
                isNotNone += 1

        self.assertTrue(isNotNone>0)


    def testExceptions(self):
        """
        Test that exceptions are raised when they should be
        """

        nStars = 10
        xpList = numpy.random.random_sample(nStars)*0.1
        ypList = numpy.random.random_sample(nStars)*0.1

        obs = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=112.0, mjd=42351.0,
                                  rotSkyPos=35.0)

        # verify that an exception is raised if you do not pass in a camera
        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpList, ypList)
        self.assertTrue('No camera defined' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, obs_metadata=obs, epoch=2000.0)
        self.assertTrue('No camera defined' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, obs_metadata=obs, epoch=2000.0)
        self.assertTrue('No camera defined' in context.exception.message)

        # verify that an exception is raised if you do not pass in a numpy array
        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(list(xpList), ypList)
        self.assertTrue('numpy arrays' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(list(xpList), ypList, obs_metadata=obs, epoch=2000.0)
        self.assertTrue('numpy arrays' in context.exception.message)
        # do not need to run the above test on chipNameFromRaDec because
        # the conversion from degrees to radians that happens inside that
        # method automatically casts lists as numpy arrays

        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpList, list(ypList))
        self.assertTrue('numpy arrays' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, list(ypList), obs_metadata=obs, epoch=2000.0)
        self.assertTrue('numpy arrays' in context.exception.message)

        # verify that an exception is raised if the two coordinate arrays contain
        # different numbers of elements
        xpDummy = numpy.random.random_sample(nStars/2)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpDummy, ypList, camera=self.camera)
        self.assertTrue('xPupils' in context.exception.message)
        self.assertTrue('yPupils' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpDummy, ypList, obs_metadata=obs, epoch=2000.0,
                                  camera=self.camera)
        self.assertTrue('RAs' in context.exception.message)
        self.assertTrue('Decs' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpDummy, ypList, obs_metadata=obs, epoch=2000.0,
                                   camera=self.camera)
        self.assertTrue('RAs' in context.exception.message)
        self.assertTrue('Decs' in context.exception.message)


        # verify that an exception is raised if you call chipNameFromRaDec
        # without an epoch
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, obs_metadata=obs, camera=self.camera)
        self.assertTrue('epoch' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, obs_metadata=obs, camera=self.camera)
        self.assertTrue('epoch' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        # verify that an exception is raised if you call chipNameFromRaDec
        # without an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, epoch=2000.0, camera=self.camera)
        self.assertTrue('ObservationMetaData' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, epoch=2000.0, camera=self.camera)
        self.assertTrue('ObservationMetaData' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        # verify that an exception is raised if you call chipNameFromRaDec
        # with an ObservationMetaData that has no mjd
        obsDummy = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-112.0,
                                       rotSkyPos=112.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertTrue('mjd' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertTrue('mjd' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        # verify that an exception is raised if you all chipNameFromRaDec
        # using an ObservationMetaData without a rotSkyPos
        obsDummy = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-112.0,
                                       mjd=52350.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertTrue('rotSkyPos' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertTrue('rotSkyPos' in context.exception.message)
        self.assertTrue('chipName' in context.exception.message)


    def testNaNbecomesNone(self):
        """
        Test that chipName maps NaNs and Nones in RA, Dec, and
        pupil coordinates to None as chip name
        """
        nStars = 100
        ra0 = 45.0
        dec0 = -112.0
        rotSkyPos=135.0
        mjd = 42350.0
        obs = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                  mjd=mjd, rotSkyPos=rotSkyPos)

        for badVal in [numpy.NaN, None]:

            raListRaw = (numpy.random.random_sample(nStars)-0.5)*5.0/3600.0 + ra0
            decListRaw = (numpy.random.random_sample(nStars)-0.5)*5.0/3600.0 + dec0

            raListRaw[5] = badVal
            raListRaw[10] = badVal
            decListRaw[10] = badVal
            decListRaw[25] = badVal

            raList, decList = observedFromICRS(raListRaw, decListRaw, obs_metadata=obs,
                                               epoch=2000.0)

            xpList, ypList = pupilCoordsFromRaDec(raList, decList,
                                                       obs_metadata=obs,
                                                       epoch=2000.0)

            names1 = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                            camera=self.camera)

            names2 = _chipNameFromRaDec(numpy.radians(raList), numpy.radians(decList),
                                            obs_metadata=obs, epoch=2000.0, camera=self.camera)

            names3 = chipNameFromPupilCoords(xpList, ypList, camera=self.camera)

            numpy.testing.assert_array_equal(names1, names2)
            numpy.testing.assert_array_equal(names1, names3)

            for ix in range(len(names1)):
                if ix != 5 and ix != 10 and ix != 25:
                    self.assertTrue(names1[ix] == 'Det22')
                    self.assertTrue(names2[ix] == 'Det22')
                    self.assertTrue(names3[ix] == 'Det22')
                else:
                    self.assertTrue(names1[ix] is None)
                    self.assertTrue(names2[ix] is None)
                    self.assertTrue(names3[ix] is None)


class PixelCoordTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    def setUp(self):
        numpy.random.seed(11324)


    def testExceptions(self):
        """
        Test that pixelCoord calculation methods raise exceptions when
        they should
        """
        nPoints = 100
        xpList = numpy.random.random_sample(nPoints)*numpy.radians(1.0)
        ypList = numpy.random.random_sample(nPoints)*numpy.radians(1.0)
        obs = ObservationMetaData(unrefractedRA=25.0,
                                  unrefractedDec=-36.0,
                                  rotSkyPos=122.0,
                                  mjd=41325.0)

        raList = numpy.random.random_sample(nPoints)*1.0+25.0
        decList = numpy.random.random_sample(nPoints)*1.0-36.0

        # check that an error is raised when you forget to
        # pass in a camera
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList)
        self.assertTrue('camera' in context.exception.message)
        self.assertTrue('pixel' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList, obs_metadata=obs,
                                 epoch=2000.0)
        self.assertTrue('camera' in context.exception.message)
        self.assertTrue('pixel' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  numpy.radians(decList),
                                  obs_metadata=obs,
                                  epoch=2000.0)
        self.assertTrue('camera' in context.exception.message)
        self.assertTrue('pixel' in context.exception.message)


        # test that an exception is raised when you pass in something
        # that is not a numpy array
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(list(xpList), ypList,
                                       camera=self.camera)
        self.assertTrue('numpy array' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(list(numpy.radians(raList)),
                                  numpy.radians(decList),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)
        self.assertTrue('numpy array' in context.exception.message)
        # do not need to run the above test on pixelCoordsFromRaDec,
        # because the conversion from degrees to radians  that happens
        # inside that method automatically casts lists as numpy arrays


        # test that an exception is raised if you pass in mis-matched
        # input arrays
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList[0:10],
                                       camera=self.camera)
        self.assertTrue(' 10 ' in context.exception.message)
        self.assertTrue(' 100 ' in context.exception.message)
        self.assertTrue(' xPupil ' in context.exception.message)
        self.assertTrue(' yPupil ' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList[0:10],
                                 obs_metadata=obs,
                                 epoch=2000.0,
                                 camera=self.camera)
        self.assertTrue(' 10 ' in context.exception.message)
        self.assertTrue(' 100 ' in context.exception.message)
        self.assertTrue(' RA ' in context.exception.message)
        self.assertTrue(' Dec ' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  numpy.radians(decList[0:10]),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)
        self.assertTrue(' 10 ' in context.exception.message)
        self.assertTrue(' 100 ' in context.exception.message)
        self.assertTrue(' RA ' in context.exception.message)
        self.assertTrue(' Dec ' in context.exception.message)


        # test that an error is raised if you pass an incorrect
        # number of chipNames to pixelCoordsFromPupilCoords
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList, chipNames=['Det22']*10,
                                 camera=self.camera)
        self.assertTrue(' 100 ' in context.exception.message)
        self.assertTrue(' 10 ' in context.exception.message)
        self.assertTrue(' points ' in context.exception.message)
        self.assertTrue(' chipNames ' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList, chipNames=['Det22']*10,
                                 camera=self.camera,
                                 obs_metadata=obs,
                                 epoch=2000.0)
        self.assertTrue(' 100 ' in context.exception.message)
        self.assertTrue(' 10 ' in context.exception.message)
        self.assertTrue(' points ' in context.exception.message)
        self.assertTrue(' chipNames ' in context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  numpy.radians(decList),
                                  chipNames=['Det22']*10,
                                  camera=self.camera,
                                  obs_metadata=obs,
                                  epoch=2000.0)
        self.assertTrue(' 100 ' in context.exception.message)
        self.assertTrue(' 10 ' in context.exception.message)
        self.assertTrue(' points ' in context.exception.message)
        self.assertTrue(' chipNames ' in context.exception.message)

        # test that an exception is raised if you call one of the
        # pixelCoordsFromRaDec methods without an epoch
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera,
                                 obs_metadata=obs)
        self.assertEqual(context.exception.message,
                         'You need to pass an epoch into ' \
                         + 'pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera,
                                  obs_metadata=obs)
        self.assertEqual(context.exception.message,
                         'You need to pass an epoch into ' \
                         + 'pixelCoordsFromRaDec')

        # test that an exception is raised if you call one of the
        # pixelCoordsFromRaDec methods without an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera, epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData into ' \
                         + 'pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera, epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData into ' \
                         + 'pixelCoordsFromRaDec')

        # test that an exception is raised if you try to use an
        # ObservationMetaData without an mjd
        obsDummy = ObservationMetaData(unrefractedRA=25.0,
                                       unrefractedDec=-36.0,
                                       rotSkyPos=112.0)
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera,
                                 epoch=2000.0,
                                 obs_metadata=obsDummy)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData ' \
                         + 'with an mjd into pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera,
                                  epoch=2000.0,
                                  obs_metadata=obsDummy)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData ' \
                         + 'with an mjd into pixelCoordsFromRaDec')

        # test that an exception is raised if you try to use an
        # ObservationMetaData without a rotSkyPos
        obsDummy = ObservationMetaData(unrefractedRA=25.0,
                                       unrefractedDec=-36.0,
                                       mjd=53000.0)
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera,
                                 epoch=2000.0,
                                 obs_metadata=obsDummy)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData ' \
                         + 'with a rotSkyPos into pixelCoordsFromRaDec')

        obsDummy = ObservationMetaData(unrefractedRA=25.0,
                                       unrefractedDec=-36.0,
                                       mjd=53000.0)
        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera,
                                  epoch=2000.0,
                                  obs_metadata=obsDummy)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData ' \
                         + 'with a rotSkyPos into pixelCoordsFromRaDec')

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(ChipNameTest)
    suites += unittest.makeSuite(PixelCoordTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

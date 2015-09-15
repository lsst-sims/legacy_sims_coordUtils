from __future__ import with_statement
import os
import numpy
import unittest
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir

from lsst.afw.cameraGeom import PUPIL, PIXELS, TAN_PIXELS, FOCAL_PLANE

from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import pupilCoordsFromRaDec
from lsst.sims.coordUtils import observedFromICRS
from lsst.sims.coordUtils import chipNameFromRaDec, \
                                 chipNameFromPupilCoords, \
                                 _chipNameFromRaDec

from lsst.sims.coordUtils import pixelCoordsFromPupilCoords, \
                                 pixelCoordsFromRaDec, \
                                 _pixelCoordsFromRaDec

from lsst.sims.coordUtils import focalPlaneCoordsFromPupilCoords, \
                                 focalPlaneCoordsFromRaDec, \
                                 _focalPlaneCoordsFromRaDec

from lsst.sims.coordUtils import pupilCoordsFromPixelCoords

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
        self.assertEqual('No camera defined.  Cannot run chipName.',
                          context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, obs_metadata=obs, epoch=2000.0)
        self.assertEqual('No camera defined.  Cannot run chipName.',
                          context.exception.message)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, obs_metadata=obs, epoch=2000.0)
        self.assertEqual('No camera defined.  Cannot run chipName.',
                          context.exception.message)

        # verify that an exception is raised if you do not pass in a numpy array
        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(list(xpList), ypList)
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of ' \
                         + 'xPupil and yPupil to chipNameFromPupilCoords')

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(list(xpList), ypList, obs_metadata=obs, epoch=2000.0)
        self.assertEqual(context.exception.message,
                        'You need to pass numpy arrays of RA and Dec to chipName')

        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpList, list(ypList))
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of ' \
                         + 'xPupil and yPupil to chipNameFromPupilCoords')

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, list(ypList), obs_metadata=obs, epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of RA and Dec to chipName')
        # do not need to run the above test on chipNameFromRaDec because
        # the conversion from degrees to radians that happens inside that
        # method automatically casts lists as numpy arrays


        # verify that an exception is raised if the two coordinate arrays contain
        # different numbers of elements
        xpDummy = numpy.random.random_sample(nStars/2)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpDummy, ypList, camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed %d xPupils and ' % (nStars/2) \
                         + '%d yPupils to chipName.' % nStars)

        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpDummy, ypList, obs_metadata=obs, epoch=2000.0,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed %d RAs and ' % (nStars/2) \
                         + '%d Decs to chipName.' % nStars)

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpDummy, ypList, obs_metadata=obs, epoch=2000.0,
                                   camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed %d RAs and ' % (nStars/2) \
                         + '%d Decs to chipName.' % nStars)


        # verify that an exception is raised if you call chipNameFromRaDec
        # without an epoch
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, obs_metadata=obs, camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an epoch into chipName')

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, obs_metadata=obs, camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an epoch into chipName')

        # verify that an exception is raised if you call chipNameFromRaDec
        # without an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData into chipName')

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData into chipName')

        # verify that an exception is raised if you call chipNameFromRaDec
        # with an ObservationMetaData that has no mjd
        obsDummy = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-112.0,
                                       rotSkyPos=112.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData with an mjd into chipName')

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData with an mjd into chipName')

        # verify that an exception is raised if you all chipNameFromRaDec
        # using an ObservationMetaData without a rotSkyPos
        obsDummy = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-112.0,
                                       mjd=52350.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData with a rotSkyPos into chipName')

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, ypList, epoch=2000.0, obs_metadata=obsDummy,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData with a rotSkyPos into chipName')


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


    def testConsistency(self):
        """
        Test that all of the pixelCoord calculation methods agree with
        each other
        """
        ra0 = 95.0
        dec0 = -33.0
        obs = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                  mjd=52350.0, rotSkyPos=27.0)

        nStars = 100
        raList = (numpy.random.random_sample(nStars)-0.5)*500.0/3600.0 + ra0
        decList = (numpy.random.random_sample(nStars)-0.5)*500.0/3600.0 + dec0

        xpList, ypList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0)

        chipNameList = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                         camera=self.camera)

        for includeDistortion in [True, False]:

            xx1, yy1 = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                            camera=self.camera, includeDistortion=includeDistortion)

            xx2, yy2 = _pixelCoordsFromRaDec(numpy.radians(raList), numpy.radians(decList),
                                             obs_metadata=obs, epoch=2000.0,
                                             camera=self.camera, includeDistortion=includeDistortion)

            xx3, yy3 = pixelCoordsFromPupilCoords(xpList, ypList, camera=self.camera,
                                                  includeDistortion=includeDistortion)

            xx4, yy4 = pixelCoordsFromPupilCoords(xpList, ypList, chipNames=chipNameList,
                                                  camera=self.camera,
                                                  includeDistortion=includeDistortion)

            xx5, yy5 = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                            camera=self.camera, includeDistortion=includeDistortion,
                                            chipNames=chipNameList)

            xx6, yy6 = _pixelCoordsFromRaDec(numpy.radians(raList), numpy.radians(decList),
                                             obs_metadata=obs, epoch=2000.0,
                                             camera=self.camera, includeDistortion=includeDistortion,
                                             chipNames=chipNameList)


            numpy.testing.assert_array_equal(xx1, xx2)
            numpy.testing.assert_array_equal(xx1, xx3)
            numpy.testing.assert_array_equal(xx1, xx4)
            numpy.testing.assert_array_equal(xx1, xx5)
            numpy.testing.assert_array_equal(xx1, xx6)

            numpy.testing.assert_array_equal(yy1, yy2)
            numpy.testing.assert_array_equal(yy1, yy3)
            numpy.testing.assert_array_equal(yy1, yy4)
            numpy.testing.assert_array_equal(yy1, yy5)
            numpy.testing.assert_array_equal(yy1, yy6)

            # make sure that objects which do not fall on a chip
            # get NaN pixel coords
            ctNaN = 0
            ctNotNaN = 0
            for x, y, name in zip(xx1, yy1, chipNameList):
                if name is None:
                    self.assertTrue(numpy.isnan(x))
                    self.assertTrue(numpy.isnan(y))
                    ctNaN += 1
                else:
                    self.assertFalse(numpy.isnan(x))
                    self.assertFalse(numpy.isnan(y))
                    ctNotNaN += 1

            self.assertTrue(ctNaN>0)
            self.assertTrue(ctNotNaN>0)


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
        self.assertEqual(context.exception.message,
                         'Camera not specified.  Cannot calculate pixel coordinates.')

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList, obs_metadata=obs,
                                 epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'Camera not specified.  Cannot calculate pixel coordinates.')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  numpy.radians(decList),
                                  obs_metadata=obs,
                                  epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'Camera not specified.  Cannot calculate pixel coordinates.')


        # test that an exception is raised when you pass in something
        # that is not a numpy array
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(list(xpList), ypList,
                                       camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of xPupil and yPupil ' \
                         + 'to pixelCoordsFromPupilCoords')

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, list(ypList),
                                       camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of xPupil and yPupil ' \
                         + 'to pixelCoordsFromPupilCoords')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(list(numpy.radians(raList)),
                                  numpy.radians(decList),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of RA and Dec ' \
                         + 'to pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  list(numpy.radians(decList)),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You need to pass numpy arrays of RA and Dec ' \
                         + 'to pixelCoordsFromRaDec')
        # do not need to run the above test on pixelCoordsFromRaDec,
        # because the conversion from degrees to radians  that happens
        # inside that method automatically casts lists as numpy arrays


        # test that an exception is raised if you pass in mis-matched
        # input arrays
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList[0:10],
                                       camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed 100 xPupil and 10 yPupil coordinates ' \
                         + 'to pixelCoordsFromPupilCoords')

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList[0:10],
                                 obs_metadata=obs,
                                 epoch=2000.0,
                                 camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed 100 RA and 10 Dec coordinates ' \
                         'to pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  numpy.radians(decList[0:10]),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed 100 RA and 10 Dec coordinates ' \
                         'to pixelCoordsFromRaDec')


        # test that an error is raised if you pass an incorrect
        # number of chipNames to pixelCoordsFromPupilCoords
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList, chipNames=['Det22']*10,
                                 camera=self.camera)
        self.assertEqual(context.exception.message,
                         'You passed 100 points but 10 chipNames to pixelCoordsFromPupilCoords')

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList, chipNames=['Det22']*10,
                                 camera=self.camera,
                                 obs_metadata=obs,
                                 epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'You passed 100 points but 10 chipNames to pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(numpy.radians(raList),
                                  numpy.radians(decList),
                                  chipNames=['Det22']*10,
                                  camera=self.camera,
                                  obs_metadata=obs,
                                  epoch=2000.0)
        self.assertEqual(context.exception.message,
                         'You passed 100 points but 10 chipNames to pixelCoordsFromRaDec')

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


    def testResults(self):
        """
        Test that the results of the pixelCoords methods make sense.  Note that the test
        camera has a platescale of 0.02 arcsec per pixel (2.0 arcsec per mm encoded in
        CameraForUnitTests.py and 10 microns per pixel encoded in cameraData/focalplanelayout.txt).
        We will use that to set the control values for our unit test.

        Note: This unit test will fail if the test camera ever changes.

        Note: Because we have already tested the self-consistency of
        pixelCoordsFromPupilCoords and pixelCoordsFromRaDec, we will
        only be testing pixelCoordsFromPupilCoords here, because it
        is easier.
        """

        arcsecPerPixel = 0.02
        arcsecPerMicron = 0.002

        #list a bunch of detector centers in radians
        x22 = 0.0
        y22 = 0.0

        x32 = radiansFromArcsec(40000.0 * arcsecPerMicron)
        y32 = 0.0

        x40 = radiansFromArcsec(80000.0 * arcsecPerMicron)
        y40 = radiansFromArcsec(-80000.0 * arcsecPerMicron)

        # assemble a bunch of displacements in pixels
        dxPixList = []
        dyPixList = []
        for xx in numpy.arange(-1999.0, 1999.0, 500.0):
            for yy in numpy.arange(-1999.0, 1999.0, 500.0):
                dxPixList.append(xx)
                dyPixList.append(yy)

        dxPixList = numpy.array(dxPixList)
        dyPixList = numpy.array(dyPixList)

        # convert to raidans
        dxPupList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyPupList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxPupList
        yPupList = y22 + dyPupList
        xPupList = numpy.append(xPupList, x32 + dxPupList)
        yPupList = numpy.append(yPupList, y32 + dyPupList)
        xPupList = numpy.append(xPupList, x40 + dxPupList)
        yPupList = numpy.append(yPupList, y40 + dyPupList)

        # this is what the chipNames ought to be for these points
        chipNameControl = numpy.array(['Det22'] * len(dxPupList))
        chipNameControl = numpy.append(chipNameControl, ['Det32'] * len(dxPupList))
        chipNameControl = numpy.append(chipNameControl, ['Det40'] * len(dxPupList))

        chipNameTest = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)

        # verify that the test points fall on the expected chips
        numpy.testing.assert_array_equal(chipNameControl, chipNameTest)

        # Note, the somewhat backwards way in which we go from dxPixList to
        # xPixControl is due to the fact that pixel coordinates are actually
        # aligned so that the x-axis is along the read-out direction, which
        # makes positive x in pixel coordinates correspond to positive y
        # in pupil coordinates
        xPixControl = 1999.5 + dyPixList
        yPixControl = 1999.5 - dxPixList
        xPixControl = numpy.append(xPixControl, 1999.5 + dyPixList)
        yPixControl = numpy.append(yPixControl, 1999.5 - dxPixList)
        xPixControl = numpy.append(xPixControl, 1999.5 + dyPixList)
        yPixControl = numpy.append(yPixControl, 1999.5 - dxPixList)

        # verify that the pixel coordinates are as expected to within 0.01 pixel
        xPixTest, yPixTest = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera,
                                                        includeDistortion=False)

        numpy.testing.assert_array_almost_equal(xPixTest, xPixControl, 2)
        numpy.testing.assert_array_almost_equal(yPixTest, yPixControl, 2)


    def testOffChipResults(self):
        """
        Test that the results of the pixelCoords methods make sense in the case
        that you specify a chip name that is not necessarily the chip on which
        the object actually fell.

        Note that the test camera has a platescale of 0.02 arcsec per pixel
        (2.0 arcsec per mm encoded in CameraForUnitTests.py and 10 microns per
        pixel encoded in cameraData/focalplanelayout.txt). We will use that to
        set the control values for our unit test.

        Note: This unit test will fail if the test camera ever changes.

        Note: Because we have already tested the self-consistency of
        pixelCoordsFromPupilCoords and pixelCoordsFromRaDec, we will
        only be testing pixelCoordsFromPupilCoords here, because it
        is easier.
        """

        arcsecPerPixel = 0.02
        arcsecPerMicron = 0.002

        #list a bunch of detector centers in radians
        x22 = 0.0
        y22 = 0.0

        x32 = radiansFromArcsec(40000.0 * arcsecPerMicron)
        y32 = 0.0

        x40 = radiansFromArcsec(80000.0 * arcsecPerMicron)
        y40 = radiansFromArcsec(-80000.0 * arcsecPerMicron)

        # assemble a bunch of displacements in pixels
        dxPixList = []
        dyPixList = []
        for xx in numpy.arange(-1999.0, 1999.0, 500.0):
            for yy in numpy.arange(-1999.0, 1999.0, 500.0):
                dxPixList.append(xx)
                dyPixList.append(yy)

        dxPixList = numpy.array(dxPixList)
        dyPixList = numpy.array(dyPixList)

        # convert to raidans
        dxPupList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyPupList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxPupList
        yPupList = y22 + dyPupList
        xPupList = numpy.append(xPupList, x32 + dxPupList)
        yPupList = numpy.append(yPupList, y32 + dyPupList)
        xPupList = numpy.append(xPupList, x40 + dxPupList)
        yPupList = numpy.append(yPupList, y40 + dyPupList)

        # this is what the chipNames ought to be for these points
        chipNameControl = numpy.array(['Det22'] * len(dxPupList))
        chipNameControl = numpy.append(chipNameControl, ['Det32'] * len(dxPupList))
        chipNameControl = numpy.append(chipNameControl, ['Det40'] * len(dxPupList))

        chipNameTest = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)

        # verify that the test points fall on the expected chips
        numpy.testing.assert_array_equal(chipNameControl, chipNameTest)

        # Note, the somewhat backwards way in which we go from dxPupList to
        # xPixControl is due to the fact that pixel coordinates are actually
        # aligned so that the x-axis is along the read-out direction, which
        # makes positive x in pixel coordinates correspond to positive y
        # in pupil coordinates
        xPixControl = 1999.5 + arcsecFromRadians(yPupList - y40)/arcsecPerPixel
        yPixControl = 1999.5 - arcsecFromRadians(xPupList - x40)/arcsecPerPixel

        # verify that the pixel coordinates are as expected to within 0.01 pixel
        inputChipNames = ['Det40'] * len(xPupList)
        xPixTest, yPixTest = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera,
                                                        includeDistortion=False,
                                                        chipNames=inputChipNames)

        numpy.testing.assert_array_almost_equal(xPixTest, xPixControl, 2)
        numpy.testing.assert_array_almost_equal(yPixTest, yPixControl, 2)



    def testNaN(self):
        """
        Verify that NaNs and Nones input to pixelCoordinate calculation methods result
        in NaNs coming out
        """
        ra0 = 25.0
        dec0 = -35.0
        obs = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                  rotSkyPos=42.0, mjd=42356.0)

        raCenter, decCenter = observedFromICRS(numpy.array([ra0]), numpy.array([dec0]),
                                               obs_metadata=obs, epoch=2000.0)

        nStars = 10
        raList = numpy.random.random_sample(100)*100.0/3600.0 + raCenter[0]
        decList = numpy.random.random_sample(100)*100.0/3600.0 + decCenter[0]
        chipNameList = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                         camera=self.camera)

        # make sure that all of the test points actually fall on chips
        for name in chipNameList:
            self.assertTrue(name is not None)

        xPupList, yPupList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0)

        # make sure that none of the test points already result in NaN pixel coordinates
        xPixList, yPixList = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs,
                                                  epoch=2000.0, camera=self.camera)

        for xx, yy in zip(xPixList, yPixList):
            self.assertFalse(numpy.isnan(xx))
            self.assertFalse(numpy.isnan(yy))
            self.assertFalse(xx is None)
            self.assertFalse(yy is None)

        for badVal in [numpy.NaN, None]:
            raList[5] = badVal
            decList[7] = badVal
            raList[9] = badVal
            decList[9] = badVal

            xPixList, yPixList = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs,
                                                      epoch=2000.0, camera=self.camera)

            for ix, (xx, yy) in enumerate(zip(xPixList, yPixList)):
                if ix in [5, 7, 9]:
                    self.assertTrue(numpy.isnan(xx))
                    self.assertTrue(numpy.isnan(yy))
                else:
                    self.assertFalse(numpy.isnan(xx))
                    self.assertFalse(numpy.isnan(yy))
                    self.assertFalse(xx is None)
                    self.assertFalse(yy is None)

            xPixList, yPixList = _pixelCoordsFromRaDec(numpy.radians(raList), numpy.radians(decList),
                                                       obs_metadata=obs, epoch=2000.0, camera=self.camera)

            for ix, (xx, yy) in enumerate(zip(xPixList, yPixList)):
                if ix in [5, 7, 9]:
                    self.assertTrue(numpy.isnan(xx))
                    self.assertTrue(numpy.isnan(yy))
                else:
                    self.assertFalse(numpy.isnan(xx))
                    self.assertFalse(numpy.isnan(yy))
                    self.assertFalse(xx is None)
                    self.assertFalse(yy is None)

            xPupList[5] = badVal
            yPupList[7] = badVal
            xPupList[9] = badVal
            yPupList[9] = badVal
            xPixList, yPixList = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)
            for ix, (xx, yy) in enumerate(zip(xPixList, yPixList)):
                if ix in [5, 7, 9]:
                    self.assertTrue(numpy.isnan(xx))
                    self.assertTrue(numpy.isnan(yy))
                else:
                    self.assertFalse(numpy.isnan(xx))
                    self.assertFalse(numpy.isnan(yy))
                    self.assertFalse(xx is None)
                    self.assertFalse(yy is None)



class FocalPlaneCoordTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    def setUp(self):
        numpy.random.seed(8374522)


    def testConsistency(self):
        """
        Test that all of the focalPlaneCoord calculation methods
        return self-consistent answers.
        """

        ra0 = 34.1
        dec0 = -23.0
        obs = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                  mjd=43257.0, rotSkyPos = 127.0)

        raCenter, decCenter = observedFromICRS(numpy.array([ra0]),
                                               numpy.array([dec0]),
                                               obs_metadata=obs,
                                               epoch=2000.0)

        nStars = 100
        raList = numpy.random.random_sample(nStars)*1000.0/3600.0 + raCenter[0]
        decList = numpy.random.random_sample(nStars)*1000.0/3600.0 + decCenter[0]

        xPupList, yPupList = pupilCoordsFromRaDec(raList, decList,
                                                  obs_metadata=obs,
                                                  epoch=2000.0)

        xf1, yf1 = focalPlaneCoordsFromRaDec(raList, decList,
                                             obs_metadata=obs,
                                             epoch=2000.0, camera=self.camera)

        xf2, yf2 = _focalPlaneCoordsFromRaDec(numpy.radians(raList),
                                              numpy.radians(decList),
                                              obs_metadata=obs,
                                              epoch=2000.0, camera=self.camera)

        xf3, yf3 = focalPlaneCoordsFromPupilCoords(xPupList, yPupList,
                                                   camera=self.camera)

        numpy.testing.assert_array_equal(xf1, xf2)
        numpy.testing.assert_array_equal(xf1, xf3)
        numpy.testing.assert_array_equal(yf1, yf2)
        numpy.testing.assert_array_equal(yf1, yf3)

        for x, y in zip(xf1, yf1):
            self.assertFalse(numpy.isnan(x))
            self.assertFalse(x is None)
            self.assertFalse(numpy.isnan(y))
            self.assertFalse(y is None)


    def testExceptions(self):
        """
        Test that the focalPlaneCoord methods raise the exceptions
        (with the correct messages) when they should.
        """

        ra0 = 34.0
        dec0 = -19.0
        obs = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                  rotSkyPos=61.0, mjd=52349.0)

        nStars = 10
        raList = (numpy.random.random_sample(nStars)-0.5) + ra0
        decList = (numpy.random.random_sample(nStars)-0.5) + dec0
        xPupList, yPupList = pupilCoordsFromRaDec(raList, decList,
                                                  obs_metadata=obs,
                                                  epoch=2000.0)

        # verify that an error is raised when you forget to pass
        # in a camera
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(xPupList, yPupList)
        self.assertEqual(context.exception.message,
                         "You cannot calculate focal plane coordinates " \
                         + "without specifying a camera")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obs,
                                               epoch=2000.0)
        self.assertEqual(context.exception.message,
                         "You cannot calculate focal plane coordinates " \
                         + "without specifying a camera")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obs,
                                                epoch=2000.0)
        self.assertEqual(context.exception.message,
                         "You cannot calculate focal plane coordinates " \
                         + "without specifying a camera")


        # test that an error is raised when you pass in something that
        # is not a numpy array
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(list(xPupList), yPupList,
                                                     camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You must pass numpy arrays of xPupil and yPupil " \
                         +"to focalPlaneCoordsFromPupilCoords")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(xPupList, list(yPupList),
                                                     camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You must pass numpy arrays of xPupil and yPupil " \
                         +"to focalPlaneCoordsFromPupilCoords")


        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(list(raList), decList,
                                                obs_metadata=obs,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You must pass numpy arrays of RA and Dec to " \
                         + "focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, list(decList),
                                                obs_metadata=obs,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You must pass numpy arrays of RA and Dec to " \
                         + "focalPlaneCoordsFromRaDec")
        # we do not have to run the test above on focalPlaneCoordsFromRaDec
        # because the conversion to radians automatically casts lists into
        # numpy arrays

        # test that an error is raised if you pass in mismatched numbers
        # of x and y coordinates
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(xPupList, yPupList[0:4],
                                                     camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You specified 10 xPupil and 4 yPupil coordinates " \
                         + "in focalPlaneCoordsFromPupilCoords")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList[0:4],
                                               obs_metadata=obs,
                                               epoch=2000.0,
                                               camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You specified 10 RAs and 4 Decs in " \
                         + "focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList[0:4],
                                                obs_metadata=obs,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You specified 10 RAs and 4 Decs in " \
                         + "focalPlaneCoordsFromRaDec")


        # test that an error is raised if you call
        # focalPlaneCoordsFromRaDec without an epoch
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obs,
                                               camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You have to specify an epoch to run " \
                         + "focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obs,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You have to specify an epoch to run " \
                         + "focalPlaneCoordsFromRaDec")


        # test that an error is raised if you call
        # focalPlaneCoordsFromRaDec without an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               epoch=2000.0,
                                               camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You have to specify an ObservationMetaData to run " \
                         + "focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You have to specify an ObservationMetaData to run " \
                         + "focalPlaneCoordsFromRaDec")


        # test that an error is raised if you pass an ObservationMetaData
        # without an mjd into focalPlaneCoordsFromRaDec
        obsDummy = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                       rotSkyPos=112.0)
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obsDummy,
                                               epoch=2000.0,
                                               camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with an " \
                         + "mjd into focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obsDummy,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with an " \
                         + "mjd into focalPlaneCoordsFromRaDec")

        # test that an error is raised if you pass an ObservationMetaData
        # without a rotSkyPos into focalPlaneCoordsFromRaDec
        obsDummy = ObservationMetaData(unrefractedRA=ra0, unrefractedDec=dec0,
                                       mjd=42356.0)
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obsDummy,
                                               epoch=2000.0,
                                               camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with a " \
                         + "rotSkyPos into focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obsDummy,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with a " \
                         + "rotSkyPos into focalPlaneCoordsFromRaDec")


    def testResults(self):
        """
        Test that the focalPlaneCoords methods give sensible results.

        Note: since we have already tested the self-consistency of
        focalPlaneCoordsFromPupilCoords and focalPlaneCoordsFromRaDec,
        we will only test focalPlaneCoordsFromPupilCoords here, since it
        is easier.

        Note that the test camera has a platescale of 0.02 arcsec per pixel
        (2.0 arcsec per mm encoded in CameraForUnitTests.py and 10 microns
        per pixel encoded in cameraData/focalplanelayout.txt). We will use
        that to set the control values for our unit test.

        Note: This unit test will fail if the test camera ever changes.
        """

        arcsecPerPixel = 0.02
        arcsecPerMicron = 0.002
        mmPerArcsec = 0.5

        #list a bunch of detector centers in radians
        x22 = 0.0
        y22 = 0.0

        x32 = radiansFromArcsec(40000.0 * arcsecPerMicron)
        y32 = 0.0

        x40 = radiansFromArcsec(80000.0 * arcsecPerMicron)
        y40 = radiansFromArcsec(-80000.0 * arcsecPerMicron)

        # assemble a bunch of displacements in pixels
        dxPixList = []
        dyPixList = []
        for xx in numpy.arange(-1999.0, 1999.0, 500.0):
            for yy in numpy.arange(-1999.0, 1999.0, 500.0):
                dxPixList.append(xx)
                dyPixList.append(yy)

        dxPixList = numpy.array(dxPixList)
        dyPixList = numpy.array(dyPixList)

        # convert to raidans
        dxPupList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyPupList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxPupList
        yPupList = y22 + dyPupList
        xPupList = numpy.append(xPupList, x32 + dxPupList)
        yPupList = numpy.append(yPupList, y32 + dyPupList)
        xPupList = numpy.append(xPupList, x40 + dxPupList)
        yPupList = numpy.append(yPupList, y40 + dyPupList)

        # this is what the chipNames ought to be for these points
        chipNameControl = numpy.array(['Det22'] * len(dxPupList))
        chipNameControl = numpy.append(chipNameControl, ['Det32'] * len(dxPupList))
        chipNameControl = numpy.append(chipNameControl, ['Det40'] * len(dxPupList))

        chipNameTest = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)

        # verify that the test points fall on the expected chips
        numpy.testing.assert_array_equal(chipNameControl, chipNameTest)

        # convert into millimeters on the focal plane
        xFocalControl = arcsecFromRadians(xPupList)*mmPerArcsec
        yFocalControl = arcsecFromRadians(yPupList)*mmPerArcsec

        xFocalTest, yFocalTest = focalPlaneCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)

        numpy.testing.assert_array_almost_equal(xFocalTest, xFocalControl, 3)
        numpy.testing.assert_array_almost_equal(yFocalTest, yFocalControl, 3)


class ConversionFromPixelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    def setUp(self):
        numpy.random.seed(543)

    def testPupCoordsException(self):
        """
        Test that pupilCoordsFromPixelCoords raises an exception when you
        call it without a camera
        """
        nStars = 100
        xPupList = radiansFromArcsec((numpy.random.random_sample(nStars)-0.5)*320.0)
        yPupList = radiansFromArcsec((numpy.random.random_sample(nStars)-0.5)*320.0)
        chipNameList = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)
        xPix, yPix = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)
        with self.assertRaises(RuntimeError) as context:
            xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList)
        self.assertEqual(context.exception.message,
                         "You cannot call pupilCoordsFromPixelCoords without specifying " \
                         + "a camera")


    def testPupCoordsResults(self):
        """
        Test that the results from pupilCoordsFromPixelCoords are consistent
        with the results from pixelCoordsFromPupilCoords
        """

        nStars = 100
        xPupList = radiansFromArcsec((numpy.random.random_sample(nStars)-0.5)*320.0)
        yPupList = radiansFromArcsec((numpy.random.random_sample(nStars)-0.5)*320.0)
        chipNameList = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)
        for includeDistortion in [True, False]:
            xPix, yPix = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera,
                                                    includeDistortion=includeDistortion)
            xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList, camera=self.camera,
                                                            includeDistortion=includeDistortion)

            dx = arcsecFromRadians(xPupTest-xPupList)
            numpy.testing.assert_array_almost_equal(dx, numpy.zeros(len(dx)), 9)
            dy = arcsecFromRadians(yPupTest-yPupList)
            numpy.testing.assert_array_almost_equal(dy, numpy.zeros(len(dy)), 9)

            ctNaN = 0
            for x, y in zip(xPupTest, yPupTest):
                if numpy.isnan(x) or numpy.isnan(y):
                    ctNaN += 1
            self.assertTrue(ctNaN<len(xPupTest)/10)

    def testPupCoordsNaN(self):
        """
        Test that points which do not have a chip return NaN for pupilCoordsFromPixelCoords
        """
        nStars = 10
        xPupList = radiansFromArcsec((numpy.random.random_sample(nStars)-0.5)*320.0)
        yPupList = radiansFromArcsec((numpy.random.random_sample(nStars)-0.5)*320.0)
        chipNameList = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)
        chipNameList[5] = None
        xPix, yPix = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)
        xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList, camera=self.camera)
        self.assertTrue(numpy.isnan(xPupTest[5]))
        self.assertTrue(numpy.isnan(yPupTest[5]))


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(ChipNameTest)
    suites += unittest.makeSuite(PixelCoordTest)
    suites += unittest.makeSuite(FocalPlaneCoordTest)
    suites += unittest.makeSuite(ConversionFromPixelTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

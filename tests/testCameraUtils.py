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
        dxList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxList
        yPupList = y22 + dyList
        xPupList = numpy.append(xPupList, x32 + dxList)
        yPupList = numpy.append(yPupList, y32 + dyList)
        xPupList = numpy.append(xPupList, x40 + dxList)
        yPupList = numpy.append(yPupList, y40 + dyList)

        # this is what the chipNames ought to be for these points
        chipNameControl = numpy.array(['Det22'] * len(dxList))
        chipNameControl = numpy.append(chipNameControl, ['Det32'] * len(dxList))
        chipNameControl = numpy.append(chipNameControl, ['Det40'] * len(dxList))

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

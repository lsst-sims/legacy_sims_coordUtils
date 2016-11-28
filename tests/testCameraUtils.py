from __future__ import with_statement
import os
import numpy as np
import unittest
import warnings
import lsst.utils.tests
from lsst.utils import getPackageDir

from lsst.sims.utils import ObservationMetaData, radiansFromArcsec, arcsecFromRadians
from lsst.sims.utils import pupilCoordsFromRaDec
from lsst.sims.utils import haversine
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.utils import pupilCoordsFromRaDec, observedFromICRS
from lsst.sims.coordUtils import (chipNameFromRaDec,
                                  chipNameFromPupilCoords,
                                  _chipNameFromRaDec)

from lsst.sims.coordUtils import (pixelCoordsFromPupilCoords,
                                  pixelCoordsFromRaDec,
                                  _pixelCoordsFromRaDec)

from lsst.sims.coordUtils import (focalPlaneCoordsFromPupilCoords,
                                  focalPlaneCoordsFromRaDec,
                                  _focalPlaneCoordsFromRaDec)

from lsst.sims.coordUtils import pupilCoordsFromPixelCoords
from lsst.sims.coordUtils import raDecFromPixelCoords, _raDecFromPixelCoords
from lsst.sims.coordUtils import getCornerPixels, _getCornerRaDec, getCornerRaDec
from lsst.sims.coordUtils import MultipleChipWarning
from lsst.obs.lsstSim import LsstSimMapper


def setup_module(module):
    lsst.utils.tests.init()


class ChipNameTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def setUp(self):
        self.rng = np.random.RandomState(45532)

    def testRuns(self):
        """
        Test that chipName runs, and that the various iterations of that method
        are all self-consistent
        """
        nStars = 100
        ra0 = 45.0
        dec0 = -112.0
        rotSkyPos = 135.0
        mjd = 42350.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=mjd, rotSkyPos=rotSkyPos)

        raList = (self.rng.random_sample(nStars)-0.5)*1000.0/3600.0 + ra0
        decList = (self.rng.random_sample(nStars)-0.5)*1000.0/3600.0 + dec0

        xpList, ypList = pupilCoordsFromRaDec(raList, decList,
                                              obs_metadata=obs,
                                              epoch=2000.0)

        names1 = chipNameFromRaDec(raList, decList,
                                   obs_metadata=obs,
                                   epoch=2000.0,
                                   camera=self.camera)

        names2 = _chipNameFromRaDec(np.radians(raList), np.radians(decList),
                                    obs_metadata=obs,
                                    epoch=2000.0,
                                    camera=self.camera)

        names3 = chipNameFromPupilCoords(xpList, ypList, camera=self.camera)

        np.testing.assert_array_equal(names1, names2)
        np.testing.assert_array_equal(names1, names3)

        isNone = 0
        isNotNone = 0
        for name in names1:
            if name is None:
                isNone += 1
            else:
                isNotNone += 1

        self.assertGreater(isNotNone, 0)

    def testExceptions(self):
        """
        Test that exceptions are raised when they should be
        """

        nStars = 10
        xpList = self.rng.random_sample(nStars)*0.1
        ypList = self.rng.random_sample(nStars)*0.1

        obs = ObservationMetaData(pointingRA=25.0, pointingDec=112.0, mjd=42351.0,
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
        self.assertIn("The arg xPupil", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(list(xpList), ypList, obs_metadata=obs, epoch=2000.0)
        self.assertIn("The arg ra", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpList, list(ypList))
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("yPupil", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpList, list(ypList), obs_metadata=obs, epoch=2000.0)
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("Dec", context.exception.args[0])

        # do not need to run the above test on chipNameFromRaDec because
        # the conversion from degrees to radians that happens inside that
        # method automatically casts lists as numpy arrays

        # verify that an exception is raised if the two coordinate arrays contain
        # different numbers of elements
        xpDummy = self.rng.random_sample(nStars/2)

        with self.assertRaises(RuntimeError) as context:
            chipNameFromPupilCoords(xpDummy, ypList, camera=self.camera)

        self.assertEqual(context.exception.message,
                         "The arrays input to chipNameFromPupilCoords all need "
                         "to have the same length")

        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDec(xpDummy, ypList, obs_metadata=obs, epoch=2000.0,
                              camera=self.camera)

        self.assertEqual(context.exception.message,
                         "The arrays input to chipNameFromRaDec all need to have the same length")

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDec(xpDummy, ypList, obs_metadata=obs, epoch=2000.0,
                               camera=self.camera)

        self.assertEqual(context.exception.message,
                         "The arrays input to chipNameFromRaDec all need to have the same length")

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
        obsDummy = ObservationMetaData(pointingRA=25.0, pointingDec=-112.0,
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
        obsDummy = ObservationMetaData(pointingRA=25.0, pointingDec=-112.0,
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
        rotSkyPos = 135.0
        mjd = 42350.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=mjd, rotSkyPos=rotSkyPos)

        for badVal in [np.NaN, None]:

            raList = (self.rng.random_sample(nStars)-0.5)*5.0/3600.0 + ra0
            decList = (self.rng.random_sample(nStars)-0.5)*5.0/3600.0 + dec0

            raList[5] = badVal
            raList[10] = badVal
            decList[10] = badVal
            decList[25] = badVal

            xpList, ypList = pupilCoordsFromRaDec(raList, decList,
                                                  obs_metadata=obs,
                                                  epoch=2000.0)

            names1 = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                       camera=self.camera)

            names2 = _chipNameFromRaDec(np.radians(raList), np.radians(decList),
                                        obs_metadata=obs, epoch=2000.0, camera=self.camera)

            names3 = chipNameFromPupilCoords(xpList, ypList, camera=self.camera)

            np.testing.assert_array_equal(names1, names2)
            np.testing.assert_array_equal(names1, names3)

            for ix in range(len(names1)):
                if ix != 5 and ix != 10 and ix != 25:
                    self.assertEqual(names1[ix], 'Det22')
                    self.assertEqual(names2[ix], 'Det22')
                    self.assertEqual(names3[ix], 'Det22')
                else:
                    self.assertIsNone(names1[ix], None)
                    self.assertIsNone(names2[ix], None)
                    self.assertIsNone(names3[ix], None)

    def testPassingFloats(self):
        """
        Test that you can pass floats of RA, Dec into chipNameFromRaDec.

        Ditto for chipNameFromPupilCoords
        """

        ra0 = 45.0
        dec0 = -112.0
        rotSkyPos = 135.0
        mjd = 42350.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=mjd, rotSkyPos=rotSkyPos)

        nStars = 100
        raList = (self.rng.random_sample(nStars)-0.5)*500.0/3600.0 + ra0
        decList = (self.rng.random_sample(nStars)-0.5)*500.0/3600.0 + dec0

        chipNameList = chipNameFromRaDec(raList, decList, camera=self.camera, obs_metadata=obs)

        n_not_none = 0
        # now iterate over the list of RA, Dec to make sure that the same name comes back
        for ix, (rr, dd) in enumerate(zip(raList, decList)):
            test_name = chipNameFromRaDec(rr, dd, camera=self.camera, obs_metadata=obs)
            self.assertIsInstance(rr, np.float)
            self.assertIsInstance(dd, np.float)
            self.assertEqual(chipNameList[ix], test_name)
            if test_name is not None:
                self.assertIsInstance(test_name, str)
                n_not_none += 1

        self.assertGreater(n_not_none, 50)

        # try it with pupil coordinates
        n_not_none = 0
        xpList, ypList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs)
        chipNameList = chipNameFromPupilCoords(xpList, ypList, camera=self.camera)
        for ix, (xp, yp) in enumerate(zip(xpList, ypList)):
            test_name = chipNameFromPupilCoords(xp, yp, camera=self.camera)
            self.assertIsInstance(xp, np.float)
            self.assertIsInstance(yp, np.float)
            self.assertEqual(chipNameList[ix], test_name)
            if test_name is not None:
                self.assertIsInstance(test_name, str)
                n_not_none += 1

        self.assertGreater(n_not_none, 50)


class PixelCoordTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def setUp(self):
        self.rng = np.random.RandomState(11324)

    def testConsistency(self):
        """
        Test that all of the pixelCoord calculation methods agree with
        each other
        """
        ra0 = 95.0
        dec0 = -33.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=52350.0, rotSkyPos=27.0)

        nStars = 100
        raList = (self.rng.random_sample(nStars)-0.5)*500.0/3600.0 + ra0
        decList = (self.rng.random_sample(nStars)-0.5)*500.0/3600.0 + dec0

        xpList, ypList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0)

        chipNameList = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                         camera=self.camera)

        for includeDistortion in [True, False]:

            xx1, yy1 = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                            camera=self.camera, includeDistortion=includeDistortion)

            xx2, yy2 = _pixelCoordsFromRaDec(np.radians(raList), np.radians(decList),
                                             obs_metadata=obs, epoch=2000.0,
                                             camera=self.camera, includeDistortion=includeDistortion)

            xx3, yy3 = pixelCoordsFromPupilCoords(xpList, ypList, camera=self.camera,
                                                  includeDistortion=includeDistortion)

            xx4, yy4 = pixelCoordsFromPupilCoords(xpList, ypList, chipName=chipNameList,
                                                  camera=self.camera,
                                                  includeDistortion=includeDistortion)

            xx5, yy5 = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                            camera=self.camera, includeDistortion=includeDistortion,
                                            chipName=chipNameList)

            xx6, yy6 = _pixelCoordsFromRaDec(np.radians(raList), np.radians(decList),
                                             obs_metadata=obs, epoch=2000.0,
                                             camera=self.camera, includeDistortion=includeDistortion,
                                             chipName=chipNameList)

            np.testing.assert_array_equal(xx1, xx2)
            np.testing.assert_array_equal(xx1, xx3)
            np.testing.assert_array_equal(xx1, xx4)
            np.testing.assert_array_equal(xx1, xx5)
            np.testing.assert_array_equal(xx1, xx6)

            np.testing.assert_array_equal(yy1, yy2)
            np.testing.assert_array_equal(yy1, yy3)
            np.testing.assert_array_equal(yy1, yy4)
            np.testing.assert_array_equal(yy1, yy5)
            np.testing.assert_array_equal(yy1, yy6)

            # make sure that objects which do not fall on a chip
            # get NaN pixel coords
            ctNaN = 0
            ctNotNaN = 0
            for x, y, name in zip(xx1, yy1, chipNameList):
                if name is None:
                    np.testing.assert_equal(x, np.NaN)
                    np.testing.assert_equal(y, np.NaN)
                    ctNaN += 1
                else:
                    self.assertFalse(np.isnan(x), msg='x is Nan; should not be')
                    self.assertFalse(np.isnan(y), msg='y is Nan; should not be')
                    ctNotNaN += 1

            self.assertGreater(ctNaN, 0)
            self.assertGreater(ctNotNaN, 0)

            # now test that passing in the points one at a time gives consistent results
            for ix in range(len(raList)):
                x_f, y_f = pixelCoordsFromRaDec(raList[ix], decList[ix], obs_metadata=obs,
                                                epoch=2000.0, camera=self.camera,
                                                includeDistortion=includeDistortion)
                self.assertIsInstance(x_f, np.float)
                self.assertIsInstance(y_f, np.float)
                if not np.isnan(x_f):
                    self.assertEqual(x_f, xx1[ix])
                    self.assertEqual(y_f, yy1[ix])
                else:
                    np.testing.assert_equal(xx1[ix], np.NaN)
                    np.testing.assert_equal(yy1[ix], np.NaN)

                x_f, y_f = pixelCoordsFromRaDec(raList[ix], decList[ix], obs_metadata=obs,
                                                epoch=2000.0, camera=self.camera,
                                                includeDistortion=includeDistortion,
                                                chipName=chipNameList[ix])
                self.assertIsInstance(x_f, np.float)
                self.assertIsInstance(y_f, np.float)
                if not np.isnan(x_f):
                    self.assertEqual(x_f, xx1[ix])
                    self.assertEqual(y_f, yy1[ix])
                else:
                    np.testing.assert_equal(xx1[ix], np.NaN)
                    np.testing.assert_equal(yy1[ix], np.NaN)

                x_f, y_f = pixelCoordsFromRaDec(raList[ix], decList[ix], obs_metadata=obs,
                                                epoch=2000.0, camera=self.camera,
                                                includeDistortion=includeDistortion,
                                                chipName=[chipNameList[ix]])
                self.assertIsInstance(x_f, np.float)
                self.assertIsInstance(y_f, np.float)
                if not np.isnan(x_f):
                    self.assertEqual(x_f, xx1[ix])
                    self.assertEqual(y_f, yy1[ix])
                else:
                    np.testing.assert_equal(xx1[ix], np.NaN)
                    np.testing.assert_equal(yy1[ix], np.NaN)

    def testSingleChipName(self):
        """
        Test that pixelCoordsFromRaDec works when a list of RA, Dec are passed in,
        but only one chipName
        """
        ra0 = 95.0
        dec0 = -33.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=52350.0, rotSkyPos=27.0)

        nStars = 100
        raList = (self.rng.random_sample(nStars)-0.5)*500.0/3600.0 + ra0
        decList = (self.rng.random_sample(nStars)-0.5)*500.0/3600.0 + dec0

        xpList, ypList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0)

        chipNameList = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                         camera=self.camera)

        chosen_chip = 'Det40'
        valid_pts = np.where(chipNameList == chosen_chip)[0]
        self.assertGreater(len(valid_pts), 1)
        xPixControl, yPixControl = pixelCoordsFromRaDec(raList[valid_pts], decList[valid_pts],
                                                        obs_metadata=obs,
                                                        includeDistortion=True,
                                                        camera=self.camera)

        xPixTest, yPixTest = pixelCoordsFromRaDec(raList[valid_pts], decList[valid_pts],
                                                  obs_metadata=obs,
                                                  includeDistortion=True,
                                                  camera=self.camera,
                                                  chipName=chosen_chip)

        np.testing.assert_array_almost_equal(xPixControl, xPixTest, 12)
        np.testing.assert_array_almost_equal(yPixControl, yPixTest, 12)

        xPixTest, yPixTest = pixelCoordsFromRaDec(raList[valid_pts], decList[valid_pts],
                                                  obs_metadata=obs,
                                                  includeDistortion=True,
                                                  camera=self.camera,
                                                  chipName=[chosen_chip])

        np.testing.assert_array_almost_equal(xPixControl, xPixTest, 12)
        np.testing.assert_array_almost_equal(yPixControl, yPixTest, 12)

        # test raDecFromPixelCoords
        raTest, decTest = raDecFromPixelCoords(xPixControl, yPixControl, chosen_chip,
                                               camera=self.camera, obs_metadata=obs,
                                               includeDistortion=True)

        distance = arcsecFromRadians(haversine(np.radians(raList[valid_pts]),
                                               np.radians(decList[valid_pts]),
                                               np.radians(raTest), np.radians(decTest)))

        self.assertLess(distance.max(), 0.004)  # because of the imprecision in
                                                # _icrsFromObserved, this is the best we can do

        raTest, decTest = raDecFromPixelCoords(xPixControl, yPixControl, [chosen_chip],
                                               camera=self.camera, obs_metadata=obs,
                                               includeDistortion=True)

        distance = arcsecFromRadians(haversine(np.radians(raList[valid_pts]),
                                               np.radians(decList[valid_pts]),
                                               np.radians(raTest), np.radians(decTest)))

        self.assertLess(distance.max(), 0.004)

    def testExceptions(self):
        """
        Test that pixelCoord calculation methods raise exceptions when
        they should
        """
        nPoints = 100
        xpList = self.rng.random_sample(nPoints)*np.radians(1.0)
        ypList = self.rng.random_sample(nPoints)*np.radians(1.0)
        obs = ObservationMetaData(pointingRA=25.0,
                                  pointingDec=-36.0,
                                  rotSkyPos=122.0,
                                  mjd=41325.0)

        raList = self.rng.random_sample(nPoints)*1.0+25.0
        decList = self.rng.random_sample(nPoints)*1.0-36.0

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
            _pixelCoordsFromRaDec(np.radians(raList),
                                  np.radians(decList),
                                  obs_metadata=obs,
                                  epoch=2000.0)

        self.assertEqual(context.exception.message,
                         'Camera not specified.  Cannot calculate pixel coordinates.')

        # test that an exception is raised when you pass in something
        # that is not a numpy array
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(list(xpList), ypList,
                                       camera=self.camera)

        self.assertIn("The arg xPupil", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, list(ypList),
                                       camera=self.camera)

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("yPupil", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(list(np.radians(raList)),
                                  np.radians(decList),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)

        self.assertIn("The arg ra", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(np.radians(raList),
                                  list(np.radians(decList)),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)

        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("dec", context.exception.args[0])

        # do not need to run the above test on pixelCoordsFromRaDec,
        # because the conversion from degrees to radians  that happens
        # inside that method automatically casts lists as numpy arrays

        # test that an exception is raised if you pass in mis-matched
        # input arrays
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList[0:10],
                                       camera=self.camera)

        self.assertEqual(context.exception.args[0],
                         "The arrays input to pixelCoordsFromPupilCoords "
                         "all need to have the same length")

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList[0:10],
                                 obs_metadata=obs,
                                 epoch=2000.0,
                                 camera=self.camera)

        self.assertEqual(context.exception.args[0],
                         "The arrays input to pixelCoordsFromRaDec all need "
                         "to have the same length")

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(np.radians(raList),
                                  np.radians(decList[0:10]),
                                  obs_metadata=obs,
                                  epoch=2000.0,
                                  camera=self.camera)

        self.assertEqual(context.exception.args[0],
                         "The arrays input to pixelCoordsFromRaDec all need "
                         "to have the same length")

        # test that an error is raised if you pass an incorrect
        # number of chipNames to pixelCoordsFromPupilCoords
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromPupilCoords(xpList, ypList, chipName=['Det22']*10,
                                       camera=self.camera)

        self.assertIn("You passed 10 chipNames", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList, chipName=['Det22']*10,
                                 camera=self.camera,
                                 obs_metadata=obs,
                                 epoch=2000.0)

        self.assertIn("You passed 10 chipNames", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(np.radians(raList),
                                  np.radians(decList),
                                  chipName=['Det22']*10,
                                  camera=self.camera,
                                  obs_metadata=obs,
                                  epoch=2000.0)

        self.assertIn("You passed 10 chipNames", context.exception.args[0])

        # test that an exception is raised if you call one of the
        # pixelCoordsFromRaDec methods without an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera, epoch=2000.0)

        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData into '
                         'pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera, epoch=2000.0)

        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData into '
                         'pixelCoordsFromRaDec')

        # test that an exception is raised if you try to use an
        # ObservationMetaData without an mjd
        obsDummy = ObservationMetaData(pointingRA=25.0,
                                       pointingDec=-36.0,
                                       rotSkyPos=112.0)

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera,
                                 epoch=2000.0,
                                 obs_metadata=obsDummy)

        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData '
                         'with an mjd into pixelCoordsFromRaDec')

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera,
                                  epoch=2000.0,
                                  obs_metadata=obsDummy)

        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData '
                         'with an mjd into pixelCoordsFromRaDec')

        # test that an exception is raised if you try to use an
        # ObservationMetaData without a rotSkyPos
        obsDummy = ObservationMetaData(pointingRA=25.0,
                                       pointingDec=-36.0,
                                       mjd=53000.0)

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDec(raList, decList,
                                 camera=self.camera,
                                 epoch=2000.0,
                                 obs_metadata=obsDummy)

        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData '
                         'with a rotSkyPos into pixelCoordsFromRaDec')

        obsDummy = ObservationMetaData(pointingRA=25.0,
                                       pointingDec=-36.0,
                                       mjd=53000.0)
        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDec(raList, decList,
                                  camera=self.camera,
                                  epoch=2000.0,
                                  obs_metadata=obsDummy)

        self.assertEqual(context.exception.message,
                         'You need to pass an ObservationMetaData '
                         'with a rotSkyPos into pixelCoordsFromRaDec')

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

        # list a bunch of detector centers in radians
        x22 = 0.0
        y22 = 0.0

        x32 = radiansFromArcsec(40000.0 * arcsecPerMicron)
        y32 = 0.0

        x40 = radiansFromArcsec(80000.0 * arcsecPerMicron)
        y40 = radiansFromArcsec(-80000.0 * arcsecPerMicron)

        # assemble a bunch of displacements in pixels
        dxPixList = []
        dyPixList = []
        for xx in np.arange(-1999.0, 1999.0, 500.0):
            for yy in np.arange(-1999.0, 1999.0, 500.0):
                dxPixList.append(xx)
                dyPixList.append(yy)

        dxPixList = np.array(dxPixList)
        dyPixList = np.array(dyPixList)

        # convert to raidans
        dxPupList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyPupList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxPupList
        yPupList = y22 + dyPupList
        xPupList = np.append(xPupList, x32 + dxPupList)
        yPupList = np.append(yPupList, y32 + dyPupList)
        xPupList = np.append(xPupList, x40 + dxPupList)
        yPupList = np.append(yPupList, y40 + dyPupList)

        # this is what the chipNames ought to be for these points
        chipNameControl = np.array(['Det22'] * len(dxPupList))
        chipNameControl = np.append(chipNameControl, ['Det32'] * len(dxPupList))
        chipNameControl = np.append(chipNameControl, ['Det40'] * len(dxPupList))

        chipNameTest = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)

        # verify that the test points fall on the expected chips
        np.testing.assert_array_equal(chipNameControl, chipNameTest)

        # Note, the somewhat backwards way in which we go from dxPixList to
        # xPixControl is due to the fact that pixel coordinates are actually
        # aligned so that the x-axis is along the read-out direction, which
        # makes positive x in pixel coordinates correspond to positive y
        # in pupil coordinates
        xPixControl = 1999.5 + dyPixList
        yPixControl = 1999.5 - dxPixList
        xPixControl = np.append(xPixControl, 1999.5 + dyPixList)
        yPixControl = np.append(yPixControl, 1999.5 - dxPixList)
        xPixControl = np.append(xPixControl, 1999.5 + dyPixList)
        yPixControl = np.append(yPixControl, 1999.5 - dxPixList)

        # verify that the pixel coordinates are as expected to within 0.01 pixel
        xPixTest, yPixTest = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera,
                                                        includeDistortion=False)

        np.testing.assert_array_almost_equal(xPixTest, xPixControl, 2)
        np.testing.assert_array_almost_equal(yPixTest, yPixControl, 2)

        # now test that we get the same results when we pass the pupil coords in
        # one at a time
        for ix in range(len(xPupList)):
            xpx_f, ypx_f = pixelCoordsFromPupilCoords(xPupList[ix], yPupList[ix],
                                                      camera=self.camera,
                                                      includeDistortion=False)
            self.assertIsInstance(xpx_f, np.float)
            self.assertIsInstance(ypx_f, np.float)
            self.assertAlmostEqual(xpx_f, xPixTest[ix], 12)
            self.assertAlmostEqual(ypx_f, yPixTest[ix], 12)

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

        # list a bunch of detector centers in radians
        x22 = 0.0
        y22 = 0.0

        x32 = radiansFromArcsec(40000.0 * arcsecPerMicron)
        y32 = 0.0

        x40 = radiansFromArcsec(80000.0 * arcsecPerMicron)
        y40 = radiansFromArcsec(-80000.0 * arcsecPerMicron)

        # assemble a bunch of displacements in pixels
        dxPixList = []
        dyPixList = []
        for xx in np.arange(-1999.0, 1999.0, 500.0):
            for yy in np.arange(-1999.0, 1999.0, 500.0):
                dxPixList.append(xx)
                dyPixList.append(yy)

        dxPixList = np.array(dxPixList)
        dyPixList = np.array(dyPixList)

        # convert to raidans
        dxPupList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyPupList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxPupList
        yPupList = y22 + dyPupList
        xPupList = np.append(xPupList, x32 + dxPupList)
        yPupList = np.append(yPupList, y32 + dyPupList)
        xPupList = np.append(xPupList, x40 + dxPupList)
        yPupList = np.append(yPupList, y40 + dyPupList)

        # this is what the chipNames ought to be for these points
        chipNameControl = np.array(['Det22'] * len(dxPupList))
        chipNameControl = np.append(chipNameControl, ['Det32'] * len(dxPupList))
        chipNameControl = np.append(chipNameControl, ['Det40'] * len(dxPupList))

        chipNameTest = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)

        # verify that the test points fall on the expected chips
        np.testing.assert_array_equal(chipNameControl, chipNameTest)

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
                                                        chipName=inputChipNames)

        np.testing.assert_array_almost_equal(xPixTest, xPixControl, 2)
        np.testing.assert_array_almost_equal(yPixTest, yPixControl, 2)

        # now test that we get the same results when we pass the pupil coords in
        # one at a time
        for ix in range(len(xPupList)):
            xpx_f, ypx_f = pixelCoordsFromPupilCoords(xPupList[ix], yPupList[ix],
                                                      camera=self.camera,
                                                      includeDistortion=False,
                                                      chipName=inputChipNames[ix])
            self.assertIsInstance(xpx_f, np.float)
            self.assertIsInstance(ypx_f, np.float)
            self.assertAlmostEqual(xpx_f, xPixTest[ix], 12)
            self.assertAlmostEqual(ypx_f, yPixTest[ix], 12)

        # We will now use this opportunity to test that pupilCoordsFromPixelCoords
        # and pixelCoordsFromPupilCoords work when we pass in a list of pixel
        # coords, but only one chip name
        xPix_one, yPix_one = pixelCoordsFromPupilCoords(xPupList, yPupList,
                                                        camera=self.camera,
                                                        includeDistortion=False,
                                                        chipName='Det40')

        np.testing.assert_array_almost_equal(xPix_one, xPixTest, 12)
        np.testing.assert_array_almost_equal(yPix_one, yPixTest, 12)

        xPix_one, yPix_one = pixelCoordsFromPupilCoords(xPupList, yPupList,
                                                        camera=self.camera,
                                                        includeDistortion=False,
                                                        chipName=['Det40'])

        np.testing.assert_array_almost_equal(xPix_one, xPixTest, 12)
        np.testing.assert_array_almost_equal(yPix_one, yPixTest, 12)

        xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPixTest, yPixTest, 'Det40',
                                                        camera=self.camera,
                                                        includeDistortion=False)

        np.testing.assert_array_almost_equal(xPupTest, xPupList, 12)
        np.testing.assert_array_almost_equal(yPupTest, yPupList, 12)

        xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPixTest, yPixTest, ['Det40'],
                                                        camera=self.camera,
                                                        includeDistortion=False)

        np.testing.assert_array_almost_equal(xPupTest, xPupList, 12)
        np.testing.assert_array_almost_equal(yPupTest, yPupList, 12)

    def testNaN(self):
        """
        Verify that NaNs and Nones input to pixelCoordinate calculation methods result
        in NaNs coming out
        """
        ra0 = 25.0
        dec0 = -35.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  rotSkyPos=42.0, mjd=42356.0)

        raList = self.rng.random_sample(100)*100.0/3600.0 + ra0
        decList = self.rng.random_sample(100)*100.0/3600.0 + dec0
        chipNameList = chipNameFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0,
                                         camera=self.camera)

        # make sure that all of the test points actually fall on chips
        for name in chipNameList:
            self.assertIsNotNone(name)

        xPupList, yPupList = pupilCoordsFromRaDec(raList, decList, obs_metadata=obs, epoch=2000.0)

        # make sure that none of the test points already result in NaN pixel coordinates
        xPixList, yPixList = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs,
                                                  epoch=2000.0, camera=self.camera)

        for xx, yy in zip(xPixList, yPixList):
            self.assertFalse(np.isnan(xx), msg='xx is NaN; should not be')
            self.assertFalse(np.isnan(yy), msg='yy is NaN; should not be')
            self.assertIsNotNone(xx, None)
            self.assertIsNotNone(yy, None)

        for badVal in [np.NaN, None]:
            raList[5] = badVal
            decList[7] = badVal
            raList[9] = badVal
            decList[9] = badVal

            xPixList, yPixList = pixelCoordsFromRaDec(raList, decList, obs_metadata=obs,
                                                      epoch=2000.0, camera=self.camera)

            for ix, (xx, yy) in enumerate(zip(xPixList, yPixList)):
                if ix in [5, 7, 9]:
                    np.testing.assert_equal(xx, np.NaN)
                    np.testing.assert_equal(yy, np.NaN)
                else:
                    self.assertFalse(np.isnan(xx), msg='xx is NaN; should not be')
                    self.assertFalse(np.isnan(yy), msg='yy is NaN; should not be')
                    self.assertIsNotNone(xx)
                    self.assertIsNotNone(yy)

            xPixList, yPixList = _pixelCoordsFromRaDec(np.radians(raList), np.radians(decList),
                                                       obs_metadata=obs, epoch=2000.0, camera=self.camera)

            for ix, (xx, yy) in enumerate(zip(xPixList, yPixList)):
                if ix in [5, 7, 9]:
                    np.testing.assert_equal(xx, np.NaN)
                    np.testing.assert_equal(yy, np.NaN)
                else:
                    self.assertFalse(np.isnan(xx), msg='xx is NaN; should not be')
                    self.assertFalse(np.isnan(yy), msg='yy is NaN; should not be')
                    self.assertIsNotNone(xx)
                    self.assertIsNotNone(yy)

            xPupList[5] = badVal
            yPupList[7] = badVal
            xPupList[9] = badVal
            yPupList[9] = badVal
            xPixList, yPixList = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)
            for ix, (xx, yy) in enumerate(zip(xPixList, yPixList)):

                # verify the same result if we had passed in pupil coords one-at-a-time
                xpx_f, ypx_f = pixelCoordsFromPupilCoords(xPupList[ix], yPupList[ix], camera=self.camera)
                self.assertIsInstance(xpx_f, np.float)
                self.assertIsInstance(ypx_f, np.float)

                if ix in [5, 7, 9]:
                    np.testing.assert_equal(xx, np.NaN)
                    np.testing.assert_equal(yy, np.NaN)
                    np.testing.assert_equal(xpx_f, np.NaN)
                    np.testing.assert_equal(ypx_f, np.NaN)
                else:
                    self.assertFalse(np.isnan(xx), msg='xx is NaN; should not be')
                    self.assertFalse(np.isnan(yy), msg='yy is NaN; should not be')
                    self.assertIsNotNone(xx)
                    self.assertIsNotNone(yy)
                    self.assertAlmostEqual(xx, xpx_f, 12)
                    self.assertAlmostEqual(yy, ypx_f, 12)

    def testDistortion(self):
        """
        Make sure that the results from pixelCoordsFromPupilCoords are different
        if includeDistortion is True as compared to if includeDistortion is False

        Note: This test passes because the test camera has a pincushion distortion.
        If we take that away, the test will no longer pass.
        """
        xp = radiansFromArcsec((self.rng.random_sample(100)-0.5)*100.0)
        yp = radiansFromArcsec((self.rng.random_sample(100)-0.5)*100.0)

        xu, yu = pixelCoordsFromPupilCoords(xp, yp, camera=self.camera, includeDistortion=False)
        xd, yd = pixelCoordsFromPupilCoords(xp, yp, camera=self.camera, includeDistortion=True)

        # just verify that the distorted versus undistorted coordinates vary in the
        # 4th decimal place
        self.assertRaises(AssertionError,
                          np.testing.assert_array_almost_equal, xu, xd, 4)

        self.assertRaises(AssertionError,
                          np.testing.assert_array_almost_equal, yu, yd, 4)

        # make sure that distortions are also present when we pass pupil coordinates in
        # one-at-a-time
        for ix in range(len(xp)):
            x_f, y_f = pixelCoordsFromPupilCoords(xp[ix], yp[ix], camera=self.camera,
                                                  includeDistortion=True)

            self.assertAlmostEqual(xd[ix], x_f, 12)
            self.assertAlmostEqual(yd[ix], y_f, 12)
            self.assertIsInstance(x_f, np.float)
            self.assertIsInstance(y_f, np.float)


class FocalPlaneCoordTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def setUp(self):
        self.rng = np.random.RandomState(8374522)

    def testConsistency(self):
        """
        Test that all of the focalPlaneCoord calculation methods
        return self-consistent answers.
        """

        ra0 = 34.1
        dec0 = -23.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=43257.0, rotSkyPos = 127.0)

        raCenter, decCenter = observedFromICRS(np.array([ra0]),
                                               np.array([dec0]),
                                               obs_metadata=obs,
                                               epoch=2000.0)

        nStars = 100
        raList = self.rng.random_sample(nStars)*1000.0/3600.0 + raCenter[0]
        decList = self.rng.random_sample(nStars)*1000.0/3600.0 + decCenter[0]

        xPupList, yPupList = pupilCoordsFromRaDec(raList, decList,
                                                  obs_metadata=obs,
                                                  epoch=2000.0)

        xf1, yf1 = focalPlaneCoordsFromRaDec(raList, decList,
                                             obs_metadata=obs,
                                             epoch=2000.0, camera=self.camera)

        xf2, yf2 = _focalPlaneCoordsFromRaDec(np.radians(raList),
                                              np.radians(decList),
                                              obs_metadata=obs,
                                              epoch=2000.0, camera=self.camera)

        xf3, yf3 = focalPlaneCoordsFromPupilCoords(xPupList, yPupList,
                                                   camera=self.camera)

        np.testing.assert_array_equal(xf1, xf2)
        np.testing.assert_array_equal(xf1, xf3)
        np.testing.assert_array_equal(yf1, yf2)
        np.testing.assert_array_equal(yf1, yf3)

        for x, y in zip(xf1, yf1):
            self.assertFalse(np.isnan(x), msg='x is NaN; should not be')
            self.assertIsNotNone(x)
            self.assertFalse(np.isnan(y), msg='y is NaN; should not be')
            self.assertIsNotNone(y)

        # now test that focalPlaneCoordsFromRaDec and
        # focalPlaneCoordsFromPupilCoords give the same results
        # when you pass the inputs in one-by-one
        for ix in range(len(xf1)):
            x_f, y_f = focalPlaneCoordsFromRaDec(raList[ix], decList[ix],
                                                 camera=self.camera,
                                                 obs_metadata=obs, epoch=2000.0)
            self.assertIsInstance(x_f, float)
            self.assertIsInstance(y_f, float)
            self.assertEqual(x_f, xf1[ix])
            self.assertEqual(y_f, yf1[ix])

            x_f, y_f = focalPlaneCoordsFromPupilCoords(xPupList[ix], yPupList[ix],
                                                       camera=self.camera)
            self.assertIsInstance(x_f, float)
            self.assertIsInstance(y_f, float)
            self.assertEqual(x_f, xf1[ix])
            self.assertEqual(y_f, yf1[ix])

    def testExceptions(self):
        """
        Test that the focalPlaneCoord methods raise the exceptions
        (with the correct messages) when they should.
        """

        ra0 = 34.0
        dec0 = -19.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  rotSkyPos=61.0, mjd=52349.0)

        nStars = 10
        raList = (self.rng.random_sample(nStars)-0.5) + ra0
        decList = (self.rng.random_sample(nStars)-0.5) + dec0
        xPupList, yPupList = pupilCoordsFromRaDec(raList, decList,
                                                  obs_metadata=obs,
                                                  epoch=2000.0)

        # verify that an error is raised when you forget to pass
        # in a camera
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(xPupList, yPupList)
        self.assertEqual(context.exception.message,
                         "You cannot calculate focal plane coordinates "
                         "without specifying a camera")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obs,
                                               epoch=2000.0)

        self.assertEqual(context.exception.message,
                         "You cannot calculate focal plane coordinates "
                         "without specifying a camera")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obs,
                                                epoch=2000.0)

        self.assertEqual(context.exception.message,
                         "You cannot calculate focal plane coordinates "
                         "without specifying a camera")

        # test that an error is raised when you pass in something that
        # is not a numpy array
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(list(xPupList), yPupList,
                                                     camera=self.camera)
        self.assertIn("The arg xPupil", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(xPupList, list(yPupList),
                                                     camera=self.camera)
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("yPupil", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(list(raList), decList,
                                                obs_metadata=obs,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertIn("The arg ra", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, list(decList),
                                                obs_metadata=obs,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("dec", context.exception.args[0])

        # we do not have to run the test above on focalPlaneCoordsFromRaDec
        # because the conversion to radians automatically casts lists into
        # numpy arrays

        # test that an error is raised if you pass in mismatched numbers
        # of x and y coordinates
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromPupilCoords(xPupList, yPupList[0:4],
                                                     camera=self.camera)
        self.assertEqual(context.exception.args[0],
                         "The arrays input to focalPlaneCoordsFromPupilCoords "
                         "all need to have the same length")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList[0:4],
                                               obs_metadata=obs,
                                               epoch=2000.0,
                                               camera=self.camera)
        self.assertEqual(context.exception.args[0],
                         "The arrays input to focalPlaneCoordsFromRaDec "
                         "all need to have the same length")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList[0:4],
                                                obs_metadata=obs,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.args[0],
                         "The arrays input to focalPlaneCoordsFromRaDec "
                         "all need to have the same length")

        # test that an error is raised if you call
        # focalPlaneCoordsFromRaDec without an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               epoch=2000.0,
                                               camera=self.camera)

        self.assertEqual(context.exception.message,
                         "You have to specify an ObservationMetaData to run "
                         "focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You have to specify an ObservationMetaData to run "
                         "focalPlaneCoordsFromRaDec")

        # test that an error is raised if you pass an ObservationMetaData
        # without an mjd into focalPlaneCoordsFromRaDec
        obsDummy = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                       rotSkyPos=112.0)
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obsDummy,
                                               epoch=2000.0,
                                               camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with an "
                         "mjd into focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obsDummy,
                                                epoch=2000.0,
                                                camera=self.camera)

        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with an "
                         "mjd into focalPlaneCoordsFromRaDec")

        # test that an error is raised if you pass an ObservationMetaData
        # without a rotSkyPos into focalPlaneCoordsFromRaDec
        obsDummy = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                       mjd=42356.0)
        with self.assertRaises(RuntimeError) as context:
            xf, yf = focalPlaneCoordsFromRaDec(raList, decList,
                                               obs_metadata=obsDummy,
                                               epoch=2000.0,
                                               camera=self.camera)

        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with a "
                         "rotSkyPos into focalPlaneCoordsFromRaDec")

        with self.assertRaises(RuntimeError) as context:
            xf, yf = _focalPlaneCoordsFromRaDec(raList, decList,
                                                obs_metadata=obsDummy,
                                                epoch=2000.0,
                                                camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You need to pass an ObservationMetaData with a "
                         "rotSkyPos into focalPlaneCoordsFromRaDec")

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

        # list a bunch of detector centers in radians
        x22 = 0.0
        y22 = 0.0

        x32 = radiansFromArcsec(40000.0 * arcsecPerMicron)
        y32 = 0.0

        x40 = radiansFromArcsec(80000.0 * arcsecPerMicron)
        y40 = radiansFromArcsec(-80000.0 * arcsecPerMicron)

        # assemble a bunch of displacements in pixels
        dxPixList = []
        dyPixList = []
        for xx in np.arange(-1999.0, 1999.0, 500.0):
            for yy in np.arange(-1999.0, 1999.0, 500.0):
                dxPixList.append(xx)
                dyPixList.append(yy)

        dxPixList = np.array(dxPixList)
        dyPixList = np.array(dyPixList)

        # convert to raidans
        dxPupList = radiansFromArcsec(dxPixList*arcsecPerPixel)
        dyPupList = radiansFromArcsec(dyPixList*arcsecPerPixel)

        # assemble a bunch of test pupil coordinate pairs
        xPupList = x22 + dxPupList
        yPupList = y22 + dyPupList
        xPupList = np.append(xPupList, x32 + dxPupList)
        yPupList = np.append(yPupList, y32 + dyPupList)
        xPupList = np.append(xPupList, x40 + dxPupList)
        yPupList = np.append(yPupList, y40 + dyPupList)

        # this is what the chipNames ought to be for these points
        chipNameControl = np.array(['Det22'] * len(dxPupList))
        chipNameControl = np.append(chipNameControl, ['Det32'] * len(dxPupList))
        chipNameControl = np.append(chipNameControl, ['Det40'] * len(dxPupList))

        chipNameTest = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)

        # verify that the test points fall on the expected chips
        np.testing.assert_array_equal(chipNameControl, chipNameTest)

        # convert into millimeters on the focal plane
        xFocalControl = arcsecFromRadians(xPupList)*mmPerArcsec
        yFocalControl = arcsecFromRadians(yPupList)*mmPerArcsec

        xFocalTest, yFocalTest = focalPlaneCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)

        np.testing.assert_array_almost_equal(xFocalTest, xFocalControl, 3)
        np.testing.assert_array_almost_equal(yFocalTest, yFocalControl, 3)


class ConversionFromPixelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def setUp(self):
        self.rng = np.random.RandomState(543)

    def testPupCoordsException(self):
        """
        Test that pupilCoordsFromPixelCoords raises an exception when you
        call it without a camera
        """
        nStars = 100
        xPupList = radiansFromArcsec((self.rng.random_sample(nStars)-0.5)*320.0)
        yPupList = radiansFromArcsec((self.rng.random_sample(nStars)-0.5)*320.0)
        chipNameList = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)
        xPix, yPix = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)
        with self.assertRaises(RuntimeError) as context:
            xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList)
        self.assertEqual(context.exception.message,
                         "You cannot call pupilCoordsFromPixelCoords without specifying "
                         "a camera")

    def testPupCoordsResults(self):
        """
        Test that the results from pupilCoordsFromPixelCoords are consistent
        with the results from pixelCoordsFromPupilCoords
        """

        nStars = 100
        xPupList = radiansFromArcsec((self.rng.random_sample(nStars)-0.5)*320.0)
        yPupList = radiansFromArcsec((self.rng.random_sample(nStars)-0.5)*320.0)
        chipNameList = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)
        for includeDistortion in [True, False]:
            xPix, yPix = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera,
                                                    includeDistortion=includeDistortion)
            xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList, camera=self.camera,
                                                            includeDistortion=includeDistortion)

            dx = arcsecFromRadians(xPupTest-xPupList)
            np.testing.assert_array_almost_equal(dx, np.zeros(len(dx)), 9)
            dy = arcsecFromRadians(yPupTest-yPupList)
            np.testing.assert_array_almost_equal(dy, np.zeros(len(dy)), 9)

            ctNaN = 0
            for x, y in zip(xPupTest, yPupTest):
                if np.isnan(x) or np.isnan(y):
                    ctNaN += 1
            self.assertLess(ctNaN, len(xPupTest)/10)

            # test passing in pixel coordinates one at a time
            for ix in range(len(xPupList)):
                xp_f, yp_f = pupilCoordsFromPixelCoords(xPix[ix], yPix[ix], chipNameList[ix],
                                                        camera=self.camera,
                                                        includeDistortion=includeDistortion)

                self.assertIsInstance(xp_f, np.float)
                self.assertIsInstance(yp_f, np.float)
                self.assertAlmostEqual(xp_f, xPupTest[ix], 12)
                self.assertAlmostEqual(yp_f, yPupTest[ix], 12)

    def testPupCoordsNaN(self):
        """
        Test that points which do not have a chip return NaN for pupilCoordsFromPixelCoords
        """
        nStars = 10
        xPupList = radiansFromArcsec((self.rng.random_sample(nStars)-0.5)*320.0)
        yPupList = radiansFromArcsec((self.rng.random_sample(nStars)-0.5)*320.0)
        chipNameList = chipNameFromPupilCoords(xPupList, yPupList, camera=self.camera)
        chipNameList[5] = None
        xPix, yPix = pixelCoordsFromPupilCoords(xPupList, yPupList, camera=self.camera)
        xPupTest, yPupTest = pupilCoordsFromPixelCoords(xPix, yPix, chipNameList, camera=self.camera)
        np.testing.assert_equal(xPupTest[5], np.NaN)
        np.testing.assert_equal(yPupTest[5], np.NaN)

    def testRaDecExceptions(self):
        """
        Test that raDecFromPupilCoords raises exceptions when it is supposed to
        """
        nStars = 20
        ra0 = 45.0
        dec0 = -19.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=43525.0, rotSkyPos=145.0)

        xPixList = self.rng.random_sample(nStars)*4000.0
        yPixList = self.rng.random_sample(nStars)*4000.0

        chipDexList = self.rng.random_integers(0, len(self.camera)-1, nStars)
        chipNameList = [self.camera[self.camera._nameDetectorDict.keys()[ii]].getName() for ii in chipDexList]

        # test that an error is raised if you do not pass in a camera
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                           obs_metadata=obs, epoch=2000.0)
        self.assertEqual(context.exception.message,
                         "You cannot call raDecFromPixelCoords without specifying a camera")

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                            obs_metadata=obs, epoch=2000.0)
        self.assertEqual(context.exception.message,
                         "You cannot call raDecFromPixelCoords without specifying a camera")

        # test that an error is raised if you do not pass in an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                           epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You cannot call raDecFromPixelCoords without an ObservationMetaData")

        # test that an error is raised if you do not pass in an ObservationMetaData
        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                            epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         "You cannot call raDecFromPixelCoords without an ObservationMetaData")

        # test that an error is raised if you pass in an ObservationMetaData
        # without an mjd
        obsDummy = ObservationMetaData(pointingRA=ra0, pointingDec=dec0, rotSkyPos=95.0)
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                           obs_metadata=obsDummy,
                                           epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         "The ObservationMetaData in raDecFromPixelCoords must have an mjd")

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                            obs_metadata=obsDummy,
                                            epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         "The ObservationMetaData in raDecFromPixelCoords must have an mjd")

        # test that an error is raised if you pass in an ObservationMetaData
        # without a rotSkyPos
        obsDummy = ObservationMetaData(pointingRA=ra0, pointingDec=dec0, mjd=43243.0)
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                           obs_metadata=obsDummy,
                                           epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         "The ObservationMetaData in raDecFromPixelCoords must have a rotSkyPos")

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, yPixList, chipNameList,
                                            obs_metadata=obsDummy,
                                            epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.message,
                         "The ObservationMetaData in raDecFromPixelCoords must have a rotSkyPos")

        # test that an error is raised if you pass in lists of pixel coordinates,
        # rather than numpy arrays
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(list(xPixList), yPixList,
                                           chipNameList, obs_metadata=obs,
                                           epoch=2000.0, camera=self.camera)
        self.assertIn("The arg xPix", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, list(yPixList),
                                           chipNameList, obs_metadata=obs,
                                           epoch=2000.0, camera=self.camera)
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("yPix", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(list(xPixList), yPixList,
                                            chipNameList, obs_metadata=obs,
                                            epoch=2000.0, camera=self.camera)
        self.assertIn("The arg xPix", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, list(yPixList),
                                            chipNameList, obs_metadata=obs,
                                            epoch=2000.0, camera=self.camera)
        self.assertIn("The input arguments:", context.exception.args[0])
        self.assertIn("yPix", context.exception.args[0])

        # test that an error is raised if you pass in mismatched lists of
        # xPix and yPix
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, yPixList[0:13], chipNameList,
                                           obs_metadata=obs, epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.args[0],
                         "The arrays input to raDecFromPixelCoords all need to have the same length")

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, yPixList[0:13], chipNameList,
                                            obs_metadata=obs, epoch=2000.0, camera=self.camera)
        self.assertEqual(context.exception.args[0],
                         "The arrays input to raDecFromPixelCoords all need to have the same length")

        # test that an error is raised if you do not pass in the same number of chipNames
        # as pixel coordinates
        with self.assertRaises(RuntimeError) as context:
            ra, dec = raDecFromPixelCoords(xPixList, yPixList, ['Det22']*22,
                                           obs_metadata=obs, epoch=2000.0, camera=self.camera)
        self.assertIn("22 chipNames", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            ra, dec = _raDecFromPixelCoords(xPixList, yPixList, ['Det22']*22,
                                            obs_metadata=obs, epoch=2000.0, camera=self.camera)
        self.assertIn("22 chipNames", context.exception.args[0])

    def testResults(self):
        """
        Test that raDecFromPixelCoords results are consistent with
        pixelCoordsFromRaDec
        """
        nStars = 200
        ra0 = 45.0
        dec0 = -19.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=43525.0, rotSkyPos=145.0)

        xPixList = self.rng.random_sample(nStars)*4000.0
        yPixList = self.rng.random_sample(nStars)*4000.0

        chipDexList = self.rng.random_integers(0, len(self.camera)-1, nStars)
        chipNameList = [self.camera[self.camera._nameDetectorDict.keys()[ii]].getName() for ii in chipDexList]

        for includeDistortion in [True, False]:

            raDeg, decDeg = raDecFromPixelCoords(xPixList, yPixList, chipNameList, obs_metadata=obs,
                                                 epoch=2000.0, camera=self.camera,
                                                 includeDistortion=includeDistortion)

            raRad, decRad = _raDecFromPixelCoords(xPixList, yPixList, chipNameList, obs_metadata=obs,
                                                  epoch=2000.0, camera=self.camera,
                                                  includeDistortion=includeDistortion)

            # first, make sure that the radians and degrees methods agree with each other
            dRa = arcsecFromRadians(raRad-np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(len(raRad)), 9)
            dDec = arcsecFromRadians(decRad-np.radians(decDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(len(decRad)), 9)

            # now make sure that the results from raDecFromPixelCoords are consistent
            # with the results from pixelCoordsFromRaDec by taking the ra and dec
            # arrays found above and feeding them back into pixelCoordsFromRaDec
            # and seeing if we get the same results
            xPixTest, yPixTest = pixelCoordsFromRaDec(raDeg, decDeg, obs_metadata=obs,
                                                      epoch=2000.0, camera=self.camera,
                                                      includeDistortion=includeDistortion)

            distance = np.sqrt(np.power(xPixTest-xPixList, 2) + np.power(yPixTest-yPixList, 2))
            self.assertLess(distance.max(), 0.2)  # because of the imprecision in _icrsFromObserved,
                                                  # this is the best we can get; note that, in our test
                                                  # camera, each pixel is 10 microns in size and the
                                                  # plate scale is 2 arcsec per mm, so 0.2 pixels is
                                                  # 0.004 arcsec

            # test passing the pixel coordinates in one at a time
            for ix in range(len(xPixList)):
                ra_f, dec_f = raDecFromPixelCoords(xPixList[ix], yPixList[ix],
                                                   chipNameList[ix], obs_metadata=obs,
                                                   epoch=2000.0, camera=self.camera,
                                                   includeDistortion=includeDistortion)
                self.assertIsInstance(ra_f, np.float)
                self.assertIsInstance(dec_f, np.float)
                self.assertAlmostEqual(ra_f, raDeg[ix], 12)
                self.assertAlmostEqual(dec_f, decDeg[ix], 12)

    def testResultsOffChip(self):
        """
        Test that raDecFromPixelCoords results are consistent with
        pixelCoordsFromRaDec with chip names are specified

        Note: this is the same test as in testResults, except that
        we are going to intentionally make the pixel coordinate lists
        fall outside the boundaries of the chips defined in chipNameList
        """
        nStars = 200
        ra0 = 45.0
        dec0 = -19.0
        obs = ObservationMetaData(pointingRA=ra0, pointingDec=dec0,
                                  mjd=43525.0, rotSkyPos=145.0)

        xPixList = self.rng.random_sample(nStars)*4000.0 + 4000.0
        yPixList = self.rng.random_sample(nStars)*4000.0 + 4000.0

        chipDexList = self.rng.random_integers(0, len(self.camera)-1, nStars)
        chipNameList = [self.camera[self.camera._nameDetectorDict.keys()[ii]].getName() for ii in chipDexList]

        for includeDistortion in [True, False]:

            raDeg, decDeg = raDecFromPixelCoords(xPixList, yPixList, chipNameList, obs_metadata=obs,
                                                 epoch=2000.0, camera=self.camera,
                                                 includeDistortion=includeDistortion)

            raRad, decRad = _raDecFromPixelCoords(xPixList, yPixList, chipNameList, obs_metadata=obs,
                                                  epoch=2000.0, camera=self.camera,
                                                  includeDistortion=includeDistortion)

            # first, make sure that the radians and degrees methods agree with each other
            dRa = arcsecFromRadians(raRad-np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(len(raRad)), 9)
            dDec = arcsecFromRadians(decRad-np.radians(decDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(len(decRad)), 9)

            # now make sure that the results from raDecFromPixelCoords are consistent
            # with the results from pixelCoordsFromRaDec by taking the ra and dec
            # arrays found above and feeding them back into pixelCoordsFromRaDec
            # and seeing if we get the same results
            xPixTest, yPixTest = pixelCoordsFromRaDec(raDeg, decDeg, chipName=chipNameList,
                                                      obs_metadata=obs,
                                                      epoch=2000.0,
                                                      camera=self.camera,
                                                      includeDistortion=includeDistortion)

            distance = np.sqrt(np.power(xPixTest-xPixList, 2) + np.power(yPixTest-yPixList, 2))
            self.assertLess(distance.max(), 0.2)  # because of the imprecision in _icrsFromObserved,
                                                  # this is the best we can get; note that, in our
                                                  # test camera, each pixel is 10 microns in size and
                                                  # the plate scale is 2 arcsec per mm, so 0.2 pixels is
                                                  # 0.004 arcsec

    def testDistortion(self):
        """
        Make sure that the results from pupilCoordsFromPixelCoords are different
        if includeDistortion is True as compared to if includeDistortion is False

        Note: This test passes because the test camera has a pincushion distortion.
        If we take that away, the test will no longer pass.
        """
        nStars = 200
        xPixList = self.rng.random_sample(nStars)*4000.0 + 4000.0
        yPixList = self.rng.random_sample(nStars)*4000.0 + 4000.0

        chipDexList = self.rng.random_integers(0, len(self.camera)-1, nStars)
        chipNameList = [self.camera[self.camera._nameDetectorDict.keys()[ii]].getName() for ii in chipDexList]

        xu, yu = pupilCoordsFromPixelCoords(xPixList, yPixList, chipNameList, camera=self.camera,
                                            includeDistortion=False)

        xd, yd = pupilCoordsFromPixelCoords(xPixList, yPixList, chipNameList, camera=self.camera,
                                            includeDistortion=True)

        # just verify that the distorted versus undistorted coordinates vary in the 4th decimal
        self.assertRaises(AssertionError,
                          np.testing.assert_array_almost_equal,
                          arcsecFromRadians(xu),
                          arcsecFromRadians(xd),
                          4)

        self.assertRaises(AssertionError,
                          np.testing.assert_array_almost_equal,
                          arcsecFromRadians(yu),
                          arcsecFromRadians(yd),
                          4)


class CornerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def testCornerPixels(self):
        """
        Test the method to get the pixel coordinates of the corner
        of a detector
        """
        det_name = self.camera[0].getName()
        corners = getCornerPixels(det_name, self.camera)
        self.assertEqual(corners[0][0], 0)
        self.assertEqual(corners[0][1], 0)
        self.assertEqual(corners[1][0], 0)
        self.assertEqual(corners[1][1], 3999)
        self.assertEqual(corners[2][0], 3999)
        self.assertEqual(corners[2][1], 0)
        self.assertEqual(corners[3][0], 3999)
        self.assertEqual(corners[3][1], 3999)
        self.assertEqual(len(corners), 4)
        for row in corners:
            self.assertEqual(len(row), 2)

    def testCornerRaDec_radians(self):
        """
        Test that the method to return the Ra, Dec values of the corner
        of a chip (in radians) works by validating its results against
        known pixel corner values and the _raDecFromPixelCoords method,
        which is tested separately.
        """
        obs = ObservationMetaData(pointingRA=23.0, pointingDec=-65.0,
                                  rotSkyPos=52.1, mjd=59582.3)

        det_name = self.camera[4].getName()
        cornerTest = _getCornerRaDec(det_name, self.camera, obs)

        ra_control, dec_control = _raDecFromPixelCoords(np.array([0, 0, 3999, 3999]),
                                                        np.array([0, 3999, 0, 3999]),
                                                        [det_name]*4,
                                                        camera=self.camera,
                                                        obs_metadata=obs,
                                                        epoch=2000.0, includeDistortion=True)

        # Loop over the control values and the corner values, using
        # haversine() method to find the angular distance between the
        # test and control values.  Assert that they are within
        # 0.01 milli-arcsecond of each other
        for rr1, dd1, cc in zip(ra_control, dec_control, cornerTest):
            dd = haversine(rr1, dd1, cc[0], cc[1])
            self.assertLess(arcsecFromRadians(dd), 0.00001)

    def testCornerRaDec_degrees(self):
        """
        Test that method to get corner RA, Dec in degrees is consistent
        with method to get corner RA, Dec in radians
        """

        obs = ObservationMetaData(pointingRA=31.0, pointingDec=-45.0,
                                  rotSkyPos=46.2, mjd=59583.4)

        det_name = self.camera[1].getName()

        cornerRad = _getCornerRaDec(det_name, self.camera, obs)
        cornerDeg = getCornerRaDec(det_name, self.camera, obs)
        for cc1, cc2 in zip(cornerRad, cornerDeg):
            dd = haversine(cc1[0], cc1[1], np.radians(cc2[0]), np.radians(cc2[1]))
            self.assertLess(arcsecFromRadians(dd), 0.000001)


class MotionTestCase(unittest.TestCase):
    """
    This class will contain test methods to verify that the camera utils
    work when proper motion, parallax, and radial velocity are not None.
    """

    @classmethod
    def setUpClass(cls):
        cameraDir = getPackageDir('sims_coordUtils')
        cameraDir = os.path.join(cameraDir, 'tests', 'cameraData')
        cls.camera = ReturnCamera(cameraDir)

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def set_data(self, seed):
        """
        Accept a seed integer.  Return an ObservationMetaData
        and numpy arrays of RA, Dec (in degrees),
        pm_ra, pm_dec, parallax (in arcsec) and v_rad (in km/s)
        centered on that bore site.
        """
        rng = np.random.RandomState(seed)
        n_obj = 100
        ra = 23.1
        dec = -15.6
        rotSkyPos = 23.56
        mjd = 59723.2
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                       rotSkyPos=rotSkyPos, mjd=mjd)
        rr = rng.random_sample(n_obj)*0.1
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = ra + rr*np.cos(theta)
        dec_list = dec + rr*np.sin(theta)
        pm_ra = rng.random_sample(n_obj)*20.0 - 10.0
        pm_dec = rng.random_sample(n_obj)*20.0 - 10.0
        parallax = rng.random_sample(n_obj)*1.0 - 0.5
        v_rad = rng.random_sample(n_obj)*600.0 - 300.0
        return obs, ra_list, dec_list, pm_ra, pm_dec, parallax, v_rad

    def test_chip_name(self):
        """
        Test that chipNameFromRaDec with non-zero proper motion etc.
        agrees with chipNameFromPupilCoords when pupilCoords are
        calculated with the same proper motion, etc.
        """
        (obs, ra_list, dec_list,
         pm_ra_list, pm_dec_list,
         parallax_list, v_rad_list) = self.set_data(8231)

        for is_none in ('pm_ra', 'pm_dec', 'parallax', 'v_rad'):
            pm_ra = pm_ra_list
            pm_dec = pm_dec_list
            parallax = parallax_list
            v_rad = v_rad_list

            if is_none == 'pm_ra':
                pm_ra = None
            elif is_none == 'pm_dec':
                pm_dec = None
            elif is_none == 'parallax':
                parallax = None
            elif is_none == 'v_rad':
                v_rad = None

            xp, yp = pupilCoordsFromRaDec(ra_list, dec_list,
                                          pm_ra=pm_ra, pm_dec=pm_dec,
                                          parallax=parallax, v_rad=v_rad,
                                          obs_metadata=obs)

            name_control = chipNameFromPupilCoords(xp, yp, camera=self.camera)

            name_test = chipNameFromRaDec(ra_list, dec_list,
                                          pm_ra=pm_ra, pm_dec=pm_dec,
                                          parallax=parallax, v_rad=v_rad,
                                          obs_metadata=obs, camera=self.camera)

            name_radians = _chipNameFromRaDec(np.radians(ra_list), np.radians(dec_list),
                                              pm_ra=radiansFromArcsec(pm_ra), pm_dec=radiansFromArcsec(pm_dec),
                                              parallax=radiansFromArcsec(parallax), v_rad=v_rad,
                                              obs_metadata=obs, camera=self.camera)

            np.testing.assert_array_equal(name_control, name_test)
            np.testing.assert_array_equal(name_control, name_radians)
            self.assertGreater(len(np.unique(name_control)), 4)
            self.assertLess(len(np.where(np.equal(name_control, None))[0]), 2*len(name_control)/3)

    def test_pixel_coords(self):
        """
        Test that pixelCoordsFromRaDec with non-zero proper motion etc.
        agrees with pixelCoordsFromPupilCoords when pupilCoords are
        calculated with the same proper motion, etc.
        """
        (obs, ra_list, dec_list,
         pm_ra_list, pm_dec_list,
         parallax_list, v_rad_list) = self.set_data(72)

        for is_none in ('pm_ra', 'pm_dec', 'parallax', 'v_rad'):
            pm_ra = pm_ra_list
            pm_dec = pm_dec_list
            parallax = parallax_list
            v_rad = v_rad_list

            if is_none == 'pm_ra':
                pm_ra = None
            elif is_none == 'pm_dec':
                pm_dec = None
            elif is_none == 'parallax':
                parallax = None
            elif is_none == 'v_rad':
                v_rad = None

            xp, yp = pupilCoordsFromRaDec(ra_list, dec_list,
                                          pm_ra=pm_ra, pm_dec=pm_dec,
                                          parallax=parallax, v_rad=v_rad,
                                          obs_metadata=obs)

            xpx_control, ypx_control = pixelCoordsFromPupilCoords(xp, yp, camera=self.camera)

            xpx_test, ypx_test = pixelCoordsFromRaDec(ra_list, dec_list,
                                                      pm_ra=pm_ra, pm_dec=pm_dec,
                                                      parallax=parallax, v_rad=v_rad,
                                                      obs_metadata=obs, camera=self.camera)

            xpx_radians, ypx_radians = _pixelCoordsFromRaDec(np.radians(ra_list), np.radians(dec_list),
                                                             pm_ra=radiansFromArcsec(pm_ra), pm_dec=radiansFromArcsec(pm_dec),
                                                             parallax=radiansFromArcsec(parallax), v_rad=v_rad,
                                                             obs_metadata=obs, camera=self.camera)

            np.testing.assert_array_equal(xpx_control, xpx_test)
            np.testing.assert_array_equal(ypx_control, ypx_test)
            np.testing.assert_array_equal(xpx_control, xpx_radians)
            np.testing.assert_array_equal(ypx_control, ypx_radians)
            self.assertLess(len(np.where(np.isnan(xpx_control))[0]), 2*len(xpx_control)/3)

    def test_focal_plane_coords(self):
        """
        Test that focalPlaneCoordsFromRaDec with non-zero proper motion etc.
        agrees with pixelCoordsFromPupilCoords when pupilCoords are
        calculated with the same proper motion, etc.
        """
        (obs, ra_list, dec_list,
         pm_ra_list, pm_dec_list,
         parallax_list, v_rad_list) = self.set_data(72)

        for is_none in ('pm_ra', 'pm_dec', 'parallax', 'v_rad'):
            pm_ra = pm_ra_list
            pm_dec = pm_dec_list
            parallax = parallax_list
            v_rad = v_rad_list

            if is_none == 'pm_ra':
                pm_ra = None
            elif is_none == 'pm_dec':
                pm_dec = None
            elif is_none == 'parallax':
                parallax = None
            elif is_none == 'v_rad':
                v_rad = None

            xp, yp = pupilCoordsFromRaDec(ra_list, dec_list,
                                          pm_ra=pm_ra, pm_dec=pm_dec,
                                          parallax=parallax, v_rad=v_rad,
                                          obs_metadata=obs)

            xf_control, yf_control = focalPlaneCoordsFromPupilCoords(xp, yp, camera=self.camera)

            xf_test, yf_test = focalPlaneCoordsFromRaDec(ra_list, dec_list,
                                                         pm_ra=pm_ra, pm_dec=pm_dec,
                                                         parallax=parallax, v_rad=v_rad,
                                                         obs_metadata=obs, camera=self.camera)

            xf_radians, yf_radians = _focalPlaneCoordsFromRaDec(np.radians(ra_list), np.radians(dec_list),
                                                                pm_ra=radiansFromArcsec(pm_ra), pm_dec=radiansFromArcsec(pm_dec),
                                                                parallax=radiansFromArcsec(parallax), v_rad=v_rad,
                                                                obs_metadata=obs, camera=self.camera)

            np.testing.assert_array_equal(xf_control, xf_test)
            np.testing.assert_array_equal(yf_control, yf_test)
            np.testing.assert_array_equal(xf_control, xf_radians)
            np.testing.assert_array_equal(yf_control, yf_radians)
            self.assertEqual(len(np.where(np.isnan(xf_control))[0]), 0)
            self.assertEqual(len(np.where(np.isnan(yf_control))[0]), 0)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

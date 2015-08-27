import os
import numpy
import unittest
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir

import lsst.afw.cameraGeom.testUtils as camTestUtils
import lsst.afw.geom as afwGeom
from lsst.sims.utils import ObservationMetaData
from lsst.sims.utils import arcsecFromRadians, radiansFromArcsec
from lsst.sims.coordUtils import findChipName, calculatePixelCoordinates
from lsst.sims.coordUtils.utils import ReturnCamera
from lsst.sims.coordUtils import raDecFromPixelCoordinates
from lsst.sims.coordUtils import pupilCoordinatesFromPixelCoordinates
from lsst.sims.coordUtils import calculateFocalPlaneCoordinates
from lsst.sims.coordUtils import observedFromICRS, calculatePupilCoordinates

class CameraUtilsUnitTest(unittest.TestCase):

    def testCameraCoordsExceptions(self):
        """
        Test to make sure that focal plane methods raise exceptions when coordinates are improperly
        specified.
        """

        obs = ObservationMetaData(unrefractedRA=46.0, unrefractedDec=27.0,
                                  rotSkyPos=82.0, mjd=52350.0)

        camera = camTestUtils.CameraWrapper().camera

        #these are just values shown heuristically to give an actual chip name
        ra = numpy.array(numpy.radians(obs.unrefractedRA) - numpy.array([1.01, 1.02])*numpy.radians(1.0/3600.0))
        dec = numpy.array(numpy.radians(obs.unrefractedDec) - numpy.array([2.02, 2.01])*numpy.radians(1.0/3600.0))

        ra, dec = observedFromICRS(ra, dec, obs_metadata=obs, epoch=2000.0)

        xPupil = numpy.array([-0.000262243770, -0.00000234])
        yPupil = numpy.array([0.000199467792, 0.000189334])

        ##########test findChipName

        name = findChipName(ra=ra, dec=dec,
                            epoch=2000.0,
                            obs_metadata=obs,
                            camera=camera)

        self.assertTrue(name[0] is not None)

        name = findChipName(xPupil=xPupil, yPupil=yPupil,
                            camera=camera)

        self.assertTrue(name[0] is not None)

        #test when specifying no coordinates
        self.assertRaises(RuntimeError, findChipName)

        #test when specifying both sets fo coordinates
        self.assertRaises(RuntimeError, findChipName, xPupil=xPupil, yPupil=yPupil,
                  ra=ra, dec=dec, camera=camera)

        #test when failing to specify camera
        self.assertRaises(RuntimeError, findChipName, ra=ra, dec=dec,
                          obs_metadata=obs, epoch=2000.0)
        self.assertRaises(RuntimeError, findChipName, xPupil=xPupil, yPupil=yPupil)

        #test when failing to specify obs_metadata
        self.assertRaises(RuntimeError, findChipName, ra=ra, dec=dec, epoch=2000.0,
                          camera=camera)

        #test when failing to specify epoch
        self.assertRaises(RuntimeError, findChipName, ra=ra, dec=dec, camera=camera,
                          obs_metadata=obs)

        #test mismatches
        self.assertRaises(RuntimeError, findChipName, ra=numpy.array([ra[0]]), dec=dec,
                            epoch=2000.0,
                            obs_metadata=obs,
                            camera=camera)

        self.assertRaises(RuntimeError, findChipName, ra=ra, dec=numpy.array([dec[0]]),
                            epoch=2000.0,
                            obs_metadata=obs,
                            camera=camera)

        self.assertRaises(RuntimeError, findChipName, xPupil=numpy.array([xPupil[0]]), yPupil=yPupil,
                                        camera=camera)
        self.assertRaises(RuntimeError, findChipName, xPupil=xPupil, yPupil=numpy.array([yPupil[0]]),
                                        camera=camera)

        #test lists
        self.assertRaises(RuntimeError, findChipName, ra=list(ra), dec=dec,
                            epoch=2000.0,
                            obs_metadata=obs,
                            camera=camera)

        self.assertRaises(RuntimeError, findChipName, ra=ra, dec=list(dec),
                            epoch=2000.0,
                            obs_metadata=obs,
                            camera=camera)

        self.assertRaises(RuntimeError, findChipName, xPupil=list(xPupil), yPupil=yPupil,
                                        camera=camera)
        self.assertRaises(RuntimeError, findChipName, xPupil=xPupil, yPupil=list(yPupil),
                                        camera=camera)


        ##########test FocalPlaneCoordinates

        #test that it actually runs
        xx, yy = calculateFocalPlaneCoordinates(xPupil=xPupil, yPupil=yPupil, camera=camera)
        xx, yy = calculateFocalPlaneCoordinates(ra=ra, dec=dec,
                                                epoch=2000.0, obs_metadata=obs,
                                                camera=camera)

        #test without any coordinates
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, camera=camera)

        #test specifying both ra,dec and xPupil,yPupil
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=ra, dec=dec,
                             xPupil=xPupil, yPupil=yPupil, camera=camera)

        #test without camera
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, xPupil=xPupil, yPupil=yPupil)
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=ra, dec=dec,
                                        epoch=2000.0, obs_metadata=obs)

        #test without epoch
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=ra, dec=dec,
                                                obs_metadata=obs,
                                                camera=camera)

        #test without obs_metadata
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=ra, dec=dec,
                                               epoch=2000.0,
                                               camera=camera)

        #test with lists
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, xPupil=list(xPupil), yPupil=yPupil,
                          camera=camera)
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, xPupil=xPupil, yPupil=list(yPupil),
                          camera=camera)

        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=list(ra), dec=dec,
                                        epoch=2000.0, camera=camera)
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=ra, dec=list(dec),
                                        epoch=2000.0,
                                        obs_metadata=obs,
                                        camera=camera)

        #test mismatches
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, xPupil=numpy.array([xPupil[0]]), yPupil=yPupil,
                          camera=camera)
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, xPupil=xPupil, yPupil=numpy.array([yPupil[0]]),
                          camera=camera)

        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=numpy.array([ra[0]]), dec=dec,
                                        epoch=2000.0, camera=camera)
        self.assertRaises(RuntimeError, calculateFocalPlaneCoordinates, ra=ra, dec=numpy.array([dec[0]]),
                                        epoch=2000.0,
                                        obs_metadata=obs,
                                        camera=camera)


        ##########test calculatePixelCoordinates
         #test that it actually runs
        xx, yy = calculatePixelCoordinates(xPupil=xPupil, yPupil=yPupil, camera=camera)
        xx, yy = calculatePixelCoordinates(ra=ra, dec=dec,
                                                epoch=2000.0, obs_metadata=obs,
                                                camera=camera)

        #test without any coordinates
        self.assertRaises(RuntimeError, calculatePixelCoordinates, camera=camera)

        #test specifying both ra,dec and xPupil,yPupil
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=dec,
                             xPupil=xPupil, yPupil=yPupil, camera=camera)

        #test without camera
        self.assertRaises(RuntimeError, calculatePixelCoordinates, xPupil=xPupil, yPupil=yPupil)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=dec,
                                        epoch=2000.0, obs_metadata=obs)

        #test without epoch
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=dec,
                                                obs_metadata=obs,
                                                camera=camera)

        #test without obs_metadata
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=dec,
                                                epoch=2000.0,
                                                camera=camera)

        #test with lists
        self.assertRaises(RuntimeError, calculatePixelCoordinates, xPupil=list(xPupil), yPupil=yPupil,
                          camera=camera)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, xPupil=xPupil, yPupil=list(yPupil),
                          camera=camera)

        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=list(ra), dec=dec,
                                        epoch=2000.0, camera=camera)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=list(dec),
                                        epoch=2000.0,
                                        obs_metadata=obs,
                                        camera=camera)

        #test mismatches
        self.assertRaises(RuntimeError, calculatePixelCoordinates, xPupil=numpy.array([xPupil[0]]), yPupil=yPupil,
                          camera=camera)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, xPupil=xPupil, yPupil=numpy.array([yPupil[0]]),
                          camera=camera)

        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=numpy.array([ra[0]]), dec=dec,
                                        epoch=2000.0, camera=camera)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=numpy.array([dec[0]]),
                                        epoch=2000.0,
                                        obs_metadata=obs,
                                        camera=camera)

        chipNames = findChipName(xPupil=xPupil, yPupil=yPupil, camera=camera)
        calculatePixelCoordinates(xPupil=xPupil, yPupil=yPupil, chipNames=chipNames, camera=camera)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, xPupil=xPupil, yPupil=yPupil,
                                        camera=camera, chipNames=[chipNames[0]])

        chipNames=findChipName(ra=ra, dec=dec, obs_metadata=obs, epoch=2000.0,
                               camera=camera)
        calculatePixelCoordinates(ra=ra, dec=dec, obs_metadata=obs, epoch=2000.0,
                                  camera=camera, chipNames=chipNames)
        self.assertRaises(RuntimeError, calculatePixelCoordinates, ra=ra, dec=dec, obs_metadata=obs,
                          epoch=2000.0,
                          camera=camera, chipNames=[chipNames[0]])


    def testPixelPos(self):
        obs = ObservationMetaData(unrefractedRA=25.0, unrefractedDec=-35.0,
                                  rotSkyPos=112.0, mjd=52350.0)
        numpy.random.seed(42)

        camera = camTestUtils.CameraWrapper().camera

        nSamples = 100
        ra = numpy.random.random_sample(nSamples)*radiansFromArcsec(100.0) + numpy.radians(obs.unrefractedRA)
        dec = numpy.random.random_sample(nSamples)*radiansFromArcsec(100.0) + numpy.radians(obs.unrefractedDec)

        pupilCoordinateArray = calculatePupilCoordinates(ra, dec, obs_metadata=obs,
                                                         epoch=2000.0)

        pixelCoordinateArray = calculatePixelCoordinates(ra=ra, dec=dec,
                                                        obs_metadata=obs,
                                                        epoch=2000.0, camera=camera)

        chipNameList = findChipName(ra=ra, dec=dec,
                                    obs_metadata=obs, epoch=2000.0, camera=camera)
        self.assertTrue(numpy.all(numpy.isfinite(pupilCoordinateArray[0])))
        self.assertTrue(numpy.all(numpy.isfinite(pupilCoordinateArray[1])))

        for x, y, cname in zip(pixelCoordinateArray[0], pixelCoordinateArray[1],
                               chipNameList):
            if cname is None:
                #make sure that x and y are not set if the object doesn't land on a chip
                self.assertTrue(not numpy.isfinite(x) and not numpy.isfinite(y))
            else:
                #make sure the pixel positions are inside the detector bounding box.
                self.assertTrue(afwGeom.Box2D(camera[cname].getBBox()).contains(afwGeom.Point2D(x,y)))



    def testPupilFromPixel(self):
        """
        Test the conversion between pixel coordinates and pupil coordinates
        """
        baseDir = os.path.join(getPackageDir('sims_coordUtils'),'tests','cameraData')
        camera = ReturnCamera(baseDir)
        epoch=2000.0
        raCenter = 25.0
        decCenter = -10.0
        obs = ObservationMetaData(unrefractedRA=raCenter,
                                  unrefractedDec=decCenter,
                                  boundType='circle',
                                  boundLength=0.1,
                                  rotSkyPos=23.0,
                                  mjd=52000.0)

        raTrue, decTrue = observedFromICRS(numpy.array([numpy.radians(raCenter)]),
                                           numpy.array([numpy.radians(decCenter)]),
                                           obs_metadata=obs, epoch=epoch)

        ra = []
        dec = []

        dx = 1.0e-4

        for rr in numpy.arange(raTrue-20.0*dx, raTrue+20.0*dx, dx):
            for dd in numpy.arange(decTrue-20.0*dx, decTrue+20.0*dx, dx):
                ra.append(rr)
                dec.append(dd)


        ra = numpy.array(ra)
        dec = numpy.array(dec)

        xp, yp = calculatePupilCoordinates(ra, dec, obs_metadata=obs, epoch=2000.0)
        chipNameList = findChipName(xPupil=xp, yPupil=yp, obs_metadata=obs, epoch=epoch,
                                    camera=camera)
        xPixList, yPixList = calculatePixelCoordinates(xPupil=xp, yPupil=yp, chipNames=chipNameList,
                                               obs_metadata=obs, epoch=epoch, camera=camera)

        xpTest, ypTest = pupilCoordinatesFromPixelCoordinates(xPixList, yPixList, chipNameList,
                                                     camera=camera, obs_metadata=obs, epoch=epoch)

        xpControl = numpy.array([xx if name is not None else numpy.NaN for (xx, name) in zip(xp, chipNameList)])
        ypControl = numpy.array([yy if name is not None else numpy.NaN for (yy, name) in zip(yp, chipNameList)])

        numpy.testing.assert_array_almost_equal(xpControl, xpTest, decimal=10)
        numpy.testing.assert_array_almost_equal(ypControl, ypTest, decimal=10)



    def testRaDecFromPixelCoordinates(self):
        """
        Test conversion from pixel coordinates to Ra, Dec
        """

        baseDir = os.path.join(getPackageDir('sims_coordUtils'),'tests','cameraData')
        camera = ReturnCamera(baseDir)
        epoch=2000.0

        raCenter = 25.0
        decCenter = -10.0
        obs = ObservationMetaData(unrefractedRA=raCenter,
                                  unrefractedDec=decCenter,
                                  boundType='circle',
                                  boundLength=0.1,
                                  rotSkyPos=23.0,
                                  mjd=52000.0)

        raTrue, decTrue = observedFromICRS(numpy.array([numpy.radians(raCenter)]),
                                           numpy.array([numpy.radians(decCenter)]),
                                           obs_metadata=obs, epoch=epoch)

        ra = []
        dec = []

        dx = 1.0e-4

        for rr in numpy.arange(raTrue-20.0*dx, raTrue+20.0*dx, dx):
            for dd in numpy.arange(decTrue-20.0*dx, decTrue+20.0*dx, dx):
                ra.append(rr)
                dec.append(dd)


        ra = numpy.array(ra)
        dec = numpy.array(dec)

        chipNameList = findChipName(ra=ra, dec=dec, obs_metadata=obs, epoch=epoch, camera=camera)
        pixelList = calculatePixelCoordinates(ra=ra, dec=dec, chipNames=chipNameList, obs_metadata=obs,
                                           epoch=epoch, camera=camera)

        raTest, decTest = raDecFromPixelCoordinates(pixelList[0], pixelList[1], chipNameList,
                                                    obs_metadata=obs, epoch=epoch, camera=camera)



        raControl = numpy.array([rr if name is not None else numpy.NaN for (rr, name) in zip(ra, chipNameList)])
        decControl = numpy.array([dd if name is not None else numpy.NaN for (dd, name) in zip(dec, chipNameList)])

        numpy.testing.assert_array_almost_equal(arcsecFromRadians(raControl), arcsecFromRadians(raTest), decimal=10)
        numpy.testing.assert_array_almost_equal(arcsecFromRadians(decControl), arcsecFromRadians(decTest), decimal=10)


    def testFindChipNameNaNPupil(self):
        """
        Test that findChipName returns 'None' for objects with NaN pupil coordinates
        """
        baseDir = os.path.join(getPackageDir('sims_coordUtils'),'tests','cameraData')
        camera = ReturnCamera(baseDir)

        obs = ObservationMetaData(unrefractedRA=45.0, unrefractedDec=87.0, rotSkyPos=65.0,
                                  mjd=43520.0)

        raCenter, decCenter = observedFromICRS(numpy.array([obs._unrefractedRA]),
                                               numpy.array([obs._unrefractedDec]),
                                               obs_metadata=obs, epoch=2000.0)

        nSamples = 100
        numpy.random.seed(42)
        xp = radiansFromArcsec(numpy.random.random_sample(nSamples)*100.0)
        yp = radiansFromArcsec(numpy.random.random_sample(nSamples)*100.0)

        nameList = findChipName(xPupil=xp, yPupil=yp, obs_metadata=obs, camera=camera, epoch=2000.0)
        for name in nameList:
            self.assertTrue(name is not None)

        xp[4] = numpy.NaN
        yp[4] = numpy.NaN

        xp[10] = numpy.NaN

        yp[15] = numpy.NaN

        nameList2 = findChipName(xPupil=xp, yPupil=yp, obs_metadata=obs, camera=camera, epoch=2000.0)
        for ix in range(len(nameList2)):
            if ix!=4 and ix!=10 and ix!=15:
                self.assertTrue(nameList2[ix]==nameList[ix])
            else:
                self.assertTrue(nameList2[ix] is None)


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(CameraUtilsUnitTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)

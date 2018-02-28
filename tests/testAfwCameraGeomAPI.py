import unittest
import lsst.utils.tests
from lsst.utils import getPackageDir

import numpy as np
import os

from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import getCornerRaDec
from lsst.sims.coordUtils import focalPlaneCoordsFromRaDec
from lsst.sims.coordUtils import pixelCoordsFromRaDecLSST
from lsst.sims.coordUtils import chipNameFromRaDecLSST
from lsst.sims.coordUtils import chipNameFromPupilCoordsLSST


def setup_module(module):
    lsst.utils.tests.init()


class AfwCameraGeomAPITestCase(unittest.TestCase):
    """
    This test case is meant to verify that we have correctly incorporated
    any API changes in afwCameraGeom by verifying RA, Dec to pixel results
    against identical results generated from the w.2017.50 version of
    afw.  If obs_lsstSim ever changes in a physically meaningful way, these
    tests will break, but hopefully we will be aware that that happened and
    we will be able to regenerate the underlying test data with

    $SIM_COORDUTILS_DIR/tests/lsstCameraData/make_test_catalog.py
    """

    @classmethod
    def setUpClass(cls):
        cls.camera = lsst_camera()
        cls.data_dir = os.path.join(getPackageDir('sims_coordUtils'),
                                   'tests', 'lsstCameraData')

        pix_dtype = np.dtype([('ra', float), ('dec', float),
                              ('name', str, 15),
                              ('focal_x', float), ('focal_y', float),
                              ('pixel_x', float), ('pixel_y', float)])

        cls.pix_data = np.genfromtxt(os.path.join(cls.data_dir,
                                                  'lsst_pixel_data.txt'),
                                     delimiter=';', dtype=pix_dtype)

        ra = 25.0
        dec = -62.0
        cls.obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                      rotSkyPos=57.2, mjd=59586.2)

    @classmethod
    def tearDownClass(cls):
        if hasattr(chipNameFromPupilCoordsLSST, '_detector_arr'):
            del chipNameFromPupilCoordsLSST._detector_arr

        del cls.camera
        if hasattr(lsst_camera, '_lsst_camera'):
            del lsst_camera._lsst_camera

    def test_chipName(self):
        """
        Verify that chipNameFromRaDecLSST has not changed.
        """
        chip_name_arr = chipNameFromRaDecLSST(self.pix_data['ra'],
                                              self.pix_data['dec'],
                                              obs_metadata=self.obs)

        np.testing.assert_array_equal(chip_name_arr, self.pix_data['name'])

    def test_pixelCoords(self):
        """
        Verify that pixelCoordsFromRaDecLSST has not changed
        """
        pix_x, pix_y = pixelCoordsFromRaDecLSST(self.pix_data['ra'],
                                                self.pix_data['dec'],
                                                obs_metadata=self.obs)

        np.testing.assert_array_almost_equal(pix_x, self.pix_data['pixel_x'],
                                             decimal=5)
        np.testing.assert_array_almost_equal(pix_y, self.pix_data['pixel_y'],
                                             decimal=5)


    def test_focalCoords(self):
        """
        Verify that focalPlaneCoordsFromRaDec has not changed
        """
        foc_x, foc_y = focalPlaneCoordsFromRaDec(self.pix_data['ra'],
                                                 self.pix_data['dec'],
                                                 camera=self.camera,
                                                 obs_metadata=self.obs)

        np.testing.assert_array_almost_equal(foc_x, self.pix_data['focal_x'],
                                             decimal=5)
        np.testing.assert_array_almost_equal(foc_y, self.pix_data['focal_y'],
                                             decimal=5)

    def test_cornerRaDec(self):
        """
        Verify that getCornerRaDec has not changed
        """
        dtype = np.dtype([('name', str, 15),
                          ('x0', float), ('y0', float),
                          ('x1', float), ('y1', float),
                          ('x2', float), ('y2', float),
                          ('x3', float), ('y3', float)])

        data = np.genfromtxt(os.path.join(self.data_dir, 'lsst_camera_corners.txt'),
                             dtype=dtype, delimiter=';')

        detector_name_list = [dd.getName() for dd in self.camera]
        detector_name_list.sort()
        x0 = np.zeros(len(detector_name_list), dtype=float)
        x1 = np.zeros(len(detector_name_list), dtype=float)
        x2 = np.zeros(len(detector_name_list), dtype=float)
        x3 = np.zeros(len(detector_name_list), dtype=float)
        y0 = np.zeros(len(detector_name_list), dtype=float)
        y1 = np.zeros(len(detector_name_list), dtype=float)
        y2 = np.zeros(len(detector_name_list), dtype=float)
        y3 = np.zeros(len(detector_name_list), dtype=float)

        for i_chip in range(len(detector_name_list)):
            name = detector_name_list[i_chip]
            corners = getCornerRaDec(name, self.camera, self.obs)
            x0[i_chip] = corners[0][0]
            x1[i_chip] = corners[1][0]
            x2[i_chip] = corners[2][0]
            x3[i_chip] = corners[3][0]
            y0[i_chip] = corners[0][1]
            y1[i_chip] = corners[1][1]
            y2[i_chip] = corners[2][1]
            y3[i_chip] = corners[3][1]

        np.testing.assert_array_almost_equal(x0, data['x0'], decimal=4)
        np.testing.assert_array_almost_equal(x1, data['x1'], decimal=4)
        np.testing.assert_array_almost_equal(x2, data['x2'], decimal=4)
        np.testing.assert_array_almost_equal(x3, data['x3'], decimal=4)
        np.testing.assert_array_almost_equal(y0, data['y0'], decimal=4)
        np.testing.assert_array_almost_equal(y1, data['y1'], decimal=4)
        np.testing.assert_array_almost_equal(y2, data['y2'], decimal=4)
        np.testing.assert_array_almost_equal(y3, data['y3'], decimal=4)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

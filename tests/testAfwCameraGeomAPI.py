import unittest
import lsst.utils.tests
from lsst.utils import getPackageDir

import numpy as np
import os

from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import getCornerPixels
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

    $SIM_COORDUTILS_DIR/workspace/dm_12447/make_test_catalog.py
    """

    @classmethod
    def setUpClass(cls):
        cls.camera = lsst_camera()
        data_dir = os.path.join(getPackageDir('sims_coordUtils'),
                                'tests', 'lsstCameraData')

        pix_dtype = np.dtype([('ra', float), ('dec', float),
                              ('name', str, 15),
                              ('focal_x', float), ('focal_y', float),
                              ('pixel_x', float), ('pixel_y', float)])

        cls.pix_data = np.genfromtxt(os.path.join(data_dir,
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


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

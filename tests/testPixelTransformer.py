import unittest
import numpy as np

import lsst.utils.tests
from lsst.sims.coordUtils import DMtoCameraPixelTransformer
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import pupilCoordsFromPixelCoords
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS


def setup_module(module):
    lsst.utils.tests.init()


class PixelTransformerTestCase(unittest.TestCase):
    """
    This unit test TestCase will exercise the class that transforms between
    DM pixels and Camera Team pixels.

    Recall that their conventions differ in that

    Camera +y = DM +x
    Camera +x = DM -y
    """

    def test_camPixFromDMpix(self):
        """
        test that trasformation between Camera Team and DM pixels works
        """
        camera_wrapper = DMtoCameraPixelTransformer()
        rng = np.random.RandomState()
        camera = lsst_camera()
        npts = 200
        for det in camera:
            det_name = det.getName()
            cam_x_in = rng.random_sample(npts)*4000.0
            cam_y_in = rng.random_sample(npts)*4000.0
            dm_x, dm_y = camera_wrapper.dmPixFromCameraPix(cam_x_in, cam_y_in, det_name)
            cam_x, cam_y = camera_wrapper.cameraPixFromDMPix(dm_x, dm_y, det_name)
            np.testing.assert_array_almost_equal(cam_x_in, cam_x, decimal=10)
            np.testing.assert_array_almost_equal(cam_y_in, cam_y, decimal=10)


            center_point = camera[det_name].getCenter(FOCAL_PLANE)
            pixel_system = camera[det_name].makeCameraSys(PIXELS)
            center_pix = camera.transform(center_point, FOCAL_PLANE, pixel_system)

            # test that DM and Camera Team pixels are correctly rotated
            # with respect to each other

            np.testing.assert_allclose(dm_x-center_pix.getX(),
                                       cam_y-center_pix.getX(),
                                       atol=1.0e-10, rtol=0.0)
            np.testing.assert_allclose(dm_y-center_pix.getY(),
                                       center_pix.getY()-cam_x,
                                       atol=1.0e-10, rtol=0.0)

        del camera_wrapper
        del lsst_camera._lsst_camera


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

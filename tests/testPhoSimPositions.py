"""
This test suite will verify that the CatSim and PhoSim camera models
are in agreement by taking the chip and pixel positions found by
PhoSim and verifying that CatSim predicts similar positions.
"""

import unittest
import numpy as np
import os
import lsst.utils.tests
from lsst.utils import getPackageDir

from lsst.sims.utils import ObservationMetaData
from lsst.sims.coordUtils import chipNameFromRaDecLSST
from lsst.sims.coordUtils import pixelCoordsFromRaDecLSST
from lsst.sims.coordUtils import chipNameFromPupilCoordsLSST
from lsst.sims.coordUtils import getCornerPixels
from lsst.sims.utils import observedFromICRS
from lsst.sims.utils import arcsecFromRadians, angularSeparation
from lsst.sims.coordUtils import lsst_camera

def setup_module(module):
    lsst.utils.tests.init()


class PhoSim_position_test_case(unittest.TestCase):

    longMessage = True

    @classmethod
    def setUpClass(cls):
        cat_name = os.path.join(getPackageDir('sims_coordUtils'),
                                'tests', 'LSST_focal_plane_data',
                                'source_position_catalog.txt')

        dtype = np.dtype([('pointingRA', float), ('pointingDec', float),
                          ('rotSkyPos', float),
                          ('objRA', float), ('objDec', float),
                          ('chipName', str, 6),
                          ('xpix', float), ('ypix', float)])

        cls.data = np.genfromtxt(cat_name, dtype=dtype)

    @classmethod
    def tearDownClass(cls):
        if hasattr(chipNameFromPupilCoordsLSST, '_detector_arr'):
            del chipNameFromPupilCoordsLSST._detector_arr
        del lsst_camera._lsst_camera

    def test_chipName(self):
        """
        Simply verify that CatSim puts the sources on the right chip
        """
        self.assertGreater(len(self.data), 10)
        for ix in range(len(self.data)):
            obs = ObservationMetaData(pointingRA=self.data['pointingRA'][ix],
                                      pointingDec=self.data['pointingDec'][ix],
                                      rotSkyPos=self.data['rotSkyPos'][ix],
                                      mjd=59580.0)

            chipName = chipNameFromRaDecLSST(self.data['objRA'][ix],
                                             self.data['objDec'][ix],
                                             obs_metadata=obs)

            self.assertEqual(chipName.replace(',','').replace(':','').replace(' ',''),
                             self.data['chipName'][ix])

    def test_pixel_positions(self):
        """
        Test that CatSim pixel positions are close to PhoSim pixel positions.

        This is complicated by the fact that PhoSim uses the camera team definition
        of pixel space, which differs from the DM definition of pixel space as follows:

        Camera +y = DM +x
        Camera +x = DM -y
        Camera +z = DM +z

        This has been verified both by consulting documentation -- the documentation for
        afwCameraGeom says that +x is along the serial readout direction; LCA-13381
        indicates that, in the Camera team's definition, the serial readout is along +y --
        and phenomenologically by comparing the relationship between
        pixelCoordsFromPupilCoords() to visual inspection of PhoSim-generated FITS images.
        """
        self.assertGreater(len(self.data), 10)
        for ix in range(len(self.data)):

            in_name = self.data['chipName'][ix]
            chip_name = in_name[0]+':'+in_name[1]+','+in_name[2]
            chip_name += ' '+in_name[3]+':'+in_name[4]+','+in_name[5]
            corner_pixels = getCornerPixels(chip_name, lsst_camera())

            x_center_dm = 0.25*(corner_pixels[0][0] + corner_pixels[1][0] +
                                corner_pixels[2][0] + corner_pixels[3][0])

            y_center_dm = 0.25*(corner_pixels[0][1] + corner_pixels[1][1] +
                                corner_pixels[2][1] + corner_pixels[3][1])

            obs = ObservationMetaData(pointingRA=self.data['pointingRA'][ix],
                                      pointingDec=self.data['pointingDec'][ix],
                                      rotSkyPos=self.data['rotSkyPos'][ix],
                                      mjd=59580.0)

            xpix, ypix = pixelCoordsFromRaDecLSST(self.data['objRA'][ix],
                                                  self.data['objDec'][ix],
                                                  obs_metadata=obs)

            raObs, decObs = observedFromICRS(self.data['objRA'][ix],
                                             self.data['objDec'][ix],
                                             obs_metadata=obs,
                                             epoch=2000.0)

            # find displacement from center of DM coordinates of the
            # objects as placed by PhoSim
            d_y_phosim = y_center_dm - self.data['xpix'][ix]
            d_x_phosim = self.data['ypix'][ix] - x_center_dm

            # displacement from center of DM coordinates as calculated
            # by DM
            d_x_dm = xpix - x_center_dm
            d_y_dm = ypix - y_center_dm

            d_pix = np.sqrt((d_x_dm - d_x_phosim)**2 +
                            (d_y_dm - d_y_phosim)**2)

            # demand that the difference between the two displacements is less
            # than 0.05 of the total displacement from the center of the object
            # as calculated by DM
            msg = 'dx %e; dy %e' % (d_x_dm, d_y_dm)
            self.assertLess(d_pix, 0.05*np.sqrt(d_x_dm**2+d_y_dm**2), msg=msg)

class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

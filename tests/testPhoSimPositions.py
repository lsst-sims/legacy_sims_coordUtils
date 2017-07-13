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
        Test that CatSim pixel positions are close to PhoSim pixel positions
        """
        self.assertGreater(len(self.data), 10)
        for ix in range(len(self.data)):
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

            d_ang = angularSeparation(self.data['objRA'][ix],
                                      self.data['objDec'][ix],
                                      raObs,decObs)
            d_ang = arcsecFromRadians(np.radians(d_ang))

            d_pix = np.sqrt((xpix-self.data['xpix'][ix])**2 +
                            (ypix-self.data['ypix'][ix])**2)

            # PhoSim will not have applied precession, nutation, etc.
            # It will have applied a more realistic refraction model
            # than CatSim has.  We will 'split the difference' by
            # demanding that the difference between CatSim positions
            # and PhoSim positions is less than the correction applied
            # to the RA, Dec by precession, nutation, etc.
            self.assertLess(d_pix*0.2, d_ang)

class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

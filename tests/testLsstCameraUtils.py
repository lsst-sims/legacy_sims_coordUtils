import unittest
import numpy as np
import lsst.utils.tests
from lsst.sims.coordUtils import (chipNameFromPupilCoords,
                                  chipNameFromPupilCoordsLSST,
                                  pupilCoordsFromPixelCoords)
from lsst.sims.utils import pupilCoordsFromRaDec
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper


def setup_module(module):
    lsst.utils.tests.init()


class ChipNameTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.camera = LsstSimMapper().camera

    @classmethod
    def tearDownClass(cls):
        del cls.camera

    def test_chip_name_from_pupil_coords(self):
        """
        Test that chipNameFromPupilCoordsLSST returns the same
        results as chipNameFromPupilCoords
        """
        n_pointings = 3
        n_obj = 10000
        rng = np.random.RandomState(8831)
        ra_pointing_list = rng.random_sample(n_pointings)*360.0
        dec_pointing_list = rng.random_sample(n_pointings)*180.0-90.0
        rot_list = rng.random_sample(n_pointings)*360.0
        mjd_list = rng.random_sample(n_pointings)*3653+59580.0
        for ra, dec, rot, mjd in zip(ra_pointing_list, dec_pointing_list, rot_list, mjd_list):
            obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                      rotSkyPos=rot, mjd=mjd)
            rr_list = rng.random_sample(n_obj)*1.75
            theta_list = rng.random_sample(n_obj)*2.0*np.pi
            ra_list = ra + rr_list*np.cos(theta_list)
            dec_list = dec + rr_list*np.sin(theta_list)
            x_pup, y_pup = pupilCoordsFromRaDec(ra_list, dec_list, obs_metadata=obs,
                                                epoch=2000.0)

            control_name_list = chipNameFromPupilCoords(x_pup, y_pup, camera=self.camera)
            test_name_list = chipNameFromPupilCoordsLSST(x_pup, y_pup)
            np.testing.assert_array_equal(control_name_list.astype(str), test_name_list.astype(str))

            # make sure we didn't accidentally get a lot of positions that don't land on chips
            self.assertLess(len(np.where(np.char.rfind(test_name_list.astype(str), 'None')>=0)[0]), n_obj/10)

    def test_multiple_chip_names(self):
        """
        Test that chipNameFromPupilCoordsLSST behaves as expected when
        objects fall on more than one chip (as they could with the
        wavefront sensors).
        """
        chipA = 'R:4,0 S:0,2,A'
        chipB = 'R:4,0 S:0,2,B'

        # from past experience, objects that appear at y=0 on
        # R:4,0 S:0,2,A also appear on R:4,0 S:0,2,B
        xpup, ypup = pupilCoordsFromPixelCoords(1500.0, 0.0, chipA, camera=self.camera)
        xpup_list = np.array([0.0, xpup])
        ypup_list = np.array([0.0, ypup])
        name = chipNameFromPupilCoordsLSST(xpup_list, ypup_list, allow_multiple_chips=False)
        self.assertIsInstance(name[1], str)
        self.assertTrue(name[1] == chipA or name[1] == chipB,
                        msg = 'got unexpected chip name %s' % name)

        name = chipNameFromPupilCoordsLSST(xpup_list, ypup_list, allow_multiple_chips=True)
        self.assertIn(chipA, name[1])
        self.assertIn(chipB, name[1])
        self.assertIsInstance(name[0], str)

class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

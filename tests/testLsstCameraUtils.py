import unittest
import numpy as np
import lsst.utils.tests
from lsst.sims.coordUtils import (chipNameFromPupilCoords,
                                  chipNameFromPupilCoordsLSST)
from lsst.sims.utils import pupilCoordsFromRaDec
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper


def setup_module(module):
    lsst.utils.tests.init()


class ChipNameTestCase(unittest.TestCase):

    def test_chip_name_from_pupil_coords(self):
        """
        Test that chipNameFromPupilCoordsLSST returns the same
        results as chipNameFromPupilCoords
        """
        camera = LsstSimMapper().camera
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

            control_name_list = chipNameFromPupilCoords(x_pup, y_pup, camera=camera)
            test_name_list = chipNameFromPupilCoordsLSST(x_pup, y_pup)
            np.testing.assert_array_equal(control_name_list.astype(str), test_name_list)

            # make sure we didn't accidentally get a lot of positions that don't land on chips
            self.assertLess(len(np.where(np.char.rfind(test_name_list, 'None')>=0)[0]), n_obj/10)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

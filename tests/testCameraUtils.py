from builtins import zip
import unittest
import numpy as np
import lsst.utils.tests
from lsst.sims.coordUtils import (chipNameFromPupilCoords,
                                  _chipNameFromRaDec, chipNameFromRaDec,
                                  _pixelCoordsFromRaDec, pixelCoordsFromRaDec,
                                  pixelCoordsFromPupilCoords)
from lsst.sims.utils import pupilCoordsFromRaDec, radiansFromArcsec
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsst.phosim import PhosimMapper
from lsst.sims.utils import angularSeparation

from lsst.sims.coordUtils import clean_up_lsst_camera

def setup_module(module):
    lsst.utils.tests.init()


class ChipNameTestCase(unittest.TestCase):

    longMessage = True

    @classmethod
    def setUpClass(cls):
        cls.camera = PhosimMapper().camera

    @classmethod
    def tearDownClass(cls):
        del cls.camera
        clean_up_lsst_camera()

    def test_outside_radius(self):
        """
        Test that methods can gracefully handle points
        outside of the focal plane
        """
        rng = np.random.RandomState(7123)
        ra = 145.0
        dec = -25.0
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  mjd=59580.0, rotSkyPos=113.0)

        rr = rng.random_sample(100)*5.0
        self.assertGreater(rr.max(), 4.5)
        theta = rng.random_sample(100)*2.0*np.pi
        ra_vec = ra + rr*np.cos(theta)
        dec_vec = dec + rr*np.sin(theta)
        chip_name_list = chipNameFromRaDec(ra_vec, dec_vec,
                                           obs_metadata=obs, camera=self.camera)

        rr = angularSeparation(ra, dec, ra_vec, dec_vec)

        ct_none = 0
        for rr, name in zip(rr, chip_name_list):
            if rr > 2.0:
                self.assertIsNone(name)
                ct_none += 1
        self.assertGreater(ct_none, 0)

    def test_chip_center(self):
        """
        Test that, if we ask for the chip at the bore site,
        we get back 'R:2,2 S:1,1'
        """

        ra = 145.0
        dec = -25.0
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  mjd=59580.0, rotSkyPos=113.0)

        name = chipNameFromRaDec(ra, dec, obs_metadata=obs, camera=self.camera)
        self.assertEqual(name, 'R22_S11')

    def test_one_by_one(self):
        """
        test that running RA, Dec pairs in one at a time gives the same
        results as running them in in batches
        """

        ra = 145.0
        dec = -25.0
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  mjd=59580.0, rotSkyPos=113.0)
        rng = np.random.RandomState(100)
        theta = rng.random_sample(100)*2.0*np.pi
        rr = rng.random_sample(len(theta))*2.0
        ra_list = ra + rr*np.cos(theta)
        dec_list = dec + rr*np.sin(theta)
        name_control = chipNameFromRaDec(ra_list, dec_list, obs_metadata=obs, camera=self.camera)
        is_none = 0
        for ra, dec, name in zip(ra_list, dec_list, name_control):
            test_name = chipNameFromRaDec(ra, dec, obs_metadata=obs, camera=self.camera)
            self.assertEqual(test_name, name)
            if test_name is None:
                is_none += 1

        self.assertGreater(is_none, 0)
        self.assertLess(is_none, (3*len(ra_list))//4)


class MotionTestCase(unittest.TestCase):
    """
    This class will contain test methods to verify that the LSST camera utils
    work correctly when proper motion, parallax, and v_rad are non-zero
    """
    @classmethod
    def setUpClass(cls):
        cls.camera = PhosimMapper().camera

    @classmethod
    def tearDownClass(cls):
        del cls.camera
        clean_up_lsst_camera()

    def set_data(self, seed):
        """
        Accept a seed integer.  Return an ObservationMetaData
        and numpy arrays of RA, Dec (in degrees),
        pm_ra, pm_dec, parallax (in arcsec) and v_rad (in km/s)
        centered on that bore site.
        """
        rng = np.random.RandomState(seed)
        n_obj = 30
        ra = 23.1
        dec = -15.6
        rotSkyPos = 23.56
        mjd = 59723.2
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  rotSkyPos=rotSkyPos, mjd=mjd)
        rr = rng.random_sample(n_obj)*1.75
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = ra + rr*np.cos(theta)
        dec_list = dec + rr*np.sin(theta)
        pm_ra = rng.random_sample(n_obj)*20.0 - 10.0
        pm_dec = rng.random_sample(n_obj)*20.0 - 10.0
        parallax = rng.random_sample(n_obj)*1.0 - 0.5
        v_rad = rng.random_sample(n_obj)*600.0 - 300.0
        return obs, ra_list, dec_list, pm_ra, pm_dec, parallax, v_rad


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
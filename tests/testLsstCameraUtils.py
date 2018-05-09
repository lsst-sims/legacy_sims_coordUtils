from __future__ import with_statement
from builtins import zip
import unittest
import numpy as np
import lsst.utils.tests
from lsst.sims.coordUtils import (chipNameFromPupilCoords,
                                  chipNameFromPupilCoordsLSST,
                                  _chipNameFromRaDec, chipNameFromRaDec,
                                  _chipNameFromRaDecLSST, chipNameFromRaDecLSST,
                                  _pixelCoordsFromRaDec, pixelCoordsFromRaDec,
                                  _pixelCoordsFromRaDecLSST, pixelCoordsFromRaDecLSST,
                                  pixelCoordsFromPupilCoords)
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import focalPlaneCoordsFromPupilCoordsLSST
from lsst.sims.utils import pupilCoordsFromRaDec, radiansFromArcsec
from lsst.sims.utils import ObservationMetaData
from lsst.obs.lsstSim import LsstSimMapper

from lsst.sims.coordUtils import clean_up_lsst_camera

def setup_module(module):
    lsst.utils.tests.init()


class ChipNameTestCase(unittest.TestCase):

    longMessage = True

    @classmethod
    def setUpClass(cls):
        cls.camera = LsstSimMapper().camera

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
        chip_name_list = chipNameFromRaDecLSST(ra_vec, dec_vec,
                                               obs_metadata=obs,
                                               band='u')

    def test_chip_center(self):
        """
        Test that, if we ask for the chip at the bore site,
        we get back 'R:2,2 S:1,1'
        """

        ra = 145.0
        dec = -25.0
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec,
                                  mjd=59580.0, rotSkyPos=113.0)

        name = chipNameFromRaDecLSST(ra, dec, obs_metadata=obs)
        self.assertEqual(name, 'R:2,2 S:1,1')

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
        name_control = chipNameFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        is_none = 0
        for ra, dec, name in zip(ra_list, dec_list, name_control):
            test_name = chipNameFromRaDecLSST(ra, dec, obs_metadata=obs)
            self.assertEqual(test_name, name)
            if test_name is None:
                is_none += 1

        self.assertGreater(is_none, 0)
        self.assertLess(is_none, (3*len(ra_list))//4)

    def test_chip_name_from_ra_dec_radians(self):
        """
        test that _chipNameFromRaDecLSST agrees with _chipNameFromRaDec
        """

        n_obj = 1000
        raP = 112.1
        decP = -34.1
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP,
                                  rotSkyPos=45.0, mjd=43000.0)

        rng = np.random.RandomState(8731)
        rr = rng.random_sample(n_obj)*1.75
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = np.radians(raP + rr*np.cos(theta))
        dec_list = np.radians(decP + rr*np.sin(theta))
        control_name_list = _chipNameFromRaDec(ra_list, dec_list,
                                               obs_metadata=obs,
                                               camera=self.camera)

        test_name_list = _chipNameFromRaDecLSST(ra_list, dec_list,
                                                obs_metadata=obs)

        try:
            np.testing.assert_array_equal(control_name_list.astype(str),
                                          test_name_list.astype(str))
        except AssertionError:
            n_problematic = 0
            for ii, (c_n, t_n) in enumerate(zip(control_name_list.astype(str), test_name_list.astype(str))):
                if c_n != t_n:
                    x_pix, y_pix = pixelCoordsFromRaDecLSST(ra_list[ii], dec_list[ii], obs_metadata=obs)
                    if c_n != 'None':
                        n_problematic += 1
            if n_problematic > 0:
                raise

        self.assertLessEqual(len(np.where(np.char.rfind(test_name_list.astype(str), 'None') >= 0)[0]),
                             n_obj/10)

        # test that exceptions are raised when incomplete ObservationMetaData are used
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, mjd=59580.0)
        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("rotSkyPos", context.exception.args[0])

        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=35.0)
        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("mjd", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDecLSST(ra_list, dec_list)
        self.assertIn("ObservationMetaData", context.exception.args[0])

        # check that exceptions are raised when ra_list, dec_list are of the wrong shape
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=24.0, mjd=43000.0)
        with self.assertRaises(RuntimeError) as context:
            _chipNameFromRaDecLSST(ra_list, dec_list[:5], obs_metadata=obs)
        self.assertIn("chipNameFromRaDecLSST", context.exception.args[0])

    def test_chip_name_from_ra_dec_degrees(self):
        """
        test that chipNameFromRaDecLSST agrees with chipNameFromRaDec
        """

        n_obj = 1000
        raP = 112.1
        decP = -34.1
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP,
                                  rotSkyPos=45.0, mjd=43000.0)

        rng = np.random.RandomState(8731)
        rr = rng.random_sample(n_obj)*1.75
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = raP + rr*np.cos(theta)
        dec_list = decP + rr*np.sin(theta)
        control_name_list = chipNameFromRaDec(ra_list, dec_list,
                                              obs_metadata=obs,
                                              camera=self.camera)

        test_name_list = chipNameFromRaDecLSST(ra_list, dec_list,
                                               obs_metadata=obs)

        try:
            np.testing.assert_array_equal(control_name_list.astype(str),
                                          test_name_list.astype(str))
        except AssertionError:
            n_problematic = 0
            for ii, (c_n, t_n) in enumerate(zip(control_name_list.astype(str), test_name_list.astype(str))):
                if c_n != t_n:
                    x_pix, y_pix = pixelCoordsFromRaDecLSST(ra_list[ii], dec_list[ii], obs_metadata=obs)
                    if c_n != 'None':
                        n_problematic += 1
            if n_problematic > 0:
                raise

        self.assertLessEqual(len(np.where(np.char.rfind(test_name_list.astype(str), 'None') >= 0)[0]),
                             n_obj/10)

        # test that exceptions are raised when incomplete ObservationMetaData are used
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, mjd=59580.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("rotSkyPos", context.exception.args[0])

        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=35.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("mjd", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDecLSST(ra_list, dec_list)
        self.assertIn("ObservationMetaData", context.exception.args[0])

        # check that exceptions are raised when ra_list, dec_list are of the wrong shape
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=24.0, mjd=43000.0)
        with self.assertRaises(RuntimeError) as context:
            chipNameFromRaDecLSST(ra_list, dec_list[:5], obs_metadata=obs)
        self.assertIn("chipNameFromRaDecLSST", context.exception.args[0])

    def test_pixel_coords_from_ra_dec_radians(self):
        """
        Test that _pixelCoordsFromRaDec and _pixelCoordsFromRaDecLSST agree
        """
        raP = 74.2
        decP = 13.0
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP,
                                  rotSkyPos=13.0, mjd=43441.0)

        n_obj = 1000
        rng = np.random.RandomState(83241)
        rr = rng.random_sample(n_obj)*1.75
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = np.radians(raP + rr*np.cos(theta))
        dec_list = np.radians(decP + rr*np.sin(theta))

        x_pix, y_pix = _pixelCoordsFromRaDec(ra_list, dec_list, obs_metadata=obs, camera=self.camera,
                                             includeDistortion=False)
        self.assertLessEqual(len(np.where(np.isnan(x_pix))[0]), n_obj/10)
        self.assertLessEqual(len(np.where(np.isnan(y_pix))[0]), n_obj/10)

        x_pix_test, y_pix_test = _pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs,
                                                           includeDistortion=False)

        try:
            np.testing.assert_array_equal(x_pix, x_pix_test)
            np.testing.assert_array_equal(y_pix, y_pix_test)
        except AssertionError:
            n_problematic = 0
            for xx, yy, xt, yt in zip(x_pix, y_pix, x_pix_test, y_pix_test):
                if xx!=xt or yy!=yt:
                    if (not np.isnan(xx) and not np.isnan(xt) and
                        not np.isnan(yy) and not np.isnan(yt)):
                        print(xx,yy,xt,yt)

                        n_problematic += 1
            if n_problematic>0:
                raise

        # test when we force a chipName
        x_pix, y_pix = _pixelCoordsFromRaDec(ra_list, dec_list, chipName=['R:2,2 S:1,1'],
                                             obs_metadata=obs, camera=self.camera,
                                             includeDistortion=False)
        self.assertLessEqual(len(np.where(np.isnan(x_pix))[0]), n_obj/10)
        self.assertLessEqual(len(np.where(np.isnan(y_pix))[0]), n_obj/10)

        x_pix_test, y_pix_test = _pixelCoordsFromRaDecLSST(ra_list, dec_list, chipName=['R:2,2 S:1,1'],
                                                           obs_metadata=obs,
                                                           includeDistortion=False)
        np.testing.assert_array_equal(x_pix, x_pix_test)
        np.testing.assert_array_equal(y_pix, y_pix_test)

        # test without distortion
        x_pix, y_pix = _pixelCoordsFromRaDec(ra_list, dec_list, obs_metadata=obs, camera=self.camera,
                                             includeDistortion=False)
        self.assertLessEqual(len(np.where(np.isnan(x_pix))[0]), n_obj/10)
        self.assertLessEqual(len(np.where(np.isnan(y_pix))[0]), n_obj/10)

        x_pix_test, y_pix_test = _pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs,
                                                           includeDistortion=False)
        try:
            np.testing.assert_array_equal(x_pix, x_pix_test)
            np.testing.assert_array_equal(y_pix, y_pix_test)
        except AssertionError:
            n_problematic = 0
            for xx, yy, xt, yt in zip(x_pix, y_pix, x_pix_test, y_pix_test):
                if xx!=xt or yy!=yt:
                    if (not np.isnan(xx) and not np.isnan(xt) and
                        not np.isnan(yy) and not np.isnan(yt)):
                        print(xx,yy,xt,yt)

                        n_problematic += 1
            if n_problematic>0:
                raise


        # test that exceptions are raised when incomplete ObservationMetaData are used
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, mjd=59580.0)
        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("rotSkyPos", context.exception.args[0])

        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=35.0)
        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("mjd", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDecLSST(ra_list, dec_list)
        self.assertIn("ObservationMetaData", context.exception.args[0])

        # check that exceptions are raised when ra_list, dec_list are of the wrong shape
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=24.0, mjd=43000.0)
        with self.assertRaises(RuntimeError) as context:
            _pixelCoordsFromRaDecLSST(ra_list, dec_list[:5], obs_metadata=obs)
        self.assertIn("same length", context.exception.args[0])

    def test_pixel_coords_from_ra_dec_degrees(self):
        """
        Test that pixelCoordsFromRaDec and pixelCoordsFromRaDecLSST agree
        """
        raP = 74.2
        decP = 13.0
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP,
                                  rotSkyPos=13.0, mjd=43441.0)

        n_obj = 1000
        rng = np.random.RandomState(83241)
        rr = rng.random_sample(n_obj)*1.75
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = raP + rr*np.cos(theta)
        dec_list = decP + rr*np.sin(theta)

        x_pix, y_pix = pixelCoordsFromRaDec(ra_list, dec_list, obs_metadata=obs, camera=self.camera,
                                            includeDistortion=False)
        self.assertLessEqual(len(np.where(np.isnan(x_pix))[0]), n_obj/10)
        self.assertLessEqual(len(np.where(np.isnan(y_pix))[0]), n_obj/10)

        x_pix_test, y_pix_test = pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs,
                                                          includeDistortion=False)
        try:
            np.testing.assert_array_equal(x_pix, x_pix_test)
            np.testing.assert_array_equal(y_pix, y_pix_test)
        except AssertionError:
            n_problematic = 0
            for xx, yy, xt, yt in zip(x_pix, y_pix, x_pix_test, y_pix_test):
                if xx!=xt or yy!=yt:
                    if (not np.isnan(xx) and not np.isnan(xt) and
                        not np.isnan(yy) and not np.isnan(yt)):
                        print(xx,yy,xt,yt)

                        n_problematic += 1
            if n_problematic>0:
                raise

        # test when we force a chipName
        x_pix, y_pix = pixelCoordsFromRaDec(ra_list, dec_list, chipName=['R:2,2 S:1,1'],
                                            obs_metadata=obs, camera=self.camera,
                                            includeDistortion=False)
        self.assertLessEqual(len(np.where(np.isnan(x_pix))[0]), n_obj/10)
        self.assertLessEqual(len(np.where(np.isnan(y_pix))[0]), n_obj/10)

        x_pix_test, y_pix_test = pixelCoordsFromRaDecLSST(ra_list, dec_list, chipName=['R:2,2 S:1,1'],
                                                          obs_metadata=obs,
                                                          includeDistortion=False)
        np.testing.assert_array_equal(x_pix, x_pix_test)
        np.testing.assert_array_equal(y_pix, y_pix_test)

        # test without distortion
        x_pix, y_pix = pixelCoordsFromRaDec(ra_list, dec_list, obs_metadata=obs, camera=self.camera,
                                            includeDistortion=False)
        self.assertLessEqual(len(np.where(np.isnan(x_pix))[0]), n_obj/10)
        self.assertLessEqual(len(np.where(np.isnan(y_pix))[0]), n_obj/10)

        x_pix_test, y_pix_test = pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs,
                                                          includeDistortion=False)
        try:
            np.testing.assert_array_equal(x_pix, x_pix_test)
            np.testing.assert_array_equal(y_pix, y_pix_test)
        except AssertionError:
            n_problematic = 0
            for xx, yy, xt, yt in zip(x_pix, y_pix, x_pix_test, y_pix_test):
                if xx!=xt or yy!=yt:
                    if (not np.isnan(xx) and not np.isnan(xt) and
                        not np.isnan(yy) and not np.isnan(yt)):
                        print(xx,yy,xt,yt)

                        n_problematic += 1
            if n_problematic>0:
                raise

        # test that exceptions are raised when incomplete ObservationMetaData are used
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, mjd=59580.0)
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("rotSkyPos", context.exception.args[0])

        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=35.0)
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        self.assertIn("mjd", context.exception.args[0])

        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDecLSST(ra_list, dec_list)
        self.assertIn("ObservationMetaData", context.exception.args[0])

        # check that exceptions are raised when ra_list, dec_list are of the wrong shape
        obs = ObservationMetaData(pointingRA=raP, pointingDec=decP, rotSkyPos=24.0, mjd=43000.0)
        with self.assertRaises(RuntimeError) as context:
            pixelCoordsFromRaDecLSST(ra_list, dec_list[:5], obs_metadata=obs)
        self.assertIn("same length", context.exception.args[0])


class MotionTestCase(unittest.TestCase):
    """
    This class will contain test methods to verify that the LSST camera utils
    work correctly when proper motion, parallax, and v_rad are non-zero
    """
    @classmethod
    def setUpClass(cls):
        cls.camera = LsstSimMapper().camera

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

    def test_chip_name(self):
        """
        Test that chipNameFromRaDecLSST with non-zero proper motion etc.
        agrees with chipNameFromPupilCoords when pupilCoords are
        calculated with the same proper motion, etc.
        """
        (obs, ra_list, dec_list,
         pm_ra_list, pm_dec_list,
         parallax_list, v_rad_list) = self.set_data(11)

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

            name_test = chipNameFromRaDecLSST(ra_list, dec_list,
                                              pm_ra=pm_ra, pm_dec=pm_dec,
                                              parallax=parallax, v_rad=v_rad,
                                              obs_metadata=obs)

            name_radians = _chipNameFromRaDecLSST(np.radians(ra_list), np.radians(dec_list),
                                                  pm_ra=radiansFromArcsec(pm_ra),
                                                  pm_dec=radiansFromArcsec(pm_dec),
                                                  parallax=radiansFromArcsec(parallax), v_rad=v_rad,
                                                  obs_metadata=obs)

            np.testing.assert_array_equal(name_control, name_test)
            np.testing.assert_array_equal(name_control, name_radians)
            self.assertGreater(len(np.unique(name_control.astype(str))), 4)
            self.assertLess(len(np.where(np.equal(name_control, None))[0]), len(name_control)/4)

    def test_pixel_coords(self):
        """
        Test that pixelCoordsFromRaDecLSST with non-zero proper motion etc.
        agrees with pixelCoordsFromPupilCoords when pupilCoords are
        calculated with the same proper motion, etc.
        """
        (obs, ra_list, dec_list,
         pm_ra_list, pm_dec_list,
         parallax_list, v_rad_list) = self.set_data(26)

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

            xpx_control, ypx_control = pixelCoordsFromPupilCoords(xp, yp, camera=self.camera,
                                                                  includeDistortion=False)

            xpx_test, ypx_test = pixelCoordsFromRaDecLSST(ra_list, dec_list,
                                                          pm_ra=pm_ra, pm_dec=pm_dec,
                                                          parallax=parallax, v_rad=v_rad,
                                                          obs_metadata=obs,
                                                          includeDistortion=False)

            xpx_radians, ypx_radians = _pixelCoordsFromRaDecLSST(np.radians(ra_list), np.radians(dec_list),
                                                                 pm_ra=radiansFromArcsec(pm_ra),
                                                                 pm_dec=radiansFromArcsec(pm_dec),
                                                                 parallax=radiansFromArcsec(parallax),
                                                                 v_rad=v_rad, obs_metadata=obs,
                                                                 includeDistortion=False)

            np.testing.assert_array_equal(xpx_control, xpx_test)
            np.testing.assert_array_equal(ypx_control, ypx_test)
            np.testing.assert_array_equal(xpx_control, xpx_radians)
            np.testing.assert_array_equal(ypx_control, ypx_radians)
            self.assertLess(len(np.where(np.isnan(xpx_control))[0]), len(xpx_test)/4)
            self.assertLess(len(np.where(np.isnan(ypx_control))[0]), len(ypx_test)/4)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

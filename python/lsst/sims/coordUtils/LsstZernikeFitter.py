import numpy as np
import os
import numbers
import palpy

from lsst.utils import getPackageDir
from lsst.sims.utils import ZernikePolynomialGenerator
from lsst.sims.utils import ZernikeRadialError
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import DMtoCameraPixelTransformer
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE, FIELD_ANGLE, SCIENCE
import lsst.afw.geom as afwGeom
from lsst.sims.utils.CodeUtilities import _validate_inputs


__all__ = ["LsstZernikeFitter"]


def _rawPupilCoordsFromObserved(ra_obs, dec_obs, ra0, dec0, rotSkyPos):
    """
    Convert Observed RA, Dec into pupil coordinates

    Parameters
    ----------
    ra_obs is the observed RA in radians

    dec_obs is the observed Dec in radians

    ra0 is the RA of the boresite in radians

    dec0 is the Dec of the boresite in radians

    rotSkyPos is in radians

    Returns
    --------
    A numpy array whose first row is the x coordinate on the pupil in
    radians and whose second row is the y coordinate in radians
    """

    are_arrays = _validate_inputs([ra_obs, dec_obs], ['ra_obs', 'dec_obs'],
                                  "pupilCoordsFromObserved")

    theta = -1.0*rotSkyPos

    ra_pointing = ra0
    dec_pointing = dec0

    # palpy.ds2tp performs the gnomonic projection on ra_in and dec_in
    # with a tangent point at (pointingRA, pointingDec)
    #
    if not are_arrays:
        try:
            x, y = palpy.ds2tp(ra_obs, dec_obs, ra_pointing, dec_pointing)
        except:
            x = np.NaN
            y = np.NaN
    else:
        try:
            x, y = palpy.ds2tpVector(ra_obs, dec_obs, ra_pointing, dec_pointing)
        except:
            # apparently, one of your ra/dec values was improper; we will have to do this
            # element-wise, putting NaN in the place of the bad values
            x = []
            y = []
            for rr, dd in zip(ra_obs, dec_obs):
                try:
                    xx, yy = palpy.ds2tp(rr, dd, ra_pointing, dec_pointing)
                except:
                    xx = np.NaN
                    yy = np.NaN
                x.append(xx)
                y.append(yy)
            x = np.array(x)
            y = np.array(y)

    # rotate the result by rotskypos (rotskypos being "the angle of the sky relative to
    # camera coordinates" according to phoSim documentation) to account for
    # the rotation of the focal plane about the telescope pointing

    x_out = x*np.cos(theta) - y*np.sin(theta)
    y_out = x*np.sin(theta) + y*np.cos(theta)

    return np.array([x_out, y_out])


class LsstZernikeFitter(object):
    """
    This class will fit and then apply the Zernike polynomials needed
    to correct the FIELD_ANGLE to FOCAL_PLANE transformation for the
    filter-dependent part.
    """

    def __init__(self):
        self._camera = lsst_camera()
        self._pixel_transformer = DMtoCameraPixelTransformer()
        self._z_gen = ZernikePolynomialGenerator()

        self._rr = 450.0  # radius in mm of circle containing LSST focal plane

        self._band_to_int = {}
        self._band_to_int['u'] = 0
        self._band_to_int['g'] = 1
        self._band_to_int['r'] = 2
        self._band_to_int['i'] = 3
        self._band_to_int['z'] = 4
        self._band_to_int['y'] = 5

        self._int_to_band = 'ugrizy'

        self._n_grid = []
        self._m_grid = []
        for n in range(4):
            for m in range(-n, n+1, 2):
                self._n_grid.append(n)
                self._m_grid.append(m)

        self._build_transformations()

    def _get_coeffs(self, x_in, y_in, x_out, y_out):
        """
        Get the coefficients of the best fit Zernike Polynomial
        expansion that transforms from x_in, y_in to x_out, y_out.

        Returns numpy arrays of the Zernike Polynomial expansion
        coefficients in x and y.  Zernike Polynomials correspond
        to the radial and angular orders stored in self._n_grida
        and self._m_grid.
        """

        polynomials ={}
        for n, m in zip(self._n_grid, self._m_grid):
            try:
                values = self._z_gen.evaluate_xy(x_in/self._rr, y_in/self._rr, n, m)
            except ZernikeRadialError:
                msg = "Some of the data you are fitting to in LsstZernikeFitter is outside "
                msg += "the r = %e mm circle circumscribing the LSST focal plane. " % self._rr
                msg += "The Zernike polynomials we are fitting to are not defined "
                msg += "outside of that circle."
                raise RuntimeError(msg)

            polynomials[(n,m)] = values

        poly_keys = list(polynomials.keys())

        dx = x_out - x_in
        dy = y_out - y_in

        b = np.array([(dx*polynomials[k]).sum() for k in poly_keys])
        m = np.array([[(polynomials[k1]*polynomials[k2]).sum() for k1 in poly_keys]
                      for k2 in poly_keys])

        alpha_x_ = np.linalg.solve(m, b)
        alpha_x = {}
        for ii, kk in enumerate(poly_keys):
            alpha_x[kk] = alpha_x_[ii]


        b = np.array([(dy*polynomials[k]).sum() for k in poly_keys])
        m = np.array([[(polynomials[k1]*polynomials[k2]).sum() for k1 in poly_keys]
                      for k2 in poly_keys])

        alpha_y_ = np.linalg.solve(m, b)
        alpha_y = {}
        for ii, kk in enumerate(poly_keys):
            alpha_y[kk] = alpha_y_[ii]

        return alpha_x, alpha_y

    def _build_transformations(self):
        """
        Solve for and store the coefficients of the Zernike
        polynomial expansion of the difference between the
        naive and the bandpass-dependent optical distortions
        in the LSST camera.
        """
        catsim_dir = os.path.join(getPackageDir('sims_data'),
                                  'FocalPlaneData',
                                  'CatSimData')

        phosim_dir = os.path.join(getPackageDir('sims_data'),
                                  'FocalPlaneData',
                                  'PhoSimData')

        # the file which contains the input sky positions of the objects
        # that were given to PhoSim
        catsim_catalog = os.path.join(catsim_dir,'predicted_positions.txt')

        with open(catsim_catalog, 'r') as input_file:
            header = input_file.readline()
        params = header.strip().split()
        ra0 = np.radians(float(params[2]))
        dec0 = np.radians(float(params[4]))
        rotSkyPos = np.radians(float(params[6]))

        catsim_dtype = np.dtype([('id', int), ('xmm_old', float), ('ymm_old', float),
                                 ('xpup', float), ('ypup', float),
                                 ('raObs', float), ('decObs', float)])

        catsim_data = np.genfromtxt(catsim_catalog, dtype=catsim_dtype)

        sorted_dex = np.argsort(catsim_data['id'])
        catsim_data=catsim_data[sorted_dex]

        # convert from RA, Dec to pupil coors/FIELD_ANGLE
        x_field, y_field = _rawPupilCoordsFromObserved(np.radians(catsim_data['raObs']),
                                                       np.radians(catsim_data['decObs']),
                                                       ra0, dec0, rotSkyPos)

        # convert from FIELD_ANGLE to FOCAL_PLANE without attempting to model
        # the optical distortions in the telescope
        field_to_focal = self._camera.getTransform(FIELD_ANGLE, FOCAL_PLANE)
        catsim_xmm = np.zeros(len(x_field), dtype=float)
        catsim_ymm = np.zeros(len(y_field), dtype=float)
        for ii, (xx, yy) in enumerate(zip(x_field, y_field)):
            focal_pt = field_to_focal.applyForward(afwGeom.Point2D(xx, yy))
            catsim_xmm[ii] = focal_pt.getX()
            catsim_ymm[ii] = focal_pt.getY()

        phosim_dtype = np.dtype([('id', int), ('phot', float),
                                 ('xpix', float), ('ypix', float)])

        self._pupil_to_focal = {}
        self._focal_to_pupil = {}

        for i_filter in range(6):
            self._pupil_to_focal[self._int_to_band[i_filter]] = {}
            self._focal_to_pupil[self._int_to_band[i_filter]] = {}
            phosim_xmm = np.zeros(len(catsim_data['ypup']), dtype=float)
            phosim_ymm = np.zeros(len(catsim_data['ypup']), dtype=float)

            for det in self._camera:
                if det.getType() != SCIENCE:
                    continue
                pixels_to_focal = det.getTransform(PIXELS, FOCAL_PLANE)
                det_name = det.getName()
                bbox = det.getBBox()
                det_name_m = det_name.replace(':','').replace(',','').replace(' ','_')

                # read in the actual pixel positions of the sources as realized
                # by PhoSim
                centroid_name = 'centroid_lsst_e_2_f%d_%s_E000.txt' % (i_filter, det_name_m)
                full_name = os.path.join(phosim_dir, centroid_name)
                phosim_data = np.genfromtxt(full_name, dtype=phosim_dtype, skip_header=1)

                # make sure that the data we are fitting to is not too close
                # to the edge of the detector
                assert phosim_data['xpix'].min() > bbox.getMinY() + 50.0
                assert phosim_data['xpix'].max() < bbox.getMaxY() - 50.0
                assert phosim_data['ypix'].min() > bbox.getMinX() + 50.0
                assert phosim_data['ypix'].max() < bbox.getMaxX() - 50.0

                xpix, ypix = self._pixel_transformer.dmPixFromCameraPix(phosim_data['xpix'],
                                                                        phosim_data['ypix'],
                                                                        det_name)
                xmm = np.zeros(len(xpix), dtype=float)
                ymm = np.zeros(len(ypix), dtype=float)
                for ii in range(len(xpix)):
                    focal_pt = pixels_to_focal.applyForward(afwGeom.Point2D(xpix[ii], ypix[ii]))
                    xmm[ii] = focal_pt.getX()
                    ymm[ii] = focal_pt.getY()
                phosim_xmm[phosim_data['id']-1] = xmm
                phosim_ymm[phosim_data['id']-1] = ymm

            # solve for the coefficients of the Zernike expansions
            # necessary to model the optical transformations and go
            # from the naive focal plane positions (catsim_xmm, catsim_ymm)
            # to the PhoSim realized focal plane positions
            alpha_x, alpha_y = self._get_coeffs(catsim_xmm, catsim_ymm,
                                                phosim_xmm, phosim_ymm)

            self._pupil_to_focal[self._int_to_band[i_filter]]['x'] = alpha_x
            self._pupil_to_focal[self._int_to_band[i_filter]]['y'] = alpha_y

            # solve for the coefficients to the Zernike expansions
            # necessary to go back from the PhoSim realized focal plane
            # positions to the naive CatSim predicted focal plane
            # positions
            alpha_x, alpha_y = self._get_coeffs(phosim_xmm, phosim_ymm,
                                                catsim_xmm, catsim_ymm)

            self._focal_to_pupil[self._int_to_band[i_filter]]['x'] = alpha_x
            self._focal_to_pupil[self._int_to_band[i_filter]]['y'] = alpha_y

    def _apply_transformation(self, transformation_dict, xmm, ymm, band):
        """
        Parameters
        ----------
        tranformation_dict -- a dict containing the coefficients
        of the Zernike decomposition to be applied

        xmm -- the input x position in mm

        ymm -- the input y position in mm

        band -- the filter in which we are operating

        Returns
        -------
        dx -- the x offset resulting from the transformation

        dy -- the y offset resulting from the transformation
        """
        if isinstance(band, int):
            band = self._int_to_band[band]

        if isinstance(xmm, numbers.Number):
            dx = 0.0
            dy = 0.0
        else:
            dx = np.zeros(len(xmm), dtype=float)
            dy = np.zeros(len(ymm), dtype=float)

        for kk in self._pupil_to_focal[band]['x']:
            try:
                values = self._z_gen.evaluate_xy(xmm/self._rr, ymm/self._rr, kk[0], kk[1])
            except ZernikeRadialError:
                msg = "Some of the points you are transforming in LsstZernikeFitter are outside "
                msg += "the r = %e mm circle circumscribing the LSST focal plane. " % self._rr
                msg += "The Zernike polynomials we are fitting to are not defined "
                msg += "outside of that circle."
                raise RuntimeError(msg)

            dx += transformation_dict[band]['x'][kk]*values
            dy += transformation_dict[band]['y'][kk]*values

        return dx, dy

    def dxdy(self, xmm, ymm, band):
        """
        Apply the transformation necessary when going from pupil
        coordinates to focal plane coordinates.

        The recipe to correctly use this method is

        xf0, yf0 = focalPlaneCoordsFromPupilCoords(xpupil, ypupil,
                                                   camera=lsst_camera())

        dx, dy = LsstZernikeFitter().dxdy(xf0, yf0, band=band)

        xf = xf0 + dx
        yf = yf0 + dy

        xf and yf are now the actual position in millimeters on the
        LSST focal plane corresponding to xpupil, ypupil

        Parameters
        ----------
        xmm -- the naive x focal plane position in mm

        ymm -- the naive y focal plane position in mm

        band -- the filter in which we are operating

        Returns
        -------
        dx -- the offset in the x focal plane position in mm

        dy -- the offset in the y focal plane position in mm
        """
        return self._apply_transformation(self._pupil_to_focal, xmm, ymm, band)

    def dxdy_inverse(self, xmm, ymm, band):
        """
        Apply the transformation necessary when going from focal
        plane coordinates to pupil coordinates.

        The recipe to correctly use this method is

        dx, dy = LsstZernikeFitter().dxdy_inverse(xf, yf, band=band)

        xp, yp = pupilCoordsFromFocalPlaneCoords(xf+dx,
                                                 yf+dy,
                                                 camera=lsst_camera()

        xp and yp are now the actual position in radians on the pupil
        corresponding to the focal plane coordinates xf, yf

        Parameters
        ----------
        xmm -- the naive x focal plane position in mm

        ymm -- the naive y focal plane position in mm

        band -- the filter in which we are operating

        Returns
        -------
        dx -- the offset in the x focal plane position in mm

        dy -- the offset in the y focal plane position in mm
        """
        return self._apply_transformation(self._focal_to_pupil, xmm, ymm, band)

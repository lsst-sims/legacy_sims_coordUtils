import numpy as np
import os
import numbers

from lsst.utils import getPackageDir
from lsst.sims.utils import ZernikePolynomialGenerator
from lsst.sims.coordUtils import lsst_camera
from lsst.sims.coordUtils import DMtoCameraPixelTransformer
from lsst.afw.cameraGeom import PIXELS, FOCAL_PLANE, SCIENCE
import lsst.afw.geom as afwGeom


__all__ = ["LsstZernikeFitter"]


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

        self._get_fit_coeffs()

    def _get_fit_coeffs(self):
        catsim_dir = os.path.join(getPackageDir('sims_data'),
                                  'FocalPlaneData',
                                  'CatSimData')

        phosim_dir = os.path.join(getPackageDir('sims_data'),
                                  'FocalPlaneData',
                                  'PhoSimData')

        catsim_catalog = os.path.join(catsim_dir,'predicted_positions.txt')

        catsim_dtype = np.dtype([('id', int), ('xmm', float), ('ymm', float),
                                 ('xpup', float), ('ypup', float),
                                 ('raObs', float), ('decObs', float)])

        catsim_data = np.genfromtxt(catsim_catalog, dtype=catsim_dtype)

        sorted_dex = np.argsort(catsim_data['id'])
        catsim_data=catsim_data[sorted_dex]

        catsim_radius = np.sqrt(catsim_data['xmm']**2 + catsim_data['ymm']**2)
        self._rr = catsim_radius.max()

        catsim_x = catsim_data['xmm']/self._rr
        catsim_y = catsim_data['ymm']/self._rr

        polynomials ={}
        for n, m in zip(self._n_grid, self._m_grid):
            values = self._z_gen.evaluate_xy(catsim_x, catsim_y, n, m)
            polynomials[(n,m)] = values

        poly_keys = list(polynomials.keys())

        phosim_dtype = np.dtype([('id', int), ('phot', float),
                                 ('xpix', float), ('ypix', float)])


        self._transformations = {}

        for i_filter in range(6):
            self._transformations[self._int_to_band[i_filter]] = {}
            dx = np.zeros(len(catsim_data['xpup']), dtype=float)
            dy = np.zeros(len(catsim_data['ypup']), dtype=float)
            phosim_xmm = np.zeros(len(catsim_data['ypup']), dtype=float)
            phosim_ymm = np.zeros(len(catsim_data['ypup']), dtype=float)

            for det in self._camera:
                if det.getType() != SCIENCE:
                    continue
                pixels_to_focal = det.getTransform(PIXELS, FOCAL_PLANE)
                det_name = det.getName()
                det_name_m = det_name.replace(':','').replace(',','').replace(' ','_')
                centroid_name = 'centroid_lsst_e_2_f%d_%s_E000.txt' % (i_filter, det_name_m)
                full_name = os.path.join(phosim_dir, centroid_name)
                phosim_data = np.genfromtxt(full_name, dtype=phosim_dtype, skip_header=1)
                xpix, ypix = self._pixel_transformer.dmPixFromCameraPix(phosim_data['xpix'],
                                                                        phosim_data['ypix'],
                                                                        det_name)
                xmm = np.zeros(len(xpix), dtype=float)
                ymm = np.zeros(len(ypix), dtype=float)
                for ii in range(len(xpix)):
                    focal_pt = pixels_to_focal.applyForward(afwGeom.Point2D(xpix[ii], ypix[ii]))
                    xmm[ii] = focal_pt.getX()
                    ymm[ii] = focal_pt.getY()
                dx[phosim_data['id']-1] = xmm - catsim_data['xmm'][phosim_data['id']-1]
                dy[phosim_data['id']-1] = ymm - catsim_data['ymm'][phosim_data['id']-1]
                phosim_xmm[phosim_data['id']-1] = xmm
                phosim_ymm[phosim_data['id']-1] = ymm

            b = np.array([(dx*polynomials[k]).sum() for k in poly_keys])
            m = np.array([[(polynomials[k1]*polynomials[k2]).sum() for k1 in poly_keys]
                          for k2 in poly_keys])

            alpha_x = np.linalg.solve(m, b)

            self._transformations[self._int_to_band[i_filter]]['x'] = {}
            for ii, kk in enumerate(poly_keys):
                self._transformations[self._int_to_band[i_filter]]['x'][kk] = alpha_x[ii]

            b = np.array([(dy*polynomials[k]).sum() for k in poly_keys])
            m = np.array([[(polynomials[k1]*polynomials[k2]).sum() for k1 in poly_keys]
                          for k2 in poly_keys])

            alpha_y = np.linalg.solve(m, b)

            self._transformations[self._int_to_band[i_filter]]['y'] = {}
            for ii, kk in enumerate(poly_keys):
                self._transformations[self._int_to_band[i_filter]]['y'][kk] = alpha_y[ii]

    def dxdy(self, xmm, ymm, band):
        """
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
        if isinstance(band, int):
            band = self._int_to_band[band]

        if isinstance(xmm, numbers.Number):
            dx = 0.0
            dy = 0.0
        else:
            dx = np.zeros(len(xmm), dtype=float)
            dy = np.zeros(len(ymm), dtype=float)

        for kk in self._transformations[band]['x']:
            values = self._z_gen.evaluate_xy(xmm/self._rr, ymm/self._rr, kk[0], kk[1])
            dx += self._transformations[band]['x'][kk]*values
            dy += self._transformations[band]['y'][kk]*values

        return dx, dy

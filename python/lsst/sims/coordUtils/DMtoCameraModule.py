import numpy as np
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, TAN_PIXELS
from lsst.afw.cameraGeom import FIELD_ANGLE
from lsst.sims.coordUtils import lsst_camera


__all__ = ["DMtoCameraPixelTransformer"]


class DMtoCameraPixelTransformer(object):

    def __init__(self):
        self._camera = lsst_camera()

    def getBBox(self, detector_name):
        """
        Return the bounding box for the detector named by detector_name
        """
        if not hasattr(self, '_bbox_cache'):
            self._bbox_cache = {}

        if detector_name not in self._bbox_cache:
            dm_bbox = self._camera[detector_name].getBBox()
            dm_min = dm_bbox.getMin()
            dm_max = dm_bbox.getMax()
            cam_bbox = afwGeom.Box2I(minimum=afwGeom.coordinates.Point2I(dm_min[1], dm_min[0]),
                                     maximum=afwGeom.coordinates.Point2I(dm_max[1], dm_max[0]))

            self._bbox_cache[detector_name] = cam_bbox

        return self._bbox_cache[detector_name]

    def getCenterPixel(self, detector_name):
         """
         Return the central pixel for the detector named by detector_name
         """
         if not hasattr(self, '_center_pixel_cache'):
             self._center_pixel_cache = {}

         if detector_name not in self._center_pixel_cache:
             centerPoint = self._camera[detector_name].getCenter(FOCAL_PLANE)
             centerPixel_dm = self._camera[detector_name].getTransform(FOCAL_PLANE, PIXELS).applyForward(centerPoint)
             centerPixel_cam = afwGeom.coordinates.Point2D(centerPixel_dm.getY(), centerPixel_dm.getX())
             self._center_pixel_cache[detector_name] = centerPixel_cam

         return self._center_pixel_cache[detector_name]

    def getTanPixelBounds(self, detector_name):
        """
        Return the min and max pixel values of a detector, assuming
        all radial distortions are set to zero (i.e. using the afwCameraGeom
        TAN_PIXELS coordinate system)

        Parameters
        ----------
        detector_name is a string denoting the name of the detector

        Returns
        -------
        xmin, xmax, ymin, ymax pixel values
        """
        if not hasattr(self, '_tan_pixel_bounds_cache'):
            self._tan_pixel_bounds_cache = {}

        if detector_name not in self._tan_pixel_bounds_cache:
            dm_xmin, dm_xmax, dm_ymin, dm_ymax = GalSimCameraWrapper.getTanPixelBounds(self, detector_name)
            self._tan_pixel_bounds_cache[detector_name] = (dm_ymin, dm_ymax, dm_xmin, dm_xmax)

        return self._tan_pixel_bounds_cache[detector_name]

    def cameraPixFromDMPix(self, dm_xPix, dm_yPix, chipName):
        """
        Convert DM pixel coordinates into camera pixel coordinates

        Parameters:
        -----------
        dm_xPix -- the x pixel coordinate in the DM system (either
        a number or an array)

        dm_yPix -- the y pixel coordinate in the DM system (either
        a number or an array)

        chipName designates the names of the chips on which the pixel
        coordinates will be reckoned.  Can be either single value, an array, or None.
        If an array, there must be as many chipNames as there are (RA, Dec) pairs.
        If a single value, all of the pixel coordinates will be reckoned on the same
        chip.  If None, this method will calculate which chip each(RA, Dec) pair actually
        falls on, and return pixel coordinates for each (RA, Dec) pair on the appropriate
        chip.  Default is None.

        Returns
        -------
        a 2-D numpy array in which the first row is the x pixel coordinate
        and the second row is the y pixel coordinate.  These pixel coordinates
        are defined in the Camera team system, rather than the DM system.
        """
        cam_yPix = dm_xPix

        if isinstance(chipName, list) or isinstance(chipName, np.ndarray):
            cam_xPix = np.zeros(len(dm_xPix))
            for ix, (det_name, yy) in enumerate(zip(chipName, dm_yPix)):
                cam_center_pix = self.getCenterPixel(det_name)
                cam_xPix[ix] = 2.0*cam_center_pix.getX() - dm_yPix
        else:
            cam_center_pix = self.getCenterPixel(chipName)
            cam_xPix = 2.0*cam_center_pix.getX() - dm_yPix

        return cam_xPix, cam_yPix

    def dmPixFromCameraPix(self, cam_x_pix, cam_y_pix, chipName):
        """
        Convert pixel coordinates from the Camera Team system to the DM system

        Parameters
        ----------
        cam_x_pix -- the x pixel coordinate in the Camera Team system
        (can be either a float or a numpy array)

        cam_y_pix -- the y pixel coordiantes in the Camera Team system
        (can be either a float or a numpy array)

        chipName -- the name of the chip(s) on which the pixel coordinates
        are defined.  This can be a list (in which case there should be one chip name
        for each (cam_xpix, cam_ypix) coordinate pair), or a single value (in which
        case, all of the (cam_xpix, cam_ypi) points will be reckoned on that chip).

        Returns
        -------
        dm_x_pix -- the x pixel coordinate(s) in the DM system (either
        a float or a numpy array)

        dm_y_pix -- the y pixel coordinate(s) in the DM system (either
        a float or a numpy array)
        """

        dm_x_pix = cam_y_pix
        if isinstance(chipName, list) or isinstance(chipName, np.ndarray):
            center_pix_dict = {}
            dm_y_pix = np.zeros(len(cam_x_pix))
            for ix, (det_name, xx) in enumerate(zip(chipName, cam_x_pix)):
                if det_name not in center_pix_dict:
                    center_pix = self.getCenterPixel(det_name)
                    center_pix_dict[det_name] = center_pix
                else:
                    center_pix = center_pix_dict[det_name]
                dm_y_pix[ix] = 2.0*center_pix[0]-xx
        else:
            center_pix = self.getCenterPixel(chipName)
            dm_y_pix = 2.0*center_pix[0] - cam_x_pix

        return dm_x_pix, dm_y_pix

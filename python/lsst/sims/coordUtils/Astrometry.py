import numpy
import ctypes
import math
import palpy as pal
import lsst.afw.geom as afwGeom
from lsst.afw.cameraGeom import PUPIL, PIXELS, FOCAL_PLANE
from lsst.afw.cameraGeom import SCIENCE
from lsst.sims.catalogs.measures.instance import compound
from lsst.sims.utils import haversine, radiansToArcsec, arcsecToRadians
from lsst.sims.utils import equatorialToGalactic, cartesianToSpherical, sphericalToCartesian

from lsst.sims.coordUtils.AstrometryUtils import appGeoFromICRS, observedFromAppGeo
from lsst.sims.coordUtils.AstrometryUtils import observedFromICRS, calculatePupilCoordinates
from lsst.sims.coordUtils.AstrometryUtils import calculateGnomonicProjection
from lsst.sims.coordUtils.CameraUtils import findChipName

__all__ = ["AstrometryBase", "AstrometryStars", "AstrometryGalaxies",
           "CameraCoords"]

class AstrometryBase(object):
    """Collection of astrometry routines that operate on numpy arrays"""

    @compound('glon','glat')
    def get_galactic_coords(self):
        """
        Getter for galactic coordinates, in case the catalog class does not provide that

        Reads in the ra and dec from the data base and returns columns with galactic
        longitude and latitude.

        All angles are in radians
        """
        ra=self.column_by_name('raJ2000')
        dec=self.column_by_name('decJ2000')

        glon, glat = self.equatorialToGalactic(ra,dec)

        return numpy.array([glon,glat])



    @compound('x_focal_nominal', 'y_focal_nominal')
    def get_gnomonicProjection(self):
        ra = self.column_by_name('raObserved')
        dec = self.column_by_name('decObserved')
        return calculateGnomonicProjection(ra, dec, obs_metadata=self.obs_metadata,
                                           epoch=self.db_obj.epoch)

    @compound('x_pupil','y_pupil')
    def get_skyToPupil(self):
        """
        Take an input RA and dec from the sky and convert it to coordinates
        in the pupil.
        """

        raObj = self.column_by_name('raObserved')
        decObj = self.column_by_name('decObserved')

        return calculatePupilCoordinates(raObj, decObj, epoch=self.db_obj.epoch,
                                         obs_metadata=self.obs_metadata)

class CameraCoords(AstrometryBase):
    """Methods for getting coordinates from the camera object"""
    camera = None
    allow_multiple_chips = False #this is a flag which, if true, would allow
                                 #findChipName to return objects that land on
                                 #multiple chips; only the first chip would be
                                 #written to the catalog


    def calculatePixelCoordinates(self, xPupil=None, yPupil=None, ra=None, dec=None, chipNames=None,
                                  obs_metadata=None, epoch=None, camera=None):
        """
        Get the pixel positions (or nan if not on a chip) for all objects in the catalog

        @param [in] xPupil a numpy array containing x pupil coordinates

        @param [in] yPupil a numpy array containing y pupil coordinates

        @param [in] ra one could alternatively provide a numpy array of ra and...

        @param [in] ...dec (both in radians)

        @param [in] chipNames a numpy array of chipNames.  If it is None, this method will call findChipName
        to find the array.  The option exists for the user to specify chipNames, just in case the user
        has already called findChipName for some reason.

        @param [in] obs_metadata is an ObservationMetaData object describing the telescope pointing
        (optional)

        @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
        (in years; optional)

            The optional arguments above are there to be passed to calculatePupilCoordinates if you choose
            to call this routine specifying ra and dec, rather than xPupil and yPupil.  If they are
            not set, calculatePupilCoordinates will try to set them from self and db_obj,
            assuming that this ro;utine is being called from an InstanceCatalog daughter class.
            If that is not the case, an exception will be raised.

        @param [in] camera is an afwCameraGeom object specifying the attributes of the camera.
        This is an optional argument to be passed to findChipName; if it is None, findChipName
        will try to set it from self.camera, assuming that the code is in an InstanceCatalog
        daughter class.  If this is not so, an exception will be raised.

        @param [out] a numpy array of pixel coordinates

        """

        if not camera:
            if hasattr(self, 'camera'):
                camera = self.camera

            if not camera:
                raise RuntimeError("No camera defined.  Cannot calculate pixel coordinates")

        specifiedPupil = False
        specifiedRaDec = False

        specifiedPupil = (xPupil is not None and yPupil is not None)
        specifiedRaDec = (ra is not None and dec is not None)

        if not specifiedPupil and not specifiedRaDec:
            raise RuntimeError("You need to specifiy either pupil coordinates or equatorial coordinates in calculatePixelCoordinates")

        if specifiedPupil and specifiedRaDec:
            raise RuntimeError("You cannot specify both pupil coordinates and equatorial coordinates in calculatePixelCoordinates")

        if specifiedRaDec:

            if epoch is None:
                if hasattr(self, 'db_obj'):
                    epoch = self.db_obj.epoch

            if obs_metadata is None:
                if hasattr(self, 'obs_metadata'):
                    obs_metadata = self.obs_metadata

            if epoch is None:
                raise RuntimeError("You need to specify an epoch to run calculatePixelCoordinates " + \
                                   "on these inputs")

            if obs_metadata is None:
                raise RuntimeError("You need to specify an ObservationMetaDAta to run " + \
                                   "calculatePixelCoordinates on these inputs")

            xPupil, yPupil = calculatePupilCoordinates(ra, dec,
                                                       obs_metadata=obs_metadata,
                                                       epoch=epoch)

        if chipNames is None:
            chipNames = findChipName(xPupil = xPupil, yPupil = yPupil, camera=camera)

        xPix = []
        yPix = []
        for name, x, y in zip(chipNames, xPupil, yPupil):
            if not name:
                xPix.append(numpy.nan)
                yPix.append(numpy.nan)
                continue
            cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            det = camera[name]
            cs = det.makeCameraSys(PIXELS)
            detPoint = camera.transform(cp, cs)
            xPix.append(detPoint.getPoint().getX())
            yPix.append(detPoint.getPoint().getY())
        return numpy.array([xPix, yPix])

    def calculateFocalPlaneCoordinates(self, xPupil=None, yPupil=None, ra=None, dec=None,
                                       obs_metadata=None, epoch=None, camera=None):
        """
        Get the focal plane coordinates for all objects in the catalog.

        @param [in] xPupil a numpy array of x pupil coordinates

        @param [in] yPupil a numpy array of y pupil coordinates

        @param [in] alternatively, one can specify numpy arrays of ra and dec (in radians)

        @param [in] obs_metadata is an ObservationMetaData object describing the telescope
        pointing (optional)

        @param [in] epoch is the julian epoch of the mean equinox used for coordinate transformations
        (in years; optional)

            The optional arguments are there to be passed to calculatePupilCoordinates if you choose
            to call this routine specifying ra and dec, rather than xPupil and yPupil.  If they are
            not set, calculatePupilCoordinates will try to set them from self and db_obj,
            assuming that this routine is being called from an InstanceCatalog daughter class.
            If that is not the case, an exception will be raised.

        @param [out] a numpy array in which the first row is the x pixel coordinates
        and the second row is the y pixel coordinates

        """

        specifiedPupil = (xPupil is not None and yPupil is not None)
        specifiedRaDec = (ra is not None and dec is not None)

        if not specifiedPupil and not specifiedRaDec:
            raise RuntimeError("You must specify either pupil coordinates or equatorial coordinates to call calculateFocalPlaneCoordinates")

        if specifiedPupil and specifiedRaDec:
            raise RuntimeError("You cannot specify both pupil and equaltorial coordinates when calling calculateFocalPlaneCoordinates")

        if not camera:
            if hasattr(self, 'camera'):
                camera = self.camera

            if not camera:
                raise RuntimeError("No camera defined.  Cannot calculate focalplane coordinates")

        if specifiedRaDec:
            if epoch is None:
                if hasattr(self, 'db_obj'):
                    epoch = self.db_obj.epoch

            if obs_metadata is None:
                if hasattr(self, 'obs_metadata'):
                    obs_metadata = self.obs_metadata

            if epoch is None:
                raise RuntimeError("You have to specify an epoch to run " + \
                                    "calculateFocalPlaneCoordinates on these inputs")

            if obs_metadata is None:
                raise RuntimeError("You have to specify an ObservationMetaData to run " + \
                                   "calculateFocalPlaneCoordinates on these inputs")

            xPupil, yPupil = calculatePupilCoordinates(ra ,dec,
                                                       obs_metadata=obs_metadata, epoch=epoch)

        xPix = []
        yPix = []
        for x, y in zip(xPupil, yPupil):
            cp = camera.makeCameraPoint(afwGeom.Point2D(x, y), PUPIL)
            fpPoint = camera.transform(cp, FOCAL_PLANE)
            xPix.append(fpPoint.getPoint().getX())
            yPix.append(fpPoint.getPoint().getY())
        return numpy.array([xPix, yPix])

    def get_chipName(self):
        """Get the chip name if there is one for each catalog entry"""
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))
        return findChipName(xPupil=xPupil, yPupil=yPupil, camera=self.camera)

    @compound('xPix', 'yPix')
    def get_pixelCoordinates(self):
        """Get the pixel positions (or nan if not on a chip) for all objects in the catalog"""
        if not self.camera:
            raise RuntimeError("No camera defined.  Cannot calculate pixel coordinates")
        chipNames = self.column_by_name('chipName')
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))

        return self.calculatePixelCoordinates(xPupil = xPupil, yPupil = yPupil, chipNames=chipNames)

    @compound('xFocalPlane', 'yFocalPlane')
    def get_focalPlaneCoordinates(self):
        """Get the focal plane coordinates for all objects in the catalog."""
        xPupil, yPupil = (self.column_by_name('x_pupil'), self.column_by_name('y_pupil'))

        return self.calculateFocalPlaneCoordinates(xPupil = xPupil, yPupil = yPupil)

class AstrometryGalaxies(AstrometryBase):
    """
    This mixin contains a getter for the corrected RA and dec which ignores parallax and proper motion
    """

    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')
        return observedFromICRS(ra, dec, includeRefraction = False, obs_metadata=self.obs_metadata,
                                epoch=self.db_obj.epoch)


    @compound('raObserved','decObserved')
    def get_observedCoordinates(self):
        """
        convert mean coordinates in the International Celestial Reference Frame
        to observed coordinates
        """
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')
        return observedFromICRS(ra, dec, obs_metadata=self.obs_metadata, epoch=self.db_obj.epoch)


class AstrometryStars(AstrometryBase):
    """
    This mixin contains a getter for the corrected RA and dec which takes account of proper motion and parallax
    """

    def observedStellarCoordinates(self, includeRefraction = True):
        """
        Getter which converts mean coordinates in the International Celestial
        Reference Frame to observed coordinates.
        """

        #TODO
        #are we going to store proper motion in raw radians per year
        #or in sky motion = cos(dec) * (radians per year)
        #PAL asks for radians per year inputs

        pr = self.column_by_name('properMotionRa') #in radians per year
        pd = self.column_by_name('properMotionDec') #in radians per year
        px = self.column_by_name('parallax') #in radians
        rv = self.column_by_name('radialVelocity') #in km/s; positive if receding
        ra = self.column_by_name('raJ2000')
        dec = self.column_by_name('decJ2000')

        return observedFromICRS(ra, dec, pm_ra = pr, pm_dec = pd, parallax = px, v_rad = rv,
                     includeRefraction = includeRefraction, obs_metadata=self.obs_metadata,
                     epoch=self.db_obj.epoch)


    @compound('raPhoSim','decPhoSim')
    def get_phoSimCoordinates(self):
        return self.observedStellarCoordinates(includeRefraction = False)

    @compound('raObserved','decObserved')
    def get_observedCoordinates(self):
        return self.observedStellarCoordinates()

import lsst.afw.cameraGeom.testUtils as testUtils

"""
Using cameraGeomExample.py in afw, we ought to be able to make an LSST-like
camera object

However, most of the methods behind findChipName assume that you are calling it
from a catalog object (with all of the attendant MJD, Site, dbobj.epoch
variables available.

Something must be done to make the method more general.

split the method up

one method will perform the calculation.  It will accept an ObservationMetaData
object, a Site object, and a reference epoch

the other method will be a getter that will actually perform the calculations

I suspect we should just split all of the astrometry methods

"""

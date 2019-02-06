import matplotlib
matplotlib.use('Agg')

from lsst.obs.lsstSim import LsstSimMapper
sim_camera = LsstSimMapper().camera
from lsst.afw.cameraGeom.utils import plotFocalPlane
plotFocalPlane(sim_camera, showFig=False, savePath='sim_camera.png')

from lsst.obs.lsst.phosim import PhosimMapper
phosim_camera = PhosimMapper().camera
plotFocalPlane(phosim_camera, showFig=False, savePath='phosim_camera.png')

from lsst.obs.lsst.imsim import ImsimMapper
imsim_camera = ImsimMapper().camera
plotFocalPlane(imsim_camera, showFig=False, savePath='imsim_camera.png')

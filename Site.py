""" Site Class

    Class defines the attributes of the site unless overridden
    ajc@astro 2/23/2010
    
    Restoring this so that the astrometry mixin in Astrometry.py
    can inherit the site information
    danielsf 1/27/2014

"""

class Site (object):
    """ 
    This class will store site information for use in Catalog objects.
    
    Defaults values are LSST site values
    """
    def __init__(self, longitude=-1.2320792, latitude=-0.517781017, height=2650, \
                 xPolar=0, yPolar=0, meanTemperature=284.655, meanPressure=749.3, \
                 meanHumidity=0.4, lapseRate=0.0065, **kwargs):
        
        self.longitude=longitude
        self.latitude=latitude
        self.height=height
        self.xPolar=xPolar
        self.yPolar=yPolar
        self.meanTemperature=meanTemperature
        self.meanPressure=meanPressure
        self.meanHumidity=meanHumidity
        self.lapseRate=lapseRate


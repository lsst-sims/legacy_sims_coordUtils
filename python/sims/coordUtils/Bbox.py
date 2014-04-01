class Bbox(object):
    def __init__(self, ramin, ramax, decmin, decmax):
        if (ramin > ramax) or (decmin > decmax):
            raise Exception("Bounding box poorly defined.  At least one minum\
                    greater than a maximum:\
                    %f,%f,%f,%f"%(ramin,ramax,decmin,decmax))
        racent = (ramax - ramin)/2.
        deccent = (decmax - decmin)/2.
        if decmin < -90:
            decmin = -90
            if((-90 - decmin) > (decmax + 90)):
                decmax = -180 - decmin
        elif decmax > 90:
            decmax = 90
            if((decmax - 90) > (90 - decmin)):
                decmin = 180 - decmax
        else:
            pass
        if deccent < -90.:
            deccent = -90
        elif deccent > 90.:
            deccent = 90.
        else:
            pass
        self.ramaxbound = ramax%360.
        self.raminbound = ramin%360. 
        self.ramin = ramin
        self.ramax = ramax
        self.decmin = decmin
        self.decmax = decmax
        self.racent = racent
        self.deccent = deccent

    def getRaMin(self):
        return self.ramin
    def getRaMax(self):
        return self.ramax
    def getDecMin(self):
        return self.decmin
    def getDecMax(self):
        return self.decmax
    def getCentDeg(self):
        return [self.racent, self.deccent]
    def getDecExtent(self):
        return self.decmax - self.decmin
    def getRaExtent(self):
        return self.ramax - self.ramin

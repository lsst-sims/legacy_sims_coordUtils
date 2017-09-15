from __future__ import with_statement

from lsst.sims.utils import raDecFromAltAz
from lsst.sims.utils import altAzPaFromRaDec
from lsst.sims.utils import ObservationMetaData

obs_pointing = ObservationMetaData(mjd=59580.0)

alt_pointing = 66.0
az_pointing = 11.0

(ra_pointing,
 dec_pointing) = raDecFromAltAz(alt_pointing, az_pointing, obs_pointing)

(alt_check,
 az_check,
 pa_pointing) = altAzPaFromRaDec(ra_pointing, dec_pointing, obs_pointing)

from lsst.sims.utils import angularSeparation

dd = angularSeparation(alt_pointing, az_pointing, alt_check, az_check)

assert dd < 1.0e-6

import numpy as np
from lsst.sims.utils import getRotTelPos

def write_header(file_handle, rot_sky, obshistid):
    file_handle.write('rightascension %.7f\n' % ra_pointing)
    file_handle.write('declination %.7f\n' % dec_pointing)
    file_handle.write('mjd %.5f\n' % obs_pointing.mjd.TAI)
    file_handle.write('altitude %.7f\n' % alt_pointing)
    file_handle.write('azimuth %.7f\n' % az_pointing)
    file_handle.write('filter 2\n')
    file_handle.write('moonalt -10.0\n')
    file_handle.write('nsnap 1\n')
    file_handle.write('vistime 30.0\n')
    rot_tel = getRotTelPos(ra_pointing, dec_pointing, obs_pointing, rot_sky)
    file_handle.write('rottelpos %.6f\n' % rot_tel)
    file_handle.write('rotskypos %.6f\n' % rot_sky)
    file_handle.write('obshistid %d\n' % obshistid)


from lsst.sims.coordUtils import chipNameFromRaDecLSST

ra_list = np.array([ra_pointing, ra_pointing, ra_pointing+0.5])
dec_list = np.array([dec_pointing, dec_pointing - 0.5, dec_pointing])

sed_name = 'starSED/kurucz/km30_5000.fits_g10_5040.gz'

with open('phosim_rot_control.txt', 'w') as control_file:
    for ix, rot_sky in enumerate(np.arange(0.0, 271.0, 30.0)):
        obs = ObservationMetaData(pointingRA=ra_pointing,
                                  pointingDec=dec_pointing,
                                  mjd=obs_pointing.mjd.TAI,
                                  rotSkyPos=rot_sky)
        file_name = 'coordutils_catalogs/phosim_rot_catalog_%.1f.txt' % rot_sky
        with open(file_name, 'w') as output_file:
            write_header(output_file, rot_sky, ix+1)
            for i_obj in range(len(ra_list)):
                output_file.write('object %d %e %e 22.0 %s 0 0 0 0 0 0 point none none\n' %
                                  (i_obj+1, ra_list[i_obj], dec_list[i_obj], sed_name))

        chip_names = chipNameFromRaDecLSST(ra_list, dec_list, obs_metadata=obs)
        for i_obj in range(len(ra_list)):
            control_file.write('%s %d %s\n' % (file_name, i_obj+1, chip_names[i_obj]))
        control_file.write('\n')

#!/usr/bin/python

# Example script to work with the footprint shapefiles


# shapefile library to read the footprints
# http://toblerity.org/fiona/manual.html
import fiona
import fiona.crs





# Geometric objects, predicates, and operations: ie. boundinbox, union, etc. of the footprints
# https://pypi.python.org/pypi/Shapely
from shapely.geometry import shape




footprints_sf = fiona.open('../data/footprints/VIA_APPIA_SITES.shp' )

print fiona.crs.to_string(footprints_sf.crs)
print footprints_sf.schema

for point in footprints_sf:
    print shape(point['geometry'])




footprints_sf.close()



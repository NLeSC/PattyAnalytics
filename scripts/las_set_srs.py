#!/usr/bin/env python

import liblas
import osgeo.osr as osr
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Add spatial reference system (SRS) metadata to a las file using the EPSG number")
    parser.add_argument("-o", "--outfile", type=str,
                        help="The output namelist", required=True)
    parser.add_argument("-i", "--infile",  type=str,
                        help="The input filename", required=True)
    parser.add_argument("-s", "--srs",     type=int,
                        help="Spatial reference system, default=4326 (latlon)",
                        default=4326)
    args = parser.parse_args()

    osrs = osr.SpatialReference()
    osrs.SetFromUserInput("EPSG:{}".format(args.srs))

    lsrs = liblas.srs.SRS()
    lsrs.set_wkt(osrs.ExportToWkt())

    f1 = liblas.file.File(args.infile)
    header = f1.header
    header.set_srs(lsrs)

    f2 = liblas.file.File(args.outfile, header=header, mode="w")
    for p in f1:
        f2.write(p)

    f1.close()
    f2.close()


if __name__ == "__main__":
    main()

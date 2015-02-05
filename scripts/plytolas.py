from patty.conversions import writeLas
import pcl
import argparse

parser = argparse.ArgumentParser(description='Convert PLY file to LAS file')
parser.add_argument('inFile' , metavar="INFILE" , help="Input PLY file")
parser.add_argument('outFile', metavar="OUTFILE", help="Output LASls  file")

args = parser.parse_args()

inFile = args.inFile
outFile = args.outFile

pc = pcl.load(inFile, loadRGB=True)
writeLas(outFile, pc)

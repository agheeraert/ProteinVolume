from os.path import join as jn
import numpy as np
from tqdm import tqdm
import pickle as pkl
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import pdist, cdist
from mpl_toolkits.mplot3d import Axes3D
import argparse
import warnings
import mdtraj as md 
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)


parser = argparse.ArgumentParser(description='Creates a plot of the Delaunay tessalation for a protein')
parser.add_argument('pdb',  type=str, default=False,
                    help='PDB file')
parser.add_argument('output',  type=str,
                    help='Folder where to put the results')  
parser.add_argument('-cutoff', type=float, default=0.55,
		    help='cutoff used for the Delaunay volume plot')    
parser.add_argument('-color',  type=str, nargs=2, default=['green', 'pink'],
                    help='Color used to draw the edges')
args = parser.parse_args()

def vwrite(vertex):
    vertex = vertex*10
    return str([*vertex]).replace("[", "").replace("]", "").replace(",", "")

def triangle_plot(v0, v1, v2, output):
    output.write("draw triangle {"+vwrite(v0)+"} {"+vwrite(v1)+"} {"+vwrite(v2)+"} \n")

def tetra_plot(tetrahedron, output):
    triangle_plot(*tetrahedron[0:3], output)
    triangle_plot(*tetrahedron[1:], output)
    triangle_plot(tetrahedron[0], *tetrahedron[2:], output)
    triangle_plot(*tetrahedron[0:2], tetrahedron[3], output)

traj = md.load_pdb(args.pdb)
coords = traj.xyz[0]
delau = Delaunay(coords)
tetrahedrons = delau.points[delau.simplices]
distances = np.array([pdist(i) for i in delau.points[delau.simplices]])
multiplicator = np.prod(distances < args.cutoff, axis=1)
tetrahedrons_notcut = tetrahedrons[np.where(multiplicator==1)]
tetrahedrons_cut = tetrahedrons[np.where(multiplicator==0)]

def out_write(tetrahedrons, out, color):
    with open(out, "w") as output:
        output.write("draw delete all \n")
        output.write("draw color "+color+" \n")
        for tetrahedron in tetrahedrons:
            tetra_plot(tetrahedron, output)

out_write(tetrahedrons_cut, args.output.replace(".tcl", "_delaunay.tcl"), args.color[1])
out_write(tetrahedrons_notcut, args.output.replace(".tcl", "_delaunay_cut"+str(args.cutoff)+".tcl"), args.color[0])

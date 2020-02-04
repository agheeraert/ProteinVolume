"""
Code for a weighted Vornoi diagram / Laguerre polyhedral decomposition / power diagram

DOESN'T WORK AT ALL
"""
from mendeleev import H, N, C, O
import mdtraj as md
from tqdm import tqdm
from scipy.spatial.distance import pdist
import numpy as np

def get_rw(table):
    table = np.asarray(table)
    return (table=='H')*H.vdw_radius+(table=='N')*N.vdw_radius+(table=='O')*O.vdw_radius+(table=='C')*C.vdw_radius


def get_lines(trajs, topo):
    traj = md.load(trajs, top=topo)
    rw = get_rw([atom.element.symbol for atom in traj.top.atoms])
    diffrw = pdist(np.expand_dims(rw, axis=-1))
          
    for frame in tqdm(range(traj.n_frames)):
        positions = traj.xyz[frame]
        distances = pdist(positions)
        lines_dist = np.divide(diffrw-np.square(distances), 2*distances)
    return lines_dist

if __name__ == '__main__':
    get_lines('/home/aghee/PDB/prot_apo_sim1_s10.dcd', '/home/aghee/PDB/prot.prmtop')

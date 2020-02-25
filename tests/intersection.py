import mdtraj as md
import numpy as np
from tqdm import tqdm
from Bio.PDB.Polypeptide import aa1, aa3
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as jn
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from collections import OrderedDict

class LocalVolume():
    def __init__(self, trajs, topo, output, delcutoff=0.55, chain_cut=252):
        self.three2one = dict(zip(aa3, aa1))
        self.traj = md.load(trajs, top=topo)
        topo = self.traj.topology.to_dataframe()[0]
        self.atom2res = pd.Series(topo['resSeq'].values, topo.index).to_dict()
        self.delcutoff = delcutoff
        self.output = output
        self.res2tetra = {res: [0]*self.traj.n_frames for res in topo['resSeq'].values} 
        self.res2tetra_shared = {res: [0]*self.traj.n_frames for res in topo['resSeq'].values}
        resId2resName = pd.Series(topo['resName'].values, topo['resSeq'].values).to_dict()
        self.resId2resName = {}
        for elt in resId2resName:
            if elt <= chain_cut:
                self.resId2resName[elt] = self.three2one[resId2resName[elt]]+str(elt+1)+':F'
            else:
                self.resId2resName[elt] = self.three2one[resId2resName[elt]]+str(elt-chain_cut)+':H'




if __name__ == '__main__':
    a = LocalVolume(['/home/aghee/PDB/prot_apo_sim1_s10.dcd', '/home/aghee/PDB/prot_prfar_sim1_s10.dcd'], '/home/aghee/PDB/prot.prmtop', '../results/run_qsdkklqsd')
    print(a.resId2resName)

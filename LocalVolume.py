import mdtraj as md
import numpy as np
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as jn
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from collections import OrderedDict
from Bio.PDB.Polypeptide import aa1, aa3

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

    def load_tet(self, path):
        self.tetrahedrons = pkl.load(open(path, 'rb'))
    
    def create_tet_frame(self, method, frame):
        positions = self.traj.xyz[frame]
        delau = Delaunay(positions)
        tetrahedrons = delau.simplices
        if method == 'cut':
            distances = np.array([pdist(i) for i in delau.points[delau.simplices]])
            multiplicator = np.prod(distances < self.delcutoff, axis=1)
            tetrahedrons = tetrahedrons[np.where(multiplicator==1)]       
        dets_comp = np.concatenate([delau.points[tetrahedrons], np.ones((*(delau.points[tetrahedrons]).shape[:2], 1))], axis=-1)
        volumes = 1/6*np.absolute(np.linalg.det(dets_comp))
        tetrahedrons = np.concatenate([tetrahedrons, np.expand_dims(volumes, axis=-1)], axis=-1)
        return tetrahedrons

    def create_tet(self, method):
        L_tet = []
        for frame in tqdm(range(self.traj.n_frames)):
            L_tet.append(self.create_tet_frame(method, frame))
        tetrahedrons = np.array(L_tet)
        self.tetrahedrons = tetrahedrons                    

    def save_tet(self):
        pkl.dump(self.tetrahedrons, open(self.output+'tetrahedrons.p', 'wb'))
    
    def label_single_tet(self, tetrahedron, frame):
        vertices = np.array([self.atom2res[int(v)] for v in tetrahedron[:-1]])
        n_diff = len(OrderedDict.fromkeys(vertices))
        if n_diff == 1:
            self.res2tetra[self.atom2res[tetrahedron[0]]][frame] += tetrahedron[-1]
        else:
            for vertex in OrderedDict.fromkeys(vertices):
                self.res2tetra_shared[vertex][frame] += tetrahedron[-1]       

    def label_tet_frame(self, frame):
        for tetrahedron in self.tetrahedrons[frame]:
            self.label_single_tet(tetrahedron, frame)

    def label_tet(self):
        for frame in tqdm(range(self.traj.n_frames)):
            self.label_tet_frame(frame)
    
    def save_labels(self):
        pkl.dump(self.res2tetra, open(self.output+'labels.p', 'wb'))
        pkl.dump(self.res2tetra_shared, open(self.output+'labels_shared.p', 'wb'))

    def _plot(self, residue, dic, plot_str):
        resname = self.resId2resName[residue]
        f = plt.figure()
        plt.title(resname)
        xtime = np.arange(0,100,0.1)
        for sim in range(self.n_plots):
            plt.subplot(self.n_plots//2, 2, sim+1)
            if sim ==0:
                plt.plot(xtime, dic[sim][residue][:1000], c='b', label='Volume ' +plot_str+' (apo)')
                plt.plot(xtime, dic[sim][residue][1000:], c='r', label='Volume ' +plot_str+' (PRFAR)')
            else:
                plt.plot(xtime, dic[sim][residue][:1000], c='b')
                plt.plot(xtime, dic[sim][residue][1000:], c='r')
            plt.ylabel('Volume ($nm^3$)')
            plt.xlabel('Time (ns)')
        f.legend(loc=8)        
        f.tight_layout()
        f.subplots_adjust(bottom=0.22)   
        plt.savefig(jn('results', 'residues_volume', plot_str, resname+'.png'))
        plt.close()

    def plot_labels_ext(self, residue_paths, shared_paths):
        self.n_plots = len(residue_paths)
        res2tetra = []
        res2tetra_shared = []
        for path1, path2 in zip(residue_paths, shared_paths):
            res2tetra.append(pkl.load(open(path1, 'rb')))
            res2tetra_shared.append(pkl.load(open(path2, 'rb')))
        for residue in self.resId2resName:
            self._plot(residue, res2tetra, 'within')
            self._plot(residue, res2tetra_shared, 'shared')


if __name__ == '__main__':
    # a = LocalVolume(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', 'results/all_run')
    # a.load_tet("results/all_runtetrahedrons.p")
    # a.label_tet()
    # a.save_labels()
    # for i in range(1,5):
    #     a = LocalVolume(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd', '/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd'], '/home/aghee/PDB/prot.prmtop', 'results/run_'+str(i))
    #     a.create_tet("cut")
    #     a.save_tet()
    #     a.label_tet()
    #     a.save_labels()
    #     del a
    a = LocalVolume(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', 'results/everything')
    a.plot_labels_ext(['results/run_'+str(i)+'labels.p' for i in range(1,5)], ['results/run_'+str(i)+'labels_shared.p' for i in range(1,5)])
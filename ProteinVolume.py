from os.path import join as jn
import mdtraj as md
import numpy as np
from tqdm import tqdm
import pickle as pkl
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import pdist, cdist
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ProteinVolume():
    def __init__(self, trajs, topo, n_frames, output, methods=['delcut'], apo_holo=True, delcutoff=0.55):
        traj = md.load(trajs, top=topo)
        self.positions = traj.xyz
        self.output = output
        if len(n_frames) == 1:
            self.n_frames=len(trajs)*n_frames
        else:
            self.n_frames=n_frames
        self.methods = methods
        self.apo_holo = apo_holo
        self.delcutoff = delcutoff

    def volume_Delaunay(self, frame):
        delau = Delaunay(self.positions[frame])
        dets_comp = np.concatenate([delau.points[delau.simplices], np.ones((*delau.simplices.shape[:2], 1))], axis=-1)
        volumes = 1/6*np.absolute(np.linalg.det(dets_comp))
        return np.sum(volumes)
    
    def volume_Delaunay_cut(self, frame):
        delau = Delaunay(self.positions[frame])
        dets_comp = np.concatenate([delau.points[delau.simplices], np.ones((*delau.simplices.shape[:2], 1))], axis=-1)
        distances = np.array([pdist(i) for i in delau.points[delau.simplices]])
        multiplicator = np.prod(distances < self.delcutoff, axis=1)
        volumes = 1/6*np.absolute(np.multiply(np.linalg.det(dets_comp), multiplicator))
        return np.sum(volumes)
        
   
    def volume(self, frame, method):
        if method == 'delaunay':
            return self.volume_Delaunay(frame)
        elif method == 'delcut':
            return self.volume_Delaunay_cut(frame)

    # def plot_vols(self):
    #     plt.subplot(2, n_trajs//2, traj)
    #     plt.title("Simulation "+str(traj+1))
    #     plt.plot()
    
    def plot_apo_holo(self, volumes):
        volumes_apo = volumes[self.n_trajs//2:]
        volumes_holo = volumes[:self.n_trajs//2]
        for i in range(len(volumes_apo)):
            if self.n_trajs//4 >= 1:
                plt.subplot(2, self.n_trajs//4, i+1)
            plt.title("Simulation "+str(i+1))
            plt.ylabel("Volume (nm$^3$)")
            plt.xlabel("Time (ns)")
            plt.plot(np.arange(0, 100, 0.1), volumes_apo[i], c='b', label="apo")
            plt.plot(np.arange(0, 100, 0.1), volumes_holo[i], c='r', label="PRFAR")

    def volume_over_sims(self):
        self.n_trajs = len(self.n_frames)
        n_maxframes = max(self.n_frames)
        for method in self.methods:
            volumes = np.zeros((self.n_trajs, n_maxframes))
            previous = 0
            for traj in range(self.n_trajs):
                for frame in tqdm(range(self.n_frames[traj])):
                    volumes[traj, frame] = self.volume(frame+previous, method)
                previous += frame

            f = plt.figure()
            pkl.dump(volumes, open(jn(self.output, 'volumes_'+method+'.p'), 'wb'))
            if self.apo_holo:
                self.plot_apo_holo(volumes)
            #else:
            #    self.plot(volumes)
            plt.tight_layout()
            plt.legend()
            plt.savefig(jn(self.output, 'volumes_'+method+'.svg'))
            plt.close()
              

if __name__ == '__main__':
    # ProteinVolume(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', [1000], '/home/aghee/ProteinVolume/results/').volume_over_sims()
    ProteinVolume(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', [1000], '/home/aghee/ProteinVolume/results/').volume_over_sims()

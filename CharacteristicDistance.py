from os.path import join as jn
import mdtraj as md
import numpy as np
import time
from tqdm import tqdm
import pickle as pkl
from scipy.spatial import Delaunay, Voronoi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import os

class CharacteristicDistance():
    def __init__(self, trajs, topo, n_frames, output, methods=['3NN', '4NN', '5NN', 'delaunay', 'delaunay2']):
        traj = md.load(trajs, top=topo)
        self.positions = traj.xyz
        self.output = output
        self.methods = methods
        if len(n_frames) == 1:
            self.n_frames=len(trajs)*n_frames
        else:
            self.n_frames=n_frames
        self.n_trajs = len(self.n_frames)


    def distances_NN(self, n_neighbors, frame):
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(self.positions[frame])
        distances = nbrs.kneighbors(self.positions[frame], n_neighbors=n_neighbors+1)[0][:,1:]
        distances = distances.reshape(-1)
        return distances
    
    def distances_delaunay(self, frame):
        distances = cdist(self.positions[frame], self.positions[frame])
        voronoi = Voronoi(self.positions[frame])
        distances = distances[[*np.transpose(voronoi.ridge_points)]]
        return distances
    
    def distances_delaunay2(self, frame):
        delau = Delaunay(self.positions[frame])
        distances = np.array([pdist(i) for i in delau.points[delau.simplices]]).reshape(-1)
        return distances
    
    def get_distance(self, frame, method):
        if method[-2:] == "NN":
            try:
                k = int(method[:-2])
                return self.distances_NN(k, frame)
            except ValueError:
                print('Error in method name')
                os.kill()
        elif method == "delaunay":
            return self.distances_delaunay(frame)
        elif method == "delaunay2":
            return self.distances_delaunay2(frame)
    
    def get_distances(self, method):
        distances = []
        for frame in tqdm(range(len(self.positions))):
            distances.append(np.expand_dims(self.get_distance(frame, method), axis=-1))
        if method[-2:] == "NN":
            return np.concatenate(distances, axis=-1)
        else:
            return distances

    def plot_apo_holo(self, table, char, method):
        f = plt.figure()
        table_apo = table[self.n_trajs//2*self.n_frames[0]:]
        table_holo = table[:self.n_trajs//2*self.n_frames[0]]
        previous = 0
        for i in range(self.n_trajs//2):
            if self.n_trajs//4 >= 1:
                plt.subplot(2, self.n_trajs//4, i+1)
            plt.title("Simulation "+str(i+1))
            if i==0:
                plt.plot(table_apo[previous:previous+self.n_frames[i]], c='b', label="apo")
                plt.plot(table_holo[previous:previous+self.n_frames[i]], c='r', label="PRFAR")
            else:
                plt.plot(table_apo[previous:previous+self.n_frames[i]], c='b')
                plt.plot(table_holo[previous:previous+self.n_frames[i]], c='r')
            plt.ylabel(char+' (nm)')
            plt.xlabel('Time (ns)')
            plt.xticks(np.arange(0, 1001, 250), np.arange(0, 101, 25))
            previous += self.n_frames[i]
        f.legend(loc=8, ncol=2)        
        f.tight_layout()
        f.subplots_adjust(bottom=0.12)
        plt.savefig(jn(self.output, char+'_'+method+'.png'))
        plt.close()
    
    def create_std_sim_plot(self):
        for method in self.methods:
            distances = self.get_distances(method)
            if method[-2:] == "NN":
                mean = np.mean(distances, axis=-1)
                std = np.std(distances, axis=-1)
            else:
                self.mean = [np.mean(i) for i in distances]
                self.std = [np.std(i) for i in distances]         
            pkl.dump(mean, open(jn(self.output, 'mean_'+method+'.p'), 'wb'))
            pkl.dump(std, open(jn(self.output, 'std_'+method+'.p'), 'wb'))
            self._plot(method)
    
    def _plot(self, method):
        self.plot_apo_holo(self.mean, 'mean', method)
        self.plot_apo_holo(self.std, 'std', method)
        self.plot_apo_holo(np.add(self.mean, self.std), 'mean+std', method)
    
    def load_std_sim_plot(self, path1, path2):
        self.mean = pkl.load(open(path1, 'rb'))
        self.std = pkl.load(open(path2, 'rb'))
        method = path1.split('.')[-2].split('_')[-1]
        self._plot(method)        


    def distances_distribution_plot(self):
        for method in self.methods:
            distances = []
        # for frame in tqdm(range(n_frames)):
            distances = self.get_distance(0, method)
            pkl.dump(distances, open(jn(self.output, 'distances_'+method+'.p'), 'wb'))
            f = plt.figure(figsize=[8.2, 4.8])
            ax1 = plt.subplot(121)
            ax1.scatter(range(distances.shape[0]), np.sort(distances), marker='+', color='k')
            ax1.set_title('Ordered distances '+ method)
            if method[0] == "d":
                plt.yticks(np.arange(0,5,0.25))
            plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
            plt.grid(linestyle=':')
            plt.ylabel('Distances (nm)')
            plt.xlabel('Order')
            ax2 = plt.subplot(122)
            ax2.set_title('Distance distribution '+ method)
            plt.xlabel('Distances (nm)')
            plt.ylabel('Occurences')
            sns.set_color_codes()
            sns.distplot(distances, ax=ax2, color='k')
            plt.tight_layout()
            plt.savefig(jn(self.output, 'distances_'+method+'.png'))
            plt.close()
            

if __name__ == '__main__':
    # CharacteristicDistance(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', [1000], '/home/aghee/ProteinVolume/results/').mean_std_sim_plot()
    CharacteristicDistance(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', [1000], '/home/aghee/ProteinVolume/results/').load_std_sim_plot('results/mean_delaunay.p', 'results/std_delaunay.p')


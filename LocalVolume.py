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
import networkx as nx
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib as mpl
from sklearn import decomposition


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
        resid2name = pd.Series(topo['resName'].values, topo['resSeq'].values).to_dict()
        self.contacts_apo = {}
        self.contacts_holo = {}
        self.resid2name = {}
        self.resname2id = {}
        self.bool2color = {True: 'red', False: 'green'}
        for elt in resid2name:
            if elt <= chain_cut:
                self.resid2name[elt] = self.three2one[resid2name[elt]]+str(elt+1)+':F'
                self.resname2id[self.three2one[resid2name[elt]]+str(elt+1)+':F'] = elt
            else:
                self.resid2name[elt] = self.three2one[resid2name[elt]]+str(elt-chain_cut)+':H'
                self.resname2id[self.three2one[resid2name[elt]]+str(elt-chain_cut)+':H'] = elt


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

    def contact_single(self, tetrahedron, dic):
        def _update(dic, u, v):
            if (u,v) in dic:
                dic[(u,v)] += tetrahedron[-1]
            else:
                dic[(u,v)] = tetrahedron[-1]
        
        vertices = np.array([self.resid2name[self.atom2res[int(v)]] for v in tetrahedron[:-1]])
        if len(OrderedDict.fromkeys(vertices)) == 2:
            u, v = [*sorted(OrderedDict.fromkeys(vertices))]
            _update(dic, u, v)

        elif len(OrderedDict.fromkeys(vertices)) == 3:
            u, v, w = [*sorted(OrderedDict.fromkeys(vertices))]
            _update(dic, u, v)
            _update(dic, u, w)
            _update(dic, v, w)

        elif len(OrderedDict.fromkeys(vertices)) == 4:
            u, v, w, x = [*sorted(OrderedDict.fromkeys(vertices))]
            _update(dic, u, v)
            _update(dic, u, w)
            _update(dic, u, x)
            _update(dic, v, w)        
            _update(dic, v, x)
            _update(dic, w, x)
    
    def contacts_frame(self, frame, dic):
        for tetrahedron in self.tetrahedrons[frame]:
            self.contact_single(tetrahedron, dic)
    
    def create_contacts_dic(self, output1='results/contacts_dic_apo.p', output2='results/contacts_dic_holo.p'):
        sims_len = self.traj.n_frames//2
        for frame in tqdm(range(sims_len)):
            self.contacts_frame(frame, self.contacts_apo)
        for frame in tqdm(range(sims_len, sims_len*2)):
            self.contacts_frame(frame, self.contacts_holo)
        pkl.dump(self.contacts_apo, open(output1, 'wb'))
        pkl.dump(self.contacts_holo, open(output2, 'wb'))

    def load_contacts_dic(self, fichier1='results/contacts_dic_apo.p', fichier2='results/contacts_dic_holo.p'):
        self.contacts_apo = pkl.load(open(fichier1, 'rb'))
        self.contacts_prfar = pkl.load(open(fichier2, 'rb'))
    
    def create_contact_graph(self, output='results/contacts_graph.p'):
        sims_len = self.traj.n_frames//2    
        self.contacts_graph = nx.Graph()
        for u, v in set(self.contacts_apo.keys()) & set(self.contacts_holo.keys()):
            self.contacts_graph.add_edge(u, v, weight= abs(self.contacts_holo[(u, v)] - self.contacts_apo[(u, v)])/sims_len, color=self.bool2color[(self.contacts_holo[(u, v)] > self.contacts_apo[(u, v)])])
        for u, v in set(self.contacts_apo.keys()) - set(self.contacts_holo.keys()):            
            self.contacts_graph.add_edge(u, v, weight=self.contacts_apo[(u, v)]/sims_len, color='green')
        for u, v in set(self.contacts_holo.keys()) - set(self.contacts_apo.keys()):
            self.contacts_graph.add_edge(u, v, weight=self.contacts_holo[(u, v)]/sims_len, color='red')
        pkl.dump(self.contacts_graph, open(output, 'wb'))

    def load_contacts_graph(self, fichier='results/contacts_graph.p'):
        self.contacts_graph = pkl.load(open(fichier, 'rb'))

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
        plt.style.use('classic')
        mpl.rcParams.update({'axes.formatter.useoffset': False})
        resname = self.resid2name[residue]
        # if resname == 'K19:F':
        #     pkl.dump(dic, open('results/k19'+plot_str+'.p', 'wb'))
        f = plt.figure()
        f.patch.set_facecolor('white')
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
            plt.ylabel('Volume (nm$^3$)')
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
        for residue in self.resid2name:
            self._plot(residue, res2tetra, 'within')
            self._plot(residue, res2tetra_shared, 'shared')
            self._plot(residue, np.add(res2tetra,res2tetra_shared), 'total')
    
    def covariance_matrix(self, residue_paths, shared_paths):
        res2tetra = []
        res2tetra_shared = []
        for path1, path2 in zip(residue_paths, shared_paths):
            res2tetra.append(pkl.load(open(path1, 'rb')))
            res2tetra_shared.append(pkl.load(open(path2, 'rb')))
        volumes_apo = []
        volumes_holo = []
        for residue in range(len(self.resid2name)):
            _volumes_apo = []
            _volumes_holo = []
            for sim in range(len(res2tetra)):
                _volumes_apo.append(np.add(res2tetra[sim][residue][:1000], res2tetra_shared[sim][residue][:1000]))
                _volumes_holo.append(np.add(res2tetra[sim][residue][1000:], res2tetra_shared[sim][residue][1000:]))
            volumes_apo.append(np.concatenate(_volumes_apo))
            volumes_holo.append(np.concatenate(_volumes_holo))
        volumes_apo = np.stack(volumes_apo)
        volumes_holo = np.stack(volumes_holo)
        volumes_all = np.concatenate([volumes_apo, volumes_holo], axis=1)
        self.cov_matrix_apo = np.cov(volumes_apo)
        self.cov_matrix_holo = np.cov(volumes_holo)
        self.cov_matrix_all = np.cov(volumes_all)
    
    def save_cov_matrix(self, output):
        self._save_matrix(self.cov_matrix_apo, output.replace('.png', '_apo.png'))
        self._save_matrix(self.cov_matrix_holo, output.replace('.png', '_holo.png'))
        self._save_matrix(self.cov_matrix_holo - self.cov_matrix_apo, output.replace('.png', '_diff.png'), divcmap=True)
        self._save_matrix(self.cov_matrix_all, output)
        
    def _save_matrix(self, mat, output, divcmap=False, lines_to_draw=None):
        def _rec(a1, a2, H=False, color='k'):
            x1 = a1-1.5 + 253*int(H)
            x2 = a2-0.5 + 253 * int(H)
            plt.plot([x1, x2, x2, x1, x1], [x1, x1, x2, x2, x1], linestyle=':', linewidth=1, color=color)

        f = plt.figure()
        if divcmap:
            plt.imshow(mat, origin='lower',cmap='bwr')
        else:
            plt.imshow(mat, origin='lower',cmap='jet')
        plt.colorbar()
        ticks = [list(range(50,253,50))+list(range(303,454,50)),list(range(50,253,50))+list(range(50,201,50))]
        plt.xticks(*ticks)
        plt.yticks(*ticks)
        #HisH and HisF separation
        plt.plot([252.5, 252.5], [0,453], linestyle=':', linewidth=1, color='k')
        #Loop1
        # _rec(15,29)
        # _rec(59, 74)
        # _rec(91,95)
        # _rec(224, 234)
        # _rec(9, 18, H=True)
        # _rec(49,52, H=True)
        # _rec(118, 124, color='b')
        # _rec(116, 121, color='b', H=True)
        if lines_to_draw:
            lines_id = np.vectorize(self.resname2id.get)(lines_to_draw)
            X, Y = [], []
            previous = lines_id[0]
            for elt in lines_id:
                mini, maxi = sorted([previous, elt])
                X.append(mini)
                Y.append(maxi)
                previous = elt
            X.append(elt)
            Y.append(elt)
            plt.plot(X, Y, color='g')
        plt.savefig(output, dpi=1000)
        plt.show()
        plt.close()

    def topk_cov_matrix(self, k, reverse=False):
        N, N = self.cov_matrix_all.shape
        if reverse:
            _indices = np.argsort(self.cov_matrix_all, axis=None)[:2*k:2] 
        else:
            _indices = np.argsort(self.cov_matrix_all, axis=None)[-1:-2*k:-2]
        indices = np.stack(np.divmod(_indices, N)).transpose()
        print(np.vectorize(self.resid2name.get)(indices))
        for elt in self.cov_matrix_all.reshape(-1)[_indices]:
            print(elt)

    def topk_aa(self, resname, k, v = True):
        idx = self.resname2id[resname]
        indices = np.argsort(self.cov_matrix_all[idx])[-1:-k:-1]
        if v:
            print("Highest covariances with residue", resname)
            print(np.vectorize(self.resid2name.get)(indices))
        else:
            return np.vectorize(self.resid2name.get)(indices)

    def iterator_cov(self, resname, depth, output=None):
        idx = self.resname2id[resname]
        step = 0
        visited = [resname]
        while step <= depth:
            indices = np.argsort(np.abs(self.cov_matrix_all[idx]))[::-1]
            names = np.vectorize(self.resid2name.get)(indices)
            for residue in names:
                if residue not in visited:
                    visited.append(residue)
                    idx = self.resname2id[residue] #used to be resname, does it work ???
                    step+=1
                    break
        if not output:
            return visited
        else:
            self._save_matrix(self.cov_matrix_all, output, lines_to_draw=visited)
    
    def pca(self, n_components):
        pca = decomposition.PCA(n_components=n_components)
        plt.imshow(pca.fit_transform(self.cov_matrix_all), aspect='auto')
        plt.show()

    def contact_cov(self, path='results/contact_cov_graph/contact0.5.npy', output='results/contact_cov_graph/0.p'):
        adjacency = np.abs(np.dot(np.load(path),self.cov_matrix_all))
        color = np.sign(self.cov_matrix_all)
        net = nx.from_numpy_matrix(adjacency)
        color_dict = {(u,v): self.bool2color[color[u,v] <=0] for u in range(454) for v in range(454)}
        nx.set_edge_attributes(net, color_dict, name='color')
        net = nx.relabel_nodes(net, self.resid2name)
        nx.write_gpickle(net, output)

        

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

    # a.plot_labels_ext(['results/run_'+str(i)+'labels.p' for i in range(1,5)], ['results/run_'+str(i)+'labels_shared.p' for i in range(1,5)])
    a.covariance_matrix(['results/run_'+str(i)+'labels.p' for i in range(1,5)], ['results/run_'+str(i)+'labels_shared.p' for i in range(1,5)])
    # a.pca(3)
    # print(a.iterator_cov('S225:F', 30))

    # a.save_cov_matrix('results/covariance_matrix.png')
    # a.topk_cov_matrix(200)
    # a.topk_aa('H120:H', 10)
    # a.topk_aa(10, 84+252)
    # a.topk_aa(10, 178+252)
    # a.topk_aa(10, 180+252)
    # a.topk_aa(10, 18)
    
    a.contact_cov()
    # a = LocalVolume(['/home/aghee/PDB/prot_apo_sim'+str(i)+'_s10.dcd' for i in range(1,5)]+['/home/aghee/PDB/prot_prfar_sim'+str(i)+'_s10.dcd' for i in range(1,5)], '/home/aghee/PDB/prot.prmtop', 'results/everything')
    # a.load_tet("results/all_runtetrahedrons.p")
    # a.create_contacts_dic()
    # a.create_contact_graph()
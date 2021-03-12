import numpy as np
import networkx as nx

class knnGraph:
    def __init__(self,connectivity_matrix):
        self.mat = connectivity_matrix.nonzero() # a csr matrix
        
    def get_params(self):
        return {'min_samples':1}
    
    def fit_predict(self,idx):
        idx = idx.flatten()
        result = np.zeros(idx.shape[0])
        idx_lookup = {index:i for i,index in enumerate(idx)}
        def find_intersection(m_list):
            for i,v in enumerate(m_list) : 
                for j,k in enumerate(m_list[i+1:],i+1):
                    if v & k:
                        m_list[i]=v.union(m_list.pop(j))
                        return find_intersection(m_list)
            return m_list
        xy, x_ind, y1_ind = np.intersect1d(idx,self.mat[0],return_indices=True)
        xy, x_ind, y2_ind = np.intersect1d(idx,self.mat[1],return_indices=True)
        xy = np.intersect1d(y1_ind,y2_ind)
        mat = np.vstack((self.mat[0][xy],self.mat[1][xy])).T.tolist()
        pairs = [set(v) for v in mat]
        clusters = find_intersection(pairs)
        for i, c in enumerate(clusters):
            for item in c:
                result[idx_lookup[item]] = i
        return result.astype(int)

def drawGraph(ax,graph,labels,title=''):
    n_colors = len(np.unique(labels))
    G = nx.Graph()
    for n1 in graph['links']:
        for n2 in graph['links'][n1]:
            G.add_edge(n1,n2)

    for n in graph['nodes']:
        G.add_node(n)
        
    piecharts = {}
    
    for node in graph['nodes']:
        targets = labels[graph['nodes'][node]]
        unique, counts = np.unique(targets, return_counts=True)
        dist = dict(zip(unique, counts))
        tmp = [0 for n in range(n_colors)]
        for k in dist:
            tmp[k] = dist[k]
        piecharts[node] = tmp
        
    # parameters for pie plot
    color_map = {0:'#4E79A7',1:'#F28E2B',2:'#59A14F'}
    pos = nx.spring_layout(G, scale=1,threshold=0.001)
    ax.set_xticks([])
    ax.set_yticks([])
    node_sizes = max([sum(piecharts[node]) for node in piecharts])
    def linear(x):
        return x / node_sizes * 0.1

    # storing attributes in a dict
    attrs = piecharts

    # draw graph and draw pieplots instead of nodes

    nx.draw_networkx_edges(G, pos=pos,ax=ax)
    for node in G.nodes:
        attributes = attrs[node]
        a = ax.pie(
            piecharts[node], # s.t. all wedges have equal size
            center=pos[node],
            wedgeprops={'linewidth':1,"edgecolor":"k"},
            colors = [color_map[i % len(color_map)] for i in range(n_colors)],
            radius=linear(np.sum(piecharts[node]))) #0.01* 
    
    ax.set_ylim(-1,1)
    ax.set_xlim(-1,1)
    ax.title.set_text(title)
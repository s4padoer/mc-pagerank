#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 17:19:40 2022

@author: patricia
"""
import numpy as np
import random
import networkx as nx

class fastppr: 
    def __init__(self, edgelist, jump):
        self.edges = edgelist
        self.vertices = np.unique(edgelist.ravel())
        self.n_vertex = len(self.vertices)
        self.jump = jump
        self.neighbours = None
        
    def get_neighbours(self):
        neigh = {}
        rev_neigh = {} # reverse neighbour
        for v in self.vertices:
            neigh[v] = self.edges[np.where(self.edges[:,0] == v)[0],1]
            rev_neigh[v] = self.edges[np.where(self.edges[:,1] == v)[0],0]
        self.neighbours = neigh
        self.reverse_neighbours = rev_neigh

    def first_walk(self, R, max_length = 100, seed = 0):
        self.column_names = ["Start", "Path" , "Step", "Nr_walk" ]
        if self.neighbours is None:
            self.get_neighbours()
        self.seed = seed 
        rw = np.zeros(shape = (0,4))
        random.seed(seed)
        # Here parallelization
        for v in self.vertices:
            # Here parallelization
            for r in range(R):
                walking = True
                niter = 0
                path = [v]
                while walking:
                    nghb = self.neighbours[path[-1]]
                    if len(nghb) == 0:
                        break
                    nxt = nghb[random.randint(0,len(nghb)-1)]
                    path.append(nxt)
                    niter = niter + 1
                    walking = (random.random() < self.jump)
                    if niter == max_length:
                        walking = False
                lp = len(path)
                df = np.column_stack( (np.full(lp,v), path, np.arange(lp), np.full(lp,r)) )

                rw = np.concatenate((rw, df), axis = 0)
        self.current_walks = rw
        self.walk_update_necessary = False
        
    # Create random walk from specific node
    def random_walk(self, node, max_length):
        # Create a random walk from a specific node:
        niter = 0
        path = [node]
        walking = True
        while walking:
            nghb = self.neighbours[path[-1]]
            if len(nghb) == 0:
                break
            nxt = nghb[random.randint(0,len(nghb)-1)]
            path.append(nxt)
            niter = niter + 1
            walking = (random.random() < self.jump)
            if niter == max_length:
                walking = False
        return path
    
    def update_graph(self, newedges = None, deletions = None): 
        self.deleted_nodes = []
        self.created_nodes = []
        changed_nodes1 = []
        changed_nodes2 = []
        if deletions is not None:
                    # First, incorporate deletions in graph
            ind = np.where( deletions[:,0] == deletions[:,1])[0]
            if len(ind) > 0:
            # these nodes point to the node to be deleted node:
                for v in deletions[ind,0]:
                    rev_neigh = self.reverse_neighbours[v]
                    neigh = self.neighbours[v]
                    del self.neighbours[v]
                    del self.reverse_neighbours[v]
                    for x in rev_neigh:
                        self.neighbours[x] = self.neighbours[x][np.where(self.neighbours[x] != v)[0]]
                        for x in neigh:
                            self.reverse_neighbours[x] = self.reverse_neighbours[x][np.where(self.reverse_neighbours[x] != v)[0]]
                    self.vertices = self.vertices[np.where(self.vertices != v)[0]]    
                    self.deleted_nodes = deletions[np.where(deletions[:,0] == deletions[:,1])[0],0]
            deletions_r = deletions[np.where(deletions[:,0] != deletions[:,1])[0]]
            changed_nodes1 = np.unique(deletions_r.ravel())
            deletions_r = deletions[np.where((deletions[:,0] not in self.deleted_nodes) and (deletions[:,1] not in self.deleted_nodes) )[0]]
            for x, y in deletions_r:
                # adjust reverse neighbours:
                    self.reverse_neighbours[y] = self.reverse_neighbours[y][np.where(self.reverse_neighbours[y] != x)[0]] 
                    # adjust neighbours
                    self.neighbours[x] = self.neighbours[x][np.where(self.neighbours[x] != y)[0]]
        if newedges is not None:
                    # Now add new edges:
            ind = np.where( newedges[:,0] == newedges[:,1])[0]
            new_nodes = newedges[ind,0]
            new_nodes = np.unique(np.concatenate((new_nodes, [x for x in np.unique(newedges.ravel()) if x not in self.vertices])) )
            self.vertices = np.concatenate((self.vertices, new_nodes))
            changed_nodes2 = np.unique(newedges.ravel())
            newedges_r = newedges[np.where(newedges[:,0] != newedges[:,1])[0],:]
            for x, y in newedges_r:
                if x not in new_nodes:
                    self.neighbours[x] = np.append(self.neighbours[x], y)
                
                else: 
                    self.neighbours[x] = np.array([y])
                    if y not in new_nodes:
                        self.reverse_neighbours[y] = np.append(self.reverse_neighbours[y], x)
                    else:
                        self.reverse_neighbours[y] = np.array([x])
            self.created_nodes = newedges[np.where(newedges[:,0] == newedges[:,1])[0],0]
        self.walk_update_necessary = True
        self.nodes_with_changed_arcs = np.unique( np.concatenate( (changed_nodes1, changed_nodes2 ) ) )
        edges = [ np.column_stack(([x]*len(y),y.tolist()) ) for x,y in self.neighbours.items()]    
        edges = np.row_stack(edges)
        self.edges = edges
        

    # deleting node x would then be [x,x] as a row in deletions (in addition )
    # indicating new nodes x is indicated using [x,x] as a row in newedges
    def update_walk(self, R, max_length, newedges = None, deletions = None, seed = 0):
        if self.walk_update_necessary == False:
            self.update_nodes(newedges, deletions)
        relevant_nodes = np.concatenate((self.deleted_nodes, self.nodes_with_changed_arcs))
        # remove the outdated parts:
        updated_walks = self.current_walks
        outdated_walks = np.zeros((0,4))
        for x in relevant_nodes:
            inds = updated_walks[:,1] == x
            dels = updated_walks[inds,:].copy()
            dels = dels[dels[:, 2].argsort(),:]
            dels = dels[np.unique(dels[:, [0,3]], return_index=True, axis = 1)[1]]
            for i in range(len(dels[:,0])):
                y = dels[i,:]
                inds2 = (updated_walks[:,3] == y[3]) & \
                    (updated_walks[:,0] == y[0]) & \
                    (updated_walks[:,2] >= y[2]-1) 
                outdated_walks = np.concatenate((outdated_walks, updated_walks[inds2,:]), axis = 0)
                updated_walks = updated_walks[~inds2,:]
        
        outdated_walks = outdated_walks[list(map(lambda x: x not in self.deleted_nodes, outdated_walks[:,0]))]
        outdated_walks = outdated_walks[np.lexsort((outdated_walks[:,0], outdated_walks[:,3], outdated_walks[:,2])),:]
        outdated_walks = outdated_walks[np.unique(outdated_walks[:,[0,3]], return_index = True, axis = 1)[1],:]
        # Create R random walks if a new node has been added:
        for v in self.created_nodes:
            for r in range(R):
                path = self.random_walk(v, max_length)
                lp = len(path)
                df = np.column_stack((np.full(lp,v), path, np.arange(lp), np.full(lp,r))) 
                updated_walks = np.concatenate((updated_walks, df), axis = 0)

        # Now: Update the partially outdated walks:
        for i in range(len(outdated_walks[:,0])):
            path = self.random_walk(outdated_walks[i,1], max_length)
            lp = len(path)
            df = np.column_stack((np.full(lp,outdated_walks[i,0]), path, np.arange(outdated_walks[i,2], outdated_walks[i,2] + lp),
                                  np.full( lp,outdated_walks[i,3])) )
            updated_walks = np.concatenate((updated_walks, df), axis = 0)
        self.current_walks = updated_walks
        self.walk_update_necessary = False
        
    def get_stationary_distribution(self):
        out = self.current_walks
        out = out[np.lexsort((-out[:,2], out[:,3], out[:,0])),:]
        rw_per_node = out[np.unique(out[:,[0,3]], return_index=True, axis = 0)[1],:]
        rw_per_node  = np.split(rw_per_node, np.unique(rw_per_node[:,0], return_index = True )[1])[1:]
        lens = [x.shape[0] for x in rw_per_node][1:]
        out_per_node = [np.unique(x[:,1], return_counts = True) for x in rw_per_node ]
        probs = dict.fromkeys( self.vertices,[] )
        for i in range(len(lens)):
            df = out_per_node[i]
            for j in range(len(df[0])):
                x = df[0][j]
                count = df[1][j]
                probs[x] = probs[x] + [count/lens[i]]
                
        probs = [sum(x)/len(self.vertices) for x in probs.values()]
        self.stationary_dist = probs
            
if __name__ == "__main__":
    G = nx.gnc_graph(30, seed = 2)
    edgelist = np.row_stack( G.edges )
    fpr = fastppr(edgelist, jump = 0.15)
    fpr.get_neighbours()
    fpr.first_walk(R = 100, max_length=10,seed=0)
    fpr.get_stationary_distribution()
    nx.draw(G, node_color = fpr.stationary_dist, with_labels = True)
    # Delete the highest ranked node
    print("removing node 0...")
    node = np.argmax(fpr.stationary_dist)
    dels = edgelist[(edgelist[:,0]==node)|(edgelist[:,1]==node),:]
    fpr.update_graph(deletions = np.row_stack( ([node,node], dels) ) )
    
    fpr.update_walk(R=100, max_length=10, deletions = dels)
    fpr.get_stationary_distribution()
    G_new = nx.from_edgelist( fpr.edges )   
    nx.draw(G_new, with_labels = True, node_color = fpr.stationary_dist)


from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field
from .graph import *

from scipy.optimize import linear_sum_assignment
def make_bipartite_graph_CBS(g: Graph, p_0, p_1, agents, conflicts, visitable):
    r = 3*len(p_1)-len(p_0)
    
    cost_matrix = [[float("inf") for _ in range(3*len(p_1))] for _ in range(r+len(p_0))]
    for idx1,nd in enumerate(p_0):
            
            for idx2,nd2 in enumerate(p_1):
                if visitable[nd[0]][nd2] == 0:
                    continue

                for k in range(3):
                    cost_matrix[idx1][3*idx2 + k] = g._cost_matrix[nd[1]][nd2]
    for idx1 in range(len(p_0),r+len(p_0)):
        for idx2,nd2 in enumerate(p_1):
                

                for k in range(3):
                    cost_matrix[idx1][3*idx2 + k] = 0
    return cost_matrix
def make_bipartite_graph(g: Graph, p_0, p_1, visitable = None):
    r = max(len(p_1),len(p_0))
    # print(p_0,p_1)
    cost_matrix = [[float("inf") for _ in range(r)] for _ in range(r)]
    for idx1,nd in enumerate(p_0):
            
            for idx2,nd2 in enumerate(p_1):
                if visitable and visitable[nd[0]][nd2] == 0:
                    continue
                # print(idx1,idx2,r,len(p_1),len(p_0))
                cost_matrix[idx1][idx2] = g._cost_matrix[nd][nd2] + g.nodes[nd].weight
    for idx1 in range(len(p_0),r):
        for idx2,nd2 in enumerate(p_1):
            cost_matrix[idx1][idx2] = 0

    for idx1 in range(len(p_1),r):
        for idx2,nd2 in enumerate(p_0):
            cost_matrix[idx1][idx2] = 0
    return cost_matrix


def bipartite_matching(cost_matrix):
        
        cost_matrix = np.array(cost_matrix)
        shape = cost_matrix.shape
        descending_order = np.argsort(cost_matrix.flatten())[::-1]
        inf_weight = 1e6
        for idx in descending_order:
            cur_max_weight = cost_matrix[idx //
                                         shape[0]][idx % shape[1]]
            cost_matrix[idx//shape[0]][idx % shape[1]] = inf_weight
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if cost_matrix[row_ind, col_ind].sum() >= inf_weight:
                return cur_max_weight, list(zip(col_ind, row_ind))

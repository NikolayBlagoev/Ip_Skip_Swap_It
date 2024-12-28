
import numpy as np
from schedulers.graph import *
from schedulers.a_star_modified import Agent
from schedulers.CBS import CBS, CBS_item
from schedulers.graph_partitioning import *
from schedulers.gcma import *
import random
import numpy as np
from sys import argv
from schedulers.com_sym import *
from schedulers.communication_costs import *
random.seed(int(argv[1]))
np.random.seed(int(argv[1]))
PAT_LENGTH = 4
MAX_MB_PER_STAGE = 9

LAYERS_PER_DEVICE = 4
SAMPLES_IN_MB = 4
MB_COUNT = 9
NUMBER_OF_NODES = 20
DP_SIZE_IN_BYTES = 1346446748
MB_SIZE_IN_BYTES = 33554858
FACTOR = DP_SIZE_IN_BYTES/(MB_SIZE_IN_BYTES*SAMPLES_IN_MB*MB_COUNT)
partition_sizes = [5,3,3,3,3,3] # Number of devices per partition
assert sum(partition_sizes) == NUMBER_OF_NODES
memory = 3 # memory per device

setting = "geo-distributed"
# options for setting:
# geo-distributed
# single-cluster
# 5-clusters

locations = get_locations(setting)
computations = get_computations(setting)
# get nodes:
node_list = []
while len(node_list) < NUMBER_OF_NODES:
    for v in locations:
        node_list.append(v)
        if len(node_list) == NUMBER_OF_NODES:
            break


# create cost matrix and computation array
cost_matrix = [[0 for _ in range(len(node_list))] for _ in range(len(node_list))]


wm = [0 for _ in range(len(node_list))]
for x in range(len(node_list)):
    wm[x] = computations[node_list[x]]*LAYERS_PER_DEVICE*SAMPLES_IN_MB
    
    for y in range(len(node_list)):
        if x == y:
            continue
        cost_matrix[x][y] = delay_map(node_list[x],node_list[y],sz=MB_SIZE_IN_BYTES*SAMPLES_IN_MB)

g = Graph(0)
output = {}
g.add_cost_matrix(cost_matrix,wm)
g.fill_incident_edges()
bst = None 
score = float("inf")
# Find best arrangement:
for _ in range(5):
    partitions, scores, _ = GCMA(g,partition_sizes=partition_sizes,trails=4000,population_size=100,factor=FACTOR)
    ret = np.argmin(scores)
    if scores[ret] < score:
        score = scores[ret]
        bst = reconstruct_partition(g,partitions[ret],partition_sizes)

# reconstruct arrangement of nodes: 
ret = bst
output["memory"] = memory
output["partitions"] = ret
output["GCMAscore"] = score
output["locations"] = node_list
MAIN_WM = wm
nds = {}
for idx in range(len(node_list)):
    nds[idx] = ComNode(MAIN_WM[idx],idx,3)
output["baseline-sends"] = MB_COUNT + (partition_sizes[0]-partition_sizes[1])*MB_COUNT/partition_sizes[1]
output["ours-sends"] = MB_COUNT
for num,idx in enumerate(ret[0]):
    if num >= partition_sizes[1]:
        break
    
    for k in range(5):
        path = {}
        for p in range(1,len(ret)):
            path[ret[p-1][num]] = ret[p][num]
        mb = MB(k+5*num,path,idx)
        nds[idx].receive(mb)
output["baseline-expected-time"] = run_simulation(nds,ret,cost_matrix)
print("EXPECTED TIME STANDARD",output["baseline-expected-time"])

tmp = []
for idx, p in enumerate(ret):
    for nd in p:
        tmp.append(nd)
        g.nodes[nd].properties["partition"] = idx
tmp.sort()


# COLLISION AWARE:
agents = []
for num,idx in enumerate(ret[0]):

    for k in range(3):
        # add microbatch/agent
        agents.append(Agent(k + 3*num, idx, k*wm[idx]))

# Run CBS
solutions: CBS_item = CBS(g,agents,lambda x1,x2: cost_matrix[x1][x2],ret,path_l=PAT_LENGTH,memory=memory,mb_per_stage_max=MAX_MB_PER_STAGE)
visits_per_node = {}
for ag_sol in solutions.solution:

    for nd in ag_sol[1]:
                
        if nd[0] not in visits_per_node:
            visits_per_node[nd[0]] = []
        visits_per_node[nd[0]].append((ag_sol[2],nd[1])) 
for v in visits_per_node.values():
    v.sort(key=lambda el: el[1])
output["ca-processing-order"] = visits_per_node




nds = {}
for idx in range(len(node_list)):
    nds[idx] = ComNode(MAIN_WM[idx],idx,3)

paths = {}
for ag in solutions.solution:
    path = {}
    prv = None
    for t in ag[1]:
        
        t = t[0]
        if prv == None:
            prv = t
            continue
        if t == agents[ag[2]].start_idx:
            break
        path[prv] = t
        prv = t
    # for t in ag[1]:
    #     print(t)
    # print(path)
    # print(ag[2], agents[ag[2]].start_idx)
    tmp = MB(ag[2],path,agents[ag[2]].start_idx)
    paths[ag[2]] = path
    nds[agents[ag[2]].start_idx].receive(tmp)
visits_per_node = {}

output["ca-expected-time"] = run_simulation(nds,ret,cost_matrix)
for k,v in nds.items():
    visits_per_node[k] = len(v.received_sent)
# output["ca-expected-time"] = solutions.dist
output["ca-mb-per-node"] = visits_per_node
output["ca-paths"] = paths
print("EXPECTED TIME WITH COLLISION AWARENESS:", output["ca-expected-time"])

agents = []
for num,idx in enumerate(ret[0]):

    for k in range(3):
        agents.append(Agent(k + 3*num, idx, k*wm[idx]))

solutions: CBS_item = CBS(g,agents,lambda x1,x2: cost_matrix[x1][x2],ret,constraints=[True,True,False],path_l=PAT_LENGTH,memory=memory,mb_per_stage_max=MAX_MB_PER_STAGE)
visits_per_node = {}
for ag_sol in solutions.solution:

    for nd in ag_sol[1]:
                
        if nd[0] not in visits_per_node:
            visits_per_node[nd[0]] = []
        visits_per_node[nd[0]].append((ag_sol[2],nd[1])) 
for v in visits_per_node.values():
    v.sort(key=lambda el: el[1])
output["non-ca-processing-order"] = visits_per_node
nds = {}
for idx in range(len(node_list)):
    nds[idx] = ComNode(MAIN_WM[idx],idx,3)

paths = {}
for ag in solutions.solution:
    path = {}
    prv = None
    for t in ag[1]:
        t = t[0]
        if prv == None:
            prv = t
            continue
        if t == agents[ag[2]].start_idx:
            break
        path[prv] = t
        prv = t
    # print(path)
    # print(ag[2], agents[ag[2]].start_idx)
    tmp = MB(ag[2],path,agents[ag[2]].start_idx)
    paths[ag[2]] = path
    nds[agents[ag[2]].start_idx].receive(tmp)


output["nonca-expected-time"] = run_simulation(nds,ret,cost_matrix)
print("EXPECTED TIME WITHOUT COLLISION AWARENESS:", output["nonca-expected-time"])

visits_per_node = {}

for k,v in nds.items():
    visits_per_node[k] = len(v.received_sent)
output["non-ca-mb-per-node"] = visits_per_node
output["non-ca-paths"] = paths
# save to JSON
import json
with open("communication.json","w") as fd:
    json.dump(output,fd,indent=2)
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field
from .graph import *
from .bipartite_matching import *
from .a_star_modified import *
import asyncio
import time
from joblib import Parallel, delayed
from copy import deepcopy
@dataclass(order=True)
class CBS_item:
    dist: int
    solution: List=field(compare=False)
    conflicts: List[Conflict]=field(compare=False)
    visitable: Dict[int,List[int]]=field(compare=False)
    visitable_stages: Any=field(compare=False)
    uniq_str: str=field(compare=False)
    speeds: List=field(compare=False)

def count_conflicts(g, sol, count_per_node):
    for k,v in g.nodes.items():    
        count_per_node[k].clear()
    for ag_sol in sol:
            first_node = ag_sol[1][0][0]
            prv_nd = first_node
            prv_tm = 0
            count = 0
            # print(len(ag_sol[1]))
            for nd in ag_sol[1]:
                
                
                
                count_per_node[nd[0]].append((ag_sol[2],nd[1],nd[2]))
                prv_nd = nd[0]
                prv_tm = nd[1]

    count_col = 0

    for k in range(len(g.nodes)):
                
                for idx, visit in enumerate(count_per_node[k]):
                    
                    for idx2, visit2 in enumerate(count_per_node[k]):
                        if idx == idx2:
                            continue
                        if visit[0] == visit2[0]:
                            continue
                        # print(count_per_node)
                        # print(visit)
                        if (visit[1] <= visit2[1] and visit[2] > visit2[1]) or (visit[1] <= visit2[2] and visit[2] > visit2[2]):
                            count_col += 0.01
    count_col /= 2
    for k,v in g.nodes.items():    
        count_per_node[k].clear()
    return count_col
import itertools
def CBS(g:Graph, agents: list[Agent], heuristic, partitions, constraints = [False, True, True], path_l = 4):
    h: List[CBS_item] = []
    heapq.heapify(h)
    visited = dict()
    visitable: Dict[int,List[int]] = np.full((len(agents),len(g.nodes)),1)
    visitable_stages = np.full((len(agents),len(partitions)),1)
    
       
    conflicts: List[Conflict] = []
    visited: Dict[str,bool] = dict()
   
    curr = time.time()
    # a_star_modified(g,agents[0].start_idx,heuristic,agents[0].idx,agents[0].dt,conflicts,visitable,visitable_stages,6)
    results = Parallel(n_jobs=len(agents))(delayed(a_star_modified)(g,a.start_idx, heuristic, a.idx, a.dt, conflicts, visitable, visitable_stages, path_l) for a in agents)
    print(time.time()-curr)
    cost = 0
    for v in results:
        cost = max(v[0],cost)
    heapq.heappush(h,CBS_item(cost,results,conflicts, visitable, visitable_stages,"",[]))
    count_per_node: Dict[int,int] = dict()
    count_per_partitions: Dict[int,int] = dict()
    for k,v in g.nodes.items():
        count_per_node[k] = []
    visits_per_node: Dict[int,int] = dict()
    for k,v in g.nodes.items():
        visits_per_node[k] = []
    for v in range(len(partitions)):
        count_per_partitions[v] = []
    solutions = []
    last_value = 0
    check_1 = False
    while len(h) > 0:
        for k,v in g.nodes.items():
            count_per_node[k].clear()
        for k,v in g.nodes.items():
            visits_per_node[k].clear()
        for v in range(len(partitions)):
            count_per_partitions[v].clear()
        if len(solutions) == 32:
            print(32,"viable solutions found")
            h = []
            heapq.heapify(h)
            for s in solutions:
                heapq.heappush(h,s)
            solutions = []
            check_1 = True
            
        sol = heapq.heappop(h)
        if sol.dist - last_value < -0.001:
            print(sol.dist,last_value)
            for c in sol.conflicts:
                if c.type != 3:
                    continue
                                    
                print(c.agidx,c.ndix,c.tmstart,c.tmend)
            for ag_sol in sol.solution:
                print(ag_sol[0])
                for nd in ag_sol[1]:
                    print(ag_sol[2],nd)
            assert sol.dist >= last_value
        last_value = sol.dist
        # print(sol.dist)
        # print(sol.visitable_stages)
        visited[sol.visitable_stages.data.tobytes() + sol.visitable.data.tobytes()] = True
        # check for conflicts
        
        

        
        # Check times on nodes:
        for ag_sol in sol.solution:
            first_node = ag_sol[1][0][0]
            prv_nd = None
            prv_tm = 0
            count = 0
            # print(len(ag_sol[1]))
            for nd in ag_sol[1]:
                if nd[0] == first_node and count == 1:
                    break
                elif nd[0] == first_node:
                    count += 1
                
                count_per_partitions[g.nodes[nd[0]].properties["partition"]].append((ag_sol[2],nd[1]))
                count_per_node[nd[0]].append((ag_sol[2],nd[1],nd[2],nd[0],prv_nd))
                prv_nd = nd[0]
                prv_tm = nd[1]
            
            prv_nd = None
            for nd in ag_sol[1]:
                
                
                visits_per_node[nd[0]].append((ag_sol[2],nd[1],nd[2],nd[0],prv_nd)) 
                prv_nd = nd[0]
                
        
        conflicts = []
        flag = False
        # CONFLICT TYPE 2:
        if constraints[1]:
            # print("CHECKING CONSTRAINT 1")
            for k in range(len(partitions)):
                
                if k == 0:
                    continue
                if len(count_per_partitions[k]) > 9:
                    flag = True
                    # print(k)
                    count_per_partitions[k].sort(key = lambda el: el[1])
                    ttl_count = len(count_per_partitions[k])
                    # print(k,count_per_partitions[k])
                    exclude_agents = []
                    for ag in agents:
                        if np.sum(sol.visitable_stages[ag.idx]) == path_l:
                            exclude_agents.append(ag.idx)
                        
                    count_per_partitions[k] = [ t for t in count_per_partitions[k] if t[0] not in exclude_agents]
                    # print("EXCLUDE", exclude_agents)
                    if len(exclude_agents) == len(agents):
                        continue
                    for comb in itertools.combinations(count_per_partitions[k][ -(len(count_per_partitions[k]) - (6-len(exclude_agents))):], ttl_count-9):
                        if len(conflicts) > 2:
                            continue
                        tmp = []
                        # print(k,comb)
                        for c in comb:
                            
                            
                            tmp.append(Conflict(c[0],k,-1000,float("inf"),2))
                        conflicts.append(tmp)
                    break
        if not flag and not check_1:
            sol.visitable_stages[:,:] = 0
            sol.visitable_stages[:,0] = 1
            for k in range(len(partitions)):
                if k == 0:
                    continue
                for t in count_per_partitions[k]:
                    # print(k,t)
                    sol.visitable_stages[t[0]][k] = 1
            

            
            solutions.append(sol)
            continue
        impossible = False
        if check_1 and not flag:
            for k in range(len(g.nodes)):
                # continue
                if flag:
                    break
                if g.nodes[k].properties["partition"] == 0:
                    continue
                if len(count_per_node[k]) > 3:
                    flag = True
                    # print(k)
                    this_partition = g.nodes[k].properties["partition"]
                    this_partition = partitions[this_partition]
                    p_0 = []
                    p_1 = []
                    # print(this_partition)
                    for p in this_partition:
                        if p in count_per_node:
                            for l in count_per_node[p]:
                                p_0.append((l[0],l[3],l[4]))
                            p_1.append(p)
                    # if len(p_0)
                    cm = make_bipartite_graph_CBS(g,p_0,p_1,sol.conflicts,agents,sol.visitable)
                    try:
                        ret = bipartite_matching(cm)
                    except ValueError:
                        ret = [float("inf")]
                    if ret[0] > 1000:
                        impossible = True
                        continue
                    tmp = []
                    
                            
                    for r in ret[1]:
                        # print(r)
                        if r[0] >= len(p_0):
                            continue
                        nd = p_1[r[1]//3]
                        for p in this_partition:
                            if p == nd:
                                continue
                            tmp.append(Conflict(p_0[r[0]][0],p,-1000,float("inf"),1))
                    conflicts.append(tmp)       
                    break
        if impossible:
            continue
        conflict_3 = False

        if constraints[2] and not flag:
            checked = []
            for ag in sol.speeds:
                
                for k in range(len(g.nodes)):
                    
                    
                    for idx, visit in enumerate(visits_per_node[k]):
                        if len(checked) > 2:
                            break
                        
                        if visit[0] in checked:
                            continue
                        if visit[0] != ag[0]:
                            continue
                        for idx2, visit2 in enumerate(visits_per_node[k]):
                            if visit2[0] in checked:
                                continue
                            if idx == idx2:
                                continue
                            if visit[0] == visit2[0]:
                                continue
                            # print(count_per_node)
                            # print(visit)
                            if (visit[1] <= visit2[1] and visit[2] > visit2[1]) or (visit2[1] <= visit[1] and visit2[2] > visit[1]):
                                flag = True
                                checked.append(visit[0])
                                checked.append(visit2[0])
                                # print("TYPE 3")
                                # print(visit,visit2,k)
                                for c in sol.conflicts:
                                    if c.type != 3:
                                        continue
                                    
                                    # print(c.agidx,c.ndix,c.tmstart,c.tmend)
                                    if c.agidx == visit[0] and c.ndix == k and c.tmstart == visit2[1] and c.tmend == visit2[2]:
                                        print("DUPLICATE", visit[0],visit2[1], visit2[2], c.tmstart, c.tmend, k)
                                        print(visit,visit2)
                                        for ag_sol in sol.solution:
                                            
                                            print(ag_sol[0],ag_sol[2])
                                            # print(len(ag_sol[1]))
                                            for nd in ag_sol[1]:
                                                print(nd)
                                        exit()
                                    if c.agidx == visit2[0] and c.ndix == k and c.tmstart == visit[1] and c.tmend == visit[2]:
                                        print("DUPLICATE", visit2[0],visit[1], visit[2], c.tmstart, c.tmend, k)
                                        print(visit,visit2)
                                        for ag_sol in sol.solution:
                                            
                                            print(ag_sol[0],ag_sol[2])
                                            # print(len(ag_sol[1]))
                                            for nd in ag_sol[1]:
                                                print(nd)
                                        exit()
                                    
                                conflict_3 = True
                                if len(conflicts) == 0:

                                    conflicts.append([Conflict(visit[0],k,visit2[1],visit2[2],3)])  
                                    conflicts.append([Conflict(visit2[0],k,visit[1],visit[2],3)])
                                else:
                                    conflicts[0] += [Conflict(visit[0],k,visit2[1],visit2[2],3)]
                                    conflicts[1] += [Conflict(visit2[0],k,visit[1],visit[2],3)]
                                break
        

        if not flag:
            print(sol.dist)
            # print(sol.visitable)
            return sol
            
        # print(len(conflicts))
        for c in conflicts:
            
            tmpvisitable_stages = sol.visitable_stages.copy()
            tmpvisitable = sol.visitable.copy()
            for conf in c:
                if conf.type == 2:
                    # print(conf.agidx,"CANNOT VISIT",conf.ndix)
                    tmpvisitable_stages[conf.agidx][conf.ndix] = 0
                    # print(tmpvisitable_stages[conf.agidx])
                    for nd_p in partitions[conf.ndix]:
                        tmpvisitable[conf.agidx][nd_p] = 0
                elif conf.type == 1:
                    tmpvisitable[conf.agidx][conf.ndix] = 0
                   
                    count = 0
                    for nd_p in partitions[g.nodes[conf.ndix].properties["partition"]]:
                        count += sol.visitable[conf.agidx][nd_p]
                    if count == 0:
                        tmpvisitable_stages[conf.agidx][g.nodes[conf.ndix].properties["partition"]] = 0
                        
                        
                        
                        
                    
            if not conflict_3 and tmpvisitable_stages.data.tobytes() + tmpvisitable.data.tobytes() in visited:
                
                continue

            
            comb = c + sol.conflicts.copy()
            
            curr = time.time()
            results = Parallel(n_jobs=len(agents))(delayed(a_star_modified)(g,a.start_idx, heuristic, a.idx, a.dt, comb, tmpvisitable, tmpvisitable_stages, path_l) for a in agents)
            # print(time.time()-curr)

            cost = 0
            speeds = []
            for v in results:
                if v == None:
                    cost = float("inf")
                    continue
                speeds.append((v[2],v[0]))
                cost = max(v[0],cost)
            if cost > 10000:
                continue
            speeds.sort(key=lambda el: el[1],reverse= True)
            # count_col = count_conflicts(g,results,count_per_node) if conflict_3 else 0
            # print(count_col)
            heapq.heappush(h,CBS_item(cost,results,comb, tmpvisitable, tmpvisitable_stages, "",speeds ))


    
        for k,v in g.nodes.items():    
            count_per_node[k].clear()
        for k in count_per_partitions.keys():
            count_per_partitions[k].clear()
    
    print("NO SOLUTION")
    return
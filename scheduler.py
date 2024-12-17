from collections import OrderedDict
from math import floor
from copy import deepcopy
class MB(object):
    def __init__(self, uniqid, path, invpath):
        self.tm = 0
        self.uniqid = uniqid
        self.path = path
        self.back = False
        self.invpath = invpath

class Node(object):
    def __init__(self,comp, ndid):
        
        self.comp = comp
        self.ndid = ndid
        self.received = []
        self.backreceived = []
        self.processed = []
        self.collisions = []
        self.process_at = []
        self.received_sent = OrderedDict()
    


    def receive(self,b: MB):
        self.received.append(b)
    def backreceive(self,b:MB):
        self.backreceived.append(b)
    def send(self, dl, nds, tmunit):
        self.received.sort(key=lambda mb: mb.tm)
        
        for idx,mb in enumerate(self.received):
            if floor(mb.tm) != tmunit:
                continue
            # print(self.ndid,mb.tm,tmunit)
            
            nxt = None
            if self.ndid in mb.path:
                
                nxt = mb.path[self.ndid]
            self.processed.append((mb.tm, mb.tm + self.comp, mb.back, mb.uniqid))

            self.received_sent[mb.uniqid] = (mb.tm, mb.tm + self.comp)
            
            if nxt != None and not mb.back :
                # print(self.ndid, "to",nxt)
                tmp = deepcopy(mb)
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            
            elif self.ndid != 0:
                # print("BACK")
                nxt = mb.invpath[self.ndid]
                # print(self.ndid, "to",nxt)
                tmp = deepcopy(mb)
                tmp.back = True
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            for idx2,mb2 in enumerate(self.received):
                if idx2 == idx:
                    continue
                if mb2.uniqid == mb.uniqid:
                    continue
                
                v = self.received_sent[mb.uniqid]
                if v[0] <= mb2.tm and v[1] > mb2.tm:
                    # print("collision")
                    self.collisions.append((mb2.uniqid,mb.uniqid,mb2.tm,v[1],mb2.back,mb.back,tmunit))
                    mb2.tm = v[1]

        
    

delay_bandwidth_dict = {
    "Oregon-Virginia": (67, 0.395),
    "Oregon-Ohio": (49, 0.55),
    "Oregon-Tokyo": (96, 0.261),
    "Oregon-Seoul": (124, 0.23),
    "Oregon-Singapore": (163, 0.17),
    "Oregon-Sydney": (139, 0.18),
    "Oregon-London": (136, 0.21),
    "Oregon-Frankfurt": (143, 0.202),
    "Oregon-Ireland": (124, 0.241),
    "Virginia-Ohio": (11, 0.56),
    "Virginia-Tokyo": (143, 0.262),
    "Virginia-Seoul": (172, 0.250),
    "Virginia-Singapore": (230, 0.182),
    "Virginia-Sydney": (197, 0.191),
    "Virginia-London": (76, 0.58),
    "Virginia-Frankfurt": (90, 0.51),
    "Virginia-Ireland": (67, 0.55),
    "Ohio-Tokyo": (130, 0.347),
    "Ohio-Seoul": (159, 0.264),
    "Ohio-Singapore": (197, 0.226),
    "Ohio-Sydney": (185, 0.242),
    "Ohio-London": (86, 0.52),
    "Ohio-Frankfurt": (99, 0.400),
    "Ohio-Ireland": (77, 0.57),
    "Tokyo-Seoul": (34, 0.55),
    "Tokyo-Singapore": (73, 0.505),
    "Tokyo-Sydney": (100, 0.380),
    "Tokyo-London": (210, 0.183),
    "Tokyo-Frankfurt": (223, 0.18),
    "Tokyo-Ireland": (199, 0.232),
    "Seoul-Singapore": (74, 0.57),
    "Seoul-Sydney": (148, 0.29),
    "Seoul-London": (238, 0.171),
    "Seoul-Frankfurt": (235, 0.179),
    "Seoul-Ireland": (228, 0.167),
    "Singapore-Sydney": (92, 0.408),
    "Singapore-London": (169, 0.25),
    "Singapore-Frankfurt": (155, 0.219),
    "Singapore-Ireland": (179, 0.246),
    "Sydney-London": (262, 0.163),
    "Sydney-Frankfurt": (265, 0.164),
    "Sydney-Ireland": (254, 0.172),
    "London-Frankfurt": (14, 0.57),
    "London-Ireland": (12, 0.54),
    "Frankfurt-Ireland": (24, 0.54),
    "Sofia-Frankfurt":(25.05,0.160),
    "Sofia-Seoul":(323.03,0.128),
    "Sofia-Sydney":(298.90,0.134),
    "Sofia-Virginia":(120.46,0.143),
    "Sofia-Oregon":(114.24,0.096),
    "Sofia-Ohio":(125.96,0.097),
    "Sofia-London":(68.07,0.150),
    "Sofia-Tokyo":(318.88,0.130),
    "Sofia-Amsterdam":(38.24,0.150),
    "Amsterdam-Frankfurt":(8.72,0.150),
    "Amsterdam-Seoul":(288.39,0.134),
    "Amsterdam-Sydney":(265.94,0.135),
    "Amsterdam-Virginia":(80.81,0.150),
    "Amsterdam-Oregon":(72.29,0.150),
    "Amsterdam-Ohio":(75.31, 0.150),
    "Amsterdam-London":(7,0.150),
    "Amsterdam-Tokyo":(278.65,0.13),
}
computational_cost = {
    "Sofia": 10,
    'Singapore': 9, 
    'Ireland': 8.4,
    'Sydney': 7.7,
    'Frankfurt': 9.2, 
    'Seoul': 10.5, 
    'Oregon': 8.8, 
    'Frankfurt': 9.1, 
    'Ohio': 8.1, 
    'Tokyo': 9.9, 
    'Amsterdam': 7.8, 
    'Virginia': 8.1

}

# for k in computational_cost.keys():
#     computational_cost[k] = 0
def delay_map(loc1,loc2):
    p1 = loc1
    p2 = loc2
    if delay_bandwidth_dict.get(p1+"-"+p2) != None:
        ret = delay_bandwidth_dict.get(p1+"-"+p2)
    elif delay_bandwidth_dict.get(p2+"-"+p1) != None:
        ret = delay_bandwidth_dict.get(p2+"-"+p1)
    else:
        ret = (1,0.9)
    return ret[0]/1000 + 250*6291908/(1024**3 * ret[1])
results = {}
results["normal"] = []
results["random"] = []
results["round robin"] = []
results["adaptive"] = []
results["adaptive--"] = []
results["hybrid"] = []
convergence = {}
convergence["normal"] = 2*5528
convergence["random"] = 2*7840
convergence["round robin"] = 2*6120
convergence["adaptive"] = 2*6120
convergence["adaptive--"] = 2*6120
from random import shuffle
BIG_WEIGHT = 1000
def best_from(dp,y,x,cost_matrix,comp_cost,locs):
    best = (-1,-1,-1,0,0,0)
    best_val = BIG_WEIGHT*100
    
    for skip in range(5):
        if x - 1 < 0:
            continue
        
        if y - skip < 0:
            continue
        if dp[y-skip][x-1][0] == BIG_WEIGHT:
            exit()
        curr_val = dp[y-skip][x-1][0] + cost_matrix[x-1-skip+y][x+y]+ comp_cost[locs[y+x]] + dp[y][x][0]
        # curr_val = max(dp[y-skip][x-1][4], cost_matrix[x-1-skip+y][x+y]+ comp_cost[locs[y+x]]) 
        
        #added = curr_val + dp[y][x][0] + dp[y-skip][x-1][5]
        added = curr_val
        last_skipped = dp[y-skip][x-1][3]
        

        for tmp in range(y-skip+x, y+x):
            if last_skipped != -1:
                curr_val += (cost_matrix[last_skipped][tmp] + comp_cost[locs[tmp]])
            last_skipped = tmp
        
            
        if best_val > added:
            best_val = added
            best = (y-skip,x-1, last_skipped,curr_val,dp[y-skip][x-1][5])
             

    return (best_val,best[0],best[1],best[2],best[3],dp[y][x][0]+best[4])
import random
import itertools
from tqdm import tqdm
random.seed(55)
# random.seed(10)
for p in range(100):
    locations = ['Sofia', 'Singapore', 'Ireland', 'Sydney', 'Frankfurt', 'Seoul', 'Oregon', 'Frankfurt', 'Ohio', 'Ohio', 'Tokyo', 'Amsterdam', 'Virginia']
    shuffle(locations)
    # evv = itertools.permutations(locations,len(locations))
    # best_cost = 400000
    # best_arrangement = None
    # for i in tqdm(evv):
    #     cost = 0
    #     for c in range(len(i)):

    #         if c == 0:
    #             continue
    #         cost += delay_map(i[c],i[c-1])
            
    #     if cost < best_cost:
    #             best_cost = cost
    #             best_arrangement = locations
    #             print(best_arrangement,cost)
        
    # print(best_arrangement)
    # print(best_cost)
    # shuffle(locations)
    normal_tm = 0
    for strt in ["normal","round robin", "adaptive","adaptive--","random","hybrid"]:
        print("--------------")
        print(strt)
        
        cost_matrix = [[0 for i in range(len(locations))] for _ in range(len(locations))]

        for idx in range(len(locations)):
            for idx2 in range(len(locations)):
                if idx == idx2:
                    continue
                cost_matrix[idx][idx2] = delay_map(locations[idx],locations[idx2])

        mb_count = 3
        nds = {}
        cp_nds = {}
        for idx in range(len(locations)):
            nds[idx] = Node(computational_cost[locations[idx]],idx)
            cp_nds[idx] = Node(computational_cost[locations[idx]],idx)

        
        must_include = []
        prev_paths = {}
        microbatches_scheduled = []
        for i in range(mb_count):
            path = {}
            invpath = {}
           
            
            if strt == "round robin":
                to_skip = [k for k in range((i*4)+1,(i*4)+5)]
                # print(to_skip)
                for k in range(len(locations) - 1):
                    if k in to_skip:
                        continue
                    if k + 1 in to_skip:
                        if to_skip[-1] + 1 < len(locations):
                            path[k] = to_skip[-1] + 1
                            invpath[to_skip[-1] + 1] = k
                    else:
                        path[k] = k + 1
                        invpath[k+1] = k
                print(path)
                act_cost = computational_cost[locations[0]]
                for k,v in path.items():
                    act_cost += cost_matrix[k][v] + computational_cost[locations[v]]
                print("ACTUAL COST: ",act_cost)
                # print(invpath)
                mb = MB(i,path,invpath)
                microbatches_scheduled.append(mb)
            
            elif strt == "random":
                to_skip = random.sample([k for k in range(1,len(locations))],4)
                # print(to_skip)
                
                for k in range(len(locations) - 1):
                    if k in to_skip:
                        continue
                    if k + 1 in to_skip:
                        j = k+1
                        while j in to_skip:
                            j += 1
                        if j >= len(locations):
                            continue
                        path[k] = j
                        invpath[j] = k
                        
                    else:
                        path[k] = k + 1
                        invpath[k+1] = k
                # print(path)
                mb = MB(i,path,invpath)
                microbatches_scheduled.append(mb)
            elif strt == "normal":
                for k in range(len(locations) - 1):
                    path[k] = k + 1
                    invpath[k+1] = k
                mb = MB(i,path,invpath)
                microbatches_scheduled.append(mb)
            elif strt == "adaptive":
                pth_len = floor(3*(len(locations) - 1)/4)
                rmv = len(locations) - pth_len
                
                dp = [[(BIG_WEIGHT,-1,-1, -1, 0,0) for _ in range(pth_len) ] for _ in range(rmv + 1)]

                for y in range(rmv + 1):
                    for x in range(pth_len):
                        if y+x in must_include:
                            
                            dp[y][x] = (0,-1,-1, -1, 0)
                
                dp[0][0] = (computational_cost[locations[0]],0,0,-1,0,0)
                # for y in range(rmv + 1):
                #     for x in range(pth_len):
                    
                #         print(dp[y][x][0],end="\t")
                #     print()
                for x in range(pth_len):
                    for y in range(rmv + 1):
                        if y == 0 and x == 0:
                            continue

                        dp[y][x] = best_from(dp,y,x,cost_matrix,computational_cost,locations)
                        
                        # print(y,x,dp[y][x])
                # print(dp)
                # for y in range(rmv + 1):
                #     for x in range(pth_len):
                #         print(round(dp[y][x][0],2),end="\t")
                #     print()
                best_path = []
                x = pth_len-1
                best_val = 100000000
                y = 0
                for ty in range(rmv + 1):
                    last_skipped = dp[ty][x][3]
                    # print(ty,x,"last skipped was",last_skipped)
                    vl = dp[ty][x][0]
                    for nd in range(ty+x+1,len(locations)):
                        if last_skipped != -1:
                            vl += cost_matrix[last_skipped][nd]
                        # print(ty,x,"need to also skip",nd)
                        last_skipped = nd
                    if vl < best_val:
                        best_val = vl
                        y = ty
                print(best_val)
                while y != 0 or x != 0:
                    best_path.append((y,x))
                    ret = dp[y][x]
                    # print("we had skipped",ret[3])
                    y = ret[1]
                    x = ret[2]
                print(best_path)
                best_path.reverse()
                tmp_path = [0]
                
                for t in best_path:
                    tmp_path.append(t[0]+t[1])
                # print(tmp_path)
                for k in range(len(tmp_path) - 1):

                    path[tmp_path[k]] = tmp_path[k + 1]
                    invpath[tmp_path[k + 1]] = tmp_path[k]
                act_cost = 0
                for k,v in path.items():
                    act_cost += cost_matrix[k][v] + computational_cost[locations[v]]
                print("ACTUAL COST: ",act_cost)
                must_include += [k for k in range(1,len(locations)) if k not in tmp_path]
                # print(must_include)
                
                mb = MB(i,path,invpath)
                microbatches_scheduled.append(mb)
                
                if i == 2:
                    
                    microbatches_scheduled.reverse()
                    best_schedule = deepcopy(microbatches_scheduled)
                    flg = True
                    best_val = None
                    while flg:
                        curr_schedule = deepcopy(best_schedule)
                        flg = False
                        tmp_cst = deepcopy(cp_nds)
                        for i,mb in enumerate(curr_schedule):
                            mb = deepcopy(mb)
                            mb.tm = 0
                            mb.uniqid = i
                            tmp_cst[0].receive(mb)
                        for tm in range(900):
                                for idx in range(len(locations)):
                                    tmp_cst[idx].send(cost_matrix,tmp_cst,tm)
                        
                        swaps = {}
                        if best_val != None:
                            for idx in range(len(locations)):
                                if idx == 0:
                                    continue
                                # print(idx,tmp_cst[idx].collisions)
                                for col in tmp_cst[idx].collisions:
                                    mb1 = curr_schedule[col[0]]
                                    mb2 = curr_schedule[col[1]]
                                    path1 = mb1.path
                                    path2 = mb2.path
                                    
                                    prev_1 = tmp_cst[mb1.invpath[idx]]
                                   
                                    prev_2 = tmp_cst[mb2.invpath[idx]]
                                    prev_send = col[2] - cost_matrix[prev_1.ndid][idx]
                                    for resolve in range(idx-2,idx+2):
                                        if swaps.get(mb.uniqid) and (resolve in swaps.get(mb.uniqid) or idx in swaps.get(mb.uniqid)):
                                            continue
                                        if idx == resolve or resolve <= 0 or resolve >= len(locations):
                                            continue
                                        if resolve in path1:
                                            
                                            flg = True
                                            
                                            for p in tmp_cst[resolve].processed:
                                                if p[0] <= prev_send + cost_matrix[resolve][prev_1.ndid] and p[1] > prev_send + cost_matrix[resolve][prev_1.ndid]:
                                                    flg = False
                                                    break

                                            

                                            if flg:
                                                
                                                tmp = [0]
                                                nxt = mb1.path[0]
                                                counter = 0
                                                idx_id = None
                                                resolve_id = None
                                                while nxt != None:
                                                    counter+=1
                                                    if nxt == idx:
                                                        idx_id = counter
                                                    if nxt == resolve:
                                                        resolve_id = counter
                                                    tmp.append(nxt)
                                                    nxt = mb1.path.get(nxt)
                                                tmp[idx_id] = resolve
                                                tmp[resolve_id] = idx
                                                if swaps.get(mb1.uniqid) == None:
                                                    swaps[mb1.uniqid] = []
                                                swaps[mb1.uniqid].append(idx)
                                                mb1.path.clear()
                                                mb1.invpath.clear()
                                                swaps[mb1.uniqid].append(resolve)
                                                for k in range(len(tmp) - 1):

                                                    mb1.path[tmp[k]] = tmp[k + 1]
                                                    mb1.invpath[tmp[k + 1]] = tmp[k]
                                                break
                                        
                                        
                                        

                                    


                                
                        tmp_cst = deepcopy(cp_nds)
                        for i,mb in enumerate(curr_schedule):
                            mb = deepcopy(mb)
                            mb.tm = 0
                            tmp_cst[0].receive(mb)
                        for tm in range(900):
                                for idx in range(len(locations)):
                                    tmp_cst[idx].send(cost_matrix,tmp_cst,tm)
                        largest = 0
        
                        for v in tmp_cst[0].received_sent.values():
                            largest = max(v[1],largest)
                        if best_val == None:
                            best_val = largest
                            flg = True
                        elif largest < best_val:
                            # print("IMPROVEMENT",largest,best_val)
                            # exit()
                            best_schedule = curr_schedule
                            best_val = largest
                            flg = True
                        else:
                            break
                    microbatches_scheduled = best_schedule
                                
                        
                        
                
            
            elif strt == "adaptive--":
                pth_len = floor(3*(len(locations) - 1)/4)
                rmv = len(locations) - pth_len
                
                dp = [[(BIG_WEIGHT,-1,-1, -1, 0,0) for _ in range(pth_len) ] for _ in range(rmv + 1)]
                for y in range(rmv + 1):
                    for x in range(pth_len):
                        if y+x in must_include:
                            
                            dp[y][x] = (0,-1,-1, -1, 0)
                dp[0][0] = (computational_cost[locations[0]],0,0,-1,0,0)
                for x in range(pth_len):
                    for y in range(rmv + 1):
                        if y == 0 and x == 0:
                            continue

                        dp[y][x] = best_from(dp,y,x,cost_matrix,computational_cost,locations)
                        
                        # print(y,x,dp[y][x])
                # print(dp)
                best_path = []
                x = pth_len-1
                best_val = 100000000
                y = 0
                for ty in range(rmv + 1):
                    last_skipped = dp[ty][x][3]
                    # print(ty,x,"last skipped was",last_skipped)
                    vl = dp[ty][x][0]
                    for nd in range(ty+x+1,len(locations)):
                        if last_skipped != -1:
                            vl += cost_matrix[last_skipped][nd]
                        # print(ty,x,"need to also skip",nd)
                        last_skipped = nd
                    if vl < best_val:
                        best_val = vl
                        y = ty
                print(best_val)

                while y != 0 or x != 0:
                    best_path.append((y,x))
                    ret = dp[y][x]
                    # print("we had skipped",ret[3])
                    y = ret[1]
                    x = ret[2]
                print(best_path)
                best_path.reverse()
                tmp_path = [0]
                
                for t in best_path:
                    tmp_path.append(t[0]+t[1])
                print(tmp_path)
                act_cost = computational_cost[locations[tmp_path[len(tmp_path)-1]]]
                for k in range(len(tmp_path)-1):
                    act_cost += cost_matrix[tmp_path[k]][tmp_path[k+1]] + computational_cost[locations[tmp_path[k]]]
                print("ACTUAL COST: ",act_cost)
                for k in range(len(tmp_path) - 1):

                    path[tmp_path[k]] = tmp_path[k + 1]
                    invpath[tmp_path[k + 1]] = tmp_path[k]
                must_include += [k for k in range(1,len(locations)) if k not in tmp_path]
                # print(must_include)
                tmp_cst = deepcopy(cp_nds)
                mb = MB(i,path,invpath)
                mb.tm = i*tmp_cst[0].comp
                tmp_cst[0].receive(mb)
                print(path)
                
                mb = MB(i,path,invpath)
                microbatches_scheduled.append(mb)
                if i == 2:
                    microbatches_scheduled.reverse()
                
        
            # print(path)
            # for k i
        for mb in microbatches_scheduled:
            
            mb.tm = 0
            
            nds[0].receive(mb)
        

        for tm in range(900):
            for idx in range(len(locations)):
                nds[idx].send(cost_matrix,nds,tm)
        largest = 0
        if strt != "hybrid":
            assert len(nds[0].received_sent.values()) == 3
        
        for v in nds[0].received_sent.values():
            largest = max(v[1],largest)

        print("LARGEST", largest)
        if strt == "normal":
            normal_tm = convergence[strt]*largest
            
        # print(strt)
        if strt == "hybrid":
            # if results["round robin"][-1] < results["adaptive"][-1]:
            #     exit()
            results[strt].append(min(results["round robin"][-1],results["adaptive"][-1]))
        # elif strt == "adaptive":
        #     results[strt].append(results["round robin"][-1])
        else:
            
            results[strt].append(convergence[strt]*largest/normal_tm)
            
            if results[strt][-1] > 2:
                print("WJAT?",strt,normal_tm,largest,results[strt])
                print(path)
                exit()
                break
            
    
import matplotlib.pyplot as plt
import numpy as np



print(results["round robin"])

plt.figure(figsize =(10, 7))
data =[]
# plt.ylim(bottom=0)
for k,v in results.items():
    
    data.append(np.array(v))
plt.boxplot(data,tick_labels=list(results.keys()))
plt.ylabel("Time to reach desired")
plt.savefig(f"schedulers.pdf")
plt.show()
# [0, 2, 3, 4, 6]
# 5 2
# [0, 1, 2, 3, 5]
# 5 2
# [0, 1, 4, 5, 6]
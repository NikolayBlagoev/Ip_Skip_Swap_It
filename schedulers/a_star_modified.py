from .graph import *
from typing import Dict, List, Union, Tuple, Any
import numpy as np
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class a_star_item_modified:
    dist: int
    time: int
    reachedfrom: int=field(compare=False)
    idx: int=field(compare=False)
    edg: Edge=field(compare=False)
    back: bool=field(compare=False)
    path: List=field(compare=False)

class Agent():
    def __init__(self,idx,start_idx,dt = 0):
        self.idx = idx
        self.start_idx = start_idx
        self.dt = dt


class Conflict():
    def __init__(self, agidx: int, ndix: int,tmstart: int, tmend: int, tp):
        self.type = tp
        self.agidx = agidx
        self.ndix = ndix
        self.tmstart = tmstart
        self.tmend = tmend
        self.str = str(agidx) + str(ndix) + str(tmstart) + str(tmend)
        
    
    
   

def a_star_modified(g: Graph, start_idx, heuristic, agidx, dt = 0, conflicts: List[Conflict] = [], visitable = dict(), visitable_stages = dict(), lng = 2):
    h: List[a_star_item_modified] = []
    
    lng -= 1
    # print("---",agidx,"---")
    heapq.heapify(h)
    heapq.heappush(h,a_star_item_modified(dt,dt,start_idx,start_idx,None,False,[]))
    
    def conflict_check(conflict: Conflict, time, weight):
        if conflict.type != 3:
            return time
        # print("TYPE 3?",conflict.agidx,conflict.ndix,time ,weight,conflict.tmstart,conflict.tmend)
        if (time >= conflict.tmstart and time <= conflict.tmend) or (time + weight >= conflict.tmstart and time + weight <= conflict.tmend):
        
            # print("TYPE 3!!!",conflict.agidx,conflict.ndix,time ,weight,conflict.tmstart,conflict.tmend)
            return conflict.tmend
        elif time <= conflict.tmstart and time + weight >= conflict.tmend:
            return conflict.tmend
        else:
            return time

    def reconstruct_path(path_edges: List[Edge],frm,dt):
        path: List[Tuple[idx,Edge]] = []
        el: Tuple[idx,Edge]= (frm,None)
        node_visits = []
        path.append(el)
        curr = frm
        for edg in path_edges:
            if edg.n1.idx == curr:
                path.append((edg.n2.idx , edg))
                curr = edg.n2.idx
            else:
                path.append((edg.n1.idx , edg))
                curr = edg.n1.idx
            
        
        t = dt
        prv = None

        for el,edg in path:
            max_offset = t
            if edg:
                max_offset += edg.w
            flg = True
            while flg:
                flg = False
                for c in conflicts:
                    
                    if c.agidx == agidx and c.ndix == el:
                        tmp_m = max_offset
                        max_offset = max(conflict_check(c,max_offset,g.nodes[el].weight),max_offset)
                        if max_offset > tmp_m:
                            flg = True

                        
            
            t = max_offset
            
            t += g.nodes[el].weight
            node_visits.append((el,max_offset,t))
        

        path.reverse()
        flg = True
        for el,edg in path:
            if flg:
                t += edg.w
                flg = False
                continue
            max_offset = t
            if not prv:
                
                prv = el
            else:
                prv = el
            flg = True
            while flg:
                flg = False
                for c in conflicts:
                    
                    if c.agidx == agidx and c.ndix == el:
                        tmp_m = max_offset
                        max_offset = max(conflict_check(c,max_offset,2*g.nodes[el].weight),max_offset)
                        if max_offset > tmp_m:
                            flg = True
            
            t = max_offset
            
            t += 2*g.nodes[el].weight
            node_visits.append((el,max_offset,t))
            if edg:
                t+=edg.w
            



        return node_visits,t
    prv_dist = 0
    while len(h) > 0:
        el = heapq.heappop(h)
        t = el.dist
        assert t >= prv_dist
        prv_dist = t
        t = el.time
        if visitable[agidx][el.idx] == 0:
            # print("cannot visit",el.idx)
            continue
        # print(t)
        max_offset = t
        if el.idx == start_idx and el.back:
            
            return el.time, el.path, agidx
        has_conflict = True
        while has_conflict:
            has_conflict = False
            for c in conflicts:
                
                if c.agidx == agidx and c.ndix == el.idx:
                    
                    max_offset = max(conflict_check(c,max_offset,g.nodes[el.idx].weight),max_offset)
                    

        if max_offset > t:
            if max_offset == float("inf"):
                continue
            heapq.heappush(h,a_star_item_modified(max_offset - t + el.dist,max_offset, el.reachedfrom,el.idx,el.edg,el.back,el.path))
            continue
        el.time += g.nodes[el.idx].weight
        
        
        if el.idx == start_idx and len(h) != 0:
            
            
            
            path,t = reconstruct_path(el.path,start_idx,dt)
            # print("END REACHED",t,agidx)
            heapq.heappush(h,a_star_item_modified(t,t,None,start_idx,None,True,path))
            continue
        number_of_swaps = 0
        frontier = 0
        partitions = [0]
        curr = start_idx
        for edg in el.path:
            
            if edg.n1.idx == curr:
                
                frontier = max(frontier,edg.n2.properties["partition"])
                
                curr = edg.n2.idx
            else:
                frontier = max(frontier,edg.n1.properties["partition"])
                
                curr = edg.n1.idx
            
            if g.nodes[curr].properties["partition"] < frontier:
                number_of_swaps += (frontier-g.nodes[curr].properties["partition"])
            partitions.append(g.nodes[curr].properties["partition"])
        if number_of_swaps > 4:
            
            continue


        for edg in g.nodes[el.idx].incident_edges.values():
            # print(len(g.nodes[el.idx].incident_edges.values()))
            if edg.directed:
                if edg.n1.idx != el.idx:
                    continue
                if len(el.path) < lng and edg.n2.idx == start_idx:
                    continue
                elif len(el.path) == lng and edg.n2.idx == start_idx:
                    heapq.heappush(h,a_star_item_modified(el.time + edg.w,el.time + edg.w, el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
                    continue
                if len(el.path) == lng:
                        continue
                if visitable_stages[agidx][g.nodes[edg.n2.idx].properties["partition"]] == 0:
                    continue
                if number_of_swaps > 3 and g.nodes[edg.n2.idx].properties["partition"] < frontier:
                    continue
                if g.nodes[edg.n2.idx].properties["partition"] in partitions:
                    
                    continue
                if abs(g.nodes[edg.n2.idx].properties["partition"] - frontier) > 3:
                    continue
                heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.time  + edg.w,el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
            else:
                if edg.n1.idx == el.idx:
                    if len(el.path) < lng and edg.n2.idx == start_idx:
                        continue
                    elif len(el.path) == lng and edg.n2.idx == start_idx:
                        heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.time + edg.w, el.idx,start_idx,edg,False,el.path.copy() + [edg]))
                        continue
                    if len(el.path) == lng:
                        continue
                    if number_of_swaps > 3 and g.nodes[edg.n2.idx].properties["partition"] < frontier:
                        continue
                    if visitable_stages[agidx][g.nodes[edg.n2.idx].properties["partition"]] == 0 :
                        # print("cannot visit",edg.n2.idx)
                        continue
                    if g.nodes[edg.n2.idx].properties["partition"] in partitions:
                        # print("cannot visit",edg.n2.idx,g.nodes[edg.n2.idx].properties["partition"],partitions)
                        continue
                    if abs(g.nodes[edg.n2.idx].properties["partition"] - frontier) > 3:
                        # print("gap too big",frontier,g.nodes[edg.n2.idx].properties["partition"],partitions)
                        continue
                    heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.time + edg.w,el.idx,edg.n2.idx,edg,False,el.path.copy() + [edg]))
                else:
                    if len(el.path) < lng and edg.n1.idx == start_idx:
                        continue
                    elif len(el.path) == lng and edg.n1.idx == start_idx:
                        heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.time + edg.w,el.idx,start_idx,edg,False,el.path.copy() + [edg]))
                        continue
                    if len(el.path) == lng:
                        continue
                    if number_of_swaps > 3 and g.nodes[edg.n1.idx].properties["partition"] < frontier:
                        continue
                    if visitable_stages[agidx][g.nodes[edg.n1.idx].properties["partition"]] == 0:
                        # print("cannot visit",edg.n1.idx)
                        continue
                    if g.nodes[edg.n1.idx].properties["partition"] in partitions:
                        # print("cannot visit",edg.n1.idx,g.nodes[edg.n1.idx].properties["partition"],partitions)
                        continue
                    if abs(g.nodes[edg.n1.idx].properties["partition"] - frontier) > 3:
                        # print("gap too big",frontier,g.nodes[edg.n1.idx].properties["partition"],partitions)
                        continue
                    
                    heapq.heappush(h,a_star_item_modified(el.time + edg.w, el.time + edg.w,el.idx,edg.n1.idx,edg,False,el.path.copy() + [edg]))
    return None


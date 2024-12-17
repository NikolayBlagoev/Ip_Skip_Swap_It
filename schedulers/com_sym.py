from collections import OrderedDict
from math import floor
from copy import deepcopy
class MB(object):
    def __init__(self, uniqid, path, first_node):
        self.tm = 0
        self.uniqid = uniqid
        self.path = path
        self.processed = False
        self.back = False
        self.first_node = first_node
        self.one_time = False
        self.invpath = {}
        for k,v in path.items():
            self.invpath[v] = k
        # print(self.path)
        # print(self.invpath)

class ComNode(object):
    def __init__(self,comp, ndid, memory):
        
        self.comp = comp
        self.ndid = ndid
        self.received = []
        self.backreceived = []
        self.processed = []
        self.collisions = []
        self.process_at = []
        self.memory = memory
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
            if mb.processed:
                continue
            if not mb.back and self.memory <= 0 and not mb.one_time:
                # print(self.ndid,"CANNOT TAKE IN",self.memory)
                
                mb.tm += 1
                continue
            # print(self.ndid,"processing ",mb.uniqid," at ",mb.tm, mb.back)
            
            nxt = None
            if self.ndid in mb.path:
                
                nxt = mb.path[self.ndid]
            
            
            if nxt != None and not mb.back and not mb.one_time:
                
                # print(self.ndid, "to",nxt)
                self.processed.append((mb.tm, mb.tm + self.comp, mb.back, mb.uniqid))

                self.received_sent[mb.uniqid] = (mb.tm, mb.tm + self.comp)
                tmp = deepcopy(mb)
                self.memory -= 1
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            
            elif not mb.back and self.ndid in mb.invpath and not mb.one_time:
                
                self.processed.append((mb.tm, mb.tm + self.comp, mb.back, mb.uniqid))

                self.received_sent[mb.uniqid] = (mb.tm, mb.tm + self.comp)
                nxt = mb.first_node
                # print("last node send to", nxt)
                tmp = deepcopy(mb)
                tmp.first_node = self.ndid
                tmp.one_time = True
                self.memory -= 1
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            elif mb.one_time:
                # print("one time send it back")
                self.processed.append((mb.tm, mb.tm + self.comp, mb.back, mb.uniqid))

                self.received_sent[mb.uniqid] = (mb.tm, mb.tm + self.comp)
                nxt = mb.first_node
                tmp = deepcopy(mb)
                tmp.first_node = self.ndid
                tmp.back = True
                tmp.one_time = False
                tmp.tm = tmp.tm + self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)

            elif self.ndid in mb.invpath:
                # print("BACK")
                self.processed.append((mb.tm, mb.tm + 2*self.comp, mb.back, mb.uniqid))

                self.received_sent[mb.uniqid] = (mb.tm, mb.tm + 2*self.comp)
                nxt = mb.invpath[self.ndid]
                self.memory += 1
                # print(self.ndid, "to",nxt)
                tmp = deepcopy(mb)
                tmp.back = True
                tmp.tm = tmp.tm + 2*self.comp + dl[self.ndid][nxt]
                nds[nxt].receive(tmp)
            else:
                # print(self.ndid,"RECEIVED")
                self.processed.append((mb.tm, mb.tm + 2*self.comp, mb.back, mb.uniqid))

                self.received_sent[mb.uniqid] = (mb.tm, mb.tm + 2*self.comp)
                self.memory += 1
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
            mb.processed = True

def run_simulation(nds, partitions, cost_matrix):
    for tm in range(10000):
        for idx in range(len(nds)):
            nds[idx].send(cost_matrix,nds,tm)
    largest = 0
    # for nd, v in nds.items():
    #     print(nd,"received",v.received_sent)
    for nd in partitions[0]:    
        # print(nd,"received", len(nds[nd].received_sent))
        for v in nds[nd].received_sent.values():
            
            largest = max(v[1],largest)
    return largest
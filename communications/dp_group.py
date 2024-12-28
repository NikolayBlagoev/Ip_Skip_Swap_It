from torch.distributed import new_group
from torch.distributed import init_process_group
import os
from time import sleep
def initialise_communication(partitions, pid, addr, world_size, delay_map):
    print(addr)
    if addr == "127.0.0.1":
        addr = "localhost"
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = "9010"
    init_process_group(backend="gloo", rank=pid, world_size=world_size)
    return DP_Group(partitions,pid, delay_map)

class DP_Group(object):
    def __init__(self, partitions, pid, delay_map):
        self.group = None
        self.pid = pid
        self.delay_map = delay_map
        
        self.g_size = 0
        
        self.worst_band = 100
        for p in partitions:
            if pid in p:
                # print("IM IN GROUP",p,pid)
                self.group = new_group(p, backend="gloo")
                for pid2 in p:
                    if pid2 == pid:
                        continue
                    b = delay_map[pid][pid2]
                    self.worst_band = min(b,self.worst_band)
                self.g_size = len(p)
            else:
                new_group(p, backend="gloo")
    
    def compensate(self, t1, t2, sz):
        desired_time = sz/(1024**3*self.worst_band)
        if t2 - t1 < desired_time:
            sleep(desired_time - (t2-t1))

    
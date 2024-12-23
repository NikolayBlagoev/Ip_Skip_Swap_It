from torch.distributed import new_group
from torch.distributed import init_process_group
import os
def initialise_communication(partitions, pid, addr, world_size):
    print(addr)
    if addr == "127.0.0.1":
        addr = "localhost"
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = "9010"
    init_process_group(backend="mpi", rank=pid, world_size=world_size)
    return DP_Group(partitions,pid)

class DP_Group(object):
    def __init__(self, partitions, pid):
        self.group = None
        self.pid = pid
        self.g_size = 0
        for p in partitions:
            if pid in p:
                self.group = new_group(p, backend="mpi")
                self.g_size = len(p)
            else:
                new_group(p, backend="mpi")
    

    
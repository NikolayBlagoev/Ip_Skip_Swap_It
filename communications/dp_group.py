from torch.distributed import new_group
from torch.distributed import init_process_group
def initialise_communication(partitions, pid, addr):
    os.environ["MASTER_ADDR"] = addr[0]
    os.environ["MASTER_PORT"] = str(addr[1])
    init_process_group(backend="gloo")
    return DP_Group(partitions,pid)

class DP_Group(object):
    def __init__(self, partitions, pid):
        self.group = None
        self.pid = pid
        for p in partitions:
            if pid in p:
                self.group = new_group(p, backend="gloo")
            else:
                new_group(p, backend="gloo")
    

    
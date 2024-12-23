import random
import time
from deccom.protocols.peerdiscovery.kademliadiscovery import KademliaDiscovery
from sys import argv
import asyncio
from deccom.cryptofuncs.hash import SHA256
from deccom.nodes import StreamNode, Node
from deccom.protocols.defaultprotocol import DefaultProtocol
from deccom.peers import Peer
from deccom.protocols.streamprotocol import StreamProtocol
from communications.llm_subp import run_p
from multiprocessing import Lock, Process, Queue, current_process
import json
from pprint import pprint
from communications.pp_protocol import PPProtocl
from schedulers.communication_costs import *
seq_l = 128
n_layers = 4
batch_size = 1
dmodel = 256
num_heads = 6
multiple_of = 32

if __name__ == '__main__':
    curr_id = int(argv[1])
    setting = argv[2]

    communication_distribution = argv[3]
    loop = asyncio.new_event_loop()
    with open("communication.json", 'r') as file:
        config = json.load(file)
    
    loc = id_to_loc(curr_id, communication_distribution)
    print(loc)
    compute_time = get_computations(communication_distribution)[loc]
    world_size = 0
    own_stage = -1
    rank_order = 0
    partitions = config["partitions"]
    memory = config["memory"]
    send_mbs = 2
    if setting == "baseline":
        send_mbs = config["baseline-mb-count"]
    for idx, v in enumerate(partitions):
        if curr_id in v:
            assert own_stage == -1
            own_stage = idx
            for idx2, v2 in enumerate(v):
                if v2 == curr_id:
                    rank_order = idx2
                    break
        world_size += len(v)
    
    assert own_stage != -1
    def commfunc(bid, ndkey):
        if setting == "baseline":
            if own_stage == len(partitions) - 1:
                return None
            if len(partitions[own_stage + 1]) <= rank_order:
                return None
            # with open(f"log_stats_proj_2_{curr_id}.txt", "a") as log:
            #     log.write(f"Paritions {partitions[own_stage + 1]}, {rank_order}, {own_stage}\n")
            return partitions[own_stage + 1][rank_order]
            
    while True:
        my_peer  = Peer(None, pub_key=str(curr_id))
        
        port = None

        protocol = DefaultProtocol()
        gossip = KademliaDiscovery([],interval=30, always_split = True)
        gossip.set_lower(protocol)
        stream = StreamProtocol(False)
        stream.set_lower(gossip)
        n = Peer(("127.0.0.1", 10015))
        if curr_id != 0:
            gossip.bootstrap_peers.append(n)
            time.sleep(1)
        



        queue_in = Queue(1024)
        queue_out = Queue(1024)
        
        subprocess = Process(target=run_p,args=(n.addr[0],partitions,queue_out,queue_in,curr_id,own_stage,seq_l,n_layers,batch_size,dmodel,multiple_of,num_heads,memory,compute_time,"cuda")) 
        trainingp = PPProtocl(world_size, own_stage, commfunc, None, len(partitions[0]), memory, queue_in, queue_out, subprocess, MB_SEND_COUNT=send_mbs, dp_order=rank_order)
        trainingp.set_lower(stream)
        subprocess.start()
        
        me = StreamNode(my_peer , trainingp,ip_addr="127.0.0.1", port = 10015 if curr_id == 0 else port)
        # print( "TCP", me.tcp_port)

        
        print("run...")
        
        loop.run_until_complete(me.listen())
        loop.run_forever()
        

from dataclasses import dataclass

import os
import random
from typing import Callable
from deccom.peers.peer import Peer
from deccom.protocols.abstractprotocol import AbstractProtocol
from deccom.protocols.wrappers import *
from datetime import datetime
import asyncio
from traceback import print_exception, format_exc
from llm_subp import *
from deccom.cryptofuncs.hash import SHA256

    

class TrainingProtocol(AbstractProtocol):
    required_lower = AbstractProtocol.required_lower + \
        ["find_peer", "get_peer", "get_peers", "connected_callback","disconnected_callback"]
    
    GRADIENT = int.from_bytes(b'\x17',byteorder="big")
    def __init__(self, world_size: int, k: int, queue_in: Queue, queue_out: Queue, subprocess:Process, submodule=None, callback: Callable[[tuple[str, int], bytes], None] = lambda : ...):
        
        super().__init__(submodule, callback)
        self.group = []
        self.k = k
        self.tag = 0
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.subprocess = subprocess
        self.disconnected_callback = lambda *args : ...
        self.connected_callback = lambda *args : ...
        self.world_size = world_size
        self.gradients_received = dict()
        self.queue_reader = None
        self.iteration = 0
        self.aggregated = 0
        self.can_acrue = False
        self.outstanding = dict()
        self.model_description = ["conv1","bn1","relu","maxpool","layer1","layer2","layer3","layer4","avgpool","fc"]
    
    
        
    @bindto("open_connection")
    async def _lower_open_connection(self, remote_ip, remote_port, node_id: bytes):
        return
    
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None
    
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None


    async def start_iteration(self):
        if self.peer.pub_key != "0":
            await asyncio.sleep(3)
        for i in range(3):
            tag = int(self.tag).to_bytes(4,byteorder="big")
            self.tag += 1
            if self.iteration % 10 != 7:
                for j in range(4):
                    if (i+1) == j and self.iteration % 10 != 0:
                        tag += int(0).to_bytes(1,byteorder="big")
                    else:
                        tag += int(1).to_bytes(1,byteorder="big")
            else:
                for j in range(4):
                    if (i+1) == j or j == 1:
                        tag += int(0).to_bytes(1,byteorder="big")
                    else:
                        tag += int(1).to_bytes(1,byteorder="big")

            self.queue_out.put(Start(tag), True)

    async def start(self, p: Peer):
        await super().start(p)
        
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
            log.write(f"===={self.peer.id_node} {self.peer.tcp} STARTING===\n")
        
        loop = asyncio.get_event_loop() 
        self.queue_reader = loop.create_task(self.read_from_queue())
        if self.peer.pub_key == "0":
            loop.create_task(self.start_iteration())
        

        
            

    async def read_from_queue(self):
        while self.started:
            while self.queue_in.empty() and self.started:
                await asyncio.sleep(0.5)
            if not self.started:
                with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                    log.write(f"====CLOSING???===\n")
                return
            task = self.queue_in.get(True)
            try:
                if isinstance(task, Forward):
                    path = task.tag[4:]
                    j = int(self.peer.pub_key) + 1
                    nxt = None
                    while j < len(path):
                        if path[j] == 1:
                            nxt = j
                            break
                        j += 1
                    if nxt == None:
                        nxt = 0
                        j = nxt
                    with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                        log.write(f"SENDINGTO {nxt} {path} {path[j]}\n")
                    p = await self._lower_find_peer(SHA256(str(nxt)))
                    asyncio.get_event_loop().create_task(self.send_stream(p.id_node,task.mb,task.tag))
                    continue
                elif isinstance(task, Backward):
                    prev = self.outstanding[task.tag]
                    asyncio.get_event_loop().create_task(self.send_stream(prev,task.mb,task.tag))
                    continue
            except Exception as e:
                with open(f'log{self.peer.pub_key}.txt', 'a') as f:
                    f.write(str(e))
                    f.write("!!!!!!!!!!!!!!!\n")
                    f.write(format_exc())
                    


    @bindfrom("connected_callback")
    def peer_connected(self, nodeid, peer: Peer):
        # print("NEW PEER")
        with open(f"log_stats_proj_2_{self.peer.pub_key}.txt", "a") as log:
                log.write(f"CONNECTED WITH {peer.pub_key}\n")
       
        return self.connected_callback(nodeid,peer)
        
    
    
    
    def process_datagram(self, addr: tuple[str, int], data: bytes):
        self.aggregate()
        return

    
    def aggregate(self):
        self.outstanding.clear()
        self.iteration += 1
        if self.peer.pub_key == "0":
            for j in range(1,4):
                p = self._lower_get_peer(SHA256(str(j)))
                asyncio.get_event_loop().create_task(self.send_datagram(b'x\00',p.addr))
        self.queue_out.put(Aggregate(b'x\01'),True)
        if self.peer.pub_key == "0":
            
            asyncio.get_event_loop().create_task(self.start_iteration())

        return


    @bindfrom("stream_callback")
    def process_data(self, data:bytes, nodeid, addr):
        tag = data[0:8]
        path = tag[4:]
        data = data[8:]
        tag = bytes(tag)
        
        try:
            if self.outstanding.get(tag) != None:
                self.queue_out.put(Backward(data,tag),True)
                if self.peer.pub_key == "0":
                    self.aggregated += 1
                    if self.aggregated == 3:
                        self.aggregated = 0
                        self.aggregate()

            else:
                self.outstanding[tag] = nodeid
                if self.peer.pub_key == "0":
                    self.queue_out.put(Loss(data,tag),True)
                else:
                    self.queue_out.put(Forward(data,tag),True)
        except Exception as e:
                with open(f'log{self.peer.pub_key}.txt', 'a') as f:
                    f.write(str(e))
                    f.write("!!!!!!!!!!!!!!!\n")
                    f.write(format_exc())
    async def send_stream(self, node_id, data, tag=b'x\00'):
        
        await self._lower_find_peer(bytes(node_id))
        p = self._lower_get_peer(node_id)
        await self._lower_open_connection(p.addr[0], p.tcp, p.id_node, port_listen = 0)
        
        await self._lower_send_stream(node_id, tag+data)
        return
    
    @bindto("open_connection")
    async def _lower_open_connection(self, remote_ip, remote_port, node_id: bytes):
        return
    @bindto("send_stream")
    async def _lower_send_stream(self, node_id, data):
        return
    @bindto("get_peer")
    def _lower_get_peer(self, node_id)->Peer:
        return None
    @bindto("find_peer")
    async def _lower_find_peer(self, id: bytes) -> Peer:
        return None
    def get_lowest_stream(self):
        submodule = self.submodule
        while submodule != None and not hasattr(submodule, "get_lowest_stream") and hasattr(submodule, "submodule") :
            submodule = submodule.submodule
        if submodule != None and hasattr(submodule, "get_lowest_stream"):
            ret = submodule.get_lowest_stream()
            if ret == None:
                return self
            else:
                return ret
        else:
            
            return self
    async def stop(self):
        
        await super().stop()
        
        self.queue_in.close()
        self.queue_out.close()
                

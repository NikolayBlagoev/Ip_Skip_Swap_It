# Training

Start training with

```
./run.sh 0 20
```

Which will start all nodes from 0 to 20 on this machine


# Communication organisation

The [create_schedule.py](/create_schedule.py) partitions the nodes across stages and estimates optimal communication paths for the collision aware and non-collision aware schedules.

# Distributed raining framework

All files related to distributed communication are located in [communications](/communications/). The framework is based on DecCom for the p2p synchronisation (when to do aggregating, when someone should receive from someone, etc).

The [llm_subp.py](/communications/llm_subp.py) file hosts a subprocess which holds the actual model, performs the training, and communicates the tensors.
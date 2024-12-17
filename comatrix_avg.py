
co_matrix = {}
for v in ["o_proj","gate_proj","up_proj", "down_proj", "k_proj", "v_proj", "q_proj"]:
    co_matrix[v]=[[0 for _ in range(32)] for _ in range(16)]

with open(f"avg.txt","r") as fd:
        
        for ln in fd.readlines():
            ln = ln.split(" ")
            ln = list(filter(lambda x: x != "", ln))
            # print(ln)
            if ln[0] != "AVG":
                continue
            ln = ln[1:]
            if ln[0] == "L1":
                continue
            pt = ln[2]
            me = int(ln[5])
            them = int(ln[6])
            diff = float(ln[7][7:-1])
            co_matrix[pt][abs(them-me)][me] = diff
            
            # print(diff)

import numpy as np 
import matplotlib.pyplot as plt

for k,v in co_matrix.items():
    plt.imshow(np.array(v))
    plt.colorbar()
    # /plt.legend()
    plt.show()

co_matrix = {}
for v in ["o_proj","gate_proj","up_proj", "down_proj", "k_proj", "v_proj", "q_proj"]:
    co_matrix[v]=[[0 for _ in range(32)] for _ in range(32)]

with open(f"out2.txt","r") as fd:
        
        for ln in fd.readlines():
            ln = ln.split(" ")
            ln = list(filter(lambda x: x != "", ln))
            # print(ln)
            if ln[0] == "AVG":
                continue
            if ln[0] == "L1":
                continue
            them = None
            if ln[0] == "RANDOM":
                ln = ln[1:]
                them = -1
            print(ln)
            pt = ln[2]
            me = int(ln[5])
            if them != -1:
                them = int(ln[6])
            else:
                them = me
            diff = float(ln[7][7:-1])
            co_matrix[pt][me][them] = diff
            co_matrix[pt][them][me] = diff
            # print(diff)

import numpy as np 
import matplotlib.pyplot as plt

for k,v in co_matrix.items():
    plt.imshow(np.array(v))
    plt.colorbar()
    plt.title(f"{k} l2 similarity")
    # /plt.legend()
    plt.savefig(f"figs/comatrix/{k}_comatrix.pdf")
    plt.show()